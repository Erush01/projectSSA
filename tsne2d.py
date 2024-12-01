

import numpy as np
from sklearn.manifold import TSNE
import pandas as pd  
import matplotlib.pyplot as plt
from mmt import MiniMegaTortoraDataset
from ssaUtils import DiscreteWaveletTransform,pad_to_size_interpolate
from sklearn import preprocessing
satelliteNumber=[60,160,300]
trackSize=1000
mmt=MiniMegaTortoraDataset(satNumber=satelliteNumber,periodic=True)
x,y=mmt.load_data_new()



x = DiscreteWaveletTransform(x,wavelet='sym5')
x=[pad_to_size_interpolate(array,trackSize) for array in x]


le = preprocessing.LabelEncoder()
y_numeric = le.fit_transform(y)  # Convert categories to unique integers

X = preprocessing.StandardScaler().fit_transform(x)
# Apply t-SNE
X_embedded = TSNE(n_components=2, perplexity=30,verbose=2).fit_transform(X)

# Plot with color-coded labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_numeric, cmap='viridis', s=50,alpha=0.5)
plt.colorbar(scatter, label='Labels')
plt.title('Time Series Data Visualization using t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Create a legend with label names
handles, _ = scatter.legend_elements()
legend_labels = [le.inverse_transform([i])[0] for i in range(len(le.classes_))]
plt.legend(handles, legend_labels, title="Categories", loc="upper right")

plt.show()