from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd  
import matplotlib.pyplot as plt
from mmt import MiniMegaTortoraDataset
from ssaUtils import DiscreteWaveletTransform,pad_to_size_interpolate
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D  # Import necessary for 3D plotting

satelliteNumber=[60,160,300]
trackSize=500
mmt=MiniMegaTortoraDataset(satNumber=satelliteNumber,periodic=True)
x,y=mmt.load_data_new()
classes=[[x] for x in np.unique(y)]

y=np.array(y)


x = DiscreteWaveletTransform(x, wavelet='db20')

x=[pad_to_size_interpolate(array,trackSize) for array in x]


X = StandardScaler().fit_transform(x)


# Convert categorical labels to numeric values
le = LabelEncoder()
y_numeric = le.fit_transform(y)  # Convert categories to unique integers
# Apply t-SNE with 3 components
X_embedded = TSNE(n_components=3, perplexity=30, random_state=0).fit_transform(X)

# Plot with color-coded labels in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y_numeric, cmap='viridis', s=50)

# Add color bar and labels
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, label='Labels')
ax.set_title('Time Series Data Visualization using t-SNE (3D)')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')

# Create a legend with label names
handles, _ = scatter.legend_elements()
legend_labels = [le.inverse_transform([i])[0] for i in range(len(le.classes_))]
ax.legend(handles, legend_labels, title="Categories", loc="upper right", bbox_to_anchor=(1.3, 1))

plt.show()
