from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd  
import matplotlib.pyplot as plt
from mmt import MiniMegaTortoraDataset
from ssaUtils import DiscreteWaveletTransform, pad_to_size_interpolate
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D  # Import necessary for 3D plotting

# Load dataset
satelliteNumber = [60, 160, 300]
trackSize = 500
mmt = MiniMegaTortoraDataset(satNumber=satelliteNumber, periodic=True)
x, y = mmt.load_data_new()

# Preprocess data
x = DiscreteWaveletTransform(x, wavelet='db20')
x = [pad_to_size_interpolate(array, trackSize) for array in x]
X = StandardScaler().fit_transform(x)

# Convert categorical labels to numeric values
le = LabelEncoder()
y_numeric = le.fit_transform(y)

# Apply t-SNE with 3 components
X_embedded = TSNE(n_components=3, perplexity=30, random_state=0).fit_transform(X)

# Detect and remove outliers
# Option 1: Z-score method
z_scores = np.abs(zscore(X_embedded, axis=0))
threshold = 3
outlier_mask = (z_scores < threshold).all(axis=1)

# Option 2: IQR method
# Q1 = np.percentile(X_embedded, 25, axis=0)
# Q3 = np.percentile(X_embedded, 75, axis=0)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# outlier_mask = np.all((X_embedded >= lower_bound) & (X_embedded <= upper_bound), axis=1)

# Filter data
X_embedded_filtered = X_embedded[outlier_mask]
y_numeric_filtered = y_numeric[outlier_mask]

# Plot the filtered data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    X_embedded_filtered[:, 0], 
    X_embedded_filtered[:, 1], 
    X_embedded_filtered[:, 2], 
    c=y_numeric_filtered, 
    cmap='viridis', 
    s=50
)

# Add color bar and labels
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, label='Labels')
ax.set_title('Time Series Data Visualization without Outliers')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')

# Create a legend with label names
handles, _ = scatter.legend_elements()
legend_labels = [le.inverse_transform([i])[0] for i in range(len(le.classes_))]
ax.legend(handles, legend_labels, title="Categories", loc="upper right", bbox_to_anchor=(1.3, 1))

plt.show()
