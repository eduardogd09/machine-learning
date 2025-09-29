from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2], [2,1], [5,8], [6,7], [8,6]])  # Your data
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)
print(kmeans.labels_)  # Cluster assignments: e.g., [0, 0, 1, 1, 1]
print(kmeans.cluster_centers_)  # Centroids