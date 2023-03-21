import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
