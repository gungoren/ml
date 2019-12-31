import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("../data/bilkav/sec4_clustering/musteriler.csv")

X = df.iloc[:, 3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.title('WCSS')
plt.plot(range(1, 11), wcss)
plt.show()
