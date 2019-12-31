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

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=123)
y_predict = kmeans.fit_predict(X)
plt.title('KMeans')
plt.scatter(X[y_predict==0,0], X[y_predict==0,1], s=100, c='red')
plt.scatter(X[y_predict==1,0], X[y_predict==1,1], s=100, c='blue')
plt.scatter(X[y_predict==2,0], X[y_predict==2,1], s=100, c='green')
plt.scatter(X[y_predict==3,0], X[y_predict==3,1], s=100, c='yellow')
plt.show()

#AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_predict = ac.fit_predict(X)
print(y_predict)

plt.title('Ag.Cls')
plt.scatter(X[y_predict==0,0], X[y_predict==0,1], s=100, c='red')
plt.scatter(X[y_predict==1,0], X[y_predict==1,1], s=100, c='blue')
plt.scatter(X[y_predict==2,0], X[y_predict==2,1], s=100, c='green')
plt.scatter(X[y_predict==3,0], X[y_predict==3,1], s=100, c='yellow')
plt.show()


import scipy.cluster.hierarchy as shc
dendrogram = shc.dendrogram(shc.linkage(X, method='ward'))
plt.show()







