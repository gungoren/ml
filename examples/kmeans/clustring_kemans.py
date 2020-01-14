#
#
# Mehmet Güngören (mehmetgungoren@lyrebirdstudio.net)
#
#
import pandas
import numpy
from matplotlib import pyplot
from sklearn.cluster import KMeans

data = pandas.read_csv('xclara.csv')

print(data.shape)
print(data.head(10))

v1 = data['V1'].values
v2 = data['V2'].values
x = data.values

pyplot.scatter(v1,v2,c='blue', s=5)
pyplot.show()

k3 = KMeans(n_clusters=3)
distribution = k3.fit_predict(x)

print(distribution)
print(set(distribution))

pyplot.scatter(v1,v2,c=distribution)
pyplot.show()

print(k3.cluster_centers_)