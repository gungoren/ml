
# 1. libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# 2. preprocessing
# 2.1. import data
df = pd.read_csv("../data/bilkav/sec2_prediction/maaslar.csv")

x = df.iloc[:, 1:2]
y = df.iloc[:, 2:]

X = x.values
Y = y.values

# linear regression
lr = LinearRegression()
lr.fit(X, Y)

# polynomial regression
# 2. degree
poly2 = PolynomialFeatures(degree=2)
x_poly2 = poly2.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(x_poly2, Y)

# 4. degree
poly4 = PolynomialFeatures(degree=4)
x_poly4 = poly4.fit_transform(X)
lr4 = LinearRegression()
lr4.fit(x_poly4, Y)

# visualization
plt.scatter(X, Y, color='red')
plt.plot(X, lr.predict(X), color='blue')
plt.show()

plt.scatter(X, Y, color='red')
plt.plot(X, lr2.predict(poly2.fit_transform(X)), color='blue')
plt.show()

plt.scatter(X, Y, color='red')
plt.plot(X, lr4.predict(poly4.fit_transform(X)), color='blue')
plt.show()
