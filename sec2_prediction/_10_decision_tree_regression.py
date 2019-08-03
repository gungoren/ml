
# 1. libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# 2. preprocessing
# 2.1. import data
df = pd.read_csv("../data/bilkav/sec2_prediction/maaslar.csv")

x = df.iloc[:, 1:2]
y = df.iloc[:, 2:]

X = x.values
Y = y.values

dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X, Y)

# visualization
plt.scatter(X, Y, color='red')
plt.plot(X, dt_reg.predict(X), color='blue')
plt.show()

print(dt_reg.predict([[11]]))
print(dt_reg.predict([[6.6]]))
