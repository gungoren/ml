
# 1. libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 2. preprocessing
# 2.1. import data
df = pd.read_csv("../data/bilkav/sec2_prediction/maaslar.csv")

x = df.iloc[:, 1:2]
y = df.iloc[:, 2:]

X = x.values
Y = y.values

rf_reg = RandomForestRegressor(random_state=0, n_estimators=10)
rf_reg.fit(X, Y)

Z = X + 0.5
K = X - 0.4

# visualization
plt.scatter(X, Y, color='red')
plt.plot(X, rf_reg.predict(X), color='blue')
plt.plot(X, rf_reg.predict(Z), color='green')
plt.plot(X, rf_reg.predict(K), color='yellow')
plt.show()

print(rf_reg.predict([[11]]))
print(rf_reg.predict([[6.6]]))
