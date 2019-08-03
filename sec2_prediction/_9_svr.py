
# 1. libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# 2. preprocessing
# 2.1. import data
df = pd.read_csv("../data/bilkav/sec2_prediction/maaslar.csv")

x = df.iloc[:, 1:2]
y = df.iloc[:, 2:]

X = x.values
Y = y.values

# 4. attribute scaling
sc = StandardScaler()
x_scaled = sc.fit_transform(X)
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(Y)

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaled, y_scaled)

# visualization
plt.scatter(x_scaled, y_scaled, color='red')
plt.plot(x_scaled, svr_reg.predict(x_scaled), color='blue')
plt.show()

print(sc.inverse_transform(svr_reg.predict(sc.fit_transform([[11]]))))
print(sc2.inverse_transform(svr_reg.predict(sc2.fit_transform([[6.6]]))))
