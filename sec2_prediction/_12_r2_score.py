
# 1. libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 2. preprocessing
# 2.1. import data
df = pd.read_csv("../data/bilkav/sec2_prediction/maaslar.csv")

x = df.iloc[:, 1:2]
y = df.iloc[:, 2:]

X = x.values
Y = y.values

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# visualization
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.show()

# polynomial regression
# 2. degree
poly2 = PolynomialFeatures(degree=2)
x_poly2 = poly2.fit_transform(X)
poly2_reg = LinearRegression()
poly2_reg.fit(x_poly2, Y)

# visualization
plt.scatter(X, Y, color='red')
plt.plot(X, poly2_reg.predict(poly2.fit_transform(X)), color='blue')
plt.show()

# polynomial regression
# 4. degree
poly4 = PolynomialFeatures(degree=4)
x_poly4 = poly4.fit_transform(X)
poly4_reg = LinearRegression()
poly4_reg.fit(x_poly4, Y)

# visualization
plt.scatter(X, Y, color='red')
plt.plot(X, poly4_reg.predict(poly4.fit_transform(X)), color='blue')
plt.show()


# SVR
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

# decision tree
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X, Y)

# visualization
plt.scatter(X, Y, color='red')
plt.plot(X, dt_reg.predict(X), color='blue')
plt.show()


# random forest
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


print('Linear Reg R2 score: ')
print(r2_score(Y, lin_reg.predict(X)))

print('Polynomial Degree 2 Reg R2 score: ')
print(r2_score(Y, poly2_reg.predict(poly2.fit_transform(X))))

print('Polynomial Degree 4 Reg R2 score: ')
print(r2_score(Y, poly4_reg.predict(poly4.fit_transform(X))))

print('SVR Reg R2 score: ')
print(r2_score(y_scaled, svr_reg.predict(x_scaled)))

print('Decision Tree Reg R2 score: ')
print(r2_score(Y, dt_reg.predict(X)))

print('Random Forest Reg R2 score: ')
print(r2_score(Y, rf_reg.predict(X)))
