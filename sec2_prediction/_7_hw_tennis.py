
# 1. libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.api as sm

# 2. preprocessing
# 2.1. import data
df = pd.read_csv("../data/bilkav/sec2_prediction/odev_tenis.csv")


# 2.3. transform categorical data
outlook = df.iloc[:,0:1]
hot = OneHotEncoder(categories='auto')
outlook = hot.fit_transform(outlook).toarray()

outlookDf = pd.DataFrame(data=outlook, columns=['overcast', 'rainy', 'sunny'], index=range(len(outlook)))


encoder = LabelEncoder()
df.iloc[:, -1:] = encoder.fit_transform(df.iloc[:, -1:])
df.iloc[:, -2:-1] = encoder.fit_transform(df.iloc[:, -2:-1])


# 2.4. concat data
data = pd.concat([outlookDf, df.iloc[:, 1:2], df.iloc[:, -2:]], axis=1)


humidity = df.iloc[:, 2:3]

# 3. split test and train data
x_train, x_test, y_train, y_test = train_test_split(data, humidity, test_size=0.33, random_state=0)


lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)


X = np.append(arr=np.ones((len(data.values), 1)).astype(int), values=data, axis=1)
X_l = data.iloc[:, [0,1,2,3,4,5]].values
r = sm.OLS(endog=humidity, exog=X_l).fit()
print(r.summary())


X_l = data.iloc[:, [0,1,2,3,5]].values
r = sm.OLS(endog=humidity, exog=X_l).fit()
print(r.summary())


X_l = data.iloc[:, [1,3]].values
r = sm.OLS(endog=humidity, exog=X_l).fit()
print(r.summary())

x_train, x_test, y_train, y_test = train_test_split(X_l, humidity, test_size=0.33, random_state=0)


lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)
