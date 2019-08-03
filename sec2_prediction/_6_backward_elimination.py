
# 1. libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.api as sm

# 2. preprocessing
# 2.1. import data
df = pd.read_csv("../data/bilkav/sec2_prediction/veriler.csv")


# 2.2. fill empty values
age = df.iloc[:,1:4]
age.fillna(age.mean(), inplace=True)
df.iloc[:,1:4] = age


# 2.3. transform categorical data
ulke = df.iloc[:,0:1]
hot = OneHotEncoder(categories='auto')
ulke = hot.fit_transform(ulke).toarray()

ulkeDf = pd.DataFrame(data=ulke, columns=['fr', 'tr', 'us'], index=range(len(ulke)))
ageDf = age

cinsiyet = df.iloc[:, -1:]

encoder = LabelEncoder()
df.iloc[:, -1:] = encoder.fit_transform(cinsiyet)

genderDf = df.iloc[:, -1:]

# 2.4. concat data
result_1 = pd.concat([ulkeDf, ageDf], axis=1)

result = pd.concat([ulkeDf, ageDf, genderDf], axis=1)

# 3. split test and train data
x_train, x_test, y_train, y_test = train_test_split(result_1, genderDf, test_size=0.33, random_state=0)


lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)

boy = result[['boy']]

left = result.iloc[:, :3]
right = result.iloc[:, 4:]

data = pd.concat([left, right], axis=1)
x_train, x_test, y_train, y_test = train_test_split(data, boy, test_size=0.33, random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)


X = np.append(arr=np.ones((len(data.values), 1)).astype(int), values=data, axis=1)
X_l = data.iloc[:, [0,1,2,3,4,5]].values
r = sm.OLS(endog=boy, exog=X_l).fit()
print(r.summary())

X_l = data.iloc[:, [0,1,2,3,5]].values
r = sm.OLS(endog=boy, exog=X_l).fit()
print(r.summary())


X_l = data.iloc[:, [0,1,2,3]].values
r = sm.OLS(endog=boy, exog=X_l).fit()
print(r.summary())

