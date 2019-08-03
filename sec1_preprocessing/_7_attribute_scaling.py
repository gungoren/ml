
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/bilkav/sec1_preprocessing/eksikveriler.csv")
print(df)

"""
imputer = SimpleImputer(missing_values='NaN', strategy='mean')

age = df.iloc[:,1:4].values
imputer = imputer.fit(age[:, 1:4])
age[:, 1:4] = imputer.transform(age[:, 1:4])
print(age)
"""

age = df.iloc[:,1:4]
age.fillna(age.mean(), inplace=True)
print(age)
df.iloc[:,1:4] = age
print(df)


"""
ulke = df.iloc[:,0:1]
print(ulke)
le = LabelEncoder()
ulke[:, 0] = le.fit_transform(ulke[:,0])
print(ulke)
"""

ulke = df.iloc[:,0:1]
hot = OneHotEncoder(categories='auto')
ulke = hot.fit_transform(ulke).toarray()
print(ulke)

print(len(ulke))

ulkeDf = pd.DataFrame(data=ulke, columns=['fr', 'tr', 'us'], index=range(len(ulke)))
ageDf = age

genderDf = df.iloc[:, -1:]
print(genderDf)

result_1 = pd.concat([ulkeDf, ageDf], axis=1)
print(result_1)


result = pd.concat([ulkeDf, ageDf, genderDf], axis=1)
print(result)

x_train, x_test, y_train, y_test = train_test_split(result_1, genderDf, test_size=0.33, random_state=0)
print(x_train)


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)











