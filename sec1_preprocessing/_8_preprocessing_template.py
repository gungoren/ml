
# 1. libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# 2. preprocessing
# 2.1. import data
df = pd.read_csv("../data/bilkav/sec1_preprocessing/eksikveriler.csv")


"""
imputer = SimpleImputer(missing_values='NaN', strategy='mean')

age = df.iloc[:,1:4].values
imputer = imputer.fit(age[:, 1:4])
age[:, 1:4] = imputer.transform(age[:, 1:4])
print(age)
"""

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

genderDf = df.iloc[:, -1:]

# 2.4. concat data
result_1 = pd.concat([ulkeDf, ageDf], axis=1)

result = pd.concat([ulkeDf, ageDf, genderDf], axis=1)

# 3. split test and train data
x_train, x_test, y_train, y_test = train_test_split(result_1, genderDf, test_size=0.33, random_state=0)

# 4. attribute scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)











