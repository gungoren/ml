
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


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
