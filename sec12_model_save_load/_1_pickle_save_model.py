#
#
# Mehmet Güngören (mehmetgungoren@lyrebirdstudio.net)
#
#
# 1. libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv("../data/bilkav/sec2_prediction/satislar.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)

lr = LinearRegression()
lr.fit(X_train, y_train)

print(lr.predict(X_test))

pickle.dump(lr, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print("-----------")
print(model.predict(X_test))

