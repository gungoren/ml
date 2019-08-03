
# 1. libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# 2. preprocessing
# 2.1. import data
df = pd.read_csv("../data/bilkav/sec2_prediction/satislar.csv")


aylar = df[['Aylar']]
satislar = df[['Satislar']]

# 3. split test and train data
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

# 4. attribute scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)




