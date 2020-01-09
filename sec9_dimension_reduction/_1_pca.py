#
#
# Mehmet Güngören (mehmetgungoren@lyrebirdstudio.net)
#
#
# 1. libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# 2. preprocessing
# 2.1. import data
df = pd.read_csv("Wine.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

# 3. split test and train data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# 4. attribute scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# dimension reducted by pca
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(X_train2, y_train)

lr2 = LogisticRegression(random_state=0)
lr2.fit(X_train, y_train)

y_pred2 = lr.predict(X_test2)
y_pred = lr2.predict(X_test)

from sklearn.metrics import confusion_matrix

#actual result
cm = confusion_matrix(y_test, y_pred)
print(cm)

#results after PCA
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

#pca vs original
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)



