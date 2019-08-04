# 1. libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("../data/bilkav/sec1_preprocessing/veriler.csv")

x = df.iloc[:, 1:4]
y = df.iloc[:, 4:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# 4. attribute scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print(cm)

print("------knn------")
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print(cm)
