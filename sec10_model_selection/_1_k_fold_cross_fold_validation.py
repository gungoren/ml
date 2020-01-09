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
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# 2. preprocessing
# 2.1. import data
df = pd.read_csv("Social_Network_Ads.csv")

X = df.iloc[:, 2:-1]
y = df.iloc[:, -1:]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 4. attribute scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

svc = SVC(kernel='rbf', random_state=0)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

cvs = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=4)
print(cvs.mean())
print(cvs.std())
