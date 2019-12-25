# 1. libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv("../data/bilkav/sec1_preprocessing/veriler.csv")

x = df.iloc[:, 1:4]
y = df.iloc[:, 4:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# 4. attribute scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


print("------lr------")
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

print("------svc------")
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print(cm)


print("------svc_rbf------")
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print(cm)

print("------svc_poly------")
svc = SVC(kernel='poly')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print(cm)


print("------Gauss NB------")
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print(cm)

"""
print("------Multional NB------")
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print(cm)
"""
print("------Benoulli NB------")
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print(cm)


print("------Decision Tree gini ------")
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print(cm)

print("------Decision Tree entropy ------")
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print(cm)

print("------Random forest  ------")
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, criterion='gini')
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print(cm)

# roc, tpr, fpr
y_proba = rfc.predict_proba(X_test)
print(y_proba[:,0])

from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_proba[:,0], pos_label='e')
print('fpr ' + str(fpr))
print('tpr ', str(tpr))
print('threshold ' + str(threshold))
