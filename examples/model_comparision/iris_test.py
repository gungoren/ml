#
#
# Mehmet Güngören (mehmetgungoren@lyrebirdstudio.net)
#
#
import sys
import pandas as pd
import matplotlib.pyplot as pyplot
from pandas.plotting import scatter_matrix
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

print(sys.version)
print(pd.__version__)

iris = datasets.load_iris()
DATA_SET = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
DATA_SET.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# DATA_SET.index +=1

print(DATA_SET.shape)
print(DATA_SET.head(10))
print(DATA_SET.describe())
print(DATA_SET.groupby('class').size())

DATA_SET.plot(kind='box', subplots=True, layout=(20,20))
pyplot.show()
DATA_SET.hist()
pyplot.show()

scatter_matrix(DATA_SET)
pyplot.show()

VALUES = DATA_SET.values
X = VALUES[:,0:4]
Y = VALUES[:,4]
TEST_SPLIT_SIZE = 0.20
SEED = 7


X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=TEST_SPLIT_SIZE, random_state=SEED)

models = []

models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNC', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=SEED)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    print(str(name) + " " + str(cv_results.mean()) + " " + str(cv_results.std()))

    results.append(cv_results)
    names.append(name)

fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(1,1,1)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))