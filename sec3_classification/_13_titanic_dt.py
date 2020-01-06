from sklearn import tree
from graphviz import Source
import pandas as pd

titanic = pd.read_csv('titanic.csv')
titanic.drop('Name', axis=1, inplace=True)
titanic['Sex'] = pd.get_dummies(titanic['Sex'])
data = titanic.values
rows, columns = data.shape

X = titanic.drop('Survived', axis=1).values
y = titanic['Survived'].values

model = tree.DecisionTreeClassifier(max_depth=2)
model.fit(X, y)

graph = Source(tree.export_graphviz(model, out_file=None,
                                    feature_names=titanic.columns[1:],
                                    class_names=['Not survived', 'Survived'],
                                    filled=True, rounded=True))
graph