import pandas as pd

titanic = pd.read_csv('titanic.csv')
titanic.drop('Name', axis=1, inplace=True)
titanic['Sex'] = pd.get_dummies(titanic['Sex'])
data = titanic.values
rows, columns = data.shape

lowest_gini = ['Feature', 'Value', 0.50]
for column in range(1, columns):
    data_ = data[data[:, column].argsort()]

    for row in range(1, rows):
        L, R = data_[:row, 0], data_[row:, 0]

        gini_L = 1 - (sum(L == 0) / len(L)) ** 2 - (sum(L == 1) / len(L)) ** 2
        gini_R = 1 - (sum(R == 0) / len(R)) ** 2 - (sum(R == 1) / len(R)) ** 2

        gini = (len(L) * gini_L + len(R) * gini_R) / (len(L) + len(R))

        if gini < lowest_gini[2]:
            lowest_gini = [titanic.columns[column], data_[row, column], gini]

print(f'Best split-feature     : {lowest_gini[0]}')
print(f'Value to split on      : {lowest_gini[1]}')
print(f'Weighted gini impurity : {round(lowest_gini[2], 5)}')