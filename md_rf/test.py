import pandas as pd


features = pd.read_csv('temps.csv')
print(features.head(5))

print('The shape of our features is:', features.shape)

print(features.describe())