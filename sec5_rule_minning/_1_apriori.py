
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../data/bilkav/sec5_rule_minning/sepet.csv", header=None)

transactions = []
for i in range(0, df.shape[0]):
    transactions.append([str(df.values[i,j]) for j in range(0, df.shape[1]) if not str(df.values[i, j]) == 'nan'])

from apyori import apriori
results = apriori(transactions, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)

for result in list(results):
    print(result)

#TODO: FPGrowth take a look