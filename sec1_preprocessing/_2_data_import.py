
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("data/bilkav/sec1_preprocessing/veriler.csv")
print(df)

height = df["boy"]

heightweight = df[["boy", "kilo"]]

print(heightweight)


class Human:
    height = 180

    def run(self, b):
        return b + 10


man = Human()
print(man.height)
print(man.run(90))

list = [2,3,4] #list

