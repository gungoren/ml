import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

N, d = veriler.shape
toplam = 0
secilenler = []

rewards = [0] * d
penalty = [0] * d

for n in range(1, N):
    ad = 0
    max_th = 0
    for i in range(0, d):
        rasbeta = random.betavariate(rewards[i] + 1, penalty[i] + 1) * d
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n, ad]
    if odul > 0:
        rewards[ad] += 1
    else:
        penalty[ad] += 1
    toplam = toplam + odul

print('Toplam Odul:')
print(toplam)

plt.hist(secilenler)
plt.show()
