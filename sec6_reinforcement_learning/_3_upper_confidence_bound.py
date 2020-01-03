import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import math
#UCB
N, d = veriler.shape
#Ri(n)
oduller = [0] * d
#Ni(n)
tiklamalar = [0] * d
toplam = 0
secilenler = []
for n in range(1,N):
    ad = 0
    max_ucb = 0
    for i in range(0,d):
        if tiklamalar[i] > 0:
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2* math.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad]+ 1
    odul = veriler.values[n, ad]
    oduller[ad] = oduller[ad]+ odul
    toplam = toplam + odul
print('Toplam Odul:')   
print(toplam)

plt.hist(secilenler)
plt.show()

