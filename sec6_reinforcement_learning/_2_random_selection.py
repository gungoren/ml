import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N, d = veriler.shape
toplam = 0
secilenler = []
total = []
for n in range(0, N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n, ad]
    toplam = toplam + odul
    total.append(toplam)

plt.hist(secilenler)
plt.show()

plt.plot(range(0,N), total, 'r-')
plt.show()