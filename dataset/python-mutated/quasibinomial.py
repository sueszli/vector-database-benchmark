import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
raw = StringIO('0.05,0.00,1.25,2.50,5.50,1.00,5.00,5.00,17.50\n0.00,0.05,1.25,0.50,1.00,5.00,0.10,10.00,25.00\n0.00,0.05,2.50,0.01,6.00,5.00,5.00,5.00,42.50\n0.10,0.30,16.60,3.00,1.10,5.00,5.00,5.00,50.00\n0.25,0.75,2.50,2.50,2.50,5.00,50.00,25.00,37.50\n0.05,0.30,2.50,0.01,8.00,5.00,10.00,75.00,95.00\n0.50,3.00,0.00,25.00,16.50,10.00,50.00,50.00,62.50\n1.30,7.50,20.00,55.00,29.50,5.00,25.00,75.00,95.00\n1.50,1.00,37.50,5.00,20.00,50.00,50.00,75.00,95.00\n1.50,12.70,26.25,40.00,43.50,75.00,75.00,75.00,95.00')
df = pd.read_csv(raw, header=None)
df = df.melt()
df['site'] = 1 + np.floor(df.index / 10).astype(int)
df['variety'] = 1 + df.index % 10
df = df.rename(columns={'value': 'blotch'})
df = df.drop('variable', axis=1)
df['blotch'] /= 100
model1 = sm.GLM.from_formula('blotch ~ 0 + C(variety) + C(site)', family=sm.families.Binomial(), data=df)
result1 = model1.fit(scale='X2')
print(result1.summary())
plt.clf()
plt.grid(True)
plt.plot(result1.predict(linear=True), result1.resid_pearson, 'o')
plt.xlabel('Linear predictor')
plt.ylabel('Residual')

class vf(sm.families.varfuncs.VarianceFunction):

    def __call__(self, mu):
        if False:
            for i in range(10):
                print('nop')
        return mu ** 2 * (1 - mu) ** 2

    def deriv(self, mu):
        if False:
            while True:
                i = 10
        return 2 * mu - 6 * mu ** 2 + 4 * mu ** 3
bin = sm.families.Binomial()
bin.variance = vf()
model2 = sm.GLM.from_formula('blotch ~ 0 + C(variety) + C(site)', family=bin, data=df)
result2 = model2.fit(scale='X2')
print(result2.summary())
plt.clf()
plt.grid(True)
plt.plot(result2.predict(linear=True), result2.resid_pearson, 'o')
plt.xlabel('Linear predictor')
plt.ylabel('Residual')