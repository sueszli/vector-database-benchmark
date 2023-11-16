import numpy as np
import statsmodels.api as sm
import pandas as pd
url = 'https://stats.idre.ucla.edu/stat/data/hsb2.csv'
hsb2 = pd.read_table(url, delimiter=',')
hsb2.head(10)
hsb2.groupby('race')['write'].mean()
from patsy.contrasts import Treatment
levels = [1, 2, 3, 4]
contrast = Treatment(reference=0).code_without_intercept(levels)
print(contrast.matrix)
hsb2.race.head(10)
print(contrast.matrix[hsb2.race - 1, :][:20])
pd.get_dummies(hsb2.race.values, drop_first=False)
from statsmodels.formula.api import ols
mod = ols('write ~ C(race, Treatment)', data=hsb2)
res = mod.fit()
print(res.summary())
from patsy.contrasts import ContrastMatrix

def _name_levels(prefix, levels):
    if False:
        i = 10
        return i + 15
    return ['[%s%s]' % (prefix, level) for level in levels]

class Simple:

    def _simple_contrast(self, levels):
        if False:
            print('Hello World!')
        nlevels = len(levels)
        contr = -1.0 / nlevels * np.ones((nlevels, nlevels - 1))
        contr[1:][np.diag_indices(nlevels - 1)] = (nlevels - 1.0) / nlevels
        return contr

    def code_with_intercept(self, levels):
        if False:
            return 10
        contrast = np.column_stack((np.ones(len(levels)), self._simple_contrast(levels)))
        return ContrastMatrix(contrast, _name_levels('Simp.', levels))

    def code_without_intercept(self, levels):
        if False:
            print('Hello World!')
        contrast = self._simple_contrast(levels)
        return ContrastMatrix(contrast, _name_levels('Simp.', levels[:-1]))
hsb2.groupby('race')['write'].mean().mean()
contrast = Simple().code_without_intercept(levels)
print(contrast.matrix)
mod = ols('write ~ C(race, Simple)', data=hsb2)
res = mod.fit()
print(res.summary())
from patsy.contrasts import Sum
contrast = Sum().code_without_intercept(levels)
print(contrast.matrix)
mod = ols('write ~ C(race, Sum)', data=hsb2)
res = mod.fit()
print(res.summary())
hsb2.groupby('race')['write'].mean().mean()
from patsy.contrasts import Diff
contrast = Diff().code_without_intercept(levels)
print(contrast.matrix)
mod = ols('write ~ C(race, Diff)', data=hsb2)
res = mod.fit()
print(res.summary())
res.params['C(race, Diff)[D.1]']
hsb2.groupby('race').mean()['write'][2] - hsb2.groupby('race').mean()['write'][1]
from patsy.contrasts import Helmert
contrast = Helmert().code_without_intercept(levels)
print(contrast.matrix)
mod = ols('write ~ C(race, Helmert)', data=hsb2)
res = mod.fit()
print(res.summary())
grouped = hsb2.groupby('race')
grouped.mean()['write'][4] - grouped.mean()['write'][:3].mean()
k = 4
1.0 / k * (grouped.mean()['write'][k] - grouped.mean()['write'][:k - 1].mean())
k = 3
1.0 / k * (grouped.mean()['write'][k] - grouped.mean()['write'][:k - 1].mean())
hsb2['readcat'] = np.asarray(pd.cut(hsb2.read, bins=4))
hsb2['readcat'] = hsb2['readcat'].astype(object)
hsb2.groupby('readcat').mean()['write']
from patsy.contrasts import Poly
levels = hsb2.readcat.unique()
contrast = Poly().code_without_intercept(levels)
print(contrast.matrix)
mod = ols('write ~ C(readcat, Poly)', data=hsb2)
res = mod.fit()
print(res.summary())