import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import VCSpec
import pandas as pd
np.random.seed(3123)

def generate_nested(n_group1=200, n_group2=20, n_rep=10, group1_sd=2, group2_sd=3, unexplained_sd=4):
    if False:
        i = 10
        return i + 15
    group1 = np.kron(np.arange(n_group1), np.ones(n_group2 * n_rep))
    u = group1_sd * np.random.normal(size=n_group1)
    effects1 = np.kron(u, np.ones(n_group2 * n_rep))
    group2 = np.kron(np.ones(n_group1), np.kron(np.arange(n_group2), np.ones(n_rep)))
    u = group2_sd * np.random.normal(size=n_group1 * n_group2)
    effects2 = np.kron(u, np.ones(n_rep))
    e = unexplained_sd * np.random.normal(size=n_group1 * n_group2 * n_rep)
    y = effects1 + effects2 + e
    df = pd.DataFrame({'y': y, 'group1': group1, 'group2': group2})
    return df
df = generate_nested()
model1 = sm.MixedLM.from_formula('y ~ 1', re_formula='1', vc_formula={'group2': '0 + C(group2)'}, groups='group1', data=df)
result1 = model1.fit()
print(result1.summary())

def f(x):
    if False:
        for i in range(10):
            print('nop')
    n = x.shape[0]
    g2 = x.group2
    u = g2.unique()
    u.sort()
    uv = {v: k for (k, v) in enumerate(u)}
    mat = np.zeros((n, len(u)))
    for i in range(n):
        mat[i, uv[g2.iloc[i]]] = 1
    colnames = ['%d' % z for z in u]
    return (mat, colnames)
vcm = df.groupby('group1').apply(f).to_list()
mats = [x[0] for x in vcm]
colnames = [x[1] for x in vcm]
names = ['group2']
vcs = VCSpec(names, [colnames], [mats])
oo = np.ones(df.shape[0])
model2 = sm.MixedLM(df.y, oo, exog_re=oo, groups=df.group1, exog_vc=vcs)
result2 = model2.fit()
print(result2.summary())

def generate_crossed(n_group1=100, n_group2=100, n_rep=4, group1_sd=2, group2_sd=3, unexplained_sd=4):
    if False:
        while True:
            i = 10
    group1 = np.kron(np.arange(n_group1, dtype=int), np.ones(n_group2 * n_rep, dtype=int))
    group1 = group1[np.random.permutation(len(group1))]
    u = group1_sd * np.random.normal(size=n_group1)
    effects1 = u[group1]
    group2 = np.kron(np.arange(n_group2, dtype=int), np.ones(n_group2 * n_rep, dtype=int))
    group2 = group2[np.random.permutation(len(group2))]
    u = group2_sd * np.random.normal(size=n_group2)
    effects2 = u[group2]
    e = unexplained_sd * np.random.normal(size=n_group1 * n_group2 * n_rep)
    y = effects1 + effects2 + e
    df = pd.DataFrame({'y': y, 'group1': group1, 'group2': group2})
    return df
df = generate_crossed()
vc = {'g1': '0 + C(group1)', 'g2': '0 + C(group2)'}
oo = np.ones(df.shape[0])
model3 = sm.MixedLM.from_formula('y ~ 1', groups=oo, vc_formula=vc, data=df)
result3 = model3.fit()
print(result3.summary())

def f(g):
    if False:
        i = 10
        return i + 15
    n = len(g)
    u = g.unique()
    u.sort()
    uv = {v: k for (k, v) in enumerate(u)}
    mat = np.zeros((n, len(u)))
    for i in range(n):
        mat[i, uv[g[i]]] = 1
    colnames = ['%d' % z for z in u]
    return ([mat], [colnames])
vcm = [f(df.group1), f(df.group2)]
mats = [x[0] for x in vcm]
colnames = [x[1] for x in vcm]
names = ['group1', 'group2']
vcs = VCSpec(names, colnames, mats)
oo = np.ones(df.shape[0])
model4 = sm.MixedLM(df.y, oo[:, None], exog_re=None, groups=oo, exog_vc=vcs)
result4 = model4.fit()
print(result4.summary())