import pandas as pd
import numpy as np
from scipy.stats.distributions import norm, poisson
import statsmodels.api as sm
import matplotlib.pyplot as plt

def negbinom(u, mu, scale):
    if False:
        for i in range(10):
            print('nop')
    p = (scale - 1) / scale
    r = mu * (1 - p) / p
    x = np.random.gamma(r, p / (1 - p), len(u))
    return poisson.ppf(u, mu=x)
n = 1000
p = 5
m = 10
r = 0.5
grp = np.kron(np.arange(n / m), np.ones(m))
x = np.random.normal(size=(n, p))
x[:, 0] = 1
x0 = x[:, 0:3]
scale = 10
coeff = [[4, 0.4, -0.2], [4, 0.4, -0.2, 0, -0.04]]
lp = [np.dot(x0, coeff[0]), np.dot(x, coeff[1])]
mu = [np.exp(lp[0]), np.exp(lp[1])]

def dosim(hyp, cov_struct=None, mcrep=500):
    if False:
        for i in range(10):
            print('nop')
    scales = [[], []]
    pv = []
    for k in range(mcrep):
        z = np.random.normal(size=n)
        u = np.random.normal(size=n // m)
        u = np.kron(u, np.ones(m))
        z = r * z + np.sqrt(1 - r ** 2) * u
        u = norm.cdf(z)
        y = negbinom(u, mu=mu[hyp], scale=scale)
        m0 = sm.GEE(y, x0, groups=grp, cov_struct=cov_struct, family=sm.families.Poisson())
        r0 = m0.fit(scale='X2')
        scales[0].append(r0.scale)
        m1 = sm.GEE(y, x, groups=grp, cov_struct=cov_struct, family=sm.families.Poisson())
        r1 = m1.fit(scale='X2')
        scales[1].append(r1.scale)
        st = m1.compare_score_test(r0)
        pv.append(st['p-value'])
    pv = np.asarray(pv)
    rslt = [np.mean(pv), np.mean(pv < 0.1)]
    return (rslt, scales)
(rslt, scales) = ([], [])
for hyp in (0, 1):
    (s, t) = dosim(hyp, sm.cov_struct.Independence())
    rslt.append(s)
    scales.append(t)
rslt = pd.DataFrame(rslt, index=['H0', 'H1'], columns=['Mean', 'Prop(p<0.1)'])
print(rslt)
_ = plt.boxplot([scales[0][0], scales[0][1], scales[1][0], scales[1][1]])
plt.ylabel('Estimated scale')
(rslt, scales) = ([], [])
for hyp in (0, 1):
    (s, t) = dosim(hyp, sm.cov_struct.Exchangeable(), mcrep=100)
    rslt.append(s)
    scales.append(t)
rslt = pd.DataFrame(rslt, index=['H0', 'H1'], columns=['Mean', 'Prop(p<0.1)'])
print(rslt)