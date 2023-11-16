from statsmodels.compat import lmap
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
norms = sm.robust.norms

def plot_weights(support, weights_func, xlabels, xticks):
    if False:
        for i in range(10):
            print('nop')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(support, weights_func(support))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=16)
    ax.set_ylim(-0.1, 1.1)
    return ax
help(norms.AndrewWave.weights)
a = 1.339
support = np.linspace(-np.pi * a, np.pi * a, 100)
andrew = norms.AndrewWave(a=a)
plot_weights(support, andrew.weights, ['$-\\pi*a$', '0', '$\\pi*a$'], [-np.pi * a, 0, np.pi * a])
help(norms.Hampel.weights)
c = 8
support = np.linspace(-3 * c, 3 * c, 1000)
hampel = norms.Hampel(a=2.0, b=4.0, c=c)
plot_weights(support, hampel.weights, ['3*c', '0', '3*c'], [-3 * c, 0, 3 * c])
help(norms.HuberT.weights)
t = 1.345
support = np.linspace(-3 * t, 3 * t, 1000)
huber = norms.HuberT(t=t)
plot_weights(support, huber.weights, ['-3*t', '0', '3*t'], [-3 * t, 0, 3 * t])
help(norms.LeastSquares.weights)
support = np.linspace(-3, 3, 1000)
lst_sq = norms.LeastSquares()
plot_weights(support, lst_sq.weights, ['-3', '0', '3'], [-3, 0, 3])
help(norms.RamsayE.weights)
a = 0.3
support = np.linspace(-3 * a, 3 * a, 1000)
ramsay = norms.RamsayE(a=a)
plot_weights(support, ramsay.weights, ['-3*a', '0', '3*a'], [-3 * a, 0, 3 * a])
help(norms.TrimmedMean.weights)
c = 2
support = np.linspace(-3 * c, 3 * c, 1000)
trimmed = norms.TrimmedMean(c=c)
plot_weights(support, trimmed.weights, ['-3*c', '0', '3*c'], [-3 * c, 0, 3 * c])
help(norms.TukeyBiweight.weights)
c = 4.685
support = np.linspace(-3 * c, 3 * c, 1000)
tukey = norms.TukeyBiweight(c=c)
plot_weights(support, tukey.weights, ['-3*c', '0', '3*c'], [-3 * c, 0, 3 * c])
x = np.array([1, 2, 3, 4, 500])
x.mean()
np.median(x)
x.std()
stats.norm.ppf(0.75)
print(x)
sm.robust.scale.mad(x)
np.array([1, 2, 3, 4, 5.0]).std()
sm.robust.scale.iqr(x)
sm.robust.scale.qn_scale(x)
np.random.seed(12345)
fat_tails = stats.t(6).rvs(40)
kde = sm.nonparametric.KDEUnivariate(fat_tails)
kde.fit()
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.plot(kde.support, kde.density)
print(fat_tails.mean(), fat_tails.std())
print(stats.norm.fit(fat_tails))
print(stats.t.fit(fat_tails, f0=6))
huber = sm.robust.scale.Huber()
(loc, scale) = huber(fat_tails)
print(loc, scale)
sm.robust.mad(fat_tails)
sm.robust.mad(fat_tails, c=stats.t(6).ppf(0.75))
sm.robust.scale.mad(fat_tails)
from statsmodels.graphics.api import abline_plot
from statsmodels.formula.api import ols, rlm
prestige = sm.datasets.get_rdataset('Duncan', 'carData', cache=True).data
print(prestige.head(10))
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(211, xlabel='Income', ylabel='Prestige')
ax1.scatter(prestige.income, prestige.prestige)
xy_outlier = prestige.loc['minister', ['income', 'prestige']]
ax1.annotate('Minister', xy_outlier, xy_outlier + 1, fontsize=16)
ax2 = fig.add_subplot(212, xlabel='Education', ylabel='Prestige')
ax2.scatter(prestige.education, prestige.prestige)
ols_model = ols('prestige ~ income + education', prestige).fit()
print(ols_model.summary())
infl = ols_model.get_influence()
student = infl.summary_frame()['student_resid']
print(student)
print(student.loc[np.abs(student) > 2])
print(infl.summary_frame().loc['minister'])
sidak = ols_model.outlier_test('sidak')
sidak.sort_values('unadj_p', inplace=True)
print(sidak)
fdr = ols_model.outlier_test('fdr_bh')
fdr.sort_values('unadj_p', inplace=True)
print(fdr)
rlm_model = rlm('prestige ~ income + education', prestige).fit()
print(rlm_model.summary())
print(rlm_model.weights)
dta = sm.datasets.get_rdataset('starsCYG', 'robustbase', cache=True).data
from matplotlib.patches import Ellipse
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, xlabel='log(Temp)', ylabel='log(Light)', title='Hertzsprung-Russell Diagram of Star Cluster CYG OB1')
ax.scatter(*dta.values.T)
e = Ellipse((3.5, 6), 0.2, 1, alpha=0.25, color='r')
ax.add_patch(e)
ax.annotate('Red giants', xy=(3.6, 6), xytext=(3.8, 6), arrowprops=dict(facecolor='black', shrink=0.05, width=2), horizontalalignment='left', verticalalignment='bottom', clip_on=True, fontsize=16)
for (i, row) in dta.loc[dta['log.Te'] < 3.8].iterrows():
    ax.annotate(i, row, row + 0.01, fontsize=14)
(xlim, ylim) = (ax.get_xlim(), ax.get_ylim())
from IPython.display import Image
Image(filename='star_diagram.png')
y = dta['log.light']
X = sm.add_constant(dta['log.Te'], prepend=True)
ols_model = sm.OLS(y, X).fit()
abline_plot(model_results=ols_model, ax=ax)
rlm_mod = sm.RLM(y, X, sm.robust.norms.TrimmedMean(0.5)).fit()
abline_plot(model_results=rlm_mod, ax=ax, color='red')
infl = ols_model.get_influence()
h_bar = 2 * (ols_model.df_model + 1) / ols_model.nobs
hat_diag = infl.summary_frame()['hat_diag']
hat_diag.loc[hat_diag > h_bar]
sidak2 = ols_model.outlier_test('sidak')
sidak2.sort_values('unadj_p', inplace=True)
print(sidak2)
fdr2 = ols_model.outlier_test('fdr_bh')
fdr2.sort_values('unadj_p', inplace=True)
print(fdr2)
l = ax.lines[-1]
l.remove()
del l
weights = np.ones(len(X))
weights[X[X['log.Te'] < 3.8].index.values - 1] = 0
wls_model = sm.WLS(y, X, weights=weights).fit()
abline_plot(model_results=wls_model, ax=ax, color='green')
yy = y.values[:, None]
xx = X['log.Te'].values[:, None]
params = [-4.969387980288108, 2.2531613477892365]
print(params[0], params[1])
abline_plot(intercept=params[0], slope=params[1], ax=ax, color='red')
np.random.seed(12345)
nobs = 200
beta_true = np.array([3, 1, 2.5, 3, -4])
X = np.random.uniform(-20, 20, size=(nobs, len(beta_true) - 1))
X = sm.add_constant(X, prepend=True)
mc_iter = 500
contaminate = 0.25
all_betas = []
for i in range(mc_iter):
    y = np.dot(X, beta_true) + np.random.normal(size=200)
    random_idx = np.random.randint(0, nobs, size=int(contaminate * nobs))
    y[random_idx] = np.random.uniform(-750, 750)
    beta_hat = sm.RLM(y, X).fit().params
    all_betas.append(beta_hat)
all_betas = np.asarray(all_betas)
se_loss = lambda x: np.linalg.norm(x, ord=2) ** 2
se_beta = lmap(se_loss, all_betas - beta_true)
np.array(se_beta).mean()
all_betas.mean(0)
beta_true
se_loss(all_betas.mean(0) - beta_true)