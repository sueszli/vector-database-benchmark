import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.multivariate.pca import PCA
plt.rc('figure', figsize=(16, 8))
plt.rc('font', size=14)
data = sm.datasets.fertility.load_pandas().data
data.head()
columns = list(map(str, range(1960, 2012)))
data.set_index('Country Name', inplace=True)
dta = data[columns]
dta = dta.dropna()
dta.head()
ax = dta.mean().plot(grid=False)
ax.set_xlabel('Year', size=17)
ax.set_ylabel('Fertility rate', size=17)
ax.set_xlim(0, 51)
pca_model = PCA(dta.T, standardize=False, demean=True)
fig = pca_model.plot_scree(log_scale=False)
(fig, ax) = plt.subplots(figsize=(8, 4))
lines = ax.plot(pca_model.factors.iloc[:, :3], lw=4, alpha=0.6)
ax.set_xticklabels(dta.columns.values[::10])
ax.set_xlim(0, 51)
ax.set_xlabel('Year', size=17)
fig.subplots_adjust(0.1, 0.1, 0.85, 0.9)
legend = fig.legend(lines, ['PC 1', 'PC 2', 'PC 3'], loc='center right')
legend.draw_frame(False)
idx = pca_model.loadings.iloc[:, 0].argsort()

def make_plot(labels):
    if False:
        while True:
            i = 10
    (fig, ax) = plt.subplots(figsize=(9, 5))
    ax = dta.loc[labels].T.plot(legend=False, grid=False, ax=ax)
    dta.mean().plot(ax=ax, grid=False, label='Mean')
    ax.set_xlim(0, 51)
    fig.subplots_adjust(0.1, 0.1, 0.75, 0.9)
    ax.set_xlabel('Year', size=17)
    ax.set_ylabel('Fertility', size=17)
    legend = ax.legend(*ax.get_legend_handles_labels(), loc='center left', bbox_to_anchor=(1, 0.5))
    legend.draw_frame(False)
labels = dta.index[idx[-5:]]
make_plot(labels)
idx = pca_model.loadings.iloc[:, 1].argsort()
make_plot(dta.index[idx[-5:]])
make_plot(dta.index[idx[:5]])
(fig, ax) = plt.subplots()
pca_model.loadings.plot.scatter(x='comp_00', y='comp_01', ax=ax)
ax.set_xlabel('PC 1', size=17)
ax.set_ylabel('PC 2', size=17)
dta.index[pca_model.loadings.iloc[:, 1] > 0.2].values