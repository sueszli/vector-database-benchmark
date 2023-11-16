"""
======================================
Poisson regression and non-normal loss
======================================

This example illustrates the use of log-linear Poisson regression on the
`French Motor Third-Party Liability Claims dataset
<https://www.openml.org/d/41214>`_ from [1]_ and compares it with a linear
model fitted with the usual least squared error and a non-linear GBRT model
fitted with the Poisson loss (and a log-link).

A few definitions:

- A **policy** is a contract between an insurance company and an individual:
  the **policyholder**, that is, the vehicle driver in this case.

- A **claim** is the request made by a policyholder to the insurer to
  compensate for a loss covered by the insurance.

- The **exposure** is the duration of the insurance coverage of a given policy,
  in years.

- The claim **frequency** is the number of claims divided by the exposure,
  typically measured in number of claims per year.

In this dataset, each sample corresponds to an insurance policy. Available
features include driver age, vehicle age, vehicle power, etc.

Our goal is to predict the expected frequency of claims following car accidents
for a new policyholder given the historical data over a population of
policyholders.

.. [1]  A. Noll, R. Salzmann and M.V. Wuthrich, Case Study: French Motor
    Third-Party Liability Claims (November 8, 2018). `doi:10.2139/ssrn.3164764
    <https://doi.org/10.2139/ssrn.3164764>`_

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
df = fetch_openml(data_id=41214, as_frame=True, parser='pandas').frame
df
df['Frequency'] = df['ClaimNb'] / df['Exposure']
print('Average Frequency = {}'.format(np.average(df['Frequency'], weights=df['Exposure'])))
print('Fraction of exposure with zero claims = {0:.1%}'.format(df.loc[df['ClaimNb'] == 0, 'Exposure'].sum() / df['Exposure'].sum()))
(fig, (ax0, ax1, ax2)) = plt.subplots(ncols=3, figsize=(16, 4))
ax0.set_title('Number of claims')
_ = df['ClaimNb'].hist(bins=30, log=True, ax=ax0)
ax1.set_title('Exposure in years')
_ = df['Exposure'].hist(bins=30, log=True, ax=ax1)
ax2.set_title('Frequency (number of claims per year)')
_ = df['Frequency'].hist(bins=30, log=True, ax=ax2)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler
log_scale_transformer = make_pipeline(FunctionTransformer(np.log, validate=False), StandardScaler())
linear_model_preprocessor = ColumnTransformer([('passthrough_numeric', 'passthrough', ['BonusMalus']), ('binned_numeric', KBinsDiscretizer(n_bins=10, subsample=int(200000.0), random_state=0), ['VehAge', 'DrivAge']), ('log_scaled_numeric', log_scale_transformer, ['Density']), ('onehot_categorical', OneHotEncoder(), ['VehBrand', 'VehPower', 'VehGas', 'Region', 'Area'])], remainder='drop')
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
(df_train, df_test) = train_test_split(df, test_size=0.33, random_state=0)
dummy = Pipeline([('preprocessor', linear_model_preprocessor), ('regressor', DummyRegressor(strategy='mean'))]).fit(df_train, df_train['Frequency'], regressor__sample_weight=df_train['Exposure'])
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

def score_estimator(estimator, df_test):
    if False:
        i = 10
        return i + 15
    'Score an estimator on the test set.'
    y_pred = estimator.predict(df_test)
    print('MSE: %.3f' % mean_squared_error(df_test['Frequency'], y_pred, sample_weight=df_test['Exposure']))
    print('MAE: %.3f' % mean_absolute_error(df_test['Frequency'], y_pred, sample_weight=df_test['Exposure']))
    mask = y_pred > 0
    if (~mask).any():
        (n_masked, n_samples) = ((~mask).sum(), mask.shape[0])
        print(f'WARNING: Estimator yields invalid, non-positive predictions  for {n_masked} samples out of {n_samples}. These predictions are ignored when computing the Poisson deviance.')
    print('mean Poisson deviance: %.3f' % mean_poisson_deviance(df_test['Frequency'][mask], y_pred[mask], sample_weight=df_test['Exposure'][mask]))
print('Constant mean frequency evaluation:')
score_estimator(dummy, df_test)
from sklearn.linear_model import Ridge
ridge_glm = Pipeline([('preprocessor', linear_model_preprocessor), ('regressor', Ridge(alpha=1e-06))]).fit(df_train, df_train['Frequency'], regressor__sample_weight=df_train['Exposure'])
print('Ridge evaluation:')
score_estimator(ridge_glm, df_test)
from sklearn.linear_model import PoissonRegressor
n_samples = df_train.shape[0]
poisson_glm = Pipeline([('preprocessor', linear_model_preprocessor), ('regressor', PoissonRegressor(alpha=1e-12, solver='newton-cholesky'))])
poisson_glm.fit(df_train, df_train['Frequency'], regressor__sample_weight=df_train['Exposure'])
print('PoissonRegressor evaluation:')
score_estimator(poisson_glm, df_test)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
tree_preprocessor = ColumnTransformer([('categorical', OrdinalEncoder(), ['VehBrand', 'VehPower', 'VehGas', 'Region', 'Area']), ('numeric', 'passthrough', ['VehAge', 'DrivAge', 'BonusMalus', 'Density'])], remainder='drop')
poisson_gbrt = Pipeline([('preprocessor', tree_preprocessor), ('regressor', HistGradientBoostingRegressor(loss='poisson', max_leaf_nodes=128))])
poisson_gbrt.fit(df_train, df_train['Frequency'], regressor__sample_weight=df_train['Exposure'])
print('Poisson Gradient Boosted Trees evaluation:')
score_estimator(poisson_gbrt, df_test)
(fig, axes) = plt.subplots(nrows=2, ncols=4, figsize=(16, 6), sharey=True)
fig.subplots_adjust(bottom=0.2)
n_bins = 20
for (row_idx, label, df) in zip(range(2), ['train', 'test'], [df_train, df_test]):
    df['Frequency'].hist(bins=np.linspace(-1, 30, n_bins), ax=axes[row_idx, 0])
    axes[row_idx, 0].set_title('Data')
    axes[row_idx, 0].set_yscale('log')
    axes[row_idx, 0].set_xlabel('y (observed Frequency)')
    axes[row_idx, 0].set_ylim([10.0, 500000.0])
    axes[row_idx, 0].set_ylabel(label + ' samples')
    for (idx, model) in enumerate([ridge_glm, poisson_glm, poisson_gbrt]):
        y_pred = model.predict(df)
        pd.Series(y_pred).hist(bins=np.linspace(-1, 4, n_bins), ax=axes[row_idx, idx + 1])
        axes[row_idx, idx + 1].set(title=model[-1].__class__.__name__, yscale='log', xlabel='y_pred (predicted expected Frequency)')
plt.tight_layout()
from sklearn.utils import gen_even_slices

def _mean_frequency_by_risk_group(y_true, y_pred, sample_weight=None, n_bins=100):
    if False:
        while True:
            i = 10
    'Compare predictions and observations for bins ordered by y_pred.\n\n    We order the samples by ``y_pred`` and split it in bins.\n    In each bin the observed mean is compared with the predicted mean.\n\n    Parameters\n    ----------\n    y_true: array-like of shape (n_samples,)\n        Ground truth (correct) target values.\n    y_pred: array-like of shape (n_samples,)\n        Estimated target values.\n    sample_weight : array-like of shape (n_samples,)\n        Sample weights.\n    n_bins: int\n        Number of bins to use.\n\n    Returns\n    -------\n    bin_centers: ndarray of shape (n_bins,)\n        bin centers\n    y_true_bin: ndarray of shape (n_bins,)\n        average y_pred for each bin\n    y_pred_bin: ndarray of shape (n_bins,)\n        average y_pred for each bin\n    '
    idx_sort = np.argsort(y_pred)
    bin_centers = np.arange(0, 1, 1 / n_bins) + 0.5 / n_bins
    y_pred_bin = np.zeros(n_bins)
    y_true_bin = np.zeros(n_bins)
    for (n, sl) in enumerate(gen_even_slices(len(y_true), n_bins)):
        weights = sample_weight[idx_sort][sl]
        y_pred_bin[n] = np.average(y_pred[idx_sort][sl], weights=weights)
        y_true_bin[n] = np.average(y_true[idx_sort][sl], weights=weights)
    return (bin_centers, y_true_bin, y_pred_bin)
print(f"Actual number of claims: {df_test['ClaimNb'].sum()}")
(fig, ax) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
plt.subplots_adjust(wspace=0.3)
for (axi, model) in zip(ax.ravel(), [ridge_glm, poisson_glm, poisson_gbrt, dummy]):
    y_pred = model.predict(df_test)
    y_true = df_test['Frequency'].values
    exposure = df_test['Exposure'].values
    (q, y_true_seg, y_pred_seg) = _mean_frequency_by_risk_group(y_true, y_pred, sample_weight=exposure, n_bins=10)
    print(f'Predicted number of claims by {model[-1]}: {np.sum(y_pred * exposure):.1f}')
    axi.plot(q, y_pred_seg, marker='x', linestyle='--', label='predictions')
    axi.plot(q, y_true_seg, marker='o', linestyle='--', label='observations')
    axi.set_xlim(0, 1.0)
    axi.set_ylim(0, 0.5)
    axi.set(title=model[-1], xlabel='Fraction of samples sorted by y_pred', ylabel='Mean Frequency (y_pred)')
    axi.legend()
plt.tight_layout()
from sklearn.metrics import auc

def lorenz_curve(y_true, y_pred, exposure):
    if False:
        for i in range(10):
            print('nop')
    (y_true, y_pred) = (np.asarray(y_true), np.asarray(y_pred))
    exposure = np.asarray(exposure)
    ranking = np.argsort(y_pred)
    ranked_frequencies = y_true[ranking]
    ranked_exposure = exposure[ranking]
    cumulated_claims = np.cumsum(ranked_frequencies * ranked_exposure)
    cumulated_claims /= cumulated_claims[-1]
    cumulated_exposure = np.cumsum(ranked_exposure)
    cumulated_exposure /= cumulated_exposure[-1]
    return (cumulated_exposure, cumulated_claims)
(fig, ax) = plt.subplots(figsize=(8, 8))
for model in [dummy, ridge_glm, poisson_glm, poisson_gbrt]:
    y_pred = model.predict(df_test)
    (cum_exposure, cum_claims) = lorenz_curve(df_test['Frequency'], y_pred, df_test['Exposure'])
    gini = 1 - 2 * auc(cum_exposure, cum_claims)
    label = '{} (Gini: {:.2f})'.format(model[-1], gini)
    ax.plot(cum_exposure, cum_claims, linestyle='-', label=label)
(cum_exposure, cum_claims) = lorenz_curve(df_test['Frequency'], df_test['Frequency'], df_test['Exposure'])
gini = 1 - 2 * auc(cum_exposure, cum_claims)
label = 'Oracle (Gini: {:.2f})'.format(gini)
ax.plot(cum_exposure, cum_claims, linestyle='-.', color='gray', label=label)
ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random baseline')
ax.set(title='Lorenz curves by model', xlabel='Cumulative proportion of exposure (from safest to riskiest)', ylabel='Cumulative proportion of claims')
ax.legend(loc='upper left')
plt.show()