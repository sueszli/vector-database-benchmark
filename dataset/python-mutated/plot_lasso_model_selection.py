"""
=================================================
Lasso model selection: AIC-BIC / cross-validation
=================================================

This example focuses on model selection for Lasso models that are
linear models with an L1 penalty for regression problems.

Indeed, several strategies can be used to select the value of the
regularization parameter: via cross-validation or using an information
criterion, namely AIC or BIC.

In what follows, we will discuss in details the different strategies.
"""
from sklearn.datasets import load_diabetes
(X, y) = load_diabetes(return_X_y=True, as_frame=True)
X.head()
import numpy as np
import pandas as pd
rng = np.random.RandomState(42)
n_random_features = 14
X_random = pd.DataFrame(rng.randn(X.shape[0], n_random_features), columns=[f'random_{i:02d}' for i in range(n_random_features)])
X = pd.concat([X, X_random], axis=1)
X[X.columns[::3]].head()
import time
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
start_time = time.time()
lasso_lars_ic = make_pipeline(StandardScaler(), LassoLarsIC(criterion='aic')).fit(X, y)
fit_time = time.time() - start_time
results = pd.DataFrame({'alphas': lasso_lars_ic[-1].alphas_, 'AIC criterion': lasso_lars_ic[-1].criterion_}).set_index('alphas')
alpha_aic = lasso_lars_ic[-1].alpha_
lasso_lars_ic.set_params(lassolarsic__criterion='bic').fit(X, y)
results['BIC criterion'] = lasso_lars_ic[-1].criterion_
alpha_bic = lasso_lars_ic[-1].alpha_

def highlight_min(x):
    if False:
        while True:
            i = 10
    x_min = x.min()
    return ['font-weight: bold' if v == x_min else '' for v in x]
results.style.apply(highlight_min)
ax = results.plot()
ax.vlines(alpha_aic, results['AIC criterion'].min(), results['AIC criterion'].max(), label='alpha: AIC estimate', linestyles='--', color='tab:blue')
ax.vlines(alpha_bic, results['BIC criterion'].min(), results['BIC criterion'].max(), label='alpha: BIC estimate', linestyle='--', color='tab:orange')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('criterion')
ax.set_xscale('log')
ax.legend()
_ = ax.set_title(f'Information-criterion for model selection (training time {fit_time:.2f}s)')
from sklearn.linear_model import LassoCV
start_time = time.time()
model = make_pipeline(StandardScaler(), LassoCV(cv=20)).fit(X, y)
fit_time = time.time() - start_time
import matplotlib.pyplot as plt
(ymin, ymax) = (2300, 3800)
lasso = model[-1]
plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=':')
plt.plot(lasso.alphas_, lasso.mse_path_.mean(axis=-1), color='black', label='Average across the folds', linewidth=2)
plt.axvline(lasso.alpha_, linestyle='--', color='black', label='alpha: CV estimate')
plt.ylim(ymin, ymax)
plt.xlabel('$\\alpha$')
plt.ylabel('Mean square error')
plt.legend()
_ = plt.title(f'Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)')
from sklearn.linear_model import LassoLarsCV
start_time = time.time()
model = make_pipeline(StandardScaler(), LassoLarsCV(cv=20)).fit(X, y)
fit_time = time.time() - start_time
lasso = model[-1]
plt.semilogx(lasso.cv_alphas_, lasso.mse_path_, ':')
plt.semilogx(lasso.cv_alphas_, lasso.mse_path_.mean(axis=-1), color='black', label='Average across the folds', linewidth=2)
plt.axvline(lasso.alpha_, linestyle='--', color='black', label='alpha CV')
plt.ylim(ymin, ymax)
plt.xlabel('$\\alpha$')
plt.ylabel('Mean square error')
plt.legend()
_ = plt.title(f'Mean square error on each fold: Lars (train time: {fit_time:.2f}s)')