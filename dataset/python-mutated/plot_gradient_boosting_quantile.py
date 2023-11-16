"""
=====================================================
Prediction Intervals for Gradient Boosting Regression
=====================================================

This example shows how quantile regression can be used to create prediction
intervals.

"""
import numpy as np
from sklearn.model_selection import train_test_split

def f(x):
    if False:
        for i in range(10):
            print('nop')
    'The function to predict.'
    return x * np.sin(x)
rng = np.random.RandomState(42)
X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
expected_y = f(X).ravel()
sigma = 0.5 + X.ravel() / 10
noise = rng.lognormal(sigma=sigma) - np.exp(sigma ** 2 / 2)
y = expected_y + noise
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error
all_models = {}
common_params = dict(learning_rate=0.05, n_estimators=200, max_depth=2, min_samples_leaf=9, min_samples_split=9)
for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(loss='quantile', alpha=alpha, **common_params)
    all_models['q %1.2f' % alpha] = gbr.fit(X_train, y_train)
gbr_ls = GradientBoostingRegressor(loss='squared_error', **common_params)
all_models['mse'] = gbr_ls.fit(X_train, y_train)
xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
import matplotlib.pyplot as plt
y_pred = all_models['mse'].predict(xx)
y_lower = all_models['q 0.05'].predict(xx)
y_upper = all_models['q 0.95'].predict(xx)
y_med = all_models['q 0.50'].predict(xx)
fig = plt.figure(figsize=(10, 10))
plt.plot(xx, f(xx), 'g:', linewidth=3, label='$f(x) = x\\,\\sin(x)$')
plt.plot(X_test, y_test, 'b.', markersize=10, label='Test observations')
plt.plot(xx, y_med, 'r-', label='Predicted median')
plt.plot(xx, y_pred, 'r-', label='Predicted mean')
plt.plot(xx, y_upper, 'k-')
plt.plot(xx, y_lower, 'k-')
plt.fill_between(xx.ravel(), y_lower, y_upper, alpha=0.4, label='Predicted 90% interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 25)
plt.legend(loc='upper left')
plt.show()
import pandas as pd

def highlight_min(x):
    if False:
        while True:
            i = 10
    x_min = x.min()
    return ['font-weight: bold' if v == x_min else '' for v in x]
results = []
for (name, gbr) in sorted(all_models.items()):
    metrics = {'model': name}
    y_pred = gbr.predict(X_train)
    for alpha in [0.05, 0.5, 0.95]:
        metrics['pbl=%1.2f' % alpha] = mean_pinball_loss(y_train, y_pred, alpha=alpha)
    metrics['MSE'] = mean_squared_error(y_train, y_pred)
    results.append(metrics)
pd.DataFrame(results).set_index('model').style.apply(highlight_min)
results = []
for (name, gbr) in sorted(all_models.items()):
    metrics = {'model': name}
    y_pred = gbr.predict(X_test)
    for alpha in [0.05, 0.5, 0.95]:
        metrics['pbl=%1.2f' % alpha] = mean_pinball_loss(y_test, y_pred, alpha=alpha)
    metrics['MSE'] = mean_squared_error(y_test, y_pred)
    results.append(metrics)
pd.DataFrame(results).set_index('model').style.apply(highlight_min)

def coverage_fraction(y, y_low, y_high):
    if False:
        i = 10
        return i + 15
    return np.mean(np.logical_and(y >= y_low, y <= y_high))
coverage_fraction(y_train, all_models['q 0.05'].predict(X_train), all_models['q 0.95'].predict(X_train))
coverage_fraction(y_test, all_models['q 0.05'].predict(X_test), all_models['q 0.95'].predict(X_test))
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from pprint import pprint
param_grid = dict(learning_rate=[0.05, 0.1, 0.2], max_depth=[2, 5, 10], min_samples_leaf=[1, 5, 10, 20], min_samples_split=[5, 10, 20, 30, 50])
alpha = 0.05
neg_mean_pinball_loss_05p_scorer = make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False)
gbr = GradientBoostingRegressor(loss='quantile', alpha=alpha, random_state=0)
search_05p = HalvingRandomSearchCV(gbr, param_grid, resource='n_estimators', max_resources=250, min_resources=50, scoring=neg_mean_pinball_loss_05p_scorer, n_jobs=2, random_state=0).fit(X_train, y_train)
pprint(search_05p.best_params_)
from sklearn.base import clone
alpha = 0.95
neg_mean_pinball_loss_95p_scorer = make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False)
search_95p = clone(search_05p).set_params(estimator__alpha=alpha, scoring=neg_mean_pinball_loss_95p_scorer)
search_95p.fit(X_train, y_train)
pprint(search_95p.best_params_)
y_lower = search_05p.predict(xx)
y_upper = search_95p.predict(xx)
fig = plt.figure(figsize=(10, 10))
plt.plot(xx, f(xx), 'g:', linewidth=3, label='$f(x) = x\\,\\sin(x)$')
plt.plot(X_test, y_test, 'b.', markersize=10, label='Test observations')
plt.plot(xx, y_upper, 'k-')
plt.plot(xx, y_lower, 'k-')
plt.fill_between(xx.ravel(), y_lower, y_upper, alpha=0.4, label='Predicted 90% interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 25)
plt.legend(loc='upper left')
plt.title('Prediction with tuned hyper-parameters')
plt.show()
coverage_fraction(y_train, search_05p.predict(X_train), search_95p.predict(X_train))
coverage_fraction(y_test, search_05p.predict(X_test), search_95p.predict(X_test))