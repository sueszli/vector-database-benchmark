"""
=========================================================================
Comparing randomized search and grid search for hyperparameter estimation
=========================================================================

Compare randomized search and grid search for optimizing hyperparameters of a
linear SVM with SGD training.
All parameters that influence the learning are searched simultaneously
(except for the number of estimators, which poses a time / quality tradeoff).

The randomized search and the grid search explore exactly the same space of
parameters. The result in parameter settings is quite similar, while the run
time for randomized search is drastically lower.

The performance is may slightly worse for the randomized search, and is likely
due to a noise effect and would not carry over to a held-out test set.

Note that in practice, one would not search over this many different parameters
simultaneously using grid search, but pick only the ones deemed most important.

"""
from time import time
import numpy as np
import scipy.stats as stats
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
(X, y) = load_digits(return_X_y=True, n_class=3)
clf = SGDClassifier(loss='hinge', penalty='elasticnet', fit_intercept=True)

def report(results, n_top=3):
    if False:
        print('Hello World!')
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(results['mean_test_score'][candidate], results['std_test_score'][candidate]))
            print('Parameters: {0}'.format(results['params'][candidate]))
            print('')
param_dist = {'average': [True, False], 'l1_ratio': stats.uniform(0, 1), 'alpha': stats.loguniform(0.01, 1.0)}
n_iter_search = 15
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)
start = time()
random_search.fit(X, y)
print('RandomizedSearchCV took %.2f seconds for %d candidates parameter settings.' % (time() - start, n_iter_search))
report(random_search.cv_results_)
param_grid = {'average': [True, False], 'l1_ratio': np.linspace(0, 1, num=10), 'alpha': np.power(10, np.arange(-2, 1, dtype=float))}
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)
print('GridSearchCV took %.2f seconds for %d candidate parameter settings.' % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)