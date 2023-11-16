from itertools import product
import numpy as np
from sklearn.base import clone
from .bootstrap_outofbag import BootstrapOutOfBag

def _check_arrays(X, y=None):
    if False:
        i = 10
        return i + 15
    if isinstance(X, list):
        raise ValueError('X must be a numpy array')
    if not len(X.shape) == 2:
        raise ValueError('X must be a 2D array. Try X[:, numpy.newaxis]')
    try:
        if y is None:
            return
    except AttributeError:
        if not len(y.shape) == 1:
            raise ValueError('y must be a 1D array.')
    if not len(y) == X.shape[0]:
        raise ValueError('X and y must contain thesame number of samples')

def no_information_rate(targets, predictions, loss_fn):
    if False:
        for i in range(10):
            print('nop')
    combinations = np.array(list(product(targets, predictions)))
    return loss_fn(combinations[:, 0], combinations[:, 1])

def accuracy(targets, predictions):
    if False:
        return 10
    return np.mean(np.array(targets) == np.array(predictions))

def mse(targets, predictions):
    if False:
        i = 10
        return i + 15
    return np.mean((np.array(targets) - np.array(predictions)) ** 2)

def bootstrap_point632_score(estimator, X, y, n_splits=200, method='.632', scoring_func=None, predict_proba=False, random_seed=None, clone_estimator=True, **fit_params):
    if False:
        for i in range(10):
            print('nop')
    '\n    Implementation of the .632 [1] and .632+ [2] bootstrap\n    for supervised learning\n\n    References:\n\n    - [1] Efron, Bradley. 1983. "Estimating the Error Rate\n      of a Prediction Rule: Improvement on Cross-Validation."\n      Journal of the American Statistical Association\n      78 (382): 316. doi:10.2307/2288636.\n    - [2] Efron, Bradley, and Robert Tibshirani. 1997.\n      "Improvements on Cross-Validation: The .632+ Bootstrap Method."\n      Journal of the American Statistical Association\n      92 (438): 548. doi:10.2307/2965703.\n\n    Parameters\n    ----------\n    estimator : object\n        An estimator for classification or regression that\n        follows the scikit-learn API and implements "fit" and "predict"\n        methods.\n\n    X : array-like\n        The data to fit. Can be, for example a list, or an array at least 2d.\n\n    y : array-like, optional, default: None\n        The target variable to try to predict in the case of\n        supervised learning.\n\n    n_splits : int (default=200)\n        Number of bootstrap iterations.\n        Must be larger than 1.\n\n    method : str (default=\'.632\')\n        The bootstrap method, which can be either\n        - 1) \'.632\' bootstrap (default)\n        - 2) \'.632+\' bootstrap\n        - 3) \'oob\' (regular out-of-bag, no weighting)\n        for comparison studies.\n\n    scoring_func : callable,\n        Score function (or loss function) with signature\n        ``scoring_func(y, y_pred, **kwargs)``.\n        If none, uses classification accuracy if the\n        estimator is a classifier and mean squared error\n        if the estimator is a regressor.\n\n    predict_proba : bool\n        Whether to use the `predict_proba` function for the\n        `estimator` argument. This is to be used in conjunction\n        with `scoring_func` which takes in probability values\n        instead of actual predictions.\n        For example, if the scoring_func is\n        :meth:`sklearn.metrics.roc_auc_score`, then use\n        `predict_proba=True`.\n        Note that this requires `estimator` to have\n        `predict_proba` method implemented.\n\n    random_seed : int (default=None)\n        If int, random_seed is the seed used by\n        the random number generator.\n\n    clone_estimator : bool (default=True)\n        Clones the estimator if true, otherwise fits\n        the original.\n\n    fit_params : additional parameters\n        Additional parameters to be passed to the .fit() function of the\n        estimator when it is fit to the bootstrap samples.\n\n\n    Returns\n    -------\n    scores : array of float, shape=(len(list(n_splits)),)\n        Array of scores of the estimator for each bootstrap\n        replicate.\n\n    Examples\n    --------\n    >>> from sklearn import datasets, linear_model\n    >>> from mlxtend.evaluate import bootstrap_point632_score\n    >>> iris = datasets.load_iris()\n    >>> X = iris.data\n    >>> y = iris.target\n    >>> lr = linear_model.LogisticRegression()\n    >>> scores = bootstrap_point632_score(lr, X, y)\n    >>> acc = np.mean(scores)\n    >>> print(\'Accuracy:\', acc)\n    0.953023146884\n    >>> lower = np.percentile(scores, 2.5)\n    >>> upper = np.percentile(scores, 97.5)\n    >>> print(\'95%% Confidence interval: [%.2f, %.2f]\' % (lower, upper))\n    95% Confidence interval: [0.90, 0.98]\n\n    For more usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap_point632_score/\n\n    '
    if not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError('Number of splits must be greater than 1. Got %s.' % n_splits)
    allowed_methods = ('.632', '.632+', 'oob')
    if not isinstance(method, str) or method not in allowed_methods:
        raise ValueError('The `method` must be in %s. Got %s.' % (allowed_methods, method))
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    _check_arrays(X, y)
    if clone_estimator:
        cloned_est = clone(estimator)
    else:
        cloned_est = estimator
    if scoring_func is None:
        if cloned_est._estimator_type == 'classifier':
            scoring_func = accuracy
        elif cloned_est._estimator_type == 'regressor':
            scoring_func = mse
        else:
            raise AttributeError('Estimator type undefined.Please provide a scoring_func argument.')
    if not predict_proba:
        predict_func = cloned_est.predict
    else:
        if not getattr(cloned_est, 'predict_proba', None):
            raise RuntimeError(f'The estimator {cloned_est} does not support predicting probabilities via `predict_proba` function.')
        predict_func = cloned_est.predict_proba
    oob = BootstrapOutOfBag(n_splits=n_splits, random_seed=random_seed)
    scores = np.empty(dtype=float, shape=(n_splits,))
    cnt = 0
    for (train, test) in oob.split(X):
        cloned_est.fit(X[train], y[train], **fit_params)
        predicted_test_val = predict_func(X[test])
        if method in ('.632', '.632+'):
            predicted_train_val = predict_func(X)
        if predict_proba:
            len_uniq = np.unique(y)
            if len(len_uniq) == 2:
                predicted_train_val = predicted_train_val[:, 1]
                predicted_test_val = predicted_test_val[:, 1]
        test_acc = scoring_func(y[test], predicted_test_val)
        if method == 'oob':
            acc = test_acc
        else:
            test_err = 1 - test_acc
            train_err = 1 - scoring_func(y, predicted_train_val)
            if method == '.632+':
                gamma = 1 - no_information_rate(y, cloned_est.predict(X), scoring_func)
                R = (test_err - train_err) / (gamma - train_err)
                weight = 0.632 / (1 - 0.368 * R)
            else:
                weight = 0.632
            acc = 1 - (weight * test_err + (1.0 - weight) * train_err)
        scores[cnt] = acc
        cnt += 1
    return scores