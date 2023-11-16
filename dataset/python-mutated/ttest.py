import numpy as np
from scipy import stats
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, train_test_split

def paired_ttest_resampled(estimator1, estimator2, X, y, num_rounds=30, test_size=0.3, scoring=None, random_seed=None):
    if False:
        while True:
            i = 10
    "\n    Implements the resampled paired t test procedure\n    to compare the performance of two models\n    (also called k-hold-out paired t test).\n\n    Parameters\n    ----------\n    estimator1 : scikit-learn classifier or regressor\n\n    estimator2 : scikit-learn classifier or regressor\n\n    X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n        Training vectors, where n_samples is the number of samples and\n        n_features is the number of features.\n\n    y : array-like, shape = [n_samples]\n        Target values.\n\n    num_rounds : int (default: 30)\n        Number of resampling iterations\n        (i.e., train/test splits)\n\n    test_size : float or int (default: 0.3)\n        If float, should be between 0.0 and 1.0 and\n        represent the proportion of the dataset to use\n        as a test set.\n        If int, represents the absolute number of test exsamples.\n\n    scoring : str, callable, or None (default: None)\n        If None (default), uses 'accuracy' for sklearn classifiers\n        and 'r2' for sklearn regressors.\n        If str, uses a sklearn scoring metric string identifier, for example\n        {accuracy, f1, precision, recall, roc_auc} for classifiers,\n        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',\n        'median_absolute_error', 'r2'} for regressors.\n        If a callable object or function is provided, it has to be conform with\n        sklearn's signature ``scorer(estimator, X, y)``; see\n        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html\n        for more information.\n\n    random_seed : int or None (default: None)\n        Random seed for creating the test/train splits.\n\n    Returns\n    ----------\n    t : float\n        The t-statistic\n\n    pvalue : float\n        Two-tailed p-value.\n        If the chosen significance level is larger\n        than the p-value, we reject the null hypothesis\n        and accept that there are significant differences\n        in the two compared models.\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_resampled/\n\n    "
    if not isinstance(test_size, int) and (not isinstance(test_size, float)):
        raise ValueError('train_size must be of type int or float. Got %s.' % type(test_size))
    rng = np.random.RandomState(random_seed)
    if scoring is None:
        if estimator1._estimator_type == 'classifier':
            scoring = 'accuracy'
        elif estimator1._estimator_type == 'regressor':
            scoring = 'r2'
        else:
            raise AttributeError('Estimator must be a Classifier or Regressor.')
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring
    score_diff = []
    for i in range(num_rounds):
        randint = rng.randint(low=0, high=32767)
        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_size, random_state=randint)
        estimator1.fit(X_train, y_train)
        estimator2.fit(X_train, y_train)
        est1_score = scorer(estimator1, X_test, y_test)
        est2_score = scorer(estimator2, X_test, y_test)
        score_diff.append(est1_score - est2_score)
    avg_diff = np.mean(score_diff)
    numerator = avg_diff * np.sqrt(num_rounds)
    denominator = np.sqrt(sum([(diff - avg_diff) ** 2 for diff in score_diff]) / (num_rounds - 1))
    t_stat = numerator / denominator
    pvalue = stats.t.sf(np.abs(t_stat), num_rounds - 1) * 2.0
    return (float(t_stat), float(pvalue))

def paired_ttest_kfold_cv(estimator1, estimator2, X, y, cv=10, scoring=None, shuffle=False, random_seed=None):
    if False:
        while True:
            i = 10
    "\n    Implements the k-fold paired t test procedure\n    to compare the performance of two models.\n\n    Parameters\n    ----------\n    estimator1 : scikit-learn classifier or regressor\n\n    estimator2 : scikit-learn classifier or regressor\n\n    X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n        Training vectors, where n_samples is the number of samples and\n        n_features is the number of features.\n\n    y : array-like, shape = [n_samples]\n        Target values.\n\n    cv : int (default: 10)\n        Number of splits and iteration for the\n        cross-validation procedure\n\n    scoring : str, callable, or None (default: None)\n        If None (default), uses 'accuracy' for sklearn classifiers\n        and 'r2' for sklearn regressors.\n        If str, uses a sklearn scoring metric string identifier, for example\n        {accuracy, f1, precision, recall, roc_auc} for classifiers,\n        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',\n        'median_absolute_error', 'r2'} for regressors.\n        If a callable object or function is provided, it has to be conform with\n        sklearn's signature ``scorer(estimator, X, y)``; see\n        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html\n        for more information.\n\n    shuffle : bool (default: True)\n        Whether to shuffle the dataset for generating\n        the k-fold splits.\n\n    random_seed : int or None (default: None)\n        Random seed for shuffling the dataset\n        for generating the k-fold splits.\n        Ignored if shuffle=False.\n\n    Returns\n    ----------\n    t : float\n        The t-statistic\n\n    pvalue : float\n        Two-tailed p-value.\n        If the chosen significance level is larger\n        than the p-value, we reject the null hypothesis\n        and accept that there are significant differences\n        in the two compared models.\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_kfold_cv/\n\n    "
    if not shuffle:
        kf = KFold(n_splits=cv, shuffle=shuffle)
    else:
        kf = KFold(n_splits=cv, random_state=random_seed, shuffle=shuffle)
    if scoring is None:
        if estimator1._estimator_type == 'classifier':
            scoring = 'accuracy'
        elif estimator1._estimator_type == 'regressor':
            scoring = 'r2'
        else:
            raise AttributeError('Estimator must be a Classifier or Regressor.')
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring
    score_diff = []
    for (train_index, test_index) in kf.split(X):
        (X_train, X_test) = (X[train_index], X[test_index])
        (y_train, y_test) = (y[train_index], y[test_index])
        estimator1.fit(X_train, y_train)
        estimator2.fit(X_train, y_train)
        est1_score = scorer(estimator1, X_test, y_test)
        est2_score = scorer(estimator2, X_test, y_test)
        score_diff.append(est1_score - est2_score)
    avg_diff = np.mean(score_diff)
    numerator = avg_diff * np.sqrt(cv)
    denominator = np.sqrt(sum([(diff - avg_diff) ** 2 for diff in score_diff]) / (cv - 1))
    t_stat = numerator / denominator
    pvalue = stats.t.sf(np.abs(t_stat), cv - 1) * 2.0
    return (float(t_stat), float(pvalue))

def paired_ttest_5x2cv(estimator1, estimator2, X, y, scoring=None, random_seed=None):
    if False:
        i = 10
        return i + 15
    "\n    Implements the 5x2cv paired t test proposed\n    by Dieterrich (1998)\n    to compare the performance of two models.\n\n    Parameters\n    ----------\n    estimator1 : scikit-learn classifier or regressor\n\n    estimator2 : scikit-learn classifier or regressor\n\n    X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n        Training vectors, where n_samples is the number of samples and\n        n_features is the number of features.\n\n    y : array-like, shape = [n_samples]\n        Target values.\n\n    scoring : str, callable, or None (default: None)\n        If None (default), uses 'accuracy' for sklearn classifiers\n        and 'r2' for sklearn regressors.\n        If str, uses a sklearn scoring metric string identifier, for example\n        {accuracy, f1, precision, recall, roc_auc} for classifiers,\n        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',\n        'median_absolute_error', 'r2'} for regressors.\n        If a callable object or function is provided, it has to be conform with\n        sklearn's signature ``scorer(estimator, X, y)``; see\n        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html\n        for more information.\n\n    random_seed : int or None (default: None)\n        Random seed for creating the test/train splits.\n\n    Returns\n    ----------\n    t : float\n        The t-statistic\n\n    pvalue : float\n        Two-tailed p-value.\n        If the chosen significance level is larger\n        than the p-value, we reject the null hypothesis\n        and accept that there are significant differences\n        in the two compared models.\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/\n\n    "
    rng = np.random.RandomState(random_seed)
    if scoring is None:
        if estimator1._estimator_type == 'classifier':
            scoring = 'accuracy'
        elif estimator1._estimator_type == 'regressor':
            scoring = 'r2'
        else:
            raise AttributeError('Estimator must be a Classifier or Regressor.')
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring
    variance_sum = 0.0
    first_diff = None

    def score_diff(X_1, X_2, y_1, y_2):
        if False:
            i = 10
            return i + 15
        estimator1.fit(X_1, y_1)
        estimator2.fit(X_1, y_1)
        est1_score = scorer(estimator1, X_2, y_2)
        est2_score = scorer(estimator2, X_2, y_2)
        score_diff = est1_score - est2_score
        return score_diff
    for i in range(5):
        randint = rng.randint(low=0, high=32767)
        (X_1, X_2, y_1, y_2) = train_test_split(X, y, test_size=0.5, random_state=randint)
        score_diff_1 = score_diff(X_1, X_2, y_1, y_2)
        score_diff_2 = score_diff(X_2, X_1, y_2, y_1)
        score_mean = (score_diff_1 + score_diff_2) / 2.0
        score_var = (score_diff_1 - score_mean) ** 2 + (score_diff_2 - score_mean) ** 2
        variance_sum += score_var
        if first_diff is None:
            first_diff = score_diff_1
    numerator = first_diff
    denominator = np.sqrt(1 / 5.0 * variance_sum)
    t_stat = numerator / denominator
    pvalue = stats.t.sf(np.abs(t_stat), 5) * 2.0
    return (float(t_stat), float(pvalue))