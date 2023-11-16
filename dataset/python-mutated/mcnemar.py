from itertools import combinations
import numpy as np
import scipy.stats

def mcnemar_table(y_target, y_model1, y_model2):
    if False:
        while True:
            i = 10
    "\n    Compute a 2x2 contigency table for McNemar's test.\n\n    Parameters\n    -----------\n    y_target : array-like, shape=[n_samples]\n        True class labels as 1D NumPy array.\n    y_model1 : array-like, shape=[n_samples]\n        Predicted class labels from model as 1D NumPy array.\n    y_model2 : array-like, shape=[n_samples]\n        Predicted class labels from model 2 as 1D NumPy array.\n\n    Returns\n    ----------\n    tb : array-like, shape=[2, 2]\n       2x2 contingency table with the following contents:\n       a: tb[0, 0]: # of samples that both models predicted correctly\n       b: tb[0, 1]: # of samples that model 1 got right and model 2 got wrong\n       c: tb[1, 0]: # of samples that model 2 got right and model 1 got wrong\n       d: tb[1, 1]: # of samples that both models predicted incorrectly\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_table/\n\n    "
    for ary in (y_target, y_model1, y_model2):
        if len(ary.shape) != 1:
            raise ValueError('One or more input arrays are not 1-dimensional.')
    if y_target.shape[0] != y_model1.shape[0]:
        raise ValueError('y_target and y_model1 contain a different number of elements.')
    if y_target.shape[0] != y_model2.shape[0]:
        raise ValueError('y_target and y_model2 contain a different number of elements.')
    m1_vs_true = (y_target == y_model1).astype(int)
    m2_vs_true = (y_target == y_model2).astype(int)
    plus_true = m1_vs_true + m2_vs_true
    minus_true = m1_vs_true - m2_vs_true
    tb = np.zeros((2, 2), dtype=int)
    tb[0, 0] = np.sum(plus_true == 2)
    tb[0, 1] = np.sum(minus_true == 1)
    tb[1, 0] = np.sum(minus_true == -1)
    tb[1, 1] = np.sum(plus_true == 0)
    return tb

def mcnemar_tables(y_target, *y_model_predictions):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute multiple 2x2 contigency tables for McNemar\'s\n    test or Cochran\'s Q test.\n\n    Parameters\n    -----------\n    y_target : array-like, shape=[n_samples]\n        True class labels as 1D NumPy array.\n\n    y_model_predictions : array-like, shape=[n_samples]\n        Predicted class labels for a model.\n\n    Returns\n    ----------\n\n    tables : dict\n        Dictionary of NumPy arrays with shape=[2, 2]. Each dictionary\n        key names the two models to be compared based on the order the\n        models were passed as `*y_model_predictions`. The number of\n        dictionary entries is equal to the number of pairwise combinations\n        between the m models, i.e., "m choose 2."\n\n        For example the following target array (containing the true labels)\n        and 3 models\n\n        - y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])\n        - y_mod0 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0])\n        - y_mod1 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0])\n        - y_mod2 = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0])\n\n        would result in the following dictionary:\n\n\n        {\'model_0 vs model_1\': array([[ 4.,  1.],\n                                      [ 2.,  3.]]),\n         \'model_0 vs model_2\': array([[ 3.,  0.],\n                                      [ 3.,  4.]]),\n         \'model_1 vs model_2\': array([[ 3.,  0.],\n                                      [ 2.,  5.]])}\n\n        Each array is structured in the following way:\n\n        - tb[0, 0]: # of samples that both models predicted correctly\n        - tb[0, 1]: # of samples that model a got right and model b got wrong\n        - tb[1, 0]: # of samples that model b got right and model a got wrong\n        - tb[1, 1]: # of samples that both models predicted incorrectly\n\n    Examples\n    -----------\n\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_tables/\n\n    '
    model_lens = set()
    y_model_predictions = list(y_model_predictions)
    for ary in [y_target] + y_model_predictions:
        if len(ary.shape) != 1:
            raise ValueError('One or more input arrays are not 1-dimensional.')
        model_lens.add(ary.shape[0])
    if len(model_lens) > 1:
        raise ValueError('Each prediction array must have the same number of samples.')
    num_models = len(y_model_predictions)
    if num_models < 2:
        raise ValueError('Provide at least 2 model prediction arrays.')
    tables = {}
    for comb in combinations(range(num_models), 2):
        tb = np.zeros((2, 2))
        model1_vs_true = (y_target == y_model_predictions[comb[0]]).astype(int)
        model2_vs_true = (y_target == y_model_predictions[comb[1]]).astype(int)
        plus_true = model1_vs_true + model2_vs_true
        minus_true = model1_vs_true - model2_vs_true
        tb[0, 0] = np.sum(plus_true == 2)
        tb[0, 1] = np.sum(minus_true == 1)
        tb[1, 0] = np.sum(minus_true == -1)
        tb[1, 1] = np.sum(plus_true == 0)
        name_str = 'model_%s vs model_%s' % (comb[0], comb[1])
        tables[name_str] = tb
    return tables

def mcnemar(ary, corrected=True, exact=False):
    if False:
        i = 10
        return i + 15
    "\n    McNemar test for paired nominal data\n\n    Parameters\n    -----------\n    ary : array-like, shape=[2, 2]\n        2 x 2 contigency table (as returned by evaluate.mcnemar_table),\n        where\n        a: ary[0, 0]: # of samples that both models predicted correctly\n        b: ary[0, 1]: # of samples that model 1 got right and model 2 got wrong\n        c: ary[1, 0]: # of samples that model 2 got right and model 1 got wrong\n        d: aryCell [1, 1]: # of samples that both models predicted incorrectly\n    corrected : array-like, shape=[n_samples] (default: True)\n        Uses Edward's continuity correction for chi-squared if `True`\n    exact : bool, (default: False)\n        If `True`, uses an exact binomial test comparing b to\n        a binomial distribution with n = b + c and p = 0.5.\n        It is highly recommended to use `exact=True` for sample sizes < 25\n        since chi-squared is not well-approximated\n        by the chi-squared distribution!\n\n    Returns\n    -----------\n    chi2, p : float or None, float\n        Returns the chi-squared value and the p-value;\n        if `exact=True` (default: `False`), `chi2` is `None`\n\n    Examples\n    -----------\n\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/\n\n    "
    if not ary.shape == (2, 2):
        raise ValueError('Input array must be a 2x2 array.')
    b = ary[0, 1]
    c = ary[1, 0]
    n = b + c
    if not exact:
        if corrected:
            chi2 = (abs(ary[0, 1] - ary[1, 0]) - 1.0) ** 2 / float(n)
        else:
            chi2 = (ary[0, 1] - ary[1, 0]) ** 2 / float(n)
        p = scipy.stats.distributions.chi2.sf(chi2, 1)
    else:
        chi2 = None
        p = min(scipy.stats.binom.cdf(min(b, c), b + c, 0.5) * 2.0, 1.0)
    return (chi2, p)