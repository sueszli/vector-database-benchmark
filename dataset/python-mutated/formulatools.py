import statsmodels.tools.data as data_util
from patsy import dmatrices, NAAction
import numpy as np
formula_handler = {}

class NAAction(NAAction):

    def _handle_NA_drop(self, values, is_NAs, origins):
        if False:
            while True:
                i = 10
        total_mask = np.zeros(is_NAs[0].shape[0], dtype=bool)
        for is_NA in is_NAs:
            total_mask |= is_NA
        good_mask = ~total_mask
        self.missing_mask = total_mask
        return [v[good_mask, ...] for v in values]

def handle_formula_data(Y, X, formula, depth=0, missing='drop'):
    if False:
        return 10
    '\n    Returns endog, exog, and the model specification from arrays and formula.\n\n    Parameters\n    ----------\n    Y : array_like\n        Either endog (the LHS) of a model specification or all of the data.\n        Y must define __getitem__ for now.\n    X : array_like\n        Either exog or None. If all the data for the formula is provided in\n        Y then you must explicitly set X to None.\n    formula : str or patsy.model_desc\n        You can pass a handler by import formula_handler and adding a\n        key-value pair where the key is the formula object class and\n        the value is a function that returns endog, exog, formula object.\n\n    Returns\n    -------\n    endog : array_like\n        Should preserve the input type of Y,X.\n    exog : array_like\n        Should preserve the input type of Y,X. Could be None.\n    '
    if isinstance(formula, tuple(formula_handler.keys())):
        return formula_handler[type(formula)]
    na_action = NAAction(on_NA=missing)
    if X is not None:
        if data_util._is_using_pandas(Y, X):
            result = dmatrices(formula, (Y, X), depth, return_type='dataframe', NA_action=na_action)
        else:
            result = dmatrices(formula, (Y, X), depth, return_type='dataframe', NA_action=na_action)
    elif data_util._is_using_pandas(Y, None):
        result = dmatrices(formula, Y, depth, return_type='dataframe', NA_action=na_action)
    else:
        result = dmatrices(formula, Y, depth, return_type='dataframe', NA_action=na_action)
    missing_mask = getattr(na_action, 'missing_mask', None)
    if not np.any(missing_mask):
        missing_mask = None
    if len(result) > 1:
        design_info = result[1].design_info
    else:
        design_info = None
    return (result, missing_mask, design_info)

def _remove_intercept_patsy(terms):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove intercept from Patsy terms.\n    '
    from patsy.desc import INTERCEPT
    if INTERCEPT in terms:
        terms.remove(INTERCEPT)
    return terms

def _has_intercept(design_info):
    if False:
        print('Hello World!')
    from patsy.desc import INTERCEPT
    return INTERCEPT in design_info.terms

def _intercept_idx(design_info):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns boolean array index indicating which column holds the intercept.\n    '
    from patsy.desc import INTERCEPT
    from numpy import array
    return array([INTERCEPT == i for i in design_info.terms])

def make_hypotheses_matrices(model_results, test_formula):
    if False:
        print('Hello World!')
    '\n    '
    from patsy.constraint import linear_constraint
    exog_names = model_results.model.exog_names
    LC = linear_constraint(test_formula, exog_names)
    return LC