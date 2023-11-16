"""
Small helpers that help find and filter missing data.
"""
import numpy as np
import warnings
from yellowbrick.exceptions import DataWarning

def filter_missing(X, y=None):
    if False:
        while True:
            i = 10
    "\n    Removes rows that contain np.nan values in data. If y is given,\n    X and y will be filtered together so that their shape remains identical.\n    For example, rows in X with nans will also remove rows in y, or rows in y\n    with np.nans will also remove corresponding rows in X.\n\n    Parameters\n    ------------\n    X : array-like\n        Data in shape (m, n) that possibly contains np.nan values\n\n    y : array-like, optional\n        Data in shape (m, 1) that possibly contains np.nan values\n\n    Returns\n    --------\n    X' : np.array\n       Possibly transformed X with any row containing np.nan removed\n\n    y' : np.array\n        If y is given, will also return possibly transformed y to match the\n        shape of X'.\n\n    Notes\n    ------\n    This function will return either a np.array if only X is passed or a tuple\n    if both X and y is passed. Because all return values are indexable, it is\n    important to recognize what is being passed to the function to determine\n    its output.\n    "
    if y is not None:
        return filter_missing_X_and_y(X, y)
    else:
        return X[~np.isnan(X).any(axis=1)]

def filter_missing_X_and_y(X, y):
    if False:
        print('Hello World!')
    'Remove rows from X and y where either contains nans.'
    y_nans = np.isnan(y)
    x_nans = np.isnan(X).any(axis=1)
    unioned_nans = np.logical_or(x_nans, y_nans)
    return (X[~unioned_nans], y[~unioned_nans])

def warn_if_nans_exist(X):
    if False:
        for i in range(10):
            print('nop')
    'Warn if nans exist in a numpy array.'
    null_count = count_rows_with_nans(X)
    total = len(X)
    percent = 100 * null_count / total
    if null_count > 0:
        warning_message = 'Warning! Found {} rows of {} ({:0.2f}%) with nan values. Only complete rows will be plotted.'.format(null_count, total, percent)
        warnings.warn(warning_message, DataWarning)

def count_rows_with_nans(X):
    if False:
        return 10
    'Count the number of rows in 2D arrays that contain any nan values.'
    if X.ndim == 2:
        return np.where(np.isnan(X).sum(axis=1) != 0, 1, 0).sum()

def count_nan_elements(data):
    if False:
        while True:
            i = 10
    'Count the number of elements in 1D arrays that are nan values.'
    if data.ndim == 1:
        return np.isnan(data).sum()