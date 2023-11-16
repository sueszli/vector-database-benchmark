from __future__ import annotations
from statsmodels.compat.python import lrange, Literal
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from statsmodels.tools.data import _is_recarray, _is_using_pandas
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.typing import NDArray
from statsmodels.tools.validation import array_like, bool_like, int_like, string_like
__all__ = ['lagmat', 'lagmat2ds', 'add_trend', 'duplication_matrix', 'elimination_matrix', 'commutation_matrix', 'vec', 'vech', 'unvec', 'unvech', 'freq_to_period']

def add_trend(x, trend='c', prepend=False, has_constant='skip'):
    if False:
        while True:
            i = 10
    "\n    Add a trend and/or constant to an array.\n\n    Parameters\n    ----------\n    x : array_like\n        Original array of data.\n    trend : str {'n', 'c', 't', 'ct', 'ctt'}\n        The trend to add.\n\n        * 'n' add no trend.\n        * 'c' add constant only.\n        * 't' add trend only.\n        * 'ct' add constant and linear trend.\n        * 'ctt' add constant and linear and quadratic trend.\n    prepend : bool\n        If True, prepends the new data to the columns of X.\n    has_constant : str {'raise', 'add', 'skip'}\n        Controls what happens when trend is 'c' and a constant column already\n        exists in x. 'raise' will raise an error. 'add' will add a column of\n        1s. 'skip' will return the data without change. 'skip' is the default.\n\n    Returns\n    -------\n    array_like\n        The original data with the additional trend columns.  If x is a\n        pandas Series or DataFrame, then the trend column names are 'const',\n        'trend' and 'trend_squared'.\n\n    See Also\n    --------\n    statsmodels.tools.tools.add_constant\n        Add a constant column to an array.\n\n    Notes\n    -----\n    Returns columns as ['ctt','ct','c'] whenever applicable. There is currently\n    no checking for an existing trend.\n    "
    prepend = bool_like(prepend, 'prepend')
    trend = string_like(trend, 'trend', options=('n', 'c', 't', 'ct', 'ctt'))
    has_constant = string_like(has_constant, 'has_constant', options=('raise', 'add', 'skip'))
    columns = ['const', 'trend', 'trend_squared']
    if trend == 'n':
        return x.copy()
    elif trend == 'c':
        columns = columns[:1]
        trendorder = 0
    elif trend == 'ct' or trend == 't':
        columns = columns[:2]
        if trend == 't':
            columns = columns[1:2]
        trendorder = 1
    elif trend == 'ctt':
        trendorder = 2
    if _is_recarray(x):
        from statsmodels.tools.sm_exceptions import recarray_exception
        raise NotImplementedError(recarray_exception)
    is_pandas = _is_using_pandas(x, None)
    if is_pandas:
        if isinstance(x, pd.Series):
            x = pd.DataFrame(x)
        else:
            x = x.copy()
    else:
        x = np.asanyarray(x)
    nobs = len(x)
    trendarr = np.vander(np.arange(1, nobs + 1, dtype=np.float64), trendorder + 1)
    trendarr = np.fliplr(trendarr)
    if trend == 't':
        trendarr = trendarr[:, 1]
    if 'c' in trend:
        if is_pandas:

            def safe_is_const(s):
                if False:
                    for i in range(10):
                        print('nop')
                try:
                    return np.ptp(s) == 0.0 and np.any(s != 0.0)
                except:
                    return False
            col_const = x.apply(safe_is_const, 0)
        else:
            ptp0 = np.ptp(np.asanyarray(x), axis=0)
            col_is_const = ptp0 == 0
            nz_const = col_is_const & (x[0] != 0)
            col_const = nz_const
        if np.any(col_const):
            if has_constant == 'raise':
                if x.ndim == 1:
                    base_err = 'x is constant.'
                else:
                    columns = np.arange(x.shape[1])[col_const]
                    if isinstance(x, pd.DataFrame):
                        columns = x.columns
                    const_cols = ', '.join([str(c) for c in columns])
                    base_err = f'x contains one or more constant columns. Column(s) {const_cols} are constant.'
                msg = f"{base_err} Adding a constant with trend='{trend}' is not allowed."
                raise ValueError(msg)
            elif has_constant == 'skip':
                columns = columns[1:]
                trendarr = trendarr[:, 1:]
    order = 1 if prepend else -1
    if is_pandas:
        trendarr = pd.DataFrame(trendarr, index=x.index, columns=columns)
        x = [trendarr, x]
        x = pd.concat(x[::order], axis=1)
    else:
        x = [trendarr, x]
        x = np.column_stack(x[::order])
    return x

def add_lag(x, col=None, lags=1, drop=False, insert=True):
    if False:
        i = 10
        return i + 15
    "\n    Returns an array with lags included given an array.\n\n    Parameters\n    ----------\n    x : array_like\n        An array or NumPy ndarray subclass. Can be either a 1d or 2d array with\n        observations in columns.\n    col : int or None\n        `col` can be an int of the zero-based column index. If it's a\n        1d array `col` can be None.\n    lags : int\n        The number of lags desired.\n    drop : bool\n        Whether to keep the contemporaneous variable for the data.\n    insert : bool or int\n        If True, inserts the lagged values after `col`. If False, appends\n        the data. If int inserts the lags at int.\n\n    Returns\n    -------\n    array : ndarray\n        Array with lags\n\n    Examples\n    --------\n\n    >>> import statsmodels.api as sm\n    >>> data = sm.datasets.macrodata.load()\n    >>> data = data.data[['year','quarter','realgdp','cpi']]\n    >>> data = sm.tsa.add_lag(data, 'realgdp', lags=2)\n\n    Notes\n    -----\n    Trims the array both forward and backward, so that the array returned\n    so that the length of the returned array is len(`X`) - lags. The lags are\n    returned in increasing order, ie., t-1,t-2,...,t-lags\n    "
    lags = int_like(lags, 'lags')
    drop = bool_like(drop, 'drop')
    x = array_like(x, 'x', ndim=2)
    if col is None:
        col = 0
    if col < 0:
        col = x.shape[1] + col
    if x.ndim == 1:
        x = x[:, None]
    contemp = x[:, col]
    if insert is True:
        ins_idx = col + 1
    elif insert is False:
        ins_idx = x.shape[1]
    else:
        if insert < 0:
            insert = x.shape[1] + insert + 1
        if insert > x.shape[1]:
            insert = x.shape[1]
            warnings.warn('insert > number of variables, inserting at the last position', ValueWarning)
        ins_idx = insert
    ndlags = lagmat(contemp, lags, trim='Both')
    first_cols = lrange(ins_idx)
    last_cols = lrange(ins_idx, x.shape[1])
    if drop:
        if col in first_cols:
            first_cols.pop(first_cols.index(col))
        else:
            last_cols.pop(last_cols.index(col))
    return np.column_stack((x[lags:, first_cols], ndlags, x[lags:, last_cols]))

def detrend(x, order=1, axis=0):
    if False:
        while True:
            i = 10
    '\n    Detrend an array with a trend of given order along axis 0 or 1.\n\n    Parameters\n    ----------\n    x : array_like, 1d or 2d\n        Data, if 2d, then each row or column is independently detrended with\n        the same trendorder, but independent trend estimates.\n    order : int\n        The polynomial order of the trend, zero is constant, one is\n        linear trend, two is quadratic trend.\n    axis : int\n        Axis can be either 0, observations by rows, or 1, observations by\n        columns.\n\n    Returns\n    -------\n    ndarray\n        The detrended series is the residual of the linear regression of the\n        data on the trend of given order.\n    '
    order = int_like(order, 'order')
    axis = int_like(axis, 'axis')
    if x.ndim == 2 and int(axis) == 1:
        x = x.T
    elif x.ndim > 2:
        raise NotImplementedError('x.ndim > 2 is not implemented until it is needed')
    nobs = x.shape[0]
    if order == 0:
        resid = x - x.mean(axis=0)
    else:
        trends = np.vander(np.arange(float(nobs)), N=order + 1)
        beta = np.linalg.pinv(trends).dot(x)
        resid = x - np.dot(trends, beta)
    if x.ndim == 2 and int(axis) == 1:
        resid = resid.T
    return resid

def lagmat(x, maxlag: int, trim: Literal['forward', 'backward', 'both', 'none']='forward', original: Literal['ex', 'sep', 'in']='ex', use_pandas: bool=False) -> NDArray | DataFrame | tuple[NDArray, NDArray] | tuple[DataFrame, DataFrame]:
    if False:
        i = 10
        return i + 15
    '\n    Create 2d array of lags.\n\n    Parameters\n    ----------\n    x : array_like\n        Data; if 2d, observation in rows and variables in columns.\n    maxlag : int\n        All lags from zero to maxlag are included.\n    trim : {\'forward\', \'backward\', \'both\', \'none\', None}\n        The trimming method to use.\n\n        * \'forward\' : trim invalid observations in front.\n        * \'backward\' : trim invalid initial observations.\n        * \'both\' : trim invalid observations on both sides.\n        * \'none\', None : no trimming of observations.\n    original : {\'ex\',\'sep\',\'in\'}\n        How the original is treated.\n\n        * \'ex\' : drops the original array returning only the lagged values.\n        * \'in\' : returns the original array and the lagged values as a single\n          array.\n        * \'sep\' : returns a tuple (original array, lagged values). The original\n                  array is truncated to have the same number of rows as\n                  the returned lagmat.\n    use_pandas : bool\n        If true, returns a DataFrame when the input is a pandas\n        Series or DataFrame.  If false, return numpy ndarrays.\n\n    Returns\n    -------\n    lagmat : ndarray\n        The array with lagged observations.\n    y : ndarray, optional\n        Only returned if original == \'sep\'.\n\n    Notes\n    -----\n    When using a pandas DataFrame or Series with use_pandas=True, trim can only\n    be \'forward\' or \'both\' since it is not possible to consistently extend\n    index values.\n\n    Examples\n    --------\n    >>> from statsmodels.tsa.tsatools import lagmat\n    >>> import numpy as np\n    >>> X = np.arange(1,7).reshape(-1,2)\n    >>> lagmat(X, maxlag=2, trim="forward", original=\'in\')\n    array([[ 1.,  2.,  0.,  0.,  0.,  0.],\n       [ 3.,  4.,  1.,  2.,  0.,  0.],\n       [ 5.,  6.,  3.,  4.,  1.,  2.]])\n\n    >>> lagmat(X, maxlag=2, trim="backward", original=\'in\')\n    array([[ 5.,  6.,  3.,  4.,  1.,  2.],\n       [ 0.,  0.,  5.,  6.,  3.,  4.],\n       [ 0.,  0.,  0.,  0.,  5.,  6.]])\n\n    >>> lagmat(X, maxlag=2, trim="both", original=\'in\')\n    array([[ 5.,  6.,  3.,  4.,  1.,  2.]])\n\n    >>> lagmat(X, maxlag=2, trim="none", original=\'in\')\n    array([[ 1.,  2.,  0.,  0.,  0.,  0.],\n       [ 3.,  4.,  1.,  2.,  0.,  0.],\n       [ 5.,  6.,  3.,  4.,  1.,  2.],\n       [ 0.,  0.,  5.,  6.,  3.,  4.],\n       [ 0.,  0.,  0.,  0.,  5.,  6.]])\n    '
    maxlag = int_like(maxlag, 'maxlag')
    use_pandas = bool_like(use_pandas, 'use_pandas')
    trim = string_like(trim, 'trim', optional=True, options=('forward', 'backward', 'both', 'none'))
    original = string_like(original, 'original', options=('ex', 'sep', 'in'))
    orig = x
    x = array_like(x, 'x', ndim=2, dtype=None)
    is_pandas = _is_using_pandas(orig, None) and use_pandas
    trim = 'none' if trim is None else trim
    trim = trim.lower()
    if is_pandas and trim in ('none', 'backward'):
        raise ValueError("trim cannot be 'none' or 'backward' when used on Series or DataFrames")
    dropidx = 0
    (nobs, nvar) = x.shape
    if original in ['ex', 'sep']:
        dropidx = nvar
    if maxlag >= nobs:
        raise ValueError('maxlag should be < nobs')
    lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)))
    for k in range(0, int(maxlag + 1)):
        lm[maxlag - k:nobs + maxlag - k, nvar * (maxlag - k):nvar * (maxlag - k + 1)] = x
    if trim in ('none', 'forward'):
        startobs = 0
    elif trim in ('backward', 'both'):
        startobs = maxlag
    else:
        raise ValueError('trim option not valid')
    if trim in ('none', 'backward'):
        stopobs = len(lm)
    else:
        stopobs = nobs
    if is_pandas:
        x = orig
        if isinstance(x, DataFrame):
            x_columns = [str(c) for c in x.columns]
            if len(set(x_columns)) != x.shape[1]:
                raise ValueError('Columns names must be distinct after conversion to string (if not already strings).')
        else:
            x_columns = [str(x.name)]
        columns = [str(col) for col in x_columns]
        for lag in range(maxlag):
            lag_str = str(lag + 1)
            columns.extend([str(col) + '.L.' + lag_str for col in x_columns])
        lm = DataFrame(lm[:stopobs], index=x.index, columns=columns)
        lags = lm.iloc[startobs:]
        if original in ('sep', 'ex'):
            leads = lags[x_columns]
            lags = lags.drop(x_columns, axis=1)
    else:
        lags = lm[startobs:stopobs, dropidx:]
        if original == 'sep':
            leads = lm[startobs:stopobs, :dropidx]
    if original == 'sep':
        return (lags, leads)
    else:
        return lags

def lagmat2ds(x, maxlag0, maxlagex=None, dropex=0, trim='forward', use_pandas=False):
    if False:
        i = 10
        return i + 15
    "\n    Generate lagmatrix for 2d array, columns arranged by variables.\n\n    Parameters\n    ----------\n    x : array_like\n        Data, 2d. Observations in rows and variables in columns.\n    maxlag0 : int\n        The first variable all lags from zero to maxlag are included.\n    maxlagex : {None, int}\n        The max lag for all other variables all lags from zero to maxlag are\n        included.\n    dropex : int\n        Exclude first dropex lags from other variables. For all variables,\n        except the first, lags from dropex to maxlagex are included.\n    trim : str\n        The trimming method to use.\n\n        * 'forward' : trim invalid observations in front.\n        * 'backward' : trim invalid initial observations.\n        * 'both' : trim invalid observations on both sides.\n        * 'none' : no trimming of observations.\n    use_pandas : bool\n        If true, returns a DataFrame when the input is a pandas\n        Series or DataFrame.  If false, return numpy ndarrays.\n\n    Returns\n    -------\n    ndarray\n        The array with lagged observations, columns ordered by variable.\n\n    Notes\n    -----\n    Inefficient implementation for unequal lags, implemented for convenience.\n    "
    maxlag0 = int_like(maxlag0, 'maxlag0')
    maxlagex = int_like(maxlagex, 'maxlagex', optional=True)
    trim = string_like(trim, 'trim', optional=True, options=('forward', 'backward', 'both', 'none'))
    if maxlagex is None:
        maxlagex = maxlag0
    maxlag = max(maxlag0, maxlagex)
    is_pandas = _is_using_pandas(x, None)
    if x.ndim == 1:
        if is_pandas:
            x = pd.DataFrame(x)
        else:
            x = x[:, None]
    elif x.ndim == 0 or x.ndim > 2:
        raise ValueError('Only supports 1 and 2-dimensional data.')
    (nobs, nvar) = x.shape
    if is_pandas and use_pandas:
        lags = lagmat(x.iloc[:, 0], maxlag, trim=trim, original='in', use_pandas=True)
        lagsli = [lags.iloc[:, :maxlag0 + 1]]
        for k in range(1, nvar):
            lags = lagmat(x.iloc[:, k], maxlag, trim=trim, original='in', use_pandas=True)
            lagsli.append(lags.iloc[:, dropex:maxlagex + 1])
        return pd.concat(lagsli, axis=1)
    elif is_pandas:
        x = np.asanyarray(x)
    lagsli = [lagmat(x[:, 0], maxlag, trim=trim, original='in')[:, :maxlag0 + 1]]
    for k in range(1, nvar):
        lagsli.append(lagmat(x[:, k], maxlag, trim=trim, original='in')[:, dropex:maxlagex + 1])
    return np.column_stack(lagsli)

def vec(mat):
    if False:
        return 10
    return mat.ravel('F')

def vech(mat):
    if False:
        print('Hello World!')
    return mat.T.take(_triu_indices(len(mat)))

def _tril_indices(n):
    if False:
        print('Hello World!')
    (rows, cols) = np.tril_indices(n)
    return rows * n + cols

def _triu_indices(n):
    if False:
        for i in range(10):
            print('nop')
    (rows, cols) = np.triu_indices(n)
    return rows * n + cols

def _diag_indices(n):
    if False:
        for i in range(10):
            print('nop')
    (rows, cols) = np.diag_indices(n)
    return rows * n + cols

def unvec(v):
    if False:
        while True:
            i = 10
    k = int(np.sqrt(len(v)))
    assert k * k == len(v)
    return v.reshape((k, k), order='F')

def unvech(v):
    if False:
        i = 10
        return i + 15
    rows = 0.5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))
    result = np.zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = result + result.T
    result[np.diag_indices(rows)] /= 2
    return result

def duplication_matrix(n):
    if False:
        return 10
    '\n    Create duplication matrix D_n which satisfies vec(S) = D_n vech(S) for\n    symmetric matrix S\n\n    Returns\n    -------\n    D_n : ndarray\n    '
    n = int_like(n, 'n')
    tmp = np.eye(n * (n + 1) // 2)
    return np.array([unvech(x).ravel() for x in tmp]).T

def elimination_matrix(n):
    if False:
        i = 10
        return i + 15
    '\n    Create the elimination matrix L_n which satisfies vech(M) = L_n vec(M) for\n    any matrix M\n\n    Parameters\n    ----------\n\n    Returns\n    -------\n    '
    n = int_like(n, 'n')
    vech_indices = vec(np.tril(np.ones((n, n))))
    return np.eye(n * n)[vech_indices != 0]

def commutation_matrix(p, q):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create the commutation matrix K_{p,q} satisfying vec(A') = K_{p,q} vec(A)\n\n    Parameters\n    ----------\n    p : int\n    q : int\n\n    Returns\n    -------\n    K : ndarray (pq x pq)\n    "
    p = int_like(p, 'p')
    q = int_like(q, 'q')
    K = np.eye(p * q)
    indices = np.arange(p * q).reshape((p, q), order='F')
    return K.take(indices.ravel(), axis=0)

def _ar_transparams(params):
    if False:
        for i in range(10):
            print('nop')
    '\n    Transforms params to induce stationarity/invertability.\n\n    Parameters\n    ----------\n    params : array_like\n        The AR coefficients\n\n    Reference\n    ---------\n    Jones(1980)\n    '
    newparams = np.tanh(params / 2)
    tmp = np.tanh(params / 2)
    for j in range(1, len(params)):
        a = newparams[j]
        for kiter in range(j):
            tmp[kiter] -= a * newparams[j - kiter - 1]
        newparams[:j] = tmp[:j]
    return newparams

def _ar_invtransparams(params):
    if False:
        while True:
            i = 10
    '\n    Inverse of the Jones reparameterization\n\n    Parameters\n    ----------\n    params : array_like\n        The transformed AR coefficients\n    '
    params = params.copy()
    tmp = params.copy()
    for j in range(len(params) - 1, 0, -1):
        a = params[j]
        for kiter in range(j):
            tmp[kiter] = (params[kiter] + a * params[j - kiter - 1]) / (1 - a ** 2)
        params[:j] = tmp[:j]
    invarcoefs = 2 * np.arctanh(params)
    return invarcoefs

def _ma_transparams(params):
    if False:
        i = 10
        return i + 15
    '\n    Transforms params to induce stationarity/invertability.\n\n    Parameters\n    ----------\n    params : ndarray\n        The ma coeffecients of an (AR)MA model.\n\n    Reference\n    ---------\n    Jones(1980)\n    '
    newparams = ((1 - np.exp(-params)) / (1 + np.exp(-params))).copy()
    tmp = ((1 - np.exp(-params)) / (1 + np.exp(-params))).copy()
    for j in range(1, len(params)):
        b = newparams[j]
        for kiter in range(j):
            tmp[kiter] += b * newparams[j - kiter - 1]
        newparams[:j] = tmp[:j]
    return newparams

def _ma_invtransparams(macoefs):
    if False:
        while True:
            i = 10
    '\n    Inverse of the Jones reparameterization\n\n    Parameters\n    ----------\n    params : ndarray\n        The transformed MA coefficients\n    '
    tmp = macoefs.copy()
    for j in range(len(macoefs) - 1, 0, -1):
        b = macoefs[j]
        for kiter in range(j):
            tmp[kiter] = (macoefs[kiter] - b * macoefs[j - kiter - 1]) / (1 - b ** 2)
        macoefs[:j] = tmp[:j]
    invmacoefs = -np.log((1 - macoefs) / (1 + macoefs))
    return invmacoefs

def unintegrate_levels(x, d):
    if False:
        print('Hello World!')
    '\n    Returns the successive differences needed to unintegrate the series.\n\n    Parameters\n    ----------\n    x : array_like\n        The original series\n    d : int\n        The number of differences of the differenced series.\n\n    Returns\n    -------\n    y : array_like\n        The increasing differences from 0 to d-1 of the first d elements\n        of x.\n\n    See Also\n    --------\n    unintegrate\n    '
    d = int_like(d, 'd')
    x = x[:d]
    return np.asarray([np.diff(x, d - i)[0] for i in range(d, 0, -1)])

def unintegrate(x, levels):
    if False:
        for i in range(10):
            print('nop')
    '\n    After taking n-differences of a series, return the original series\n\n    Parameters\n    ----------\n    x : array_like\n        The n-th differenced series\n    levels : list\n        A list of the first-value in each differenced series, for\n        [first-difference, second-difference, ..., n-th difference]\n\n    Returns\n    -------\n    y : array_like\n        The original series de-differenced\n\n    Examples\n    --------\n    >>> x = np.array([1, 3, 9., 19, 8.])\n    >>> levels = unintegrate_levels(x, 2)\n    >>> levels\n    array([ 1.,  2.])\n    >>> unintegrate(np.diff(x, 2), levels)\n    array([  1.,   3.,   9.,  19.,   8.])\n    '
    levels = list(levels)[:]
    if len(levels) > 1:
        x0 = levels.pop(-1)
        return unintegrate(np.cumsum(np.r_[x0, x]), levels)
    x0 = levels[0]
    return np.cumsum(np.r_[x0, x])

def freq_to_period(freq: str | offsets.DateOffset) -> int:
    if False:
        return 10
    '\n    Convert a pandas frequency to a periodicity\n\n    Parameters\n    ----------\n    freq : str or offset\n        Frequency to convert\n\n    Returns\n    -------\n    int\n        Periodicity of freq\n\n    Notes\n    -----\n    Annual maps to 1, quarterly maps to 4, monthly to 12, weekly to 52.\n    '
    if not isinstance(freq, offsets.DateOffset):
        freq = to_offset(freq)
    assert isinstance(freq, offsets.DateOffset)
    freq = freq.rule_code.upper()
    if freq in ('A', 'Y') or freq.startswith(('A-', 'AS-', 'Y-', 'YS-')):
        return 1
    elif freq == 'Q' or freq.startswith(('Q-', 'QS', 'QE')):
        return 4
    elif freq == 'M' or freq.startswith(('M-', 'MS', 'ME')):
        return 12
    elif freq == 'W' or freq.startswith('W-'):
        return 52
    elif freq == 'D':
        return 7
    elif freq == 'B':
        return 5
    elif freq == 'H':
        return 24
    else:
        raise ValueError('freq {} not understood. Please report if you think this is in error.'.format(freq))