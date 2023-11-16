"""
Bspines and smoothing splines.

General references:

    Craven, P. and Wahba, G. (1978) "Smoothing noisy data with spline functions.
    Estimating the correct degree of smoothing by
    the method of generalized cross-validation."
    Numerische Mathematik, 31(4), 377-403.

    Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical
    Learning." Springer-Verlag. 536 pages.

    Hutchison, M. and Hoog, F. "Smoothing noisy data with spline functions."
    Numerische Mathematik, 47(1), 99-106.
"""
import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline
import warnings
_msg = '\nThe bspline code is technology preview and requires significant work\non the public API and documentation. The API will likely change in the future\n'
warnings.warn(_msg, FutureWarning)

def _band2array(a, lower=0, symmetric=False, hermitian=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Take an upper or lower triangular banded matrix and return a\n    numpy array.\n\n    INPUTS:\n       a         -- a matrix in upper or lower triangular banded matrix\n       lower     -- is the matrix upper or lower triangular?\n       symmetric -- if True, return the original result plus its transpose\n       hermitian -- if True (and symmetric False), return the original\n                    result plus its conjugate transposed\n    '
    n = a.shape[1]
    r = a.shape[0]
    _a = 0
    if not lower:
        for j in range(r):
            _b = np.diag(a[r - 1 - j], k=j)[j:n + j, j:n + j]
            _a += _b
            if symmetric and j > 0:
                _a += _b.T
            elif hermitian and j > 0:
                _a += _b.conjugate().T
    else:
        for j in range(r):
            _b = np.diag(a[j], k=j)[0:n, 0:n]
            _a += _b
            if symmetric and j > 0:
                _a += _b.T
            elif hermitian and j > 0:
                _a += _b.conjugate().T
        _a = _a.T
    return _a

def _upper2lower(ub):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert upper triangular banded matrix to lower banded form.\n\n    INPUTS:\n       ub  -- an upper triangular banded matrix\n\n    OUTPUTS: lb\n       lb  -- a lower triangular banded matrix with same entries\n              as ub\n    '
    lb = np.zeros(ub.shape, ub.dtype)
    (nrow, ncol) = ub.shape
    for i in range(ub.shape[0]):
        lb[i, 0:ncol - i] = ub[nrow - 1 - i, i:ncol]
        lb[i, ncol - i:] = ub[nrow - 1 - i, 0:i]
    return lb

def _lower2upper(lb):
    if False:
        while True:
            i = 10
    '\n    Convert lower triangular banded matrix to upper banded form.\n\n    INPUTS:\n       lb  -- a lower triangular banded matrix\n\n    OUTPUTS: ub\n       ub  -- an upper triangular banded matrix with same entries\n              as lb\n    '
    ub = np.zeros(lb.shape, lb.dtype)
    (nrow, ncol) = lb.shape
    for i in range(lb.shape[0]):
        ub[nrow - 1 - i, i:ncol] = lb[i, 0:ncol - i]
        ub[nrow - 1 - i, 0:i] = lb[i, ncol - i:]
    return ub

def _triangle2unit(tb, lower=0):
    if False:
        return 10
    "\n    Take a banded triangular matrix and return its diagonal and the\n    unit matrix: the banded triangular matrix with 1's on the diagonal,\n    i.e. each row is divided by the corresponding entry on the diagonal.\n\n    INPUTS:\n       tb    -- a lower triangular banded matrix\n       lower -- if True, then tb is assumed to be lower triangular banded,\n                in which case return value is also lower triangular banded.\n\n    OUTPUTS: d, b\n       d     -- diagonal entries of tb\n       b     -- unit matrix: if lower is False, b is upper triangular\n                banded and its rows of have been divided by d,\n                else lower is True, b is lower triangular banded\n                and its columns have been divieed by d.\n    "
    if lower:
        d = tb[0].copy()
    else:
        d = tb[-1].copy()
    if lower:
        return (d, tb / d)
    else:
        lnum = _upper2lower(tb)
        return (d, _lower2upper(lnum / d))

def _trace_symbanded(a, b, lower=0):
    if False:
        return 10
    '\n    Compute the trace(ab) for two upper or banded real symmetric matrices\n    stored either in either upper or lower form.\n\n    INPUTS:\n       a, b    -- two banded real symmetric matrices (either lower or upper)\n       lower   -- if True, a and b are assumed to be the lower half\n\n\n    OUTPUTS: trace\n       trace   -- trace(ab)\n    '
    if lower:
        t = _zero_triband(a * b, lower=1)
        return t[0].sum() + 2 * t[1:].sum()
    else:
        t = _zero_triband(a * b, lower=0)
        return t[-1].sum() + 2 * t[:-1].sum()

def _zero_triband(a, lower=0):
    if False:
        print('Hello World!')
    '\n    Explicitly zero out unused elements of a real symmetric banded matrix.\n\n    INPUTS:\n       a   -- a real symmetric banded matrix (either upper or lower hald)\n       lower   -- if True, a is assumed to be the lower half\n    '
    (nrow, ncol) = a.shape
    if lower:
        for i in range(nrow):
            a[i, ncol - i:] = 0.0
    else:
        for i in range(nrow):
            a[i, 0:i] = 0.0
    return a

class BSpline:
    """

    Bsplines of a given order and specified knots.

    Implementation is based on description in Chapter 5 of

    Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical
    Learning." Springer-Verlag. 536 pages.


    INPUTS:
       knots  -- a sorted array of knots with knots[0] the lower boundary,
                 knots[1] the upper boundary and knots[1:-1] the internal
                 knots.
       order  -- order of the Bspline, default is 4 which yields cubic
                 splines
       M      -- number of additional boundary knots, if None it defaults
                 to order
       coef   -- an optional array of real-valued coefficients for the Bspline
                 of shape (knots.shape + 2 * (M - 1) - order,).
       x      -- an optional set of x values at which to evaluate the
                 Bspline to avoid extra evaluation in the __call__ method

    """

    def __init__(self, knots, order=4, M=None, coef=None, x=None):
        if False:
            print('Hello World!')
        knots = np.squeeze(np.unique(np.asarray(knots)))
        if knots.ndim != 1:
            raise ValueError('expecting 1d array for knots')
        self.m = order
        if M is None:
            M = self.m
        self.M = M
        self.tau = np.hstack([[knots[0]] * (self.M - 1), knots, [knots[-1]] * (self.M - 1)])
        self.K = knots.shape[0] - 2
        if coef is None:
            self.coef = np.zeros(self.K + 2 * self.M - self.m, np.float64)
        else:
            self.coef = np.squeeze(coef)
            if self.coef.shape != self.K + 2 * self.M - self.m:
                raise ValueError('coefficients of Bspline have incorrect shape')
        if x is not None:
            self.x = x

    def _setx(self, x):
        if False:
            while True:
                i = 10
        self._x = x
        self._basisx = self.basis(self._x)

    def _getx(self):
        if False:
            return 10
        return self._x
    x = property(_getx, _setx)

    def __call__(self, *args):
        if False:
            while True:
                i = 10
        '\n        Evaluate the BSpline at a given point, yielding\n        a matrix B and return\n\n        B * self.coef\n\n\n        INPUTS:\n           args -- optional arguments. If None, it returns self._basisx,\n                   the BSpline evaluated at the x values passed in __init__.\n                   Otherwise, return the BSpline evaluated at the\n                   first argument args[0].\n\n        OUTPUTS: y\n           y    -- value of Bspline at specified x values\n\n        BUGS:\n           If self has no attribute x, an exception will be raised\n           because self has no attribute _basisx.\n        '
        if not args:
            b = self._basisx.T
        else:
            x = args[0]
            b = np.asarray(self.basis(x)).T
        return np.squeeze(np.dot(b, self.coef))

    def basis_element(self, x, i, d=0):
        if False:
            while True:
                i = 10
        '\n        Evaluate a particular basis element of the BSpline,\n        or its derivative.\n\n        INPUTS:\n           x  -- x values at which to evaluate the basis element\n           i  -- which element of the BSpline to return\n           d  -- the order of derivative\n\n        OUTPUTS: y\n           y  -- value of d-th derivative of the i-th basis element\n                 of the BSpline at specified x values\n        '
        x = np.asarray(x, np.float64)
        _shape = x.shape
        if _shape == ():
            x.shape = (1,)
        x.shape = (np.product(_shape, axis=0),)
        if i < self.tau.shape[0] - 1:
            v = _hbspline.evaluate(x, self.tau, self.m, d, i, i + 1)
        else:
            return np.zeros(x.shape, np.float64)
        if i == self.tau.shape[0] - self.m:
            v = np.where(np.equal(x, self.tau[-1]), 1, v)
        v.shape = _shape
        return v

    def basis(self, x, d=0, lower=None, upper=None):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate the basis of the BSpline or its derivative.\n        If lower or upper is specified, then only\n        the [lower:upper] elements of the basis are returned.\n\n        INPUTS:\n           x     -- x values at which to evaluate the basis element\n           i     -- which element of the BSpline to return\n           d     -- the order of derivative\n           lower -- optional lower limit of the set of basis\n                    elements\n           upper -- optional upper limit of the set of basis\n                    elements\n\n        OUTPUTS: y\n           y  -- value of d-th derivative of the basis elements\n                 of the BSpline at specified x values\n        '
        x = np.asarray(x)
        _shape = x.shape
        if _shape == ():
            x.shape = (1,)
        x.shape = (np.product(_shape, axis=0),)
        if upper is None:
            upper = self.tau.shape[0] - self.m
        if lower is None:
            lower = 0
        upper = min(upper, self.tau.shape[0] - self.m)
        lower = max(0, lower)
        d = np.asarray(d)
        if d.shape == ():
            v = _hbspline.evaluate(x, self.tau, self.m, int(d), lower, upper)
        else:
            if d.shape[0] != 2:
                raise ValueError('if d is not an integer, expecting a jx2                    array with first row indicating order                    of derivative, second row coefficient in front.')
            v = 0
            for i in range(d.shape[1]):
                v += d[1, i] * _hbspline.evaluate(x, self.tau, self.m, d[0, i], lower, upper)
        v.shape = (upper - lower,) + _shape
        if upper == self.tau.shape[0] - self.m:
            v[-1] = np.where(np.equal(x, self.tau[-1]), 1, v[-1])
        return v

    def gram(self, d=0):
        if False:
            while True:
                i = 10
        '\n        Compute Gram inner product matrix, storing it in lower\n        triangular banded form.\n\n        The (i,j) entry is\n\n        G_ij = integral b_i^(d) b_j^(d)\n\n        where b_i are the basis elements of the BSpline and (d) is the\n        d-th derivative.\n\n        If d is a matrix then, it is assumed to specify a differential\n        operator as follows: the first row represents the order of derivative\n        with the second row the coefficient corresponding to that order.\n\n        For instance:\n\n        [[2, 3],\n         [3, 1]]\n\n        represents 3 * f^(2) + 1 * f^(3).\n\n        INPUTS:\n           d    -- which derivative to apply to each basis element,\n                   if d is a matrix, it is assumed to specify\n                   a differential operator as above\n\n        OUTPUTS: gram\n           gram -- the matrix of inner products of (derivatives)\n                   of the BSpline elements\n        '
        d = np.squeeze(d)
        if np.asarray(d).shape == ():
            self.g = _hbspline.gram(self.tau, self.m, int(d), int(d))
        else:
            d = np.asarray(d)
            if d.shape[0] != 2:
                raise ValueError('if d is not an integer, expecting a jx2                    array with first row indicating order                    of derivative, second row coefficient in front.')
            if d.shape == (2,):
                d.shape = (2, 1)
            self.g = 0
            for i in range(d.shape[1]):
                for j in range(d.shape[1]):
                    self.g += d[1, i] * d[1, j] * _hbspline.gram(self.tau, self.m, int(d[0, i]), int(d[0, j]))
        self.g = self.g.T
        self.d = d
        return np.nan_to_num(self.g)

class SmoothingSpline(BSpline):
    penmax = 30.0
    method = 'target_df'
    target_df = 5
    default_pen = 0.001
    optimize = True
    '\n    A smoothing spline, which can be used to smooth scatterplots, i.e.\n    a list of (x,y) tuples.\n\n    See fit method for more information.\n\n    '

    def fit(self, y, x=None, weights=None, pen=0.0):
        if False:
            i = 10
            return i + 15
        '\n        Fit the smoothing spline to a set of (x,y) pairs.\n\n        INPUTS:\n           y       -- response variable\n           x       -- if None, uses self.x\n           weights -- optional array of weights\n           pen     -- constant in front of Gram matrix\n\n        OUTPUTS: None\n           The smoothing spline is determined by self.coef,\n           subsequent calls of __call__ will be the smoothing spline.\n\n        ALGORITHM:\n           Formally, this solves a minimization:\n\n           fhat = ARGMIN_f SUM_i=1^n (y_i-f(x_i))^2 + pen * int f^(2)^2\n\n           int is integral. pen is lambda (from Hastie)\n\n           See Chapter 5 of\n\n           Hastie, Tibshirani and Friedman (2001). "The Elements of Statistical\n           Learning." Springer-Verlag. 536 pages.\n\n           for more details.\n\n        TODO:\n           Should add arbitrary derivative penalty instead of just\n           second derivative.\n        '
        banded = True
        if x is None:
            x = self._x
            bt = self._basisx.copy()
        else:
            bt = self.basis(x)
        if pen == 0.0:
            banded = False
        if x.shape != y.shape:
            raise ValueError("x and y shape do not agree, by default x are                the Bspline's internal knots")
        if pen >= self.penmax:
            pen = self.penmax
        if weights is not None:
            self.weights = weights
        else:
            self.weights = 1.0
        _w = np.sqrt(self.weights)
        bt *= _w
        mask = np.flatnonzero(1 - np.all(np.equal(bt, 0), axis=0))
        bt = bt[:, mask]
        y = y[mask]
        self.df_total = y.shape[0]
        bty = np.squeeze(np.dot(bt, _w * y))
        self.N = y.shape[0]
        if not banded:
            self.btb = np.dot(bt, bt.T)
            _g = _band2array(self.g, lower=1, symmetric=True)
            (self.coef, _, self.rank) = L.lstsq(self.btb + pen * _g, bty)[0:3]
            self.rank = min(self.rank, self.btb.shape[0])
            del _g
        else:
            self.btb = np.zeros(self.g.shape, np.float64)
            (nband, nbasis) = self.g.shape
            for i in range(nbasis):
                for k in range(min(nband, nbasis - i)):
                    self.btb[k, i] = (bt[i] * bt[i + k]).sum()
            bty.shape = (1, bty.shape[0])
            self.pen = pen
            (self.chol, self.coef) = solveh_banded(self.btb + pen * self.g, bty, lower=1)
        self.coef = np.squeeze(self.coef)
        self.resid = y * self.weights - np.dot(self.coef, bt)
        self.pen = pen
        del bty
        del mask
        del bt

    def smooth(self, y, x=None, weights=None):
        if False:
            return 10
        if self.method == 'target_df':
            if hasattr(self, 'pen'):
                self.fit(y, x=x, weights=weights, pen=self.pen)
            else:
                self.fit_target_df(y, x=x, weights=weights, df=self.target_df)
        elif self.method == 'optimize_gcv':
            self.fit_optimize_gcv(y, x=x, weights=weights)

    def gcv(self):
        if False:
            print('Hello World!')
        '\n        Generalized cross-validation score of current fit.\n\n        Craven, P. and Wahba, G.  "Smoothing noisy data with spline functions.\n        Estimating the correct degree of smoothing by\n        the method of generalized cross-validation."\n        Numerische Mathematik, 31(4), 377-403.\n        '
        norm_resid = (self.resid ** 2).sum()
        return norm_resid / (self.df_total - self.trace())

    def df_resid(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Residual degrees of freedom in the fit.\n\n        self.N - self.trace()\n\n        where self.N is the number of observations of last fit.\n        '
        return self.N - self.trace()

    def df_fit(self):
        if False:
            i = 10
            return i + 15
        '\n        How many degrees of freedom used in the fit?\n\n        self.trace()\n        '
        return self.trace()

    def trace(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Trace of the smoothing matrix S(pen)\n\n        TODO: addin a reference to Wahba, and whoever else I used.\n        '
        if self.pen > 0:
            _invband = _hbspline.invband(self.chol.copy())
            tr = _trace_symbanded(_invband, self.btb, lower=1)
            return tr
        else:
            return self.rank

    def fit_target_df(self, y, x=None, df=None, weights=None, tol=0.001, apen=0, bpen=0.001):
        if False:
            while True:
                i = 10
        '\n        Fit smoothing spline with approximately df degrees of freedom\n        used in the fit, i.e. so that self.trace() is approximately df.\n\n        Uses binary search strategy.\n\n        In general, df must be greater than the dimension of the null space\n        of the Gram inner product. For cubic smoothing splines, this means\n        that df > 2.\n\n        INPUTS:\n           y       -- response variable\n           x       -- if None, uses self.x\n           df      -- target degrees of freedom\n           weights -- optional array of weights\n           tol     -- (relative) tolerance for convergence\n           apen    -- lower bound of penalty for binary search\n           bpen    -- upper bound of penalty for binary search\n\n        OUTPUTS: None\n           The smoothing spline is determined by self.coef,\n           subsequent calls of __call__ will be the smoothing spline.\n        '
        df = df or self.target_df
        olddf = y.shape[0] - self.m
        if hasattr(self, 'pen'):
            self.fit(y, x=x, weights=weights, pen=self.pen)
            curdf = self.trace()
            if np.fabs(curdf - df) / df < tol:
                return
            if curdf > df:
                (apen, bpen) = (self.pen, 2 * self.pen)
            else:
                (apen, bpen) = (0.0, self.pen)
        while True:
            curpen = 0.5 * (apen + bpen)
            self.fit(y, x=x, weights=weights, pen=curpen)
            curdf = self.trace()
            if curdf > df:
                (apen, bpen) = (curpen, 2 * curpen)
            else:
                (apen, bpen) = (apen, curpen)
            if apen >= self.penmax:
                raise ValueError('penalty too large, try setting penmax                    higher or decreasing df')
            if np.fabs(curdf - df) / df < tol:
                break

    def fit_optimize_gcv(self, y, x=None, weights=None, tol=0.001, brack=(-100, 20)):
        if False:
            i = 10
            return i + 15
        '\n        Fit smoothing spline trying to optimize GCV.\n\n        Try to find a bracketing interval for scipy.optimize.golden\n        based on bracket.\n\n        It is probably best to use target_df instead, as it is\n        sometimes difficult to find a bracketing interval.\n\n        INPUTS:\n           y       -- response variable\n           x       -- if None, uses self.x\n           df      -- target degrees of freedom\n           weights -- optional array of weights\n           tol     -- (relative) tolerance for convergence\n           brack   -- an initial guess at the bracketing interval\n\n        OUTPUTS: None\n           The smoothing spline is determined by self.coef,\n           subsequent calls of __call__ will be the smoothing spline.\n        '

        def _gcv(pen, y, x):
            if False:
                print('Hello World!')
            self.fit(y, x=x, pen=np.exp(pen))
            a = self.gcv()
            return a
        a = golden(_gcv, args=(y, x), brack=brack, tol=tol)