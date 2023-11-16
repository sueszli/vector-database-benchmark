""" Helper and filter functions for VAR and VARMA, and basic VAR class

Created on Mon Jan 11 11:04:23 2010
Author: josef-pktd
License: BSD

This is a new version, I did not look at the old version again, but similar
ideas.

not copied/cleaned yet:
 * fftn based filtering, creating samples with fft
 * Tests: I ran examples but did not convert them to tests
   examples look good for parameter estimate and forecast, and filter functions

main TODOs:
* result statistics
* see whether Bayesian dummy observation can be included without changing
  the single call to linalg.lstsq
* impulse response function does not treat correlation, see Hamilton and jplv

Extensions
* constraints, Bayesian priors/penalization
* Error Correction Form and Cointegration
* Factor Models Stock-Watson,  ???


see also VAR section in Notes.txt

"""
import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat

def varfilter(x, a):
    if False:
        for i in range(10):
            print('nop')
    'apply an autoregressive filter to a series x\n\n    Warning: I just found out that convolve does not work as I\n       thought, this likely does not work correctly for\n       nvars>3\n\n\n    x can be 2d, a can be 1d, 2d, or 3d\n\n    Parameters\n    ----------\n    x : array_like\n        data array, 1d or 2d, if 2d then observations in rows\n    a : array_like\n        autoregressive filter coefficients, ar lag polynomial\n        see Notes\n\n    Returns\n    -------\n    y : ndarray, 2d\n        filtered array, number of columns determined by x and a\n\n    Notes\n    -----\n\n    In general form this uses the linear filter ::\n\n        y = a(L)x\n\n    where\n    x : nobs, nvars\n    a : nlags, nvars, npoly\n\n    Depending on the shape and dimension of a this uses different\n    Lag polynomial arrays\n\n    case 1 : a is 1d or (nlags,1)\n        one lag polynomial is applied to all variables (columns of x)\n    case 2 : a is 2d, (nlags, nvars)\n        each series is independently filtered with its own\n        lag polynomial, uses loop over nvar\n    case 3 : a is 3d, (nlags, nvars, npoly)\n        the ith column of the output array is given by the linear filter\n        defined by the 2d array a[:,:,i], i.e. ::\n\n            y[:,i] = a(.,.,i)(L) * x\n            y[t,i] = sum_p sum_j a(p,j,i)*x(t-p,j)\n                     for p = 0,...nlags-1, j = 0,...nvars-1,\n                     for all t >= nlags\n\n\n    Note: maybe convert to axis=1, Not\n\n    TODO: initial conditions\n\n    '
    x = np.asarray(x)
    a = np.asarray(a)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim > 2:
        raise ValueError('x array has to be 1d or 2d')
    nvar = x.shape[1]
    nlags = a.shape[0]
    ntrim = nlags // 2
    if a.ndim == 1:
        return signal.convolve(x, a[:, None], mode='valid')
    elif a.ndim == 2:
        if min(a.shape) == 1:
            return signal.convolve(x, a, mode='valid')
        result = np.zeros((x.shape[0] - nlags + 1, nvar))
        for i in range(nvar):
            result[:, i] = signal.convolve(x[:, i], a[:, i], mode='valid')
        return result
    elif a.ndim == 3:
        yf = signal.convolve(x[:, :, None], a)
        yvalid = yf[ntrim:-ntrim, yf.shape[1] // 2, :]
        return yvalid

def varinversefilter(ar, nobs, version=1):
    if False:
        print('Hello World!')
    'creates inverse ar filter (MA representation) recursively\n\n    The VAR lag polynomial is defined by ::\n\n        ar(L) y_t = u_t  or\n        y_t = -ar_{-1}(L) y_{t-1} + u_t\n\n    the returned lagpolynomial is arinv(L)=ar^{-1}(L) in ::\n\n        y_t = arinv(L) u_t\n\n\n\n    Parameters\n    ----------\n    ar : ndarray, (nlags,nvars,nvars)\n        matrix lagpolynomial, currently no exog\n        first row should be identity\n\n    Returns\n    -------\n    arinv : ndarray, (nobs,nvars,nvars)\n\n\n    Notes\n    -----\n\n    '
    (nlags, nvars, nvarsex) = ar.shape
    if nvars != nvarsex:
        print('exogenous variables not implemented not tested')
    arinv = np.zeros((nobs + 1, nvarsex, nvars))
    arinv[0, :, :] = ar[0]
    arinv[1:nlags, :, :] = -ar[1:]
    if version == 1:
        for i in range(2, nobs + 1):
            tmp = np.zeros((nvars, nvars))
            for p in range(1, nlags):
                tmp += np.dot(-ar[p], arinv[i - p, :, :])
            arinv[i, :, :] = tmp
    if version == 0:
        for i in range(nlags + 1, nobs + 1):
            print(ar[1:].shape, arinv[i - 1:i - nlags:-1, :, :].shape)
            raise NotImplementedError('waiting for generalized ufuncs or something')
    return arinv

def vargenerate(ar, u, initvalues=None):
    if False:
        return 10
    'generate an VAR process with errors u\n\n    similar to gauss\n    uses loop\n\n    Parameters\n    ----------\n    ar : array (nlags,nvars,nvars)\n        matrix lagpolynomial\n    u : array (nobs,nvars)\n        exogenous variable, error term for VAR\n\n    Returns\n    -------\n    sar : array (1+nobs,nvars)\n        sample of var process, inverse filtered u\n        does not trim initial condition y_0 = 0\n\n    Examples\n    --------\n    # generate random sample of VAR\n    nobs, nvars = 10, 2\n    u = numpy.random.randn(nobs,nvars)\n    a21 = np.array([[[ 1. ,  0. ],\n                     [ 0. ,  1. ]],\n\n                    [[-0.8,  0. ],\n                     [ 0.,  -0.6]]])\n    vargenerate(a21,u)\n\n    # Impulse Response to an initial shock to the first variable\n    imp = np.zeros((nobs, nvars))\n    imp[0,0] = 1\n    vargenerate(a21,imp)\n\n    '
    (nlags, nvars, nvarsex) = ar.shape
    nlagsm1 = nlags - 1
    nobs = u.shape[0]
    if nvars != nvarsex:
        print('exogenous variables not implemented not tested')
    if u.shape[1] != nvars:
        raise ValueError('u needs to have nvars columns')
    if initvalues is None:
        sar = np.zeros((nobs + nlagsm1, nvars))
        start = nlagsm1
    else:
        start = max(nlagsm1, initvalues.shape[0])
        sar = np.zeros((nobs + start, nvars))
        sar[start - initvalues.shape[0]:start] = initvalues
    sar[start:] = u
    for i in range(start, start + nobs):
        for p in range(1, nlags):
            sar[i] += np.dot(sar[i - p, :], -ar[p])
    return sar

def padone(x, front=0, back=0, axis=0, fillvalue=0):
    if False:
        i = 10
        return i + 15
    'pad with zeros along one axis, currently only axis=0\n\n\n    can be used sequentially to pad several axis\n\n    Examples\n    --------\n    >>> padone(np.ones((2,3)),1,3,axis=1)\n    array([[ 0.,  1.,  1.,  1.,  0.,  0.,  0.],\n           [ 0.,  1.,  1.,  1.,  0.,  0.,  0.]])\n\n    >>> padone(np.ones((2,3)),1,1, fillvalue=np.nan)\n    array([[ NaN,  NaN,  NaN],\n           [  1.,   1.,   1.],\n           [  1.,   1.,   1.],\n           [ NaN,  NaN,  NaN]])\n    '
    shape = np.array(x.shape)
    shape[axis] += front + back
    shapearr = np.array(x.shape)
    out = np.empty(shape)
    out.fill(fillvalue)
    startind = np.zeros(x.ndim)
    startind[axis] = front
    endind = startind + shapearr
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    out[tuple(myslice)] = x
    return out

def trimone(x, front=0, back=0, axis=0):
    if False:
        i = 10
        return i + 15
    'trim number of array elements along one axis\n\n\n    Examples\n    --------\n    >>> xp = padone(np.ones((2,3)),1,3,axis=1)\n    >>> xp\n    array([[ 0.,  1.,  1.,  1.,  0.,  0.,  0.],\n           [ 0.,  1.,  1.,  1.,  0.,  0.,  0.]])\n    >>> trimone(xp,1,3,1)\n    array([[ 1.,  1.,  1.],\n           [ 1.,  1.,  1.]])\n    '
    shape = np.array(x.shape)
    shape[axis] -= front + back
    shapearr = np.array(x.shape)
    startind = np.zeros(x.ndim)
    startind[axis] = front
    endind = startind + shape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return x[tuple(myslice)]

def ar2full(ar):
    if False:
        while True:
            i = 10
    'make reduced lagpolynomial into a right side lagpoly array\n    '
    (nlags, nvar, nvarex) = ar.shape
    return np.r_[np.eye(nvar, nvarex)[None, :, :], -ar]

def ar2lhs(ar):
    if False:
        i = 10
        return i + 15
    'convert full (rhs) lagpolynomial into a reduced, left side lagpoly array\n\n    this is mainly a reminder about the definition\n    '
    return -ar[1:]

class _Var:
    """obsolete VAR class, use tsa.VAR instead, for internal use only


    Examples
    --------

    >>> v = Var(ar2s)
    >>> v.fit(1)
    >>> v.arhat
    array([[[ 1.        ,  0.        ],
            [ 0.        ,  1.        ]],

           [[-0.77784898,  0.01726193],
            [ 0.10733009, -0.78665335]]])

    """

    def __init__(self, y):
        if False:
            i = 10
            return i + 15
        self.y = y
        (self.nobs, self.nvars) = y.shape

    def fit(self, nlags):
        if False:
            while True:
                i = 10
        'estimate parameters using ols\n\n        Parameters\n        ----------\n        nlags : int\n            number of lags to include in regression, same for all variables\n\n        Returns\n        -------\n        None, but attaches\n\n        arhat : array (nlags, nvar, nvar)\n            full lag polynomial array\n        arlhs : array (nlags-1, nvar, nvar)\n            reduced lag polynomial for left hand side\n        other statistics as returned by linalg.lstsq : need to be completed\n\n\n\n        This currently assumes all parameters are estimated without restrictions.\n        In this case SUR is identical to OLS\n\n        estimation results are attached to the class instance\n\n\n        '
        self.nlags = nlags
        nvars = self.nvars
        lmat = lagmat(self.y, nlags, trim='both', original='in')
        self.yred = lmat[:, :nvars]
        self.xred = lmat[:, nvars:]
        res = np.linalg.lstsq(self.xred, self.yred, rcond=-1)
        self.estresults = res
        self.arlhs = res[0].reshape(nlags, nvars, nvars)
        self.arhat = ar2full(self.arlhs)
        self.rss = res[1]
        self.xredrank = res[2]

    def predict(self):
        if False:
            for i in range(10):
                print('nop')
        'calculate estimated timeseries (yhat) for sample\n\n        '
        if not hasattr(self, 'yhat'):
            self.yhat = varfilter(self.y, self.arhat)
        return self.yhat

    def covmat(self):
        if False:
            for i in range(10):
                print('nop')
        " covariance matrix of estimate\n        # not sure it's correct, need to check orientation everywhere\n        # looks ok, display needs getting used to\n        >>> v.rss[None,None,:]*np.linalg.inv(np.dot(v.xred.T,v.xred))[:,:,None]\n        array([[[ 0.37247445,  0.32210609],\n                [ 0.1002642 ,  0.08670584]],\n\n               [[ 0.1002642 ,  0.08670584],\n                [ 0.45903637,  0.39696255]]])\n        >>>\n        >>> v.rss[0]*np.linalg.inv(np.dot(v.xred.T,v.xred))\n        array([[ 0.37247445,  0.1002642 ],\n               [ 0.1002642 ,  0.45903637]])\n        >>> v.rss[1]*np.linalg.inv(np.dot(v.xred.T,v.xred))\n        array([[ 0.32210609,  0.08670584],\n               [ 0.08670584,  0.39696255]])\n       "
        self.paramcov = self.rss[None, None, :] * np.linalg.inv(np.dot(self.xred.T, self.xred))[:, :, None]

    def forecast(self, horiz=1, u=None):
        if False:
            while True:
                i = 10
        'calculates forcast for horiz number of periods at end of sample\n\n        Parameters\n        ----------\n        horiz : int (optional, default=1)\n            forecast horizon\n        u : array (horiz, nvars)\n            error term for forecast periods. If None, then u is zero.\n\n        Returns\n        -------\n        yforecast : array (nobs+horiz, nvars)\n            this includes the sample and the forecasts\n        '
        if u is None:
            u = np.zeros((horiz, self.nvars))
        return vargenerate(self.arhat, u, initvalues=self.y)

class VarmaPoly:
    """class to keep track of Varma polynomial format


    Examples
    --------

    ar23 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.6,  0. ],
                     [ 0.2, -0.6]],

                    [[-0.1,  0. ],
                     [ 0.1, -0.1]]])

    ma22 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[ 0.4,  0. ],
                     [ 0.2, 0.3]]])


    """

    def __init__(self, ar, ma=None):
        if False:
            print('Hello World!')
        self.ar = ar
        self.ma = ma
        (nlags, nvarall, nvars) = ar.shape
        (self.nlags, self.nvarall, self.nvars) = (nlags, nvarall, nvars)
        self.isstructured = not (ar[0, :nvars] == np.eye(nvars)).all()
        if self.ma is None:
            self.ma = np.eye(nvars)[None, ...]
            self.isindependent = True
        else:
            self.isindependent = not (ma[0] == np.eye(nvars)).all()
        self.malags = ar.shape[0]
        self.hasexog = nvarall > nvars
        self.arm1 = -ar[1:]

    def vstack(self, a=None, name='ar'):
        if False:
            for i in range(10):
                print('nop')
        'stack lagpolynomial vertically in 2d array\n\n        '
        if a is not None:
            a = a
        elif name == 'ar':
            a = self.ar
        elif name == 'ma':
            a = self.ma
        else:
            raise ValueError('no array or name given')
        return a.reshape(-1, self.nvarall)

    def hstack(self, a=None, name='ar'):
        if False:
            return 10
        'stack lagpolynomial horizontally in 2d array\n\n        '
        if a is not None:
            a = a
        elif name == 'ar':
            a = self.ar
        elif name == 'ma':
            a = self.ma
        else:
            raise ValueError('no array or name given')
        return a.swapaxes(1, 2).reshape(-1, self.nvarall).T

    def stacksquare(self, a=None, name='ar', orientation='vertical'):
        if False:
            return 10
        'stack lagpolynomial vertically in 2d square array with eye\n\n        '
        if a is not None:
            a = a
        elif name == 'ar':
            a = self.ar
        elif name == 'ma':
            a = self.ma
        else:
            raise ValueError('no array or name given')
        astacked = a.reshape(-1, self.nvarall)
        (lenpk, nvars) = astacked.shape
        amat = np.eye(lenpk, k=nvars)
        amat[:, :nvars] = astacked
        return amat

    def vstackarma_minus1(self):
        if False:
            for i in range(10):
                print('nop')
        'stack ar and lagpolynomial vertically in 2d array\n\n        '
        a = np.concatenate((self.ar[1:], self.ma[1:]), 0)
        return a.reshape(-1, self.nvarall)

    def hstackarma_minus1(self):
        if False:
            return 10
        'stack ar and lagpolynomial vertically in 2d array\n\n        this is the Kalman Filter representation, I think\n        '
        a = np.concatenate((self.ar[1:], self.ma[1:]), 0)
        return a.swapaxes(1, 2).reshape(-1, self.nvarall)

    def getisstationary(self, a=None):
        if False:
            for i in range(10):
                print('nop')
        'check whether the auto-regressive lag-polynomial is stationary\n\n        Returns\n        -------\n        isstationary : bool\n\n        *attaches*\n\n        areigenvalues : complex array\n            eigenvalues sorted by absolute value\n\n        References\n        ----------\n        formula taken from NAG manual\n\n        '
        if a is not None:
            a = a
        elif self.isstructured:
            a = -self.reduceform(self.ar)[1:]
        else:
            a = -self.ar[1:]
        amat = self.stacksquare(a)
        ev = np.sort(np.linalg.eigvals(amat))[::-1]
        self.areigenvalues = ev
        return (np.abs(ev) < 1).all()

    def getisinvertible(self, a=None):
        if False:
            print('Hello World!')
        'check whether the auto-regressive lag-polynomial is stationary\n\n        Returns\n        -------\n        isinvertible : bool\n\n        *attaches*\n\n        maeigenvalues : complex array\n            eigenvalues sorted by absolute value\n\n        References\n        ----------\n        formula taken from NAG manual\n\n        '
        if a is not None:
            a = a
        elif self.isindependent:
            a = self.reduceform(self.ma)[1:]
        else:
            a = self.ma[1:]
        if a.shape[0] == 0:
            self.maeigenvalues = np.array([], np.complex)
            return True
        amat = self.stacksquare(a)
        ev = np.sort(np.linalg.eigvals(amat))[::-1]
        self.maeigenvalues = ev
        return (np.abs(ev) < 1).all()

    def reduceform(self, apoly):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        this assumes no exog, todo\n\n        '
        if apoly.ndim != 3:
            raise ValueError('apoly needs to be 3d')
        (nlags, nvarsex, nvars) = apoly.shape
        a = np.empty_like(apoly)
        try:
            a0inv = np.linalg.inv(a[0, :nvars, :])
        except np.linalg.LinAlgError:
            raise ValueError('matrix not invertible', 'ask for implementation of pinv')
        for lag in range(nlags):
            a[lag] = np.dot(a0inv, apoly[lag])
        return a
if __name__ == '__main__':
    a21 = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-0.8, 0.0], [0.0, -0.6]]])
    a22 = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-0.8, 0.0], [0.1, -0.8]]])
    a23 = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-0.8, 0.2], [0.1, -0.6]]])
    a24 = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-0.6, 0.0], [0.2, -0.6]], [[-0.1, 0.0], [0.1, -0.1]]])
    a31 = np.r_[np.eye(3)[None, :, :], 0.8 * np.eye(3)[None, :, :]]
    a32 = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.8, 0.0, 0.0], [0.1, 0.6, 0.0], [0.0, 0.0, 0.9]]])
    ut = np.random.randn(1000, 2)
    ar2s = vargenerate(a22, ut)
    res = np.linalg.lstsq(lagmat(ar2s, 1), ar2s, rcond=-1)
    bhat = res[0].reshape(1, 2, 2)
    arhat = ar2full(bhat)
    v = _Var(ar2s)
    v.fit(1)
    v.forecast()
    v.forecast(25)[-30:]
    ar23 = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-0.6, 0.0], [0.2, -0.6]], [[-0.1, 0.0], [0.1, -0.1]]])
    ma22 = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.4, 0.0], [0.2, 0.3]]])
    ar23ns = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-1.9, 0.0], [0.4, -0.6]], [[0.3, 0.0], [0.1, -0.1]]])
    vp = VarmaPoly(ar23, ma22)
    print(vars(vp))
    print(vp.vstack())
    print(vp.vstack(a24))
    print(vp.hstackarma_minus1())
    print(vp.getisstationary())
    print(vp.getisinvertible())
    vp2 = VarmaPoly(ar23ns)
    print(vp2.getisstationary())
    print(vp2.getisinvertible())