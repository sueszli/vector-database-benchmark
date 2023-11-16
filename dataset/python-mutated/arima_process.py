"""ARMA process and estimation with scipy.signal.lfilter

Notes
-----
* written without textbook, works but not sure about everything
  briefly checked and it looks to be standard least squares, see below

* theoretical autocorrelation function of general ARMA
  Done, relatively easy to guess solution, time consuming to get
  theoretical test cases, example file contains explicit formulas for
  acovf of MA(1), MA(2) and ARMA(1,1)

Properties:
Judge, ... (1985): The Theory and Practise of Econometrics

Author: josefpktd
License: BSD
"""
import warnings
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
__all__ = ['arma_acf', 'arma_acovf', 'arma_generate_sample', 'arma_impulse_response', 'arma2ar', 'arma2ma', 'deconvolve', 'lpol2index', 'index2lpol']
NONSTATIONARY_ERROR = "The model's autoregressive parameters (ar) indicate that the process\n is non-stationary. arma_acovf can only be used with stationary processes.\n"

def arma_generate_sample(ar, ma, nsample, scale=1, distrvs=None, axis=0, burnin=0):
    if False:
        return 10
    "\n    Simulate data from an ARMA.\n\n    Parameters\n    ----------\n    ar : array_like\n        The coefficient for autoregressive lag polynomial, including zero lag.\n    ma : array_like\n        The coefficient for moving-average lag polynomial, including zero lag.\n    nsample : int or tuple of ints\n        If nsample is an integer, then this creates a 1d timeseries of\n        length size. If nsample is a tuple, creates a len(nsample)\n        dimensional time series where time is indexed along the input\n        variable ``axis``. All series are unless ``distrvs`` generates\n        dependent data.\n    scale : float\n        The standard deviation of noise.\n    distrvs : function, random number generator\n        A function that generates the random numbers, and takes ``size``\n        as argument. The default is np.random.standard_normal.\n    axis : int\n        See nsample for details.\n    burnin : int\n        Number of observation at the beginning of the sample to drop.\n        Used to reduce dependence on initial values.\n\n    Returns\n    -------\n    ndarray\n        Random sample(s) from an ARMA process.\n\n    Notes\n    -----\n    As mentioned above, both the AR and MA components should include the\n    coefficient on the zero-lag. This is typically 1. Further, due to the\n    conventions used in signal processing used in signal.lfilter vs.\n    conventions in statistics for ARMA processes, the AR parameters should\n    have the opposite sign of what you might expect. See the examples below.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> np.random.seed(12345)\n    >>> arparams = np.array([.75, -.25])\n    >>> maparams = np.array([.65, .35])\n    >>> ar = np.r_[1, -arparams] # add zero-lag and negate\n    >>> ma = np.r_[1, maparams] # add zero-lag\n    >>> y = sm.tsa.arma_generate_sample(ar, ma, 250)\n    >>> model = sm.tsa.ARIMA(y, (2, 0, 2), trend='n').fit(disp=0)\n    >>> model.params\n    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])\n    "
    distrvs = np.random.standard_normal if distrvs is None else distrvs
    if np.ndim(nsample) == 0:
        nsample = [nsample]
    if burnin:
        newsize = list(nsample)
        newsize[axis] += burnin
        newsize = tuple(newsize)
        fslice = [slice(None)] * len(newsize)
        fslice[axis] = slice(burnin, None, None)
        fslice = tuple(fslice)
    else:
        newsize = tuple(nsample)
        fslice = tuple([slice(None)] * np.ndim(newsize))
    eta = scale * distrvs(size=newsize)
    return signal.lfilter(ma, ar, eta, axis=axis)[fslice]

def arma_acovf(ar, ma, nobs=10, sigma2=1, dtype=None):
    if False:
        i = 10
        return i + 15
    '\n    Theoretical autocovariances of stationary ARMA processes\n\n    Parameters\n    ----------\n    ar : array_like, 1d\n        The coefficients for autoregressive lag polynomial, including zero lag.\n    ma : array_like, 1d\n        The coefficients for moving-average lag polynomial, including zero lag.\n    nobs : int\n        The number of terms (lags plus zero lag) to include in returned acovf.\n    sigma2 : float\n        Variance of the innovation term.\n\n    Returns\n    -------\n    ndarray\n        The autocovariance of ARMA process given by ar, ma.\n\n    See Also\n    --------\n    arma_acf : Autocorrelation function for ARMA processes.\n    acovf : Sample autocovariance estimation.\n\n    References\n    ----------\n    .. [*] Brockwell, Peter J., and Richard A. Davis. 2009. Time Series:\n        Theory and Methods. 2nd ed. 1991. New York, NY: Springer.\n    '
    if dtype is None:
        dtype = np.common_type(np.array(ar), np.array(ma), np.array(sigma2))
    p = len(ar) - 1
    q = len(ma) - 1
    m = max(p, q) + 1
    if sigma2.real < 0:
        raise ValueError('Must have positive innovation variance.')
    if p == q == 0:
        out = np.zeros(nobs, dtype=dtype)
        out[0] = sigma2
        return out
    elif p > 0 and np.max(np.abs(np.roots(ar))) >= 1:
        raise ValueError(NONSTATIONARY_ERROR)
    ma_coeffs = arma2ma(ar, ma, lags=m)
    A = np.zeros((m, m), dtype=dtype)
    b = np.zeros((m, 1), dtype=dtype)
    tmp_ar = np.zeros(m, dtype=dtype)
    tmp_ar[:p + 1] = ar
    for k in range(m):
        A[k, :k + 1] = tmp_ar[:k + 1][::-1]
        A[k, 1:m - k] += tmp_ar[k + 1:m]
        b[k] = sigma2 * np.dot(ma[k:q + 1], ma_coeffs[:max(q + 1 - k, 0)])
    acovf = np.zeros(max(nobs, m), dtype=dtype)
    try:
        acovf[:m] = np.linalg.solve(A, b)[:, 0]
    except np.linalg.LinAlgError:
        raise ValueError(NONSTATIONARY_ERROR)
    if nobs > m:
        zi = signal.lfiltic([1], ar, acovf[:m][::-1])
        acovf[m:] = signal.lfilter([1], ar, np.zeros(nobs - m, dtype=dtype), zi=zi)[0]
    return acovf[:nobs]

def arma_acf(ar, ma, lags=10):
    if False:
        i = 10
        return i + 15
    '\n    Theoretical autocorrelation function of an ARMA process.\n\n    Parameters\n    ----------\n    ar : array_like\n        Coefficients for autoregressive lag polynomial, including zero lag.\n    ma : array_like\n        Coefficients for moving-average lag polynomial, including zero lag.\n    lags : int\n        The number of terms (lags plus zero lag) to include in returned acf.\n\n    Returns\n    -------\n    ndarray\n        The autocorrelations of ARMA process given by ar and ma.\n\n    See Also\n    --------\n    arma_acovf : Autocovariances from ARMA processes.\n    acf : Sample autocorrelation function estimation.\n    acovf : Sample autocovariance function estimation.\n    '
    acovf = arma_acovf(ar, ma, lags)
    return acovf / acovf[0]

def arma_pacf(ar, ma, lags=10):
    if False:
        return 10
    '\n    Theoretical partial autocorrelation function of an ARMA process.\n\n    Parameters\n    ----------\n    ar : array_like, 1d\n        The coefficients for autoregressive lag polynomial, including zero lag.\n    ma : array_like, 1d\n        The coefficients for moving-average lag polynomial, including zero lag.\n    lags : int\n        The number of terms (lags plus zero lag) to include in returned pacf.\n\n    Returns\n    -------\n    ndarrray\n        The partial autocorrelation of ARMA process given by ar and ma.\n\n    Notes\n    -----\n    Solves yule-walker equation for each lag order up to nobs lags.\n\n    not tested/checked yet\n    '
    apacf = np.zeros(lags)
    acov = arma_acf(ar, ma, lags=lags + 1)
    apacf[0] = 1.0
    for k in range(2, lags + 1):
        r = acov[:k]
        apacf[k - 1] = linalg.solve(linalg.toeplitz(r[:-1]), r[1:])[-1]
    return apacf

def arma_periodogram(ar, ma, worN=None, whole=0):
    if False:
        i = 10
        return i + 15
    '\n    Periodogram for ARMA process given by lag-polynomials ar and ma.\n\n    Parameters\n    ----------\n    ar : array_like\n        The autoregressive lag-polynomial with leading 1 and lhs sign.\n    ma : array_like\n        The moving average lag-polynomial with leading 1.\n    worN : {None, int}, optional\n        An option for scipy.signal.freqz (read "w or N").\n        If None, then compute at 512 frequencies around the unit circle.\n        If a single integer, the compute at that many frequencies.\n        Otherwise, compute the response at frequencies given in worN.\n    whole : {0,1}, optional\n        An options for scipy.signal.freqz/\n        Normally, frequencies are computed from 0 to pi (upper-half of\n        unit-circle.  If whole is non-zero compute frequencies from 0 to 2*pi.\n\n    Returns\n    -------\n    w : ndarray\n        The frequencies.\n    sd : ndarray\n        The periodogram, also known as the spectral density.\n\n    Notes\n    -----\n    Normalization ?\n\n    This uses signal.freqz, which does not use fft. There is a fft version\n    somewhere.\n    '
    (w, h) = signal.freqz(ma, ar, worN=worN, whole=whole)
    sd = np.abs(h) ** 2 / np.sqrt(2 * np.pi)
    if np.any(np.isnan(h)):
        import warnings
        warnings.warn('Warning: nan in frequency response h, maybe a unit root', RuntimeWarning, stacklevel=2)
    return (w, sd)

def arma_impulse_response(ar, ma, leads=100):
    if False:
        i = 10
        return i + 15
    '\n    Compute the impulse response function (MA representation) for ARMA process.\n\n    Parameters\n    ----------\n    ar : array_like, 1d\n        The auto regressive lag polynomial.\n    ma : array_like, 1d\n        The moving average lag polynomial.\n    leads : int\n        The number of observations to calculate.\n\n    Returns\n    -------\n    ndarray\n        The impulse response function with nobs elements.\n\n    Notes\n    -----\n    This is the same as finding the MA representation of an ARMA(p,q).\n    By reversing the role of ar and ma in the function arguments, the\n    returned result is the AR representation of an ARMA(p,q), i.e\n\n    ma_representation = arma_impulse_response(ar, ma, leads=100)\n    ar_representation = arma_impulse_response(ma, ar, leads=100)\n\n    Fully tested against matlab\n\n    Examples\n    --------\n    AR(1)\n\n    >>> arma_impulse_response([1.0, -0.8], [1.], leads=10)\n    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,\n            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])\n\n    this is the same as\n\n    >>> 0.8**np.arange(10)\n    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,\n            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])\n\n    MA(2)\n\n    >>> arma_impulse_response([1.0], [1., 0.5, 0.2], leads=10)\n    array([ 1. ,  0.5,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ])\n\n    ARMA(1,2)\n\n    >>> arma_impulse_response([1.0, -0.8], [1., 0.5, 0.2], leads=10)\n    array([ 1.        ,  1.3       ,  1.24      ,  0.992     ,  0.7936    ,\n            0.63488   ,  0.507904  ,  0.4063232 ,  0.32505856,  0.26004685])\n    '
    impulse = np.zeros(leads)
    impulse[0] = 1.0
    return signal.lfilter(ma, ar, impulse)

def arma2ma(ar, ma, lags=100):
    if False:
        print('Hello World!')
    '\n    A finite-lag approximate MA representation of an ARMA process.\n\n    Parameters\n    ----------\n    ar : ndarray\n        The auto regressive lag polynomial.\n    ma : ndarray\n        The moving average lag polynomial.\n    lags : int\n        The number of coefficients to calculate.\n\n    Returns\n    -------\n    ndarray\n        The coefficients of AR lag polynomial with nobs elements.\n\n    Notes\n    -----\n    Equivalent to ``arma_impulse_response(ma, ar, leads=100)``\n    '
    return arma_impulse_response(ar, ma, leads=lags)

def arma2ar(ar, ma, lags=100):
    if False:
        i = 10
        return i + 15
    '\n    A finite-lag AR approximation of an ARMA process.\n\n    Parameters\n    ----------\n    ar : array_like\n        The auto regressive lag polynomial.\n    ma : array_like\n        The moving average lag polynomial.\n    lags : int\n        The number of coefficients to calculate.\n\n    Returns\n    -------\n    ndarray\n        The coefficients of AR lag polynomial with nobs elements.\n\n    Notes\n    -----\n    Equivalent to ``arma_impulse_response(ma, ar, leads=100)``\n    '
    return arma_impulse_response(ma, ar, leads=lags)

def ar2arma(ar_des, p, q, n=20, mse='ar', start=None):
    if False:
        while True:
            i = 10
    "\n    Find arma approximation to ar process.\n\n    This finds the ARMA(p,q) coefficients that minimize the integrated\n    squared difference between the impulse_response functions (MA\n    representation) of the AR and the ARMA process. This does not  check\n    whether the MA lag polynomial of the ARMA process is invertible, neither\n    does it check the roots of the AR lag polynomial.\n\n    Parameters\n    ----------\n    ar_des : array_like\n        The coefficients of original AR lag polynomial, including lag zero.\n    p : int\n        The length of desired AR lag polynomials.\n    q : int\n        The length of desired MA lag polynomials.\n    n : int\n        The number of terms of the impulse_response function to include in the\n        objective function for the approximation.\n    mse : str, 'ar'\n        Not used.\n    start : ndarray\n        Initial values to use when finding the approximation.\n\n    Returns\n    -------\n    ar_app : ndarray\n        The coefficients of the AR lag polynomials of the approximation.\n    ma_app : ndarray\n        The coefficients of the MA lag polynomials of the approximation.\n    res : tuple\n        The result of optimize.leastsq.\n\n    Notes\n    -----\n    Extension is possible if we want to match autocovariance instead\n    of impulse response function.\n    "

    def msear_err(arma, ar_des):
        if False:
            i = 10
            return i + 15
        (ar, ma) = (np.r_[1, arma[:p - 1]], np.r_[1, arma[p - 1:]])
        ar_approx = arma_impulse_response(ma, ar, n)
        return ar_des - ar_approx
    if start is None:
        arma0 = np.r_[-0.9 * np.ones(p - 1), np.zeros(q - 1)]
    else:
        arma0 = start
    res = optimize.leastsq(msear_err, arma0, ar_des, maxfev=5000)
    arma_app = np.atleast_1d(res[0])
    ar_app = (np.r_[1, arma_app[:p - 1]],)
    ma_app = np.r_[1, arma_app[p - 1:]]
    return (ar_app, ma_app, res)
_arma_docs = {'ar': arma2ar.__doc__, 'ma': arma2ma.__doc__}

def lpol2index(ar):
    if False:
        while True:
            i = 10
    '\n    Remove zeros from lag polynomial\n\n    Parameters\n    ----------\n    ar : array_like\n        coefficients of lag polynomial\n\n    Returns\n    -------\n    coeffs : ndarray\n        non-zero coefficients of lag polynomial\n    index : ndarray\n        index (lags) of lag polynomial with non-zero elements\n    '
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.ComplexWarning)
        ar = array_like(ar, 'ar')
    index = np.nonzero(ar)[0]
    coeffs = ar[index]
    return (coeffs, index)

def index2lpol(coeffs, index):
    if False:
        print('Hello World!')
    '\n    Expand coefficients to lag poly\n\n    Parameters\n    ----------\n    coeffs : ndarray\n        non-zero coefficients of lag polynomial\n    index : ndarray\n        index (lags) of lag polynomial with non-zero elements\n\n    Returns\n    -------\n    ar : array_like\n        coefficients of lag polynomial\n    '
    n = max(index)
    ar = np.zeros(n + 1)
    ar[index] = coeffs
    return ar

def lpol_fima(d, n=20):
    if False:
        while True:
            i = 10
    'MA representation of fractional integration\n\n    .. math:: (1-L)^{-d} for |d|<0.5  or |d|<1 (?)\n\n    Parameters\n    ----------\n    d : float\n        fractional power\n    n : int\n        number of terms to calculate, including lag zero\n\n    Returns\n    -------\n    ma : ndarray\n        coefficients of lag polynomial\n    '
    from scipy.special import gammaln
    j = np.arange(n)
    return np.exp(gammaln(d + j) - gammaln(j + 1) - gammaln(d))

def lpol_fiar(d, n=20):
    if False:
        return 10
    'AR representation of fractional integration\n\n    .. math:: (1-L)^{d} for |d|<0.5  or |d|<1 (?)\n\n    Parameters\n    ----------\n    d : float\n        fractional power\n    n : int\n        number of terms to calculate, including lag zero\n\n    Returns\n    -------\n    ar : ndarray\n        coefficients of lag polynomial\n\n    Notes:\n    first coefficient is 1, negative signs except for first term,\n    ar(L)*x_t\n    '
    from scipy.special import gammaln
    j = np.arange(n)
    ar = -np.exp(gammaln(-d + j) - gammaln(j + 1) - gammaln(-d))
    ar[0] = 1
    return ar

def lpol_sdiff(s):
    if False:
        return 10
    'return coefficients for seasonal difference (1-L^s)\n\n    just a trivial convenience function\n\n    Parameters\n    ----------\n    s : int\n        number of periods in season\n\n    Returns\n    -------\n    sdiff : list, length s+1\n    '
    return [1] + [0] * (s - 1) + [-1]

def deconvolve(num, den, n=None):
    if False:
        i = 10
        return i + 15
    'Deconvolves divisor out of signal, division of polynomials for n terms\n\n    calculates den^{-1} * num\n\n    Parameters\n    ----------\n    num : array_like\n        signal or lag polynomial\n    denom : array_like\n        coefficients of lag polynomial (linear filter)\n    n : None or int\n        number of terms of quotient\n\n    Returns\n    -------\n    quot : ndarray\n        quotient or filtered series\n    rem : ndarray\n        remainder\n\n    Notes\n    -----\n    If num is a time series, then this applies the linear filter den^{-1}.\n    If both num and den are both lag polynomials, then this calculates the\n    quotient polynomial for n terms and also returns the remainder.\n\n    This is copied from scipy.signal.signaltools and added n as optional\n    parameter.\n    '
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)
    N = len(num)
    D = len(den)
    if D > N and n is None:
        quot = []
        rem = num
    else:
        if n is None:
            n = N - D + 1
        input = np.zeros(n, float)
        input[0] = 1
        quot = signal.lfilter(num, den, input)
        num_approx = signal.convolve(den, quot, mode='full')
        if len(num) < len(num_approx):
            num = np.concatenate((num, np.zeros(len(num_approx) - len(num))))
        rem = num - num_approx
    return (quot, rem)
_generate_sample_doc = Docstring(arma_generate_sample.__doc__)
_generate_sample_doc.remove_parameters(['ar', 'ma'])
_generate_sample_doc.replace_block('Notes', [])
_generate_sample_doc.replace_block('Examples', [])

class ArmaProcess:
    """
    Theoretical properties of an ARMA process for specified lag-polynomials.

    Parameters
    ----------
    ar : array_like
        Coefficient for autoregressive lag polynomial, including zero lag.
        Must be entered using the signs from the lag polynomial representation.
        See the notes for more information about the sign.
    ma : array_like
        Coefficient for moving-average lag polynomial, including zero lag.
    nobs : int, optional
        Length of simulated time series. Used, for example, if a sample is
        generated. See example.

    Notes
    -----
    Both the AR and MA components must include the coefficient on the
    zero-lag. In almost all cases these values should be 1. Further, due to
    using the lag-polynomial representation, the AR parameters should
    have the opposite sign of what one would write in the ARMA representation.
    See the examples below.

    The ARMA(p,q) process is described by

    .. math::

        y_{t}=\\phi_{1}y_{t-1}+\\ldots+\\phi_{p}y_{t-p}+\\theta_{1}\\epsilon_{t-1}
               +\\ldots+\\theta_{q}\\epsilon_{t-q}+\\epsilon_{t}

    and the parameterization used in this function uses the lag-polynomial
    representation,

    .. math::

        \\left(1-\\phi_{1}L-\\ldots-\\phi_{p}L^{p}\\right)y_{t} =
            \\left(1+\\theta_{1}L+\\ldots+\\theta_{q}L^{q}\\right)\\epsilon_{t}

    Examples
    --------
    ARMA(2,2) with AR coefficients 0.75 and -0.25, and MA coefficients 0.65 and 0.35

    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -arparams] # add zero-lag and negate
    >>> ma = np.r_[1, maparams] # add zero-lag
    >>> arma_process = sm.tsa.ArmaProcess(ar, ma)
    >>> arma_process.isstationary
    True
    >>> arma_process.isinvertible
    True
    >>> arma_process.arroots
    array([1.5-1.32287566j, 1.5+1.32287566j])
    >>> y = arma_process.generate_sample(250)
    >>> model = sm.tsa.ARIMA(y, (2, 0, 2), trend='n').fit(disp=0)
    >>> model.params
    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])

    The same ARMA(2,2) Using the from_coeffs class method

    >>> arma_process = sm.tsa.ArmaProcess.from_coeffs(arparams, maparams)
    >>> arma_process.arroots
    array([1.5-1.32287566j, 1.5+1.32287566j])
    """

    def __init__(self, ar=None, ma=None, nobs=100):
        if False:
            print('Hello World!')
        if ar is None:
            ar = np.array([1.0])
        if ma is None:
            ma = np.array([1.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.ComplexWarning)
            self.ar = array_like(ar, 'ar')
            self.ma = array_like(ma, 'ma')
        self.arcoefs = -self.ar[1:]
        self.macoefs = self.ma[1:]
        self.arpoly = np.polynomial.Polynomial(self.ar)
        self.mapoly = np.polynomial.Polynomial(self.ma)
        self.nobs = nobs

    @classmethod
    def from_roots(cls, maroots=None, arroots=None, nobs=100):
        if False:
            return 10
        '\n        Create ArmaProcess from AR and MA polynomial roots.\n\n        Parameters\n        ----------\n        maroots : array_like, optional\n            Roots for the MA polynomial\n            1 + theta_1*z + theta_2*z^2 + ..... + theta_n*z^n\n        arroots : array_like, optional\n            Roots for the AR polynomial\n            1 - phi_1*z - phi_2*z^2 - ..... - phi_n*z^n\n        nobs : int, optional\n            Length of simulated time series. Used, for example, if a sample\n            is generated.\n\n        Returns\n        -------\n        ArmaProcess\n            Class instance initialized with arcoefs and macoefs.\n\n        Examples\n        --------\n        >>> arroots = [.75, -.25]\n        >>> maroots = [.65, .35]\n        >>> arma_process = sm.tsa.ArmaProcess.from_roots(arroots, maroots)\n        >>> arma_process.isstationary\n        True\n        >>> arma_process.isinvertible\n        True\n        '
        if arroots is not None and len(arroots):
            arpoly = np.polynomial.polynomial.Polynomial.fromroots(arroots)
            arcoefs = arpoly.coef[1:] / arpoly.coef[0]
        else:
            arcoefs = []
        if maroots is not None and len(maroots):
            mapoly = np.polynomial.polynomial.Polynomial.fromroots(maroots)
            macoefs = mapoly.coef[1:] / mapoly.coef[0]
        else:
            macoefs = []
        return cls(np.r_[1, arcoefs], np.r_[1, macoefs], nobs=nobs)

    @classmethod
    def from_coeffs(cls, arcoefs=None, macoefs=None, nobs=100):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create ArmaProcess from an ARMA representation.\n\n        Parameters\n        ----------\n        arcoefs : array_like\n            Coefficient for autoregressive lag polynomial, not including zero\n            lag. The sign is inverted to conform to the usual time series\n            representation of an ARMA process in statistics. See the class\n            docstring for more information.\n        macoefs : array_like\n            Coefficient for moving-average lag polynomial, excluding zero lag.\n        nobs : int, optional\n            Length of simulated time series. Used, for example, if a sample\n            is generated.\n\n        Returns\n        -------\n        ArmaProcess\n            Class instance initialized with arcoefs and macoefs.\n\n        Examples\n        --------\n        >>> arparams = [.75, -.25]\n        >>> maparams = [.65, .35]\n        >>> arma_process = sm.tsa.ArmaProcess.from_coeffs(ar, ma)\n        >>> arma_process.isstationary\n        True\n        >>> arma_process.isinvertible\n        True\n        '
        arcoefs = [] if arcoefs is None else arcoefs
        macoefs = [] if macoefs is None else macoefs
        return cls(np.r_[1, -np.asarray(arcoefs)], np.r_[1, np.asarray(macoefs)], nobs=nobs)

    @classmethod
    def from_estimation(cls, model_results, nobs=None):
        if False:
            i = 10
            return i + 15
        '\n        Create an ArmaProcess from the results of an ARIMA estimation.\n\n        Parameters\n        ----------\n        model_results : ARIMAResults instance\n            A fitted model.\n        nobs : int, optional\n            If None, nobs is taken from the results.\n\n        Returns\n        -------\n        ArmaProcess\n            Class instance initialized from model_results.\n\n        See Also\n        --------\n        statsmodels.tsa.arima.model.ARIMA\n            The models class used to create the ArmaProcess\n        '
        nobs = nobs or model_results.nobs
        return cls(model_results.polynomial_reduced_ar, model_results.polynomial_reduced_ma, nobs=nobs)

    def __mul__(self, oth):
        if False:
            print('Hello World!')
        if isinstance(oth, self.__class__):
            ar = (self.arpoly * oth.arpoly).coef
            ma = (self.mapoly * oth.mapoly).coef
        else:
            try:
                (aroth, maoth) = oth
                arpolyoth = np.polynomial.Polynomial(aroth)
                mapolyoth = np.polynomial.Polynomial(maoth)
                ar = (self.arpoly * arpolyoth).coef
                ma = (self.mapoly * mapolyoth).coef
            except:
                raise TypeError('Other type is not a valid type')
        return self.__class__(ar, ma, nobs=self.nobs)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        msg = 'ArmaProcess({0}, {1}, nobs={2}) at {3}'
        return msg.format(self.ar.tolist(), self.ma.tolist(), self.nobs, hex(id(self)))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'ArmaProcess\nAR: {0}\nMA: {1}'.format(self.ar.tolist(), self.ma.tolist())

    @Appender(remove_parameters(arma_acovf.__doc__, ['ar', 'ma', 'sigma2']))
    def acovf(self, nobs=None):
        if False:
            i = 10
            return i + 15
        nobs = nobs or self.nobs
        return arma_acovf(self.ar, self.ma, nobs=nobs)

    @Appender(remove_parameters(arma_acf.__doc__, ['ar', 'ma']))
    def acf(self, lags=None):
        if False:
            while True:
                i = 10
        lags = lags or self.nobs
        return arma_acf(self.ar, self.ma, lags=lags)

    @Appender(remove_parameters(arma_pacf.__doc__, ['ar', 'ma']))
    def pacf(self, lags=None):
        if False:
            return 10
        lags = lags or self.nobs
        return arma_pacf(self.ar, self.ma, lags=lags)

    @Appender(remove_parameters(arma_periodogram.__doc__, ['ar', 'ma', 'worN', 'whole']))
    def periodogram(self, nobs=None):
        if False:
            return 10
        nobs = nobs or self.nobs
        return arma_periodogram(self.ar, self.ma, worN=nobs)

    @Appender(remove_parameters(arma_impulse_response.__doc__, ['ar', 'ma']))
    def impulse_response(self, leads=None):
        if False:
            return 10
        leads = leads or self.nobs
        return arma_impulse_response(self.ar, self.ma, leads=leads)

    @Appender(remove_parameters(arma2ma.__doc__, ['ar', 'ma']))
    def arma2ma(self, lags=None):
        if False:
            for i in range(10):
                print('nop')
        lags = lags or self.lags
        return arma2ma(self.ar, self.ma, lags=lags)

    @Appender(remove_parameters(arma2ar.__doc__, ['ar', 'ma']))
    def arma2ar(self, lags=None):
        if False:
            print('Hello World!')
        lags = lags or self.lags
        return arma2ar(self.ar, self.ma, lags=lags)

    @property
    def arroots(self):
        if False:
            i = 10
            return i + 15
        'Roots of autoregressive lag-polynomial'
        return self.arpoly.roots()

    @property
    def maroots(self):
        if False:
            i = 10
            return i + 15
        'Roots of moving average lag-polynomial'
        return self.mapoly.roots()

    @property
    def isstationary(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Arma process is stationary if AR roots are outside unit circle.\n\n        Returns\n        -------\n        bool\n             True if autoregressive roots are outside unit circle.\n        '
        if np.all(np.abs(self.arroots) > 1.0):
            return True
        else:
            return False

    @property
    def isinvertible(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Arma process is invertible if MA roots are outside unit circle.\n\n        Returns\n        -------\n        bool\n             True if moving average roots are outside unit circle.\n        '
        if np.all(np.abs(self.maroots) > 1):
            return True
        else:
            return False

    def invertroots(self, retnew=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make MA polynomial invertible by inverting roots inside unit circle.\n\n        Parameters\n        ----------\n        retnew : bool\n            If False (default), then return the lag-polynomial as array.\n            If True, then return a new instance with invertible MA-polynomial.\n\n        Returns\n        -------\n        manew : ndarray\n           A new invertible MA lag-polynomial, returned if retnew is false.\n        wasinvertible : bool\n           True if the MA lag-polynomial was already invertible, returned if\n           retnew is false.\n        armaprocess : new instance of class\n           If retnew is true, then return a new instance with invertible\n           MA-polynomial.\n        '
        pr = self.maroots
        mainv = self.ma
        invertible = self.isinvertible
        if not invertible:
            pr[np.abs(pr) < 1] = 1.0 / pr[np.abs(pr) < 1]
            pnew = np.polynomial.Polynomial.fromroots(pr)
            mainv = pnew.coef / pnew.coef[0]
        if retnew:
            return self.__class__(self.ar, mainv, nobs=self.nobs)
        else:
            return (mainv, invertible)

    @Appender(str(_generate_sample_doc))
    def generate_sample(self, nsample=100, scale=1.0, distrvs=None, axis=0, burnin=0):
        if False:
            return 10
        return arma_generate_sample(self.ar, self.ma, nsample, scale, distrvs, axis=axis, burnin=burnin)