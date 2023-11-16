"""trying to verify theoretical acf of arma

explicit functions for autocovariance functions of ARIMA(1,1), MA(1), MA(2)
plus 3 functions from nitime.utils

"""
import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
ar = [1.0, -0.6]
ma = [1.0, 0.4]
mod = ''
x = arma_generate_sample(ar, ma, 5000)
x_acf = acf(x)[:10]
x_ir = arma_impulse_response(ar, ma)

def detrend(x, key=None):
    if False:
        i = 10
        return i + 15
    if key is None or key == 'constant':
        return detrend_mean(x)
    elif key == 'linear':
        return detrend_linear(x)

def demean(x, axis=0):
    if False:
        for i in range(10):
            print('nop')
    'Return x minus its mean along the specified axis'
    x = np.asarray(x)
    if axis:
        ind = [slice(None)] * axis
        ind.append(np.newaxis)
        return x - x.mean(axis)[ind]
    return x - x.mean(axis)

def detrend_mean(x):
    if False:
        for i in range(10):
            print('nop')
    'Return x minus the mean(x)'
    return x - x.mean()

def detrend_none(x):
    if False:
        for i in range(10):
            print('nop')
    'Return x: no detrending'
    return x

def detrend_linear(y):
    if False:
        i = 10
        return i + 15
    "Return y minus best fit line; 'linear' detrending "
    x = np.arange(len(y), dtype=np.float_)
    C = np.cov(x, y, bias=1)
    b = C[0, 1] / C[0, 0]
    a = y.mean() - b * x.mean()
    return y - (b * x + a)

def acovf_explicit(ar, ma, nobs):
    if False:
        while True:
            i = 10
    'add correlation of MA representation explicitely\n\n    '
    ir = arma_impulse_response(ar, ma)
    acovfexpl = [np.dot(ir[:nobs - t], ir[t:nobs]) for t in range(10)]
    return acovfexpl

def acovf_arma11(ar, ma):
    if False:
        i = 10
        return i + 15
    a = -ar[1]
    b = ma[1]
    rho = [(1.0 + b ** 2 + 2 * a * b) / (1.0 - a ** 2)]
    rho.append((1 + a * b) * (a + b) / (1.0 - a ** 2))
    for _ in range(8):
        last = rho[-1]
        rho.append(a * last)
    return np.array(rho)

def acovf_ma2(ma):
    if False:
        return 10
    b1 = -ma[1]
    b2 = -ma[2]
    rho = np.zeros(10)
    rho[0] = 1 + b1 ** 2 + b2 ** 2
    rho[1] = -b1 + b1 * b2
    rho[2] = -b2
    return rho

def acovf_ma1(ma):
    if False:
        for i in range(10):
            print('nop')
    b = -ma[1]
    rho = np.zeros(10)
    rho[0] = 1 + b ** 2
    rho[1] = -b
    return rho
ar1 = [1.0, -0.8]
ar0 = [1.0, 0.0]
ma1 = [1.0, 0.4]
ma2 = [1.0, 0.4, 0.6]
ma0 = [1.0, 0.0]
comparefn = dict([('ma1', acovf_ma1), ('ma2', acovf_ma2), ('arma11', acovf_arma11), ('ar1', acovf_arma11)])
cases = [('ma1', (ar0, ma1)), ('ma2', (ar0, ma2)), ('arma11', (ar1, ma1)), ('ar1', (ar1, ma0))]
for (c, args) in cases:
    (ar, ma) = args
    print('')
    print(c, ar, ma)
    myacovf = arma_acovf(ar, ma, nobs=10)
    myacf = arma_acf(ar, ma, lags=10)
    if c[:2] == 'ma':
        othacovf = comparefn[c](ma)
    else:
        othacovf = comparefn[c](ar, ma)
    print(myacovf[:5])
    print(othacovf[:5])
    assert_array_almost_equal(myacovf, othacovf, 10)
    assert_array_almost_equal(myacf, othacovf / othacovf[0], 10)

def ar_generator(N=512, sigma=1.0):
    if False:
        while True:
            i = 10
    taps = np.array([2.7607, -3.8106, 2.6535, -0.9238])
    v = np.random.normal(size=N, scale=sigma ** 0.5)
    u = np.zeros(N)
    P = len(taps)
    for l in range(P):
        u[l] = v[l] + np.dot(u[:l][::-1], taps[:l])
    for l in range(P, N):
        u[l] = v[l] + np.dot(u[l - P:l][::-1], taps)
    return (u, v, taps)

def autocorr(s, axis=-1):
    if False:
        for i in range(10):
            print('nop')
    'Returns the autocorrelation of signal s at all lags. Adheres to the\ndefinition r(k) = E{s(n)s*(n-k)} where E{} is the expectation operator.\n'
    N = s.shape[axis]
    S = np.fft.fft(s, n=2 * N - 1, axis=axis)
    sxx = np.fft.ifft(S * S.conjugate(), axis=axis).real[:N]
    return sxx / N

def norm_corr(x, y, mode='valid'):
    if False:
        return 10
    "Returns the correlation between two ndarrays, by calling np.correlate in\n'same' mode and normalizing the result by the std of the arrays and by\ntheir lengths. This results in a correlation = 1 for an auto-correlation"
    return np.correlate(x, y, mode) / (np.std(x) * np.std(y) * x.shape[-1])

def pltacorr(self, x, **kwargs):
    if False:
        return 10
    "\n    call signature::\n\n        acorr(x, normed=True, detrend=detrend_none, usevlines=True,\n              maxlags=10, **kwargs)\n\n    Plot the autocorrelation of *x*.  If *normed* = *True*,\n    normalize the data by the autocorrelation at 0-th lag.  *x* is\n    detrended by the *detrend* callable (default no normalization).\n\n    Data are plotted as ``plot(lags, c, **kwargs)``\n\n    Return value is a tuple (*lags*, *c*, *line*) where:\n\n      - *lags* are a length 2*maxlags+1 lag vector\n\n      - *c* is the 2*maxlags+1 auto correlation vector\n\n      - *line* is a :class:`~matplotlib.lines.Line2D` instance\n        returned by :meth:`plot`\n\n    The default *linestyle* is None and the default *marker* is\n    ``'o'``, though these can be overridden with keyword args.\n    The cross correlation is performed with\n    :func:`numpy.correlate` with *mode* = 2.\n\n    If *usevlines* is *True*, :meth:`~matplotlib.axes.Axes.vlines`\n    rather than :meth:`~matplotlib.axes.Axes.plot` is used to draw\n    vertical lines from the origin to the acorr.  Otherwise, the\n    plot style is determined by the kwargs, which are\n    :class:`~matplotlib.lines.Line2D` properties.\n\n    *maxlags* is a positive integer detailing the number of lags\n    to show.  The default value of *None* will return all\n    :math:`2 \\mathrm{len}(x) - 1` lags.\n\n    The return value is a tuple (*lags*, *c*, *linecol*, *b*)\n    where\n\n    - *linecol* is the\n      :class:`~matplotlib.collections.LineCollection`\n\n    - *b* is the *x*-axis.\n\n    .. seealso::\n\n        :meth:`~matplotlib.axes.Axes.plot` or\n        :meth:`~matplotlib.axes.Axes.vlines`\n           For documentation on valid kwargs.\n\n    **Example:**\n\n    :func:`~matplotlib.pyplot.xcorr` above, and\n    :func:`~matplotlib.pyplot.acorr` below.\n\n    **Example:**\n\n    .. plot:: mpl_examples/pylab_examples/xcorr_demo.py\n    "
    return self.xcorr(x, x, **kwargs)

def pltxcorr(self, x, y, normed=True, detrend=detrend_none, usevlines=True, maxlags=10, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    call signature::\n\n        def xcorr(self, x, y, normed=True, detrend=detrend_none,\n          usevlines=True, maxlags=10, **kwargs):\n\n    Plot the cross correlation between *x* and *y*.  If *normed* =\n    *True*, normalize the data by the cross correlation at 0-th\n    lag.  *x* and y are detrended by the *detrend* callable\n    (default no normalization).  *x* and *y* must be equal length.\n\n    Data are plotted as ``plot(lags, c, **kwargs)``\n\n    Return value is a tuple (*lags*, *c*, *line*) where:\n\n      - *lags* are a length ``2*maxlags+1`` lag vector\n\n      - *c* is the ``2*maxlags+1`` auto correlation vector\n\n      - *line* is a :class:`~matplotlib.lines.Line2D` instance\n         returned by :func:`~matplotlib.pyplot.plot`.\n\n    The default *linestyle* is *None* and the default *marker* is\n    'o', though these can be overridden with keyword args.  The\n    cross correlation is performed with :func:`numpy.correlate`\n    with *mode* = 2.\n\n    If *usevlines* is *True*:\n\n       :func:`~matplotlib.pyplot.vlines`\n       rather than :func:`~matplotlib.pyplot.plot` is used to draw\n       vertical lines from the origin to the xcorr.  Otherwise the\n       plotstyle is determined by the kwargs, which are\n       :class:`~matplotlib.lines.Line2D` properties.\n\n       The return value is a tuple (*lags*, *c*, *linecol*, *b*)\n       where *linecol* is the\n       :class:`matplotlib.collections.LineCollection` instance and\n       *b* is the *x*-axis.\n\n    *maxlags* is a positive integer detailing the number of lags to show.\n    The default value of *None* will return all ``(2*len(x)-1)`` lags.\n\n    **Example:**\n\n    :func:`~matplotlib.pyplot.xcorr` above, and\n    :func:`~matplotlib.pyplot.acorr` below.\n\n    **Example:**\n\n    .. plot:: mpl_examples/pylab_examples/xcorr_demo.py\n    "
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    x = detrend(np.asarray(x))
    y = detrend(np.asarray(y))
    c = np.correlate(x, y, mode=2)
    if normed:
        c /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    if maxlags is None:
        maxlags = Nx - 1
    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly positive < %d' % Nx)
    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    if usevlines:
        a = self.vlines(lags, [0], c, **kwargs)
        b = self.axhline(**kwargs)
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        d = self.plot(lags, c, **kwargs)
    else:
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        (a,) = self.plot(lags, c, **kwargs)
        b = None
    return (lags, c, a, b)
arrvs = ar_generator()
arma = ARIMA(arrvs[0])
res = arma.fit((4, 0, 0))
print(res[0])
acf1 = acf(arrvs[0])
acovf1b = acovf(arrvs[0], unbiased=False)
acf2 = autocorr(arrvs[0])
acf2m = autocorr(arrvs[0] - arrvs[0].mean())
print(acf1[:10])
print(acovf1b[:10])
print(acf2[:10])
print(acf2m[:10])
x = arma_generate_sample([1.0, -0.8], [1.0], 500)
print(acf(x)[:20])
print(regression.yule_walker(x, 10))
plt.plot(x)
plt.figure()
pltxcorr(plt, x, x)
plt.figure()
pltxcorr(plt, x, x, usevlines=False)
plt.figure()
plot_acf(plt, acf1[:20], np.arange(len(acf1[:20])), usevlines=True)
plt.figure()
ax = plt.subplot(211)
plot_acf(ax, acf1[:20], usevlines=True)
ax = plt.subplot(212)
plot_acf(ax, acf1[:20], np.arange(len(acf1[:20])), usevlines=False)