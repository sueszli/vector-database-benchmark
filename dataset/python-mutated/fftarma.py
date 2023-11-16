"""
Created on Mon Dec 14 19:53:25 2009

Author: josef-pktd

generate arma sample using fft with all the lfilter it looks slow
to get the ma representation first

apply arma filter (in ar representation) to time series to get white noise
but seems slow to be useful for fast estimation for nobs=10000

change/check: instead of using marep, use fft-transform of ar and ma
    separately, use ratio check theory is correct and example works
    DONE : feels much faster than lfilter
    -> use for estimation of ARMA
    -> use pade (scipy.interpolate) approximation to get starting polynomial
       from autocorrelation (is autocorrelation of AR(p) related to marep?)
       check if pade is fast, not for larger arrays ?
       maybe pade does not do the right thing for this, not tried yet
       scipy.pade([ 1.    ,  0.6,  0.25, 0.125, 0.0625, 0.1],2)
       raises LinAlgError: singular matrix
       also does not have roots inside unit circle ??
    -> even without initialization, it might be fast for estimation
    -> how do I enforce stationarity and invertibility,
       need helper function

get function drop imag if close to zero from numpy/scipy source, where?

"""
import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess

class ArmaFft(ArmaProcess):
    """fft tools for arma processes

    This class contains several methods that are providing the same or similar
    returns to try out and test different implementations.

    Notes
    -----
    TODO:
    check whether we do not want to fix maxlags, and create new instance if
    maxlag changes. usage for different lengths of timeseries ?
    or fix frequency and length for fft

    check default frequencies w, terminology norw  n_or_w

    some ffts are currently done without padding with zeros

    returns for spectral density methods needs checking, is it always the power
    spectrum hw*hw.conj()

    normalization of the power spectrum, spectral density: not checked yet, for
    example no variance of underlying process is used

    """

    def __init__(self, ar, ma, n):
        if False:
            for i in range(10):
                print('nop')
        super(ArmaFft, self).__init__(ar, ma)
        self.ar = np.asarray(ar)
        self.ma = np.asarray(ma)
        self.nobs = n
        self.arpoly = np.polynomial.Polynomial(ar)
        self.mapoly = np.polynomial.Polynomial(ma)
        self.nar = len(ar)
        self.nma = len(ma)

    def padarr(self, arr, maxlag, atend=True):
        if False:
            return 10
        'pad 1d array with zeros at end to have length maxlag\n        function that is a method, no self used\n\n        Parameters\n        ----------\n        arr : array_like, 1d\n            array that will be padded with zeros\n        maxlag : int\n            length of array after padding\n        atend : bool\n            If True (default), then the zeros are added to the end, otherwise\n            to the front of the array\n\n        Returns\n        -------\n        arrp : ndarray\n            zero-padded array\n\n        Notes\n        -----\n        This is mainly written to extend coefficient arrays for the lag-polynomials.\n        It returns a copy.\n\n        '
        if atend:
            return np.r_[arr, np.zeros(maxlag - len(arr))]
        else:
            return np.r_[np.zeros(maxlag - len(arr)), arr]

    def pad(self, maxlag):
        if False:
            print('Hello World!')
        'construct AR and MA polynomials that are zero-padded to a common length\n\n        Parameters\n        ----------\n        maxlag : int\n            new length of lag-polynomials\n\n        Returns\n        -------\n        ar : ndarray\n            extended AR polynomial coefficients\n        ma : ndarray\n            extended AR polynomial coefficients\n\n        '
        arpad = np.r_[self.ar, np.zeros(maxlag - self.nar)]
        mapad = np.r_[self.ma, np.zeros(maxlag - self.nma)]
        return (arpad, mapad)

    def fftar(self, n=None):
        if False:
            for i in range(10):
                print('nop')
        'Fourier transform of AR polynomial, zero-padded at end to n\n\n        Parameters\n        ----------\n        n : int\n            length of array after zero-padding\n\n        Returns\n        -------\n        fftar : ndarray\n            fft of zero-padded ar polynomial\n        '
        if n is None:
            n = len(self.ar)
        return fft.fft(self.padarr(self.ar, n))

    def fftma(self, n):
        if False:
            while True:
                i = 10
        'Fourier transform of MA polynomial, zero-padded at end to n\n\n        Parameters\n        ----------\n        n : int\n            length of array after zero-padding\n\n        Returns\n        -------\n        fftar : ndarray\n            fft of zero-padded ar polynomial\n        '
        if n is None:
            n = len(self.ar)
        return fft.fft(self.padarr(self.ma, n))

    def fftarma(self, n=None):
        if False:
            return 10
        'Fourier transform of ARMA polynomial, zero-padded at end to n\n\n        The Fourier transform of the ARMA process is calculated as the ratio\n        of the fft of the MA polynomial divided by the fft of the AR polynomial.\n\n        Parameters\n        ----------\n        n : int\n            length of array after zero-padding\n\n        Returns\n        -------\n        fftarma : ndarray\n            fft of zero-padded arma polynomial\n        '
        if n is None:
            n = self.nobs
        return self.fftma(n) / self.fftar(n)

    def spd(self, npos):
        if False:
            for i in range(10):
                print('nop')
        'raw spectral density, returns Fourier transform\n\n        n is number of points in positive spectrum, the actual number of points\n        is twice as large. different from other spd methods with fft\n        '
        n = npos
        w = fft.fftfreq(2 * n) * 2 * np.pi
        hw = self.fftarma(2 * n)
        return ((hw * hw.conj()).real * 0.5 / np.pi, w)

    def spdshift(self, n):
        if False:
            while True:
                i = 10
        'power spectral density using fftshift\n\n        currently returns two-sided according to fft frequencies, use first half\n        '
        mapadded = self.padarr(self.ma, n)
        arpadded = self.padarr(self.ar, n)
        hw = fft.fft(fft.fftshift(mapadded)) / fft.fft(fft.fftshift(arpadded))
        w = fft.fftfreq(n) * 2 * np.pi
        wslice = slice(n // 2 - 1, None, None)
        return ((hw * hw.conj()).real, w)

    def spddirect(self, n):
        if False:
            return 10
        'power spectral density using padding to length n done by fft\n\n        currently returns two-sided according to fft frequencies, use first half\n        '
        hw = fft.fft(self.ma, n) / fft.fft(self.ar, n)
        w = fft.fftfreq(n) * 2 * np.pi
        wslice = slice(None, n // 2, None)
        return (np.abs(hw) ** 2 * 0.5 / np.pi, w)

    def _spddirect2(self, n):
        if False:
            i = 10
            return i + 15
        'this looks bad, maybe with an fftshift\n        '
        hw = fft.fft(np.r_[self.ma[::-1], self.ma], n) / fft.fft(np.r_[self.ar[::-1], self.ar], n)
        return hw * hw.conj()

    def spdroots(self, w):
        if False:
            print('Hello World!')
        'spectral density for frequency using polynomial roots\n\n        builds two arrays (number of roots, number of frequencies)\n        '
        return self._spdroots(self.arroots, self.maroots, w)

    def _spdroots(self, arroots, maroots, w):
        if False:
            while True:
                i = 10
        'spectral density for frequency using polynomial roots\n\n        builds two arrays (number of roots, number of frequencies)\n\n        Parameters\n        ----------\n        arroots : ndarray\n            roots of ar (denominator) lag-polynomial\n        maroots : ndarray\n            roots of ma (numerator) lag-polynomial\n        w : array_like\n            frequencies for which spd is calculated\n\n        Notes\n        -----\n        this should go into a function\n        '
        w = np.atleast_2d(w).T
        cosw = np.cos(w)
        maroots = 1.0 / maroots
        arroots = 1.0 / arroots
        num = 1 + maroots ** 2 - 2 * maroots * cosw
        den = 1 + arroots ** 2 - 2 * arroots * cosw
        hw = 0.5 / np.pi * num.prod(-1) / den.prod(-1)
        return (np.squeeze(hw), w.squeeze())

    def spdpoly(self, w, nma=50):
        if False:
            print('Hello World!')
        'spectral density from MA polynomial representation for ARMA process\n\n        References\n        ----------\n        Cochrane, section 8.3.3\n        '
        mpoly = np.polynomial.Polynomial(self.arma2ma(nma))
        hw = mpoly(np.exp(1j * w))
        spd = np.real_if_close(hw * hw.conj() * 0.5 / np.pi)
        return (spd, w)

    def filter(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        filter a timeseries with the ARMA filter\n\n        padding with zero is missing, in example I needed the padding to get\n        initial conditions identical to direct filter\n\n        Initial filtered observations differ from filter2 and signal.lfilter, but\n        at end they are the same.\n\n        See Also\n        --------\n        tsa.filters.fftconvolve\n\n        '
        n = x.shape[0]
        if n == self.fftarma:
            fftarma = self.fftarma
        else:
            fftarma = self.fftma(n) / self.fftar(n)
        tmpfft = fftarma * fft.fft(x)
        return fft.ifft(tmpfft)

    def filter2(self, x, pad=0):
        if False:
            return 10
        'filter a time series using fftconvolve3 with ARMA filter\n\n        padding of x currently works only if x is 1d\n        in example it produces same observations at beginning as lfilter even\n        without padding.\n\n        TODO: this returns 1 additional observation at the end\n        '
        from statsmodels.tsa.filters import fftconvolve3
        if not pad:
            pass
        elif pad == 'auto':
            x = self.padarr(x, x.shape[0] + 2 * (self.nma + self.nar), atend=False)
        else:
            x = self.padarr(x, x.shape[0] + int(pad), atend=False)
        return fftconvolve3(x, self.ma, self.ar)

    def acf2spdfreq(self, acovf, nfreq=100, w=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        not really a method\n        just for comparison, not efficient for large n or long acf\n\n        this is also similarly use in tsa.stattools.periodogram with window\n        '
        if w is None:
            w = np.linspace(0, np.pi, nfreq)[:, None]
        nac = len(acovf)
        hw = 0.5 / np.pi * (acovf[0] + 2 * (acovf[1:] * np.cos(w * np.arange(1, nac))).sum(1))
        return hw

    def invpowerspd(self, n):
        if False:
            return 10
        'autocovariance from spectral density\n\n        scaling is correct, but n needs to be large for numerical accuracy\n        maybe padding with zero in fft would be faster\n        without slicing it returns 2-sided autocovariance with fftshift\n\n        >>> ArmaFft([1, -0.5], [1., 0.4], 40).invpowerspd(2**8)[:10]\n        array([ 2.08    ,  1.44    ,  0.72    ,  0.36    ,  0.18    ,  0.09    ,\n                0.045   ,  0.0225  ,  0.01125 ,  0.005625])\n        >>> ArmaFft([1, -0.5], [1., 0.4], 40).acovf(10)\n        array([ 2.08    ,  1.44    ,  0.72    ,  0.36    ,  0.18    ,  0.09    ,\n                0.045   ,  0.0225  ,  0.01125 ,  0.005625])\n        '
        hw = self.fftarma(n)
        return np.real_if_close(fft.ifft(hw * hw.conj()), tol=200)[:n]

    def spdmapoly(self, w, twosided=False):
        if False:
            for i in range(10):
                print('nop')
        'ma only, need division for ar, use LagPolynomial\n        '
        if w is None:
            w = np.linspace(0, np.pi, nfreq)
        return 0.5 / np.pi * self.mapoly(np.exp(w * 1j))

    def plot4(self, fig=None, nobs=100, nacf=20, nfreq=100):
        if False:
            i = 10
            return i + 15
        'Plot results'
        rvs = self.generate_sample(nsample=100, burnin=500)
        acf = self.acf(nacf)[:nacf]
        pacf = self.pacf(nacf)
        w = np.linspace(0, np.pi, nfreq)
        (spdr, wr) = self.spdroots(w)
        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        ax.plot(rvs)
        ax.set_title('Random Sample \nar=%s, ma=%s' % (self.ar, self.ma))
        ax = fig.add_subplot(2, 2, 2)
        ax.plot(acf)
        ax.set_title('Autocorrelation \nar=%s, ma=%rs' % (self.ar, self.ma))
        ax = fig.add_subplot(2, 2, 3)
        ax.plot(wr, spdr)
        ax.set_title('Power Spectrum \nar=%s, ma=%s' % (self.ar, self.ma))
        ax = fig.add_subplot(2, 2, 4)
        ax.plot(pacf)
        ax.set_title('Partial Autocorrelation \nar=%s, ma=%s' % (self.ar, self.ma))
        return fig

def spdar1(ar, w):
    if False:
        while True:
            i = 10
    if np.ndim(ar) == 0:
        rho = ar
    else:
        rho = -ar[1]
    return 0.5 / np.pi / (1 + rho * rho - 2 * rho * np.cos(w))
if __name__ == '__main__':

    def maxabs(x, y):
        if False:
            while True:
                i = 10
        return np.max(np.abs(x - y))
    nobs = 200
    ar = [1, 0.0]
    ma = [1, 0.0]
    ar2 = np.zeros(nobs)
    ar2[:2] = [1, -0.9]
    uni = np.zeros(nobs)
    uni[0] = 1.0
    arcomb = np.convolve(ar, ar2, mode='same')
    marep = signal.lfilter(ma, arcomb, uni)
    print(marep[:10])
    mafr = fft.fft(marep)
    rvs = np.random.normal(size=nobs)
    datafr = fft.fft(rvs)
    y = fft.ifft(mafr * datafr)
    print(np.corrcoef(np.c_[y[2:], y[1:-1], y[:-2]], rowvar=0))
    arrep = signal.lfilter([1], marep, uni)
    print(arrep[:20])
    arfr = fft.fft(arrep)
    yfr = fft.fft(y)
    x = fft.ifft(arfr * yfr).real
    print(x[:5])
    print(rvs[:5])
    print(np.corrcoef(np.c_[x[2:], x[1:-1], x[:-2]], rowvar=0))
    arcombp = np.zeros(nobs)
    arcombp[:len(arcomb)] = arcomb
    map_ = np.zeros(nobs)
    map_[:len(ma)] = ma
    ar0fr = fft.fft(arcombp)
    ma0fr = fft.fft(map_)
    y2 = fft.ifft(ma0fr / ar0fr * datafr)
    print(y2[:10])
    print(y[:10])
    print(maxabs(y, y2))
    ar = [1, -0.4]
    ma = [1, 0.2]
    arma1 = ArmaFft([1, -0.5, 0, 0, 0, 0, -0.7, 0.3], [1, 0.8], nobs)
    nfreq = nobs
    w = np.linspace(0, np.pi, nfreq)
    w2 = np.linspace(0, 2 * np.pi, nfreq)
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure()
    (spd1, w1) = arma1.spd(2 ** 10)
    print(spd1.shape)
    _ = plt.plot(spd1)
    plt.title('spd fft complex')
    plt.figure()
    (spd2, w2) = arma1.spdshift(2 ** 10)
    print(spd2.shape)
    _ = plt.plot(w2, spd2)
    plt.title('spd fft shift')
    plt.figure()
    (spd3, w3) = arma1.spddirect(2 ** 10)
    print(spd3.shape)
    _ = plt.plot(w3, spd3)
    plt.title('spd fft direct')
    plt.figure()
    spd3b = arma1._spddirect2(2 ** 10)
    print(spd3b.shape)
    _ = plt.plot(spd3b)
    plt.title('spd fft direct mirrored')
    plt.figure()
    (spdr, wr) = arma1.spdroots(w)
    print(spdr.shape)
    plt.plot(w, spdr)
    plt.title('spd from roots')
    plt.figure()
    spdar1_ = spdar1(arma1.ar, w)
    print(spdar1_.shape)
    _ = plt.plot(w, spdar1_)
    plt.title('spd ar1')
    plt.figure()
    (wper, spdper) = arma1.periodogram(nfreq)
    print(spdper.shape)
    _ = plt.plot(w, spdper)
    plt.title('periodogram')
    startup = 1000
    rvs = arma1.generate_sample(startup + 10000)[startup:]
    import matplotlib.mlab as mlb
    plt.figure()
    (sdm, wm) = mlb.psd(x)
    print('sdm.shape', sdm.shape)
    sdm = sdm.ravel()
    plt.plot(wm, sdm)
    plt.title('matplotlib')
    from nitime.algorithms import LD_AR_est
    (wnt, spdnt) = LD_AR_est(rvs, 10, 512)
    plt.figure()
    print('spdnt.shape', spdnt.shape)
    _ = plt.plot(spdnt.ravel())
    print(spdnt[:10])
    plt.title('nitime')
    fig = plt.figure()
    arma1.plot4(fig)