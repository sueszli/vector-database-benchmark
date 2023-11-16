import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
_E128 = 128
_EP128 = np.ldexp(np.longdouble(1), _E128)
_EM128 = np.ldexp(np.longdouble(1), -_E128)
_SQRT2PI = np.sqrt(2 * np.pi)
_LOG_2PI = np.log(2 * np.pi)
_MIN_LOG = -708
_SQRT3 = np.sqrt(3)
_PI_SQUARED = np.pi ** 2
_PI_FOUR = np.pi ** 4
_PI_SIX = np.pi ** 6
_STIRLING_COEFFS = [-0.029550653594771242, 0.00641025641025641, -0.0019175269175269176, 0.0008417508417508417, -0.0005952380952380953, 0.0007936507936507937, -0.002777777777777778, 0.08333333333333333]

def _log_nfactorial_div_n_pow_n(n):
    if False:
        i = 10
        return i + 15
    rn = 1.0 / n
    return np.log(n) / 2 - n + _LOG_2PI / 2 + rn * np.polyval(_STIRLING_COEFFS, rn / n)

def _clip_prob(p):
    if False:
        return 10
    'clips a probability to range 0<=p<=1.'
    return np.clip(p, 0.0, 1.0)

def _select_and_clip_prob(cdfprob, sfprob, cdf=True):
    if False:
        i = 10
        return i + 15
    'Selects either the CDF or SF, and then clips to range 0<=p<=1.'
    p = np.where(cdf, cdfprob, sfprob)
    return _clip_prob(p)

def _kolmogn_DMTW(n, d, cdf=True):
    if False:
        return 10
    'Computes the Kolmogorov CDF:  Pr(D_n <= d) using the MTW approach to\n    the Durbin matrix algorithm.\n\n    Durbin (1968); Marsaglia, Tsang, Wang (2003). [1], [3].\n    '
    if d >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf)
    nd = n * d
    if nd <= 0.5:
        return _select_and_clip_prob(0.0, 1.0, cdf)
    k = int(np.ceil(nd))
    h = k - nd
    m = 2 * k - 1
    H = np.zeros([m, m])
    intm = np.arange(1, m + 1)
    v = 1.0 - h ** intm
    w = np.empty(m)
    fac = 1.0
    for j in intm:
        w[j - 1] = fac
        fac /= j
        v[j - 1] *= fac
    tt = max(2 * h - 1.0, 0) ** m - 2 * h ** m
    v[-1] = (1.0 + tt) * fac
    for i in range(1, m):
        H[i - 1:, i] = w[:m - i + 1]
    H[:, 0] = v
    H[-1, :] = np.flip(v, axis=0)
    Hpwr = np.eye(np.shape(H)[0])
    nn = n
    expnt = 0
    Hexpnt = 0
    while nn > 0:
        if nn % 2:
            Hpwr = np.matmul(Hpwr, H)
            expnt += Hexpnt
        H = np.matmul(H, H)
        Hexpnt *= 2
        if np.abs(H[k - 1, k - 1]) > _EP128:
            H /= _EP128
            Hexpnt += _E128
        nn = nn // 2
    p = Hpwr[k - 1, k - 1]
    for i in range(1, n + 1):
        p = i * p / n
        if np.abs(p) < _EM128:
            p *= _EP128
            expnt -= _E128
    if expnt != 0:
        p = np.ldexp(p, expnt)
    return _select_and_clip_prob(p, 1.0 - p, cdf)

def _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf):
    if False:
        for i in range(10):
            print('nop')
    'Compute the endpoints of the interval for row i.'
    if i == 0:
        (j1, j2) = (-ll - ceilf - 1, ll + ceilf - 1)
    else:
        (ip1div2, ip1mod2) = divmod(i + 1, 2)
        if ip1mod2 == 0:
            if ip1div2 == n + 1:
                (j1, j2) = (n - ll - ceilf - 1, n + ll + ceilf - 1)
            else:
                (j1, j2) = (ip1div2 - 1 - ll - roundf - 1, ip1div2 + ll - 1 + ceilf - 1)
        else:
            (j1, j2) = (ip1div2 - 1 - ll - 1, ip1div2 + ll + roundf - 1)
    return (max(j1 + 2, 0), min(j2, n))

def _kolmogn_Pomeranz(n, x, cdf=True):
    if False:
        i = 10
        return i + 15
    'Computes Pr(D_n <= d) using the Pomeranz recursion algorithm.\n\n    Pomeranz (1974) [2]\n    '
    t = n * x
    ll = int(np.floor(t))
    f = 1.0 * (t - ll)
    g = min(f, 1.0 - f)
    ceilf = 1 if f > 0 else 0
    roundf = 1 if f > 0.5 else 0
    npwrs = 2 * (ll + 1)
    gpower = np.empty(npwrs)
    twogpower = np.empty(npwrs)
    onem2gpower = np.empty(npwrs)
    gpower[0] = 1.0
    twogpower[0] = 1.0
    onem2gpower[0] = 1.0
    expnt = 0
    (g_over_n, two_g_over_n, one_minus_two_g_over_n) = (g / n, 2 * g / n, (1 - 2 * g) / n)
    for m in range(1, npwrs):
        gpower[m] = gpower[m - 1] * g_over_n / m
        twogpower[m] = twogpower[m - 1] * two_g_over_n / m
        onem2gpower[m] = onem2gpower[m - 1] * one_minus_two_g_over_n / m
    V0 = np.zeros([npwrs])
    V1 = np.zeros([npwrs])
    V1[0] = 1
    (V0s, V1s) = (0, 0)
    (j1, j2) = _pomeranz_compute_j1j2(0, n, ll, ceilf, roundf)
    for i in range(1, 2 * n + 2):
        k1 = j1
        (V0, V1) = (V1, V0)
        (V0s, V1s) = (V1s, V0s)
        V1.fill(0.0)
        (j1, j2) = _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf)
        if i == 1 or i == 2 * n + 1:
            pwrs = gpower
        else:
            pwrs = twogpower if i % 2 else onem2gpower
        ln2 = j2 - k1 + 1
        if ln2 > 0:
            conv = np.convolve(V0[k1 - V0s:k1 - V0s + ln2], pwrs[:ln2])
            conv_start = j1 - k1
            conv_len = j2 - j1 + 1
            V1[:conv_len] = conv[conv_start:conv_start + conv_len]
            if 0 < np.max(V1) < _EM128:
                V1 *= _EP128
                expnt -= _E128
            V1s = V0s + j1 - k1
    ans = V1[n - V1s]
    for m in range(1, n + 1):
        if np.abs(ans) > _EP128:
            ans *= _EM128
            expnt += _E128
        ans *= m
    if expnt != 0:
        ans = np.ldexp(ans, expnt)
    ans = _select_and_clip_prob(ans, 1.0 - ans, cdf)
    return ans

def _kolmogn_PelzGood(n, x, cdf=True):
    if False:
        i = 10
        return i + 15
    'Computes the Pelz-Good approximation to Prob(Dn <= x) with 0<=x<=1.\n\n    Start with Li-Chien, Korolyuk approximation:\n        Prob(Dn <= x) ~ K0(z) + K1(z)/sqrt(n) + K2(z)/n + K3(z)/n**1.5\n    where z = x*sqrt(n).\n    Transform each K_(z) using Jacobi theta functions into a form suitable\n    for small z.\n    Pelz-Good (1976). [6]\n    '
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)
    z = np.sqrt(n) * x
    (zsquared, zthree, zfour, zsix) = (z ** 2, z ** 3, z ** 4, z ** 6)
    qlog = -_PI_SQUARED / 8 / zsquared
    if qlog < _MIN_LOG:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    q = np.exp(qlog)
    k1a = -zsquared
    k1b = _PI_SQUARED / 4
    k2a = 6 * zsix + 2 * zfour
    k2b = (2 * zfour - 5 * zsquared) * _PI_SQUARED / 4
    k2c = _PI_FOUR * (1 - 2 * zsquared) / 16
    k3d = _PI_SIX * (5 - 30 * zsquared) / 64
    k3c = _PI_FOUR * (-60 * zsquared + 212 * zfour) / 16
    k3b = _PI_SQUARED * (135 * zfour - 96 * zsix) / 4
    k3a = -30 * zsix - 90 * z ** 8
    K0to3 = np.zeros(4)
    maxk = int(np.ceil(16 * z / np.pi))
    for k in range(maxk, 0, -1):
        m = 2 * k - 1
        (msquared, mfour, msix) = (m ** 2, m ** 4, m ** 6)
        qpower = np.power(q, 8 * k)
        coeffs = np.array([1.0, k1a + k1b * msquared, k2a + k2b * msquared + k2c * mfour, k3a + k3b * msquared + k3c * mfour + k3d * msix])
        K0to3 *= qpower
        K0to3 += coeffs
    K0to3 *= q
    K0to3 *= _SQRT2PI
    K0to3 /= np.array([z, 6 * zfour, 72 * z ** 7, 6480 * z ** 10])
    q = np.exp(-_PI_SQUARED / 2 / zsquared)
    ks = np.arange(maxk, 0, -1)
    ksquared = ks ** 2
    sqrt3z = _SQRT3 * z
    kspi = np.pi * ks
    qpwers = q ** ksquared
    k2extra = np.sum(ksquared * qpwers)
    k2extra *= _PI_SQUARED * _SQRT2PI / (-36 * zthree)
    K0to3[2] += k2extra
    k3extra = np.sum((sqrt3z + kspi) * (sqrt3z - kspi) * ksquared * qpwers)
    k3extra *= _PI_SQUARED * _SQRT2PI / (216 * zsix)
    K0to3[3] += k3extra
    powers_of_n = np.power(n * 1.0, np.arange(len(K0to3)) / 2.0)
    K0to3 /= powers_of_n
    if not cdf:
        K0to3 *= -1
        K0to3[0] += 1
    Ksum = sum(K0to3)
    return Ksum

def _kolmogn(n, x, cdf=True):
    if False:
        while True:
            i = 10
    "Computes the CDF(or SF) for the two-sided Kolmogorov-Smirnov statistic.\n\n    x must be of type float, n of type integer.\n\n    Simard & L'Ecuyer (2011) [7].\n    "
    if np.isnan(n):
        return n
    if int(n) != n or n <= 0:
        return np.nan
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    t = n * x
    if t <= 1.0:
        if t <= 0.5:
            return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
        if n <= 140:
            prob = np.prod(np.arange(1, n + 1) * (1.0 / n) * (2 * t - 1))
        else:
            prob = np.exp(_log_nfactorial_div_n_pow_n(n) + n * np.log(2 * t - 1))
        return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
    if t >= n - 1:
        prob = 2 * (1.0 - x) ** n
        return _select_and_clip_prob(1 - prob, prob, cdf=cdf)
    if x >= 0.5:
        prob = 2 * scipy.special.smirnov(n, x)
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)
    nxsquared = t * x
    if n <= 140:
        if nxsquared <= 0.754693:
            prob = _kolmogn_DMTW(n, x, cdf=True)
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
        if nxsquared <= 4:
            prob = _kolmogn_Pomeranz(n, x, cdf=True)
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
        prob = 2 * scipy.special.smirnov(n, x)
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)
    if not cdf:
        if nxsquared >= 370.0:
            return 0.0
        if nxsquared >= 2.2:
            prob = 2 * scipy.special.smirnov(n, x)
            return _clip_prob(prob)
    if nxsquared >= 18.0:
        cdfprob = 1.0
    elif n <= 100000 and n * x ** 1.5 <= 1.4:
        cdfprob = _kolmogn_DMTW(n, x, cdf=True)
    else:
        cdfprob = _kolmogn_PelzGood(n, x, cdf=True)
    return _select_and_clip_prob(cdfprob, 1.0 - cdfprob, cdf=cdf)

def _kolmogn_p(n, x):
    if False:
        print('Hello World!')
    'Computes the PDF for the two-sided Kolmogorov-Smirnov statistic.\n\n    x must be of type float, n of type integer.\n    '
    if np.isnan(n):
        return n
    if int(n) != n or n <= 0:
        return np.nan
    if x >= 1.0 or x <= 0:
        return 0
    t = n * x
    if t <= 1.0:
        if t <= 0.5:
            return 0.0
        if n <= 140:
            prd = np.prod(np.arange(1, n) * (1.0 / n) * (2 * t - 1))
        else:
            prd = np.exp(_log_nfactorial_div_n_pow_n(n) + (n - 1) * np.log(2 * t - 1))
        return prd * 2 * n ** 2
    if t >= n - 1:
        return 2 * (1.0 - x) ** (n - 1) * n
    if x >= 0.5:
        return 2 * scipy.stats.ksone.pdf(x, n)
    delta = x / 2.0 ** 16
    delta = min(delta, x - 1.0 / n)
    delta = min(delta, 0.5 - x)

    def _kk(_x):
        if False:
            print('Hello World!')
        return kolmogn(n, _x)
    return _derivative(_kk, x, dx=delta, order=5)

def _kolmogni(n, p, q):
    if False:
        for i in range(10):
            print('nop')
    'Computes the PPF/ISF of kolmogn.\n\n    n of type integer, n>= 1\n    p is the CDF, q the SF, p+q=1\n    '
    if np.isnan(n):
        return n
    if int(n) != n or n <= 0:
        return np.nan
    if p <= 0:
        return 1.0 / n
    if q <= 0:
        return 1.0
    delta = np.exp((np.log(p) - scipy.special.loggamma(n + 1)) / n)
    if delta <= 1.0 / n:
        return (delta + 1.0 / n) / 2
    x = -np.expm1(np.log(q / 2.0) / n)
    if x >= 1 - 1.0 / n:
        return x
    x1 = scu._kolmogci(p) / np.sqrt(n)
    x1 = min(x1, 1.0 - 1.0 / n)

    def _f(x):
        if False:
            return 10
        return _kolmogn(n, x) - p
    return scipy.optimize.brentq(_f, 1.0 / n, x1, xtol=1e-14)

def kolmogn(n, x, cdf=True):
    if False:
        print('Hello World!')
    'Computes the CDF for the two-sided Kolmogorov-Smirnov distribution.\n\n    The two-sided Kolmogorov-Smirnov distribution has as its CDF Pr(D_n <= x),\n    for a sample of size n drawn from a distribution with CDF F(t), where\n    D_n &= sup_t |F_n(t) - F(t)|, and\n    F_n(t) is the Empirical Cumulative Distribution Function of the sample.\n\n    Parameters\n    ----------\n    n : integer, array_like\n        the number of samples\n    x : float, array_like\n        The K-S statistic, float between 0 and 1\n    cdf : bool, optional\n        whether to compute the CDF(default=true) or the SF.\n\n    Returns\n    -------\n    cdf : ndarray\n        CDF (or SF it cdf is False) at the specified locations.\n\n    The return value has shape the result of numpy broadcasting n and x.\n    '
    it = np.nditer([n, x, cdf, None], op_dtypes=[None, np.float64, np.bool_, np.float64])
    for (_n, _x, _cdf, z) in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        z[...] = _kolmogn(int(_n), _x, cdf=_cdf)
    result = it.operands[-1]
    return result

def kolmognp(n, x):
    if False:
        for i in range(10):
            print('nop')
    'Computes the PDF for the two-sided Kolmogorov-Smirnov distribution.\n\n    Parameters\n    ----------\n    n : integer, array_like\n        the number of samples\n    x : float, array_like\n        The K-S statistic, float between 0 and 1\n\n    Returns\n    -------\n    pdf : ndarray\n        The PDF at the specified locations\n\n    The return value has shape the result of numpy broadcasting n and x.\n    '
    it = np.nditer([n, x, None])
    for (_n, _x, z) in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        z[...] = _kolmogn_p(int(_n), _x)
    result = it.operands[-1]
    return result

def kolmogni(n, q, cdf=True):
    if False:
        print('Hello World!')
    'Computes the PPF(or ISF) for the two-sided Kolmogorov-Smirnov distribution.\n\n    Parameters\n    ----------\n    n : integer, array_like\n        the number of samples\n    q : float, array_like\n        Probabilities, float between 0 and 1\n    cdf : bool, optional\n        whether to compute the PPF(default=true) or the ISF.\n\n    Returns\n    -------\n    ppf : ndarray\n        PPF (or ISF if cdf is False) at the specified locations\n\n    The return value has shape the result of numpy broadcasting n and x.\n    '
    it = np.nditer([n, q, cdf, None])
    for (_n, _q, _cdf, z) in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        (_pcdf, _psf) = (_q, 1 - _q) if _cdf else (1 - _q, _q)
        z[...] = _kolmogni(int(_n), _pcdf, _psf)
    result = it.operands[-1]
    return result