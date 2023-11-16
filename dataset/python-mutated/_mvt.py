import math
import numpy as np
from scipy import special
from scipy.stats._qmc import primes_from_2_to

def _primes(n):
    if False:
        while True:
            i = 10
    return primes_from_2_to(math.ceil(n))

def _gaminv(a, b):
    if False:
        return 10
    return special.gammaincinv(b, a)

def _qsimvtv(m, nu, sigma, a, b, rng):
    if False:
        for i in range(10):
            print('nop')
    'Estimates the multivariate t CDF using randomized QMC\n\n    Parameters\n    ----------\n    m : int\n        The number of points\n    nu : float\n        Degrees of freedom\n    sigma : ndarray\n        A 2D positive semidefinite covariance matrix\n    a : ndarray\n        Lower integration limits\n    b : ndarray\n        Upper integration limits.\n    rng : Generator\n        Pseudorandom number generator\n\n    Returns\n    -------\n    p : float\n        The estimated CDF.\n    e : float\n        An absolute error estimate.\n\n    '
    sn = max(1, math.sqrt(nu))
    (ch, az, bz) = _chlrps(sigma, a / sn, b / sn)
    n = len(sigma)
    N = 10
    P = math.ceil(m / N)
    on = np.ones(P)
    p = 0
    e = 0
    ps = np.sqrt(_primes(5 * n * math.log(n + 4) / 4))
    q = ps[:, np.newaxis]
    c = None
    dc = None
    for S in range(N):
        vp = on.copy()
        s = np.zeros((n, P))
        for i in range(n):
            x = np.abs(2 * np.mod(q[i] * np.arange(1, P + 1) + rng.random(), 1) - 1)
            if i == 0:
                r = on
                if nu > 0:
                    r = np.sqrt(2 * _gaminv(x, nu / 2))
            else:
                y = _Phinv(c + x * dc)
                s[i:] += ch[i:, i - 1:i] * y
            si = s[i, :]
            c = on.copy()
            ai = az[i] * r - si
            d = on.copy()
            bi = bz[i] * r - si
            c[ai <= -9] = 0
            tl = abs(ai) < 9
            c[tl] = _Phi(ai[tl])
            d[bi <= -9] = 0
            tl = abs(bi) < 9
            d[tl] = _Phi(bi[tl])
            dc = d - c
            vp = vp * dc
        d = (np.mean(vp) - p) / (S + 1)
        p = p + d
        e = (S - 1) * e / (S + 1) + d ** 2
    e = math.sqrt(e)
    return (p, e)

def _Phi(z):
    if False:
        while True:
            i = 10
    return special.ndtr(z)

def _Phinv(p):
    if False:
        return 10
    return special.ndtri(p)

def _chlrps(R, a, b):
    if False:
        i = 10
        return i + 15
    '\n    Computes permuted and scaled lower Cholesky factor c for R which may be\n    singular, also permuting and scaling integration limit vectors a and b.\n    '
    ep = 1e-10
    eps = np.finfo(R.dtype).eps
    n = len(R)
    c = R.copy()
    ap = a.copy()
    bp = b.copy()
    d = np.sqrt(np.maximum(np.diag(c), 0))
    for i in range(n):
        if d[i] > 0:
            c[:, i] /= d[i]
            c[i, :] /= d[i]
            ap[i] /= d[i]
            bp[i] /= d[i]
    y = np.zeros((n, 1))
    sqtp = math.sqrt(2 * math.pi)
    for k in range(n):
        im = k
        ckk = 0
        dem = 1
        s = 0
        for i in range(k, n):
            if c[i, i] > eps:
                cii = math.sqrt(max(c[i, i], 0))
                if i > 0:
                    s = c[i, :k] @ y[:k]
                ai = (ap[i] - s) / cii
                bi = (bp[i] - s) / cii
                de = _Phi(bi) - _Phi(ai)
                if de <= dem:
                    ckk = cii
                    dem = de
                    am = ai
                    bm = bi
                    im = i
        if im > k:
            ap[[im, k]] = ap[[k, im]]
            bp[[im, k]] = bp[[k, im]]
            c[im, im] = c[k, k]
            t = c[im, :k].copy()
            c[im, :k] = c[k, :k]
            c[k, :k] = t
            t = c[im + 1:, im].copy()
            c[im + 1:, im] = c[im + 1:, k]
            c[im + 1:, k] = t
            t = c[k + 1:im, k].copy()
            c[k + 1:im, k] = c[im, k + 1:im].T
            c[im, k + 1:im] = t.T
        if ckk > ep * (k + 1):
            c[k, k] = ckk
            c[k, k + 1:] = 0
            for i in range(k + 1, n):
                c[i, k] = c[i, k] / ckk
                c[i, k + 1:i + 1] = c[i, k + 1:i + 1] - c[i, k] * c[k + 1:i + 1, k].T
            if abs(dem) > ep:
                y[k] = (np.exp(-am ** 2 / 2) - np.exp(-bm ** 2 / 2)) / (sqtp * dem)
            else:
                y[k] = (am + bm) / 2
                if am < -10:
                    y[k] = bm
                elif bm > 10:
                    y[k] = am
            c[k, :k + 1] /= ckk
            ap[k] /= ckk
            bp[k] /= ckk
        else:
            c[k:, k] = 0
            y[k] = (ap[k] + bp[k]) / 2
        pass
    return (c, ap, bp)