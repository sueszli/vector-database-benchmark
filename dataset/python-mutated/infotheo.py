"""
Information Theoretic and Entropy Measures

References
----------
Golan, As. 2008. "Information and Entropy Econometrics -- A Review and
    Synthesis." Foundations And Trends in Econometrics 2(1-2), 1-145.

Golan, A., Judge, G., and Miller, D.  1996.  Maximum Entropy Econometrics.
    Wiley & Sons, Chichester.
"""
from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp

def logsumexp(a, axis=None):
    if False:
        print('Hello World!')
    '\n    Compute the log of the sum of exponentials log(e^{a_1}+...e^{a_n}) of a\n\n    Avoids numerical overflow.\n\n    Parameters\n    ----------\n    a : array_like\n        The vector to exponentiate and sum\n    axis : int, optional\n        The axis along which to apply the operation.  Defaults is None.\n\n    Returns\n    -------\n    sum(log(exp(a)))\n\n    Notes\n    -----\n    This function was taken from the mailing list\n    http://mail.scipy.org/pipermail/scipy-user/2009-October/022931.html\n\n    This should be superceded by the ufunc when it is finished.\n    '
    if axis is None:
        return sp_logsumexp(a)
    a = np.asarray(a)
    shp = list(a.shape)
    shp[axis] = 1
    a_max = a.max(axis=axis)
    s = np.log(np.exp(a - a_max.reshape(shp)).sum(axis=axis))
    lse = a_max + s
    return lse

def _isproperdist(X):
    if False:
        return 10
    '\n    Checks to see if `X` is a proper probability distribution\n    '
    X = np.asarray(X)
    if not np.allclose(np.sum(X), 1) or not np.all(X >= 0) or (not np.all(X <= 1)):
        return False
    else:
        return True

def discretize(X, method='ef', nbins=None):
    if False:
        i = 10
        return i + 15
    '\n    Discretize `X`\n\n    Parameters\n    ----------\n    bins : int, optional\n        Number of bins.  Default is floor(sqrt(N))\n    method : str\n        "ef" is equal-frequency binning\n        "ew" is equal-width binning\n\n    Examples\n    --------\n    '
    nobs = len(X)
    if nbins is None:
        nbins = np.floor(np.sqrt(nobs))
    if method == 'ef':
        discrete = np.ceil(nbins * stats.rankdata(X) / nobs)
    if method == 'ew':
        width = np.max(X) - np.min(X)
        width = np.floor(width / nbins)
        (svec, ivec) = stats.fastsort(X)
        discrete = np.zeros(nobs)
        binnum = 1
        base = svec[0]
        discrete[ivec[0]] = binnum
        for i in range(1, nobs):
            if svec[i] < base + width:
                discrete[ivec[i]] = binnum
            else:
                base = svec[i]
                binnum += 1
                discrete[ivec[i]] = binnum
    return discrete

def logbasechange(a, b):
    if False:
        i = 10
        return i + 15
    '\n    There is a one-to-one transformation of the entropy value from\n    a log base b to a log base a :\n\n    H_{b}(X)=log_{b}(a)[H_{a}(X)]\n\n    Returns\n    -------\n    log_{b}(a)\n    '
    return np.log(b) / np.log(a)

def natstobits(X):
    if False:
        while True:
            i = 10
    '\n    Converts from nats to bits\n    '
    return logbasechange(np.e, 2) * X

def bitstonats(X):
    if False:
        while True:
            i = 10
    '\n    Converts from bits to nats\n    '
    return logbasechange(2, np.e) * X

def shannonentropy(px, logbase=2):
    if False:
        while True:
            i = 10
    "\n    This is Shannon's entropy\n\n    Parameters\n    ----------\n    logbase, int or np.e\n        The base of the log\n    px : 1d or 2d array_like\n        Can be a discrete probability distribution, a 2d joint distribution,\n        or a sequence of probabilities.\n\n    Returns\n    -----\n    For log base 2 (bits) given a discrete distribution\n        H(p) = sum(px * log2(1/px) = -sum(pk*log2(px)) = E[log2(1/p(X))]\n\n    For log base 2 (bits) given a joint distribution\n        H(px,py) = -sum_{k,j}*w_{kj}log2(w_{kj})\n\n    Notes\n    -----\n    shannonentropy(0) is defined as 0\n    "
    px = np.asarray(px)
    if not np.all(px <= 1) or not np.all(px >= 0):
        raise ValueError('px does not define proper distribution')
    entropy = -np.sum(np.nan_to_num(px * np.log2(px)))
    if logbase != 2:
        return logbasechange(2, logbase) * entropy
    else:
        return entropy

def shannoninfo(px, logbase=2):
    if False:
        i = 10
        return i + 15
    "\n    Shannon's information\n\n    Parameters\n    ----------\n    px : float or array_like\n        `px` is a discrete probability distribution\n\n    Returns\n    -------\n    For logbase = 2\n    np.log2(px)\n    "
    px = np.asarray(px)
    if not np.all(px <= 1) or not np.all(px >= 0):
        raise ValueError('px does not define proper distribution')
    if logbase != 2:
        return -logbasechange(2, logbase) * np.log2(px)
    else:
        return -np.log2(px)

def condentropy(px, py, pxpy=None, logbase=2):
    if False:
        while True:
            i = 10
    '\n    Return the conditional entropy of X given Y.\n\n    Parameters\n    ----------\n    px : array_like\n    py : array_like\n    pxpy : array_like, optional\n        If pxpy is None, the distributions are assumed to be independent\n        and conendtropy(px,py) = shannonentropy(px)\n    logbase : int or np.e\n\n    Returns\n    -------\n    sum_{kj}log(q_{j}/w_{kj}\n\n    where q_{j} = Y[j]\n    and w_kj = X[k,j]\n    '
    if not _isproperdist(px) or not _isproperdist(py):
        raise ValueError('px or py is not a proper probability distribution')
    if pxpy is not None and (not _isproperdist(pxpy)):
        raise ValueError('pxpy is not a proper joint distribtion')
    if pxpy is None:
        pxpy = np.outer(py, px)
    condent = np.sum(pxpy * np.nan_to_num(np.log2(py / pxpy)))
    if logbase == 2:
        return condent
    else:
        return logbasechange(2, logbase) * condent

def mutualinfo(px, py, pxpy, logbase=2):
    if False:
        print('Hello World!')
    '\n    Returns the mutual information between X and Y.\n\n    Parameters\n    ----------\n    px : array_like\n        Discrete probability distribution of random variable X\n    py : array_like\n        Discrete probability distribution of random variable Y\n    pxpy : 2d array_like\n        The joint probability distribution of random variables X and Y.\n        Note that if X and Y are independent then the mutual information\n        is zero.\n    logbase : int or np.e, optional\n        Default is 2 (bits)\n\n    Returns\n    -------\n    shannonentropy(px) - condentropy(px,py,pxpy)\n    '
    if not _isproperdist(px) or not _isproperdist(py):
        raise ValueError('px or py is not a proper probability distribution')
    if pxpy is not None and (not _isproperdist(pxpy)):
        raise ValueError('pxpy is not a proper joint distribtion')
    if pxpy is None:
        pxpy = np.outer(py, px)
    return shannonentropy(px, logbase=logbase) - condentropy(px, py, pxpy, logbase=logbase)

def corrent(px, py, pxpy, logbase=2):
    if False:
        print('Hello World!')
    '\n    An information theoretic correlation measure.\n\n    Reflects linear and nonlinear correlation between two random variables\n    X and Y, characterized by the discrete probability distributions px and py\n    respectively.\n\n    Parameters\n    ----------\n    px : array_like\n        Discrete probability distribution of random variable X\n    py : array_like\n        Discrete probability distribution of random variable Y\n    pxpy : 2d array_like, optional\n        Joint probability distribution of X and Y.  If pxpy is None, X and Y\n        are assumed to be independent.\n    logbase : int or np.e, optional\n        Default is 2 (bits)\n\n    Returns\n    -------\n    mutualinfo(px,py,pxpy,logbase=logbase)/shannonentropy(py,logbase=logbase)\n\n    Notes\n    -----\n    This is also equivalent to\n\n    corrent(px,py,pxpy) = 1 - condent(px,py,pxpy)/shannonentropy(py)\n    '
    if not _isproperdist(px) or not _isproperdist(py):
        raise ValueError('px or py is not a proper probability distribution')
    if pxpy is not None and (not _isproperdist(pxpy)):
        raise ValueError('pxpy is not a proper joint distribtion')
    if pxpy is None:
        pxpy = np.outer(py, px)
    return mutualinfo(px, py, pxpy, logbase=logbase) / shannonentropy(py, logbase=logbase)

def covent(px, py, pxpy, logbase=2):
    if False:
        return 10
    '\n    An information theoretic covariance measure.\n\n    Reflects linear and nonlinear correlation between two random variables\n    X and Y, characterized by the discrete probability distributions px and py\n    respectively.\n\n    Parameters\n    ----------\n    px : array_like\n        Discrete probability distribution of random variable X\n    py : array_like\n        Discrete probability distribution of random variable Y\n    pxpy : 2d array_like, optional\n        Joint probability distribution of X and Y.  If pxpy is None, X and Y\n        are assumed to be independent.\n    logbase : int or np.e, optional\n        Default is 2 (bits)\n\n    Returns\n    -------\n    condent(px,py,pxpy,logbase=logbase) + condent(py,px,pxpy,\n            logbase=logbase)\n\n    Notes\n    -----\n    This is also equivalent to\n\n    covent(px,py,pxpy) = condent(px,py,pxpy) + condent(py,px,pxpy)\n    '
    if not _isproperdist(px) or not _isproperdist(py):
        raise ValueError('px or py is not a proper probability distribution')
    if pxpy is not None and (not _isproperdist(pxpy)):
        raise ValueError('pxpy is not a proper joint distribtion')
    if pxpy is None:
        pxpy = np.outer(py, px)
    return condent(px, py, pxpy, logbase=logbase) + condent(py, px, pxpy, logbase=logbase)

def renyientropy(px, alpha=1, logbase=2, measure='R'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Renyi\'s generalized entropy\n\n    Parameters\n    ----------\n    px : array_like\n        Discrete probability distribution of random variable X.  Note that\n        px is assumed to be a proper probability distribution.\n    logbase : int or np.e, optional\n        Default is 2 (bits)\n    alpha : float or inf\n        The order of the entropy.  The default is 1, which in the limit\n        is just Shannon\'s entropy.  2 is Renyi (Collision) entropy.  If\n        the string "inf" or numpy.inf is specified the min-entropy is returned.\n    measure : str, optional\n        The type of entropy measure desired.  \'R\' returns Renyi entropy\n        measure.  \'T\' returns the Tsallis entropy measure.\n\n    Returns\n    -------\n    1/(1-alpha)*log(sum(px**alpha))\n\n    In the limit as alpha -> 1, Shannon\'s entropy is returned.\n\n    In the limit as alpha -> inf, min-entropy is returned.\n    '
    if not _isproperdist(px):
        raise ValueError('px is not a proper probability distribution')
    alpha = float(alpha)
    if alpha == 1:
        genent = shannonentropy(px)
        if logbase != 2:
            return logbasechange(2, logbase) * genent
        return genent
    elif 'inf' in str(alpha).lower() or alpha == np.inf:
        return -np.log(np.max(px))
    px = px ** alpha
    genent = np.log(px.sum())
    if logbase == 2:
        return 1 / (1 - alpha) * genent
    else:
        return 1 / (1 - alpha) * logbasechange(2, logbase) * genent

def gencrossentropy(px, py, pxpy, alpha=1, logbase=2, measure='T'):
    if False:
        i = 10
        return i + 15
    "\n    Generalized cross-entropy measures.\n\n    Parameters\n    ----------\n    px : array_like\n        Discrete probability distribution of random variable X\n    py : array_like\n        Discrete probability distribution of random variable Y\n    pxpy : 2d array_like, optional\n        Joint probability distribution of X and Y.  If pxpy is None, X and Y\n        are assumed to be independent.\n    logbase : int or np.e, optional\n        Default is 2 (bits)\n    measure : str, optional\n        The measure is the type of generalized cross-entropy desired. 'T' is\n        the cross-entropy version of the Tsallis measure.  'CR' is Cressie-Read\n        measure.\n    "
if __name__ == '__main__':
    print('From Golan (2008) "Information and Entropy Econometrics -- A Review and Synthesis')
    print('Table 3.1')
    X = [0.2, 0.2, 0.2, 0.2, 0.2]
    Y = [0.322, 0.072, 0.511, 0.091, 0.004]
    for i in X:
        print(shannoninfo(i))
    for i in Y:
        print(shannoninfo(i))
    print(shannonentropy(X))
    print(shannonentropy(Y))
    p = [1e-05, 0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    plt.subplot(111)
    plt.ylabel('Information')
    plt.xlabel('Probability')
    x = np.linspace(0, 1, 100001)
    plt.plot(x, shannoninfo(x))
    plt.subplot(111)
    plt.ylabel('Entropy')
    plt.xlabel('Probability')
    x = np.linspace(0, 1, 101)
    plt.plot(x, lmap(shannonentropy, lzip(x, 1 - x)))
    w = np.array([[0, 0, 1.0 / 3], [1 / 9.0, 1 / 9.0, 1 / 9.0], [1 / 18.0, 1 / 9.0, 1 / 6.0]])
    px = w.sum(0)
    py = w.sum(1)
    H_X = shannonentropy(px)
    H_Y = shannonentropy(py)
    H_XY = shannonentropy(w)
    H_XgivenY = condentropy(px, py, w)
    H_YgivenX = condentropy(py, px, w)
    D_YX = logbasechange(2, np.e) * stats.entropy(px, py)
    D_XY = logbasechange(2, np.e) * stats.entropy(py, px)
    I_XY = mutualinfo(px, py, w)
    print('Table 3.3')
    print(H_X, H_Y, H_XY, H_XgivenY, H_YgivenX, D_YX, D_XY, I_XY)
    print('discretize functions')
    X = np.array([21.2, 44.5, 31.0, 19.5, 40.6, 38.7, 11.1, 15.8, 31.9, 25.8, 20.2, 14.2, 24.0, 21.0, 11.3, 18.0, 16.3, 22.2, 7.8, 27.8, 16.3, 35.1, 14.9, 17.1, 28.2, 16.4, 16.5, 46.0, 9.5, 18.8, 32.1, 26.1, 16.1, 7.3, 21.4, 20.0, 29.3, 14.9, 8.3, 22.5, 12.8, 26.9, 25.5, 22.9, 11.2, 20.7, 26.2, 9.3, 10.8, 15.6])
    discX = discretize(X)
    print
    print('Example in section 3.6 of Golan, using table 3.3')
    print("Bounding errors using Fano's inequality")
    print('H(P_{e}) + P_{e}log(K-1) >= H(X|Y)')
    print('or, a weaker inequality')
    print('P_{e} >= [H(X|Y) - 1]/log(K)')
    print('P(x) = %s' % px)
    print('X = 3 has the highest probability, so this is the estimate Xhat')
    pe = 1 - px[2]
    print('The probability of error Pe is 1 - p(X=3) = %0.4g' % pe)
    H_pe = shannonentropy([pe, 1 - pe])
    print('H(Pe) = %0.4g and K=3' % H_pe)
    print('H(Pe) + Pe*log(K-1) = %0.4g >= H(X|Y) = %0.4g' % (H_pe + pe * np.log2(2), H_XgivenY))
    print('or using the weaker inequality')
    print('Pe = %0.4g >= [H(X) - 1]/log(K) = %0.4g' % (pe, (H_X - 1) / np.log2(3)))
    print('Consider now, table 3.5, where there is additional information')
    print('The conditional probabilities of P(X|Y=y) are ')
    w2 = np.array([[0.0, 0.0, 1.0], [1 / 3.0, 1 / 3.0, 1 / 3.0], [1 / 6.0, 1 / 3.0, 1 / 2.0]])
    print(w2)
    print('The probability of error given this information is')
    print('Pe = [H(X|Y) -1]/log(K) = %0.4g' % ((np.mean([0, shannonentropy(w2[1]), shannonentropy(w2[2])]) - 1) / np.log2(3)))
    print('such that more information lowers the error')
    markovchain = np.array([[0.553, 0.284, 0.163], [0.465, 0.312, 0.223], [0.42, 0.322, 0.258]])