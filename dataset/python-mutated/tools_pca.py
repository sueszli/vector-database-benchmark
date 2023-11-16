"""Principal Component Analysis


Created on Tue Sep 29 20:11:23 2009
Author: josef-pktd

TODO : add class for better reuse of results
"""
import numpy as np

def pca(data, keepdim=0, normalize=0, demean=True):
    if False:
        i = 10
        return i + 15
    'principal components with eigenvector decomposition\n    similar to princomp in matlab\n\n    Parameters\n    ----------\n    data : ndarray, 2d\n        data with observations by rows and variables in columns\n    keepdim : int\n        number of eigenvectors to keep\n        if keepdim is zero, then all eigenvectors are included\n    normalize : bool\n        if true, then eigenvectors are normalized by sqrt of eigenvalues\n    demean : bool\n        if true, then the column mean is subtracted from the data\n\n    Returns\n    -------\n    xreduced : ndarray, 2d, (nobs, nvars)\n        projection of the data x on the kept eigenvectors\n    factors : ndarray, 2d, (nobs, nfactors)\n        factor matrix, given by np.dot(x, evecs)\n    evals : ndarray, 2d, (nobs, nfactors)\n        eigenvalues\n    evecs : ndarray, 2d, (nobs, nfactors)\n        eigenvectors, normalized if normalize is true\n\n    Notes\n    -----\n\n    See Also\n    --------\n    pcasvd : principal component analysis using svd\n\n    '
    x = np.array(data)
    if demean:
        m = x.mean(0)
    else:
        m = np.zeros(x.shape[1])
    x -= m
    xcov = np.cov(x, rowvar=0)
    (evals, evecs) = np.linalg.eig(xcov)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:, indices]
    evals = evals[indices]
    if keepdim > 0 and keepdim < x.shape[1]:
        evecs = evecs[:, :keepdim]
        evals = evals[:keepdim]
    if normalize:
        evecs = evecs / np.sqrt(evals)
    factors = np.dot(x, evecs)
    xreduced = np.dot(factors, evecs.T) + m
    return (xreduced, factors, evals, evecs)

def pcasvd(data, keepdim=0, demean=True):
    if False:
        i = 10
        return i + 15
    'principal components with svd\n\n    Parameters\n    ----------\n    data : ndarray, 2d\n        data with observations by rows and variables in columns\n    keepdim : int\n        number of eigenvectors to keep\n        if keepdim is zero, then all eigenvectors are included\n    demean : bool\n        if true, then the column mean is subtracted from the data\n\n    Returns\n    -------\n    xreduced : ndarray, 2d, (nobs, nvars)\n        projection of the data x on the kept eigenvectors\n    factors : ndarray, 2d, (nobs, nfactors)\n        factor matrix, given by np.dot(x, evecs)\n    evals : ndarray, 2d, (nobs, nfactors)\n        eigenvalues\n    evecs : ndarray, 2d, (nobs, nfactors)\n        eigenvectors, normalized if normalize is true\n\n    See Also\n    --------\n    pca : principal component analysis using eigenvector decomposition\n\n    Notes\n    -----\n    This does not have yet the normalize option of pca.\n\n    '
    (nobs, nvars) = data.shape
    x = np.array(data)
    if demean:
        m = x.mean(0)
    else:
        m = 0
    x -= m
    (U, s, v) = np.linalg.svd(x.T, full_matrices=1)
    factors = np.dot(U.T, x.T).T
    if keepdim:
        xreduced = np.dot(factors[:, :keepdim], U[:, :keepdim].T) + m
    else:
        xreduced = data
        keepdim = nvars
        ('print reassigning keepdim to max', keepdim)
    evals = s ** 2 / (x.shape[0] - 1)
    return (xreduced, factors[:, :keepdim], evals[:keepdim], U[:, :keepdim])
__all__ = ['pca', 'pcasvd']