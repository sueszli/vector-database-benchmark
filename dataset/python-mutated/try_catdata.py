"""
Working with categorical data
=============================

use of dummy variables, group statistics, within and between statistics
examples for efficient matrix algebra

dummy versions require that the number of unique groups or categories is not too large
group statistics with scipy.ndimage can handle large number of observations and groups
scipy.ndimage stats is missing count

new: np.bincount can also be used for calculating values per label
"""
from statsmodels.compat.python import lrange
import numpy as np
from scipy import ndimage

def labelmeanfilter(y, x):
    if False:
        return 10
    labelsunique = np.arange(np.max(y) + 1)
    labelmeans = np.array(ndimage.mean(x, labels=y, index=labelsunique))
    return labelmeans[y]

def labelmeanfilter_nd(y, x):
    if False:
        return 10
    labelsunique = np.arange(np.max(y) + 1)
    labmeansdata = []
    labmeans = []
    for xx in x.T:
        labelmeans = np.array(ndimage.mean(xx, labels=y, index=labelsunique))
        labmeansdata.append(labelmeans[y])
        labmeans.append(labelmeans)
    labelcount = np.array(ndimage.histogram(y, labelsunique[0], labelsunique[-1] + 1, 1, labels=y, index=labelsunique))
    return (labelcount, np.array(labmeans), np.array(labmeansdata).T)

def labelmeanfilter_str(ys, x):
    if False:
        i = 10
        return i + 15
    (unil, unilinv) = np.unique(ys, return_index=False, return_inverse=True)
    labelmeans = np.array(ndimage.mean(x, labels=unilinv, index=np.arange(np.max(unil) + 1)))
    arr3 = labelmeans[unilinv]
    return arr3

def groupstatsbin(factors, values):
    if False:
        return 10
    'uses np.bincount, assumes factors/labels are integers\n    '
    n = len(factors)
    (ix, rind) = np.unique(factors, return_inverse=1)
    gcount = np.bincount(rind)
    gmean = np.bincount(rind, weights=values) / (1.0 * gcount)
    meanarr = gmean[rind]
    withinvar = np.bincount(rind, weights=(values - meanarr) ** 2) / (1.0 * gcount)
    withinvararr = withinvar[rind]
    return (gcount, gmean, meanarr, withinvar, withinvararr)

def convertlabels(ys, indices=None):
    if False:
        return 10
    'convert labels based on multiple variables or string labels to unique\n    index labels 0,1,2,...,nk-1 where nk is the number of distinct labels\n    '
    if indices is None:
        ylabel = ys
    else:
        idx = np.array(indices)
        if idx.size > 1 and ys.ndim == 2:
            ylabel = np.array(['@%s@' % ii[:2].tostring() for ii in ys])[:, np.newaxis]
        else:
            ylabel = ys
    (unil, unilinv) = np.unique(ylabel, return_index=False, return_inverse=True)
    return (unilinv, np.arange(len(unil)), unil)

def groupsstats_1d(y, x, labelsunique):
    if False:
        print('Hello World!')
    'use ndimage to get fast mean and variance'
    labelmeans = np.array(ndimage.mean(x, labels=y, index=labelsunique))
    labelvars = np.array(ndimage.var(x, labels=y, index=labelsunique))
    return (labelmeans, labelvars)

def cat2dummy(y, nonseq=0):
    if False:
        while True:
            i = 10
    if nonseq or (y.ndim == 2 and y.shape[1] > 1):
        (ycat, uniques, unitransl) = convertlabels(y, lrange(y.shape[1]))
    else:
        ycat = y.copy()
        ymin = y.min()
        uniques = np.arange(ymin, y.max() + 1)
    if ycat.ndim == 1:
        ycat = ycat[:, np.newaxis]
    dummy = (ycat == uniques).astype(int)
    return dummy

def groupsstats_dummy(y, x, nonseq=0):
    if False:
        print('Hello World!')
    if x.ndim == 1:
        x = x[:, np.newaxis]
    dummy = cat2dummy(y, nonseq=nonseq)
    countgr = dummy.sum(0, dtype=float)
    meangr = np.dot(x.T, dummy) / countgr
    meandata = np.dot(dummy, meangr.T)
    xdevmeangr = x - meandata
    vargr = np.dot((xdevmeangr * xdevmeangr).T, dummy) / countgr
    return (meangr, vargr, xdevmeangr, countgr)