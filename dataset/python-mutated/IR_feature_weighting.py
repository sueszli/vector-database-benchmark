"""
Created on 10/04/18

@author: Maurizio Ferrari Dacrema
"""
import scipy.sparse as sps
import numpy as np

def okapi_BM_25(dataMatrix, K1=1.2, B=0.75):
    if False:
        i = 10
        return i + 15
    '\n    Items are assumed to be on rows\n    :param dataMatrix:\n    :param K1:\n    :param B:\n    :return:\n    '
    assert B > 0 and B < 1, 'okapi_BM_25: B must be in (0,1)'
    assert K1 > 0, 'okapi_BM_25: K1 must be > 0'
    assert np.all(np.isfinite(dataMatrix.data)), 'okapi_BM_25: Data matrix contains {} non finite values'.format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))
    dataMatrix = sps.coo_matrix(dataMatrix)
    N = float(dataMatrix.shape[0])
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))
    row_sums = np.ravel(dataMatrix.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = 1.0 - B + B * row_sums / average_length
    denominator = K1 * length_norm[dataMatrix.row] + dataMatrix.data
    denominator[denominator == 0.0] += 1e-09
    dataMatrix.data = dataMatrix.data * (K1 + 1.0) / denominator * idf[dataMatrix.col]
    return dataMatrix.tocsr()

def TF_IDF(dataMatrix):
    if False:
        while True:
            i = 10
    '\n    Items are assumed to be on rows\n    :param dataMatrix:\n    :return:\n    '
    assert np.all(np.isfinite(dataMatrix.data)), 'TF_IDF: Data matrix contains {} non finite values.'.format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))
    assert np.all(dataMatrix.data >= 0.0), 'TF_IDF: Data matrix contains {} negative values, computing the square root is not possible.'.format(np.sum(dataMatrix.data < 0.0))
    dataMatrix = sps.coo_matrix(dataMatrix)
    N = float(dataMatrix.shape[0])
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))
    dataMatrix.data = np.sqrt(dataMatrix.data) * idf[dataMatrix.col]
    return dataMatrix.tocsr()