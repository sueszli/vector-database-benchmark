from scipy import sparse as sps
import numpy as np

def combine(ICM: sps.csr_matrix, URM: sps.csr_matrix):
    if False:
        print('Hello World!')
    return sps.hstack((URM.T, ICM), format='csr')

def binarize(x):
    if False:
        for i in range(10):
            print('nop')
    if x != 0:
        return 1
    return x

def binarize_ICM(ICM: sps.csr_matrix):
    if False:
        return 10
    vbinarize = np.vectorize(binarize)
    ICM.data = vbinarize(ICM.data)