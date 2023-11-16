from Utils.Recommender_utils import check_matrix
import numpy as np

def linear_scaling_confidence(URM_train, alpha):
    if False:
        i = 10
        return i + 15
    C = check_matrix(URM_train, format='csr', dtype=np.float32)
    C.data = 1.0 + alpha * C.data
    return C