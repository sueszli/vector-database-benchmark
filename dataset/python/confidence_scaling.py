# from ..Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
# from ..Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping

from Utils.Recommender_utils import check_matrix
import numpy as np


def linear_scaling_confidence(URM_train, alpha):
    C = check_matrix(URM_train, format="csr", dtype=np.float32)
    C.data = 1.0 + alpha * C.data

    return C
