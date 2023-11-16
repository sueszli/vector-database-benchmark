"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import similarityMatrixTopK, check_matrix
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from sklearn.preprocessing import normalize
import numpy as np
import time
import scipy.sparse as sps
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity

class EASE_R_Recommender(BaseItemSimilarityMatrixRecommender):
    """ EASE_R_Recommender

        https://arxiv.org/pdf/1905.03375.pdf

     @article{steck2019embarrassingly,
      title={Embarrassingly Shallow Autoencoders for Sparse Data},
      author={Steck, Harald},
      journal={arXiv preprint arXiv:1905.03375},
      year={2019}
    }

    """
    RECOMMENDER_NAME = 'EASE_R_Recommender'

    def __init__(self, URM_train, sparse_threshold_quota=None):
        if False:
            print('Hello World!')
        super(EASE_R_Recommender, self).__init__(URM_train)
        self.sparse_threshold_quota = sparse_threshold_quota

    def fit(self, topK=None, l2_norm=1000.0, normalize_matrix=False, verbose=True):
        if False:
            for i in range(10):
                print('nop')
        self.verbose = verbose
        start_time = time.time()
        self._print('Fitting model... ')
        if normalize_matrix:
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)
            self.URM_train = sps.csr_matrix(self.URM_train)
        similarity = Compute_Similarity(self.URM_train, shrink=0, topK=self.URM_train.shape[1], normalize=False, similarity='cosine')
        grahm_matrix = similarity.compute_similarity().toarray()
        diag_indices = np.diag_indices(grahm_matrix.shape[0])
        item_popularity = np.ediff1d(self.URM_train.tocsc().indptr)
        grahm_matrix[diag_indices] = item_popularity + l2_norm
        P = np.linalg.inv(grahm_matrix)
        B = P / -np.diag(P)
        B[diag_indices] = 0.0
        (new_time_value, new_time_unit) = seconds_to_biggest_unit(time.time() - start_time)
        self._print('Fitting model... done in {:.2f} {}'.format(new_time_value, new_time_unit))
        if topK is not None:
            B = similarityMatrixTopK(B, k=topK, verbose=False)
        if self._is_content_sparse_check(B):
            self._print('Detected model matrix to be sparse, changing format.')
            self.W_sparse = check_matrix(B, format='csr', dtype=np.float32)
        else:
            self.W_sparse = check_matrix(B, format='npy', dtype=np.float32)
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense

    def _is_content_sparse_check(self, matrix):
        if False:
            i = 10
            return i + 15
        if self.sparse_threshold_quota is None:
            return False
        if sps.issparse(matrix):
            nonzero = matrix.nnz
        else:
            nonzero = np.count_nonzero(matrix)
        return nonzero / matrix.shape[0] ** 2 <= self.sparse_threshold_quota

    def _compute_score_W_dense(self, user_id_array, items_to_compute=None):
        if False:
            while True:
                i = 10
        '\n        URM_train and W_sparse must have the same format, CSR\n        :param user_id_array:\n        :param items_to_compute:\n        :return:\n        '
        self._check_format()
        user_profile_array = self.URM_train[user_id_array]
        if items_to_compute is not None:
            item_scores = -np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)
        return item_scores

    def load_model(self, folder_path, file_name=None):
        if False:
            print('Hello World!')
        super(EASE_R_Recommender, self).load_model(folder_path, file_name=file_name)
        if not sps.issparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense