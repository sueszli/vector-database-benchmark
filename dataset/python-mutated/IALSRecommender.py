"""
Created on 23/03/2019

@author: Maurizio Ferrari Dacrema
"""
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_utils import check_matrix
import numpy as np

class IALSRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """

    Binary/Implicit Alternating Least Squares (IALS)
    See:
    Y. Hu, Y. Koren and C. Volinsky, Collaborative filtering for implicit feedback datasets, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf

    R. Pan et al., One-class collaborative filtering, ICDM 2008.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf

    Factorization model for binary feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \\operatornamewithlimits{argmin}\\limits_{x*,y*}\x0crac{1}{2}\\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \\lambda(\\sum_{i}{||x_i||^2} + \\sum_{j}{||y_j||^2})}
    """
    RECOMMENDER_NAME = 'IALSRecommender'
    AVAILABLE_CONFIDENCE_SCALING = ['linear', 'log']

    def fit(self, epochs=300, num_factors=20, confidence_scaling='linear', alpha=1.0, epsilon=1.0, reg=0.001, init_mean=0.0, init_std=0.1, **earlystopping_kwargs):
        if False:
            print('Hello World!')
        '\n\n        :param epochs:\n        :param num_factors:\n        :param confidence_scaling: supported scaling modes for the observed values: \'linear\' or \'log\'\n        :param alpha: Confidence weight, confidence c = 1 + alpha*r where r is the observed "rating".\n        :param reg: Regularization constant.\n        :param epsilon: epsilon used in log scaling only\n        :param init_mean: mean used to initialize the latent factors\n        :param init_std: standard deviation used to initialize the latent factors\n        :return:\n        '
        if confidence_scaling not in self.AVAILABLE_CONFIDENCE_SCALING:
            raise ValueError("Value for 'confidence_scaling' not recognized. Acceptable values are {}, provided was '{}'".format(self.AVAILABLE_CONFIDENCE_SCALING, confidence_scaling))
        self.num_factors = num_factors
        self.alpha = alpha
        self.epsilon = epsilon
        self.reg = reg
        self.USER_factors = self._init_factors(self.n_users, False)
        self.ITEM_factors = self._init_factors(self.n_items)
        self._build_confidence_matrix(confidence_scaling)
        warm_user_mask = np.ediff1d(self.URM_train.indptr) > 0
        warm_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) > 0
        self.warm_users = np.arange(0, self.n_users, dtype=np.int32)[warm_user_mask]
        self.warm_items = np.arange(0, self.n_items, dtype=np.int32)[warm_item_mask]
        self.regularization_diagonal = np.diag(self.reg * np.ones(self.num_factors))
        self._update_best_model()
        self._train_with_early_stopping(epochs, algorithm_name=self.RECOMMENDER_NAME, **earlystopping_kwargs)
        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

    def _build_confidence_matrix(self, confidence_scaling):
        if False:
            for i in range(10):
                print('nop')
        if confidence_scaling == 'linear':
            self.C = self._linear_scaling_confidence()
        else:
            self.C = self._log_scaling_confidence()
        self.C_csc = check_matrix(self.C.copy(), format='csc', dtype=np.float32)

    def _linear_scaling_confidence(self):
        if False:
            for i in range(10):
                print('nop')
        C = check_matrix(self.URM_train, format='csr', dtype=np.float32)
        C.data = 1.0 + self.alpha * C.data
        return C

    def _log_scaling_confidence(self):
        if False:
            print('Hello World!')
        C = check_matrix(self.URM_train, format='csr', dtype=np.float32)
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / self.epsilon)
        return C

    def _prepare_model_for_validation(self):
        if False:
            while True:
                i = 10
        pass

    def _update_best_model(self):
        if False:
            i = 10
            return i + 15
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()

    def _run_epoch(self, num_epoch):
        if False:
            i = 10
            return i + 15
        VV = self.ITEM_factors.T.dot(self.ITEM_factors)
        for user_id in self.warm_users:
            start_pos = self.C.indptr[user_id]
            end_pos = self.C.indptr[user_id + 1]
            user_profile = self.C.indices[start_pos:end_pos]
            user_confidence = self.C.data[start_pos:end_pos]
            self.USER_factors[user_id, :] = self._update_row(user_profile, user_confidence, self.ITEM_factors, VV)
        UU = self.USER_factors.T.dot(self.USER_factors)
        for item_id in self.warm_items:
            start_pos = self.C_csc.indptr[item_id]
            end_pos = self.C_csc.indptr[item_id + 1]
            item_profile = self.C_csc.indices[start_pos:end_pos]
            item_confidence = self.C_csc.data[start_pos:end_pos]
            self.ITEM_factors[item_id, :] = self._update_row(item_profile, item_confidence, self.USER_factors, UU)

    def _update_row(self, interaction_profile, interaction_confidence, Y, YtY):
        if False:
            print('Hello World!')
        '\n        Update latent factors for a single user or item.\n\n        Y = |n_interactions|x|n_factors|\n        YtY =   |n_factors|x|n_factors|\n        '
        Y_interactions = Y[interaction_profile, :]
        A = Y_interactions.T.dot(((interaction_confidence - 1) * Y_interactions.T).T)
        B = YtY + A + self.regularization_diagonal
        return np.dot(np.linalg.inv(B), Y_interactions.T.dot(interaction_confidence))

    def _init_factors(self, num_factors, assign_values=True):
        if False:
            return 10
        if assign_values:
            return self.num_factors ** (-0.5) * np.random.random_sample((num_factors, self.num_factors))
        else:
            return np.empty((num_factors, self.num_factors))