"""

@author: Massimo Quadrana
"""
import numpy as np
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.DataIO import DataIO

class TopPop(BaseRecommender):
    """Top Popular recommender"""
    RECOMMENDER_NAME = 'TopPopRecommender'

    def __init__(self, URM_train):
        if False:
            while True:
                i = 10
        super(TopPop, self).__init__(URM_train)

    def fit(self):
        if False:
            for i in range(10):
                print('nop')
        self.item_pop = np.ediff1d(self.URM_train.tocsc().indptr)
        self.n_items = self.URM_train.shape[1]

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if False:
            for i in range(10):
                print('nop')
        if items_to_compute is not None:
            item_pop_to_copy = -np.ones(self.n_items, dtype=np.float32) * np.inf
            item_pop_to_copy[items_to_compute] = self.item_pop[items_to_compute].copy()
        else:
            item_pop_to_copy = self.item_pop.copy()
        item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis=0)
        return item_scores

    def save_model(self, folder_path, file_name=None):
        if False:
            for i in range(10):
                print('nop')
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        data_dict_to_save = {'item_pop': self.item_pop}
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)
        self._print('Saving complete')

class GlobalEffects(BaseRecommender):
    """docstring for GlobalEffects"""
    RECOMMENDER_NAME = 'GlobalEffectsRecommender'

    def __init__(self, URM_train: object) -> object:
        if False:
            return 10
        super(GlobalEffects, self).__init__(URM_train)

    def fit(self, lambda_user=10, lambda_item=25):
        if False:
            while True:
                i = 10
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.n_items = self.URM_train.shape[1]
        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        self.mu = self.URM_train.data.sum(dtype=np.float32) / self.URM_train.data.shape[0]
        col_nnz = np.diff(self.URM_train.indptr)
        URM_train_unbiased = self.URM_train.copy()
        URM_train_unbiased.data -= self.mu
        self.item_bias = URM_train_unbiased.sum(axis=0) / (col_nnz + self.lambda_item)
        self.item_bias = np.asarray(self.item_bias).ravel()
        URM_train_unbiased.data -= np.repeat(self.item_bias, col_nnz)
        URM_train_unbiased_csr = URM_train_unbiased.tocsr()
        row_nnz = np.diff(URM_train_unbiased_csr.indptr)
        self.user_bias = URM_train_unbiased_csr.sum(axis=1).ravel() / (row_nnz + self.lambda_user)
        self.URM_train = check_matrix(self.URM_train, 'csr', dtype=np.float32)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if False:
            print('Hello World!')
        if items_to_compute is not None:
            item_bias_to_copy = -np.ones(self.n_items, dtype=np.float32) * np.inf
            item_bias_to_copy[items_to_compute] = self.item_bias[items_to_compute].copy()
        else:
            item_bias_to_copy = self.item_bias.copy()
        item_scores = np.array(item_bias_to_copy, dtype=np.float).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis=0)
        return item_scores

    def save_model(self, folder_path, file_name=None):
        if False:
            while True:
                i = 10
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        data_dict_to_save = {'item_bias': self.item_bias}
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)
        self._print('Saving complete')

class Random(BaseRecommender):
    """Random recommender"""
    RECOMMENDER_NAME = 'RandomRecommender'

    def __init__(self, URM_train):
        if False:
            while True:
                i = 10
        super(Random, self).__init__(URM_train)

    def fit(self, random_seed=42):
        if False:
            while True:
                i = 10
        np.random.seed(random_seed)
        self.n_items = self.URM_train.shape[1]

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if False:
            i = 10
            return i + 15
        if items_to_compute is not None:
            item_scores = -np.ones((len(user_id_array), self.n_items), dtype=np.float32) * np.inf
            item_scores[:, items_to_compute] = np.random.rand(len(user_id_array), len(items_to_compute))
        else:
            item_scores = np.random.rand(len(user_id_array), self.n_items)
        return item_scores

    def save_model(self, folder_path, file_name=None):
        if False:
            print('Hello World!')
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        data_dict_to_save = {}
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)
        self._print('Saving complete')