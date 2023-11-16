import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from recommenders.utils.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL

class RLRMCdataset(object):
    """RLRMC dataset implementation. Creates sparse data structures for RLRMC algorithm."""

    def __init__(self, train, validation=None, test=None, mean_center=True, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_timestamp=DEFAULT_TIMESTAMP_COL):
        if False:
            return 10
        'Initialize parameters.\n\n        Args:\n            train (pandas.DataFrame: training data with at least columns (col_user, col_item, col_rating)\n            validation (pandas.DataFrame): validation data with at least columns (col_user, col_item, col_rating). validation can be None, if so, we only process the training data\n            mean_center (bool): flag to mean center the ratings in train (and validation) data\n            col_user (str): user column name\n            col_item (str): item column name\n            col_rating (str): rating column name\n            col_timestamp (str): timestamp column name\n        '
        self.user_idx = None
        self.item_idx = None
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        self._data_processing(train, validation, test, mean_center)

    def _data_processing(self, train, validation=None, test=None, mean_center=True):
        if False:
            print('Hello World!')
        'Process the dataset to reindex userID and itemID\n\n        Args:\n            train (pandas.DataFrame): training data with at least columns (col_user, col_item, col_rating)\n            validation (pandas.DataFrame): validation data with at least columns (col_user, col_item, col_rating). validation can be None, if so, we only process the training data\n            mean_center (bool): flag to mean center the ratings in train (and validation) data\n\n        Returns:\n            list: train and validation pandas.DataFrame Dataset, which have been reindexed.\n\n        '
        df = train if validation is None else train.append(validation)
        df = df if test is None else df.append(test)
        if self.user_idx is None:
            user_idx = df[[self.col_user]].drop_duplicates().reindex()
            user_idx[self.col_user + '_idx'] = np.arange(len(user_idx))
            self.n_users = len(user_idx)
            self.user_idx = user_idx
            self.user2id = dict(zip(user_idx[self.col_user], user_idx[self.col_user + '_idx']))
            self.id2user = {self.user2id[k]: k for k in self.user2id}
        if self.item_idx is None:
            item_idx = df[[self.col_item]].drop_duplicates()
            item_idx[self.col_item + '_idx'] = np.arange(len(item_idx))
            self.n_items = len(item_idx)
            self.item_idx = item_idx
            self.item2id = dict(zip(item_idx[self.col_item], item_idx[self.col_item + '_idx']))
            self.id2item = {self.item2id[k]: k for k in self.item2id}
        df_train = self._reindex(train)
        d = len(user_idx)
        T = len(item_idx)
        rows_train = df_train['userID'].values
        cols_train = df_train['itemID'].values
        entries_omega = df_train['rating'].values
        if mean_center:
            train_mean = np.mean(entries_omega)
        else:
            train_mean = 0.0
        entries_train = entries_omega - train_mean
        self.model_param = {'num_row': d, 'num_col': T, 'train_mean': train_mean}
        self.train = csr_matrix((entries_train.T.ravel(), (rows_train, cols_train)), shape=(d, T))
        if validation is not None:
            df_validation = self._reindex(validation)
            rows_validation = df_validation['userID'].values
            cols_validation = df_validation['itemID'].values
            entries_validation = df_validation['rating'].values - train_mean
            self.validation = csr_matrix((entries_validation.T.ravel(), (rows_validation, cols_validation)), shape=(d, T))
        else:
            self.validation = None

    def _reindex(self, df):
        if False:
            for i in range(10):
                print('nop')
        'Process dataset to reindex userID and itemID\n\n        Args:\n            df (pandas.DataFrame): dataframe with at least columns (col_user, col_item, col_rating)\n\n        Returns:\n            list: train and validation pandas.DataFrame Dataset, which have been reindexed.\n\n        '
        if df is None:
            return None
        df = pd.merge(df, self.user_idx, on=self.col_user, how='left')
        df = pd.merge(df, self.item_idx, on=self.col_item, how='left')
        df_reindex = df[[self.col_user + '_idx', self.col_item + '_idx', self.col_rating]]
        df_reindex.columns = [self.col_user, self.col_item, self.col_rating]
        return df_reindex