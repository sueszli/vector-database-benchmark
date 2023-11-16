import numpy as np
import pandas as pd
import logging
from scipy import sparse
from recommenders.utils.python_utils import cosine_similarity, inclusion_index, jaccard, lexicographers_mutual_information, lift, mutual_information, exponential_decay, get_top_k_scored_items, rescale
from recommenders.utils import constants
SIM_COOCCUR = 'cooccurrence'
SIM_COSINE = 'cosine'
SIM_INCLUSION_INDEX = 'inclusion index'
SIM_JACCARD = 'jaccard'
SIM_LEXICOGRAPHERS_MUTUAL_INFORMATION = 'lexicographers mutual information'
SIM_LIFT = 'lift'
SIM_MUTUAL_INFORMATION = 'mutual information'
logger = logging.getLogger()

class SARSingleNode:
    """Simple Algorithm for Recommendations (SAR) implementation

    SAR is a fast scalable adaptive algorithm for personalized recommendations based on user transaction history
    and items description. The core idea behind SAR is to recommend items like those that a user already has
    demonstrated an affinity to. It does this by 1) estimating the affinity of users for items, 2) estimating
    similarity across items, and then 3) combining the estimates to generate a set of recommendations for a given user.
    """

    def __init__(self, col_user=constants.DEFAULT_USER_COL, col_item=constants.DEFAULT_ITEM_COL, col_rating=constants.DEFAULT_RATING_COL, col_timestamp=constants.DEFAULT_TIMESTAMP_COL, col_prediction=constants.DEFAULT_PREDICTION_COL, similarity_type=SIM_JACCARD, time_decay_coefficient=30, time_now=None, timedecay_formula=False, threshold=1, normalize=False):
        if False:
            return 10
        "Initialize model parameters\n\n        Args:\n            col_user (str): user column name\n            col_item (str): item column name\n            col_rating (str): rating column name\n            col_timestamp (str): timestamp column name\n            col_prediction (str): prediction column name\n            similarity_type (str): ['cooccurrence', 'cosine', 'inclusion index', 'jaccard',\n              'lexicographers mutual information', 'lift', 'mutual information'] option for\n              computing item-item similarity\n            time_decay_coefficient (float): number of days till ratings are decayed by 1/2\n            time_now (int | None): current time for time decay calculation\n            timedecay_formula (bool): flag to apply time decay\n            threshold (int): item-item co-occurrences below this threshold will be removed\n            normalize (bool): option for normalizing predictions to scale of original ratings\n        "
        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction
        available_similarity_types = [SIM_COOCCUR, SIM_COSINE, SIM_INCLUSION_INDEX, SIM_JACCARD, SIM_LIFT, SIM_MUTUAL_INFORMATION, SIM_LEXICOGRAPHERS_MUTUAL_INFORMATION]
        if similarity_type not in available_similarity_types:
            raise ValueError('Similarity type must be one of ["' + '" | "'.join(available_similarity_types) + '"]')
        self.similarity_type = similarity_type
        self.time_decay_half_life = time_decay_coefficient * 24 * 60 * 60
        self.time_decay_flag = timedecay_formula
        self.time_now = time_now
        self.threshold = threshold
        self.user_affinity = None
        self.item_similarity = None
        self.item_frequencies = None
        self.user_frequencies = None
        if self.threshold <= 0:
            raise ValueError('Threshold cannot be < 1')
        self.normalize = normalize
        self.col_unity_rating = '_unity_rating'
        self.unity_user_affinity = None
        self.col_item_id = '_indexed_items'
        self.col_user_id = '_indexed_users'
        self.n_users = None
        self.n_items = None
        self.rating_min = None
        self.rating_max = None
        self.user2index = None
        self.item2index = None
        self.index2item = None
        self.index2user = None

    def compute_affinity_matrix(self, df, rating_col):
        if False:
            i = 10
            return i + 15
        "Affinity matrix.\n\n        The user-affinity matrix can be constructed by treating the users and items as\n        indices in a sparse matrix, and the events as the data. Here, we're treating\n        the ratings as the event weights.  We convert between different sparse-matrix\n        formats to de-duplicate user-item pairs, otherwise they will get added up.\n\n        Args:\n            df (pandas.DataFrame): Indexed df of users and items\n            rating_col (str): Name of column to use for ratings\n\n        Returns:\n            sparse.csr: Affinity matrix in Compressed Sparse Row (CSR) format.\n        "
        return sparse.coo_matrix((df[rating_col], (df[self.col_user_id], df[self.col_item_id])), shape=(self.n_users, self.n_items)).tocsr()

    def compute_time_decay(self, df, decay_column):
        if False:
            return 10
        'Compute time decay on provided column.\n\n        Args:\n            df (pandas.DataFrame): DataFrame of users and items\n            decay_column (str): column to decay\n\n        Returns:\n            pandas.DataFrame: with column decayed\n        '
        if self.time_now is None:
            self.time_now = df[self.col_timestamp].max()
        df[decay_column] *= exponential_decay(value=df[self.col_timestamp], max_val=self.time_now, half_life=self.time_decay_half_life)
        return df.groupby([self.col_user, self.col_item]).sum().reset_index()

    def compute_cooccurrence_matrix(self, df):
        if False:
            i = 10
            return i + 15
        "Co-occurrence matrix.\n\n        The co-occurrence matrix is defined as :math:`C = U^T * U`\n\n        where U is the user_affinity matrix with 1's as values (instead of ratings).\n\n        Args:\n            df (pandas.DataFrame): DataFrame of users and items\n\n        Returns:\n            numpy.ndarray: Co-occurrence matrix\n        "
        user_item_hits = sparse.coo_matrix((np.repeat(1, df.shape[0]), (df[self.col_user_id], df[self.col_item_id])), shape=(self.n_users, self.n_items)).tocsr()
        item_cooccurrence = user_item_hits.transpose().dot(user_item_hits)
        item_cooccurrence = item_cooccurrence.multiply(item_cooccurrence >= self.threshold)
        return item_cooccurrence.astype(df[self.col_rating].dtype)

    def set_index(self, df):
        if False:
            return 10
        'Generate continuous indices for users and items to reduce memory usage.\n\n        Args:\n            df (pandas.DataFrame): dataframe with user and item ids\n        '
        self.index2item = dict(enumerate(df[self.col_item].unique()))
        self.index2user = dict(enumerate(df[self.col_user].unique()))
        self.item2index = {v: k for (k, v) in self.index2item.items()}
        self.user2index = {v: k for (k, v) in self.index2user.items()}
        self.n_users = len(self.user2index)
        self.n_items = len(self.index2item)

    def fit(self, df):
        if False:
            i = 10
            return i + 15
        'Main fit method for SAR.\n\n        .. note::\n\n        Please make sure that `df` has no duplicates.\n\n        Args:\n            df (pandas.DataFrame): User item rating dataframe (without duplicates).\n        '
        select_columns = [self.col_user, self.col_item, self.col_rating]
        if self.time_decay_flag:
            select_columns += [self.col_timestamp]
        if df[select_columns].duplicated().any():
            raise ValueError('There should not be duplicates in the dataframe')
        if self.index2item is None:
            self.set_index(df)
        logger.info('Collecting user affinity matrix')
        if not np.issubdtype(df[self.col_rating].dtype, np.number):
            raise TypeError('Rating column data type must be numeric')
        temp_df = df[select_columns].copy()
        if self.time_decay_flag:
            logger.info('Calculating time-decayed affinities')
            temp_df = self.compute_time_decay(df=temp_df, decay_column=self.col_rating)
        logger.info('Creating index columns')
        temp_df.loc[:, self.col_item_id] = temp_df[self.col_item].apply(lambda item: self.item2index.get(item, np.NaN))
        temp_df.loc[:, self.col_user_id] = temp_df[self.col_user].apply(lambda user: self.user2index.get(user, np.NaN))
        if self.normalize:
            self.rating_min = temp_df[self.col_rating].min()
            self.rating_max = temp_df[self.col_rating].max()
            logger.info('Calculating normalization factors')
            temp_df[self.col_unity_rating] = 1.0
            if self.time_decay_flag:
                temp_df = self.compute_time_decay(df=temp_df, decay_column=self.col_unity_rating)
            self.unity_user_affinity = self.compute_affinity_matrix(df=temp_df, rating_col=self.col_unity_rating)
        logger.info('Building user affinity sparse matrix')
        self.user_affinity = self.compute_affinity_matrix(df=temp_df, rating_col=self.col_rating)
        logger.info('Calculating item co-occurrence')
        item_cooccurrence = self.compute_cooccurrence_matrix(df=temp_df)
        del temp_df
        self.item_frequencies = item_cooccurrence.diagonal()
        logger.info('Calculating item similarity')
        if self.similarity_type == SIM_COOCCUR:
            logger.info('Using co-occurrence based similarity')
            self.item_similarity = item_cooccurrence
        elif self.similarity_type == SIM_COSINE:
            logger.info('Using cosine similarity')
            self.item_similarity = cosine_similarity(item_cooccurrence)
        elif self.similarity_type == SIM_INCLUSION_INDEX:
            logger.info('Using inclusion index')
            self.item_similarity = inclusion_index(item_cooccurrence)
        elif self.similarity_type == SIM_JACCARD:
            logger.info('Using jaccard based similarity')
            self.item_similarity = jaccard(item_cooccurrence)
        elif self.similarity_type == SIM_LEXICOGRAPHERS_MUTUAL_INFORMATION:
            logger.info('Using lexicographers mutual information similarity')
            self.item_similarity = lexicographers_mutual_information(item_cooccurrence)
        elif self.similarity_type == SIM_LIFT:
            logger.info('Using lift based similarity')
            self.item_similarity = lift(item_cooccurrence)
        elif self.similarity_type == SIM_MUTUAL_INFORMATION:
            logger.info('Using mutual information similarity')
            self.item_similarity = mutual_information(item_cooccurrence)
        else:
            raise ValueError('Unknown similarity type: {}'.format(self.similarity_type))
        del item_cooccurrence
        logger.info('Done training')

    def score(self, test, remove_seen=False):
        if False:
            i = 10
            return i + 15
        'Score all items for test users.\n\n        Args:\n            test (pandas.DataFrame): user to test\n            remove_seen (bool): flag to remove items seen in training from recommendation\n\n        Returns:\n            numpy.ndarray: Value of interest of all items for the users.\n        '
        user_ids = list(map(lambda user: self.user2index.get(user, np.NaN), test[self.col_user].unique()))
        if any(np.isnan(user_ids)):
            raise ValueError('SAR cannot score users that are not in the training set')
        logger.info('Calculating recommendation scores')
        test_scores = self.user_affinity[user_ids, :].dot(self.item_similarity)
        if isinstance(test_scores, sparse.spmatrix):
            test_scores = test_scores.toarray()
        if self.normalize:
            counts = self.unity_user_affinity[user_ids, :].dot(self.item_similarity)
            user_min_scores = np.tile(counts.min(axis=1)[:, np.newaxis], test_scores.shape[1]) * self.rating_min
            user_max_scores = np.tile(counts.max(axis=1)[:, np.newaxis], test_scores.shape[1]) * self.rating_max
            test_scores = rescale(test_scores, self.rating_min, self.rating_max, user_min_scores, user_max_scores)
        if remove_seen:
            logger.info('Removing seen items')
            test_scores += self.user_affinity[user_ids, :] * -np.inf
        return test_scores

    def get_popularity_based_topk(self, top_k=10, sort_top_k=True, items=True):
        if False:
            print('Hello World!')
        'Get top K most frequently occurring items across all users.\n\n        Args:\n            top_k (int): number of top items to recommend.\n            sort_top_k (bool): flag to sort top k results.\n            items (bool): if false, return most frequent users instead\n\n        Returns:\n            pandas.DataFrame: top k most popular items.\n        '
        if items:
            frequencies = self.item_frequencies
            col = self.col_item
            idx = self.index2item
        else:
            if self.user_frequencies is None:
                self.user_frequencies = self.user_affinity.getnnz(axis=1).astype('int64')
            frequencies = self.user_frequencies
            col = self.col_user
            idx = self.index2user
        test_scores = np.array([frequencies])
        logger.info('Getting top K')
        (top_components, top_scores) = get_top_k_scored_items(scores=test_scores, top_k=top_k, sort_top_k=sort_top_k)
        return pd.DataFrame({col: [idx[item] for item in top_components.flatten()], self.col_prediction: top_scores.flatten()})

    def get_item_based_topk(self, items, top_k=10, sort_top_k=True):
        if False:
            print('Hello World!')
        'Get top K similar items to provided seed items based on similarity metric defined.\n        This method will take a set of items and use them to recommend the most similar items to that set\n        based on the similarity matrix fit during training.\n        This allows recommendations for cold-users (unseen during training), note - the model is not updated.\n\n        The following options are possible based on information provided in the items input:\n        1. Single user or seed of items: only item column (ratings are assumed to be 1)\n        2. Single user or seed of items w/ ratings: item column and rating column\n        3. Separate users or seeds of items: item and user column (user ids are only used to separate item sets)\n        4. Separate users or seeds of items with ratings: item, user and rating columns provided\n\n        Args:\n            items (pandas.DataFrame): DataFrame with item, user (optional), and rating (optional) columns\n            top_k (int): number of top items to recommend\n            sort_top_k (bool): flag to sort top k results\n\n        Returns:\n            pandas.DataFrame: sorted top k recommendation items\n        '
        item_ids = np.asarray(list(map(lambda item: self.item2index.get(item, np.NaN), items[self.col_item].values)))
        if self.col_rating in items.columns:
            ratings = items[self.col_rating]
        else:
            ratings = pd.Series(np.ones_like(item_ids))
        if self.col_user in items.columns:
            test_users = items[self.col_user]
            user2index = {x[1]: x[0] for x in enumerate(items[self.col_user].unique())}
            user_ids = test_users.map(user2index)
        else:
            test_users = pd.Series(np.zeros_like(item_ids))
            user_ids = test_users
        n_users = user_ids.drop_duplicates().shape[0]
        pseudo_affinity = sparse.coo_matrix((ratings, (user_ids, item_ids)), shape=(n_users, self.n_items)).tocsr()
        test_scores = pseudo_affinity.dot(self.item_similarity)
        test_scores[user_ids, item_ids] = -np.inf
        (top_items, top_scores) = get_top_k_scored_items(scores=test_scores, top_k=top_k, sort_top_k=sort_top_k)
        df = pd.DataFrame({self.col_user: np.repeat(test_users.drop_duplicates().values, top_items.shape[1]), self.col_item: [self.index2item[item] for item in top_items.flatten()], self.col_prediction: top_scores.flatten()})
        return df.replace(-np.inf, np.nan).dropna()

    def get_topk_most_similar_users(self, user, top_k, sort_top_k=True):
        if False:
            return 10
        'Based on user affinity towards items, calculate the most similar users to the given user.\n\n        Args:\n            user (int): user to retrieve most similar users for\n            top_k (int): number of top items to recommend\n            sort_top_k (bool): flag to sort top k results\n\n        Returns:\n            pandas.DataFrame: top k most similar users and their scores\n        '
        user_idx = self.user2index[user]
        similarities = self.user_affinity[user_idx].dot(self.user_affinity.T).toarray()
        similarities[0, user_idx] = -np.inf
        (top_items, top_scores) = get_top_k_scored_items(scores=similarities, top_k=top_k, sort_top_k=sort_top_k)
        df = pd.DataFrame({self.col_user: [self.index2user[user] for user in top_items.flatten()], self.col_prediction: top_scores.flatten()})
        return df.replace(-np.inf, np.nan).dropna()

    def recommend_k_items(self, test, top_k=10, sort_top_k=True, remove_seen=False):
        if False:
            while True:
                i = 10
        'Recommend top K items for all users which are in the test set\n\n        Args:\n            test (pandas.DataFrame): users to test\n            top_k (int): number of top items to recommend\n            sort_top_k (bool): flag to sort top k results\n            remove_seen (bool): flag to remove items seen in training from recommendation\n\n        Returns:\n            pandas.DataFrame: top k recommendation items for each user\n        '
        test_scores = self.score(test, remove_seen=remove_seen)
        (top_items, top_scores) = get_top_k_scored_items(scores=test_scores, top_k=top_k, sort_top_k=sort_top_k)
        df = pd.DataFrame({self.col_user: np.repeat(test[self.col_user].drop_duplicates().values, top_items.shape[1]), self.col_item: [self.index2item[item] for item in top_items.flatten()], self.col_prediction: top_scores.flatten()})
        return df.replace(-np.inf, np.nan).dropna()

    def predict(self, test):
        if False:
            while True:
                i = 10
        'Output SAR scores for only the users-items pairs which are in the test set\n\n        Args:\n            test (pandas.DataFrame): DataFrame that contains users and items to test\n\n        Returns:\n            pandas.DataFrame: DataFrame contains the prediction results\n        '
        test_scores = self.score(test)
        user_ids = np.asarray(list(map(lambda user: self.user2index.get(user, np.NaN), test[self.col_user].values)))
        item_ids = np.asarray(list(map(lambda item: self.item2index.get(item, np.NaN), test[self.col_item].values)))
        nans = np.isnan(item_ids)
        if any(nans):
            logger.warning('Items found in test not seen during training, new items will have score of 0')
            test_scores = np.append(test_scores, np.zeros((self.n_users, 1)), axis=1)
            item_ids[nans] = self.n_items
            item_ids = item_ids.astype('int64')
        df = pd.DataFrame({self.col_user: test[self.col_user].values, self.col_item: test[self.col_item].values, self.col_prediction: test_scores[user_ids, item_ids]})
        return df