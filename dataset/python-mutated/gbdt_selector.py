"""
gbdt_selector.py including:
    class GBDTSelector
"""
import random
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from nni.feature_engineering.feature_selector import FeatureSelector

class GBDTSelector(FeatureSelector):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.selected_features_ = None
        self.X = None
        self.y = None
        self.feature_importance = None
        self.lgb_params = None
        self.eval_ratio = None
        self.early_stopping_rounds = None
        self.importance_type = None
        self.num_boost_round = None
        self.model = None

    def fit(self, X, y, **kwargs):
        if False:
            print('Hello World!')
        "\n        Fit the training data to FeatureSelector\n\n        Paramters\n        ---------\n        X : array-like numpy matrix\n            The training input samples, which shape is [n_samples, n_features].\n        y : array-like numpy matrix\n            The target values (class labels in classification, real numbers in\n            regression). Which shape is [n_samples].\n        lgb_params : dict\n            Parameters of lightgbm\n        eval_ratio : float\n            The ratio of data size. It's used for split the eval data and train data from self.X.\n        early_stopping_rounds : int\n            The early stopping setting in lightgbm.\n        importance_type : str\n            Supporting type is 'gain' or 'split'.\n        num_boost_round : int\n            num_boost_round in lightgbm.\n        "
        assert kwargs['lgb_params']
        assert kwargs['eval_ratio']
        assert kwargs['early_stopping_rounds']
        assert kwargs['importance_type']
        assert kwargs['num_boost_round']
        self.X = X
        self.y = y
        self.lgb_params = kwargs['lgb_params']
        self.eval_ratio = kwargs['eval_ratio']
        self.early_stopping_rounds = kwargs['early_stopping_rounds']
        self.importance_type = kwargs['importance_type']
        self.num_boost_round = kwargs['num_boost_round']
        (X_train, X_test, y_train, y_test) = train_test_split(self.X, self.y, test_size=self.eval_ratio, random_state=random.seed(41))
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        self.model = lgb.train(self.lgb_params, lgb_train, num_boost_round=self.num_boost_round, valid_sets=lgb_eval, early_stopping_rounds=self.early_stopping_rounds)
        self.feature_importance = self.model.feature_importance(self.importance_type)

    def get_selected_features(self, topk):
        if False:
            while True:
                i = 10
        '\n        Fit the training data to FeatureSelector\n\n        Returns\n        -------\n        list :\n                Return the index of imprtant feature.\n        '
        assert topk > 0
        self.selected_features_ = self.feature_importance.argsort()[-topk:][::-1]
        return self.selected_features_