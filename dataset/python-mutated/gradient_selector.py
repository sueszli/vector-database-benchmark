import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
import torch
from nni.feature_engineering.feature_selector import FeatureSelector
from . import constants
from .fginitialize import PrepareData
from .fgtrain import _train

class FeatureGradientSelector(FeatureSelector, BaseEstimator, SelectorMixin):

    def __init__(self, order=4, penalty=1, n_features=None, max_features=None, learning_rate=0.1, init='zero', n_epochs=1, shuffle=True, batch_size=1000, target_batch_size=1000, max_time=np.inf, classification=True, ordinal=False, balanced=True, preprocess='zscore', soft_grouping=False, verbose=0, device='cpu'):
        if False:
            for i in range(10):
                print('nop')
        '\n            FeatureGradientSelector is a class that selects features for a machine\n            learning model using a gradient based search.\n\n            Parameters\n            ----------\n            order : int\n                What order of interactions to include. Higher orders\n                may be more accurate but increase the run time. 12 is the maximum allowed order.\n            penatly : int\n                Constant that multiplies the regularization term.\n            n_features: int\n                If None, will automatically choose number of features based on search.\n                Otherwise, number of top features to select.\n            max_features : int\n                If not None, will use the \'elbow method\' to determine the number of features\n                with max_features as the upper limit.\n            learning_rate : float\n            init : str\n                How to initialize the vector of scores. \'zero\' is the default.\n                Options: {\'zero\', \'on\', \'off\', \'onhigh\', \'offhigh\', \'sklearn\'}\n            n_epochs : int\n                number of epochs to run\n            shuffle : bool\n                Shuffle "rows" prior to an epoch.\n            batch_size : int\n                Nnumber of "rows" to process at a time\n            target_batch_size : int\n                Number of "rows" to accumulate gradients over.\n                Useful when many rows will not fit into memory but are needed for accurate estimation.\n            classification : bool\n                If True, problem is classification, else regression.\n            ordinal : bool\n                If True, problem is ordinal classification. Requires classification to be True.\n            balanced : bool\n                If true, each class is weighted equally in optimization, otherwise\n                weighted is done via support of each class. Requires classification to be True.\n            prerocess : str\n                \'zscore\' which refers to centering and normalizing data to unit variance or\n                \'center\' which only centers the data to 0 mean\n            soft_grouping : bool\n                if True, groups represent features that come from the same source.\n                Used to encourage sparsity of groups and features within groups.\n            verbose : int\n                Controls the verbosity when fitting. Set to 0 for no printing\n                1 or higher for printing every verbose number of gradient steps.\n            device : str\n                \'cpu\' to run on CPU and \'cuda\' to run on GPU. Runs much faster on GPU\n        '
        assert order <= 12 and order >= 1, 'order must be an integer between 1 and 12, inclusive'
        assert n_features is None or max_features is None, 'only specify one of n_features and max_features at a time'
        self.order = order
        self.penalty = penalty
        self.n_features = n_features
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.init = init
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.target_batch_size = target_batch_size
        self.max_time = max_time
        self.dftol_stop = -1
        self.freltol_stop = -1
        self.classification = classification
        self.ordinal = ordinal
        self.balanced = balanced
        self.preprocess = preprocess
        self.soft_grouping = soft_grouping
        self.verbose = verbose
        self.device = device
        self.model_ = None
        self.scores_ = None
        self._prev_checkpoint = None
        self._data_train = None

    def partial_fit(self, X, y, n_classes=None, groups=None):
        if False:
            return 10
        "\n        Select Features via a gradient based search on (X, y) on the given samples.\n        Can be called repeatedly with different X and y to handle streaming datasets.\n\n        Parameters\n        ----------\n        X : array-like\n            Shape = [n_samples, n_features]\n            The training input samples.\n        y :  array-like\n            Shape = [n_samples]\n            The target values (class labels in classification, real numbers in\n            regression).\n        n_classes : int\n            Number of classes\n            Classes across all calls to partial_fit.\n            Can be obtained by via `np.unique(y_all).shape[0]`, where y_all is the\n            target vector of the entire dataset.\n            This argument is expected for the first call to partial_fit,\n            otherwise will assume all classes are present in the batch of y given.\n            It will be ignored in the subsequent calls.\n            Note that y doesn't need to contain all labels in `classes`.\n        groups : array-like\n            Optional, shape = [n_features]\n            Groups of columns that must be selected as a unit\n            e.g. [0, 0, 1, 2] specifies the first two columns are part of a group.\n            This argument is expected for the first call to partial_fit,\n            otherwise will assume all classes are present in the batch of y given.\n            It will be ignored in the subsequent calls.\n        "
        try:
            self._partial_fit(X, y, n_classes=n_classes, groups=groups)
        except constants.NanError:
            if hasattr(self, '_prev_checkpoint'):
                print('failed fitting this batch, loss was nan')
            else:
                if self.verbose:
                    print('Loss was nan, trying with Doubles')
                self._reset()
                torch.set_default_tensor_type(torch.DoubleTensor)
                self._partial_fit(X, y, n_classes=n_classes, groups=groups)
        return self

    def _partial_fit(self, X, y, n_classes=None, groups=None):
        if False:
            while True:
                i = 10
        '\n        Private function for partial_fit to enable trying floats before doubles.\n        '
        if hasattr(self, '_data_train'):
            self._data_train.X = X.values if isinstance(X, pd.DataFrame) else X
            (self._data_train.N, self._data_train.D) = self._data_train.X.shape
            self._data_train.dense_size_gb = self._data_train.get_dense_size()
            self._data_train.set_dense_X()
            self._data_train.y = y.values if isinstance(y, pd.Series) else y
            self._data_train.y = torch.as_tensor(y, dtype=torch.get_default_dtype())
        else:
            data_train = self._prepare_data(X, y, n_classes=n_classes)
            self._data_train = data_train
        (batch_size, _, accum_steps, max_iter) = self._set_batch_size(self._data_train)
        rng = None
        debug = 0
        dn_logs = None
        path_save = None
        (m, solver) = _train(self._data_train, batch_size, self.order, self.penalty, rng, self.learning_rate, debug, max_iter, self.max_time, self.init, self.dftol_stop, self.freltol_stop, dn_logs, accum_steps, path_save, self.shuffle, device=self.device, verbose=self.verbose, prev_checkpoint=self._prev_checkpoint if hasattr(self, '_prev_checkpoint') else None, groups=groups if not self.soft_grouping else None, soft_groups=groups if self.soft_grouping else None)
        self._prev_checkpoint = m
        self._process_results(m, solver, X, groups=groups)
        return self

    def fit(self, X, y, groups=None):
        if False:
            while True:
                i = 10
        '\n        Select Features via a gradient based search on (X, y).\n\n        Parameters\n        ----------\n        X : array-like\n            Shape = [n_samples, n_features]\n            The training input samples.\n        y : array-like\n            Shape = [n_samples]\n            The target values (class labels in classification, real numbers in\n            regression).\n        groups : array-like\n            Optional, shape = [n_features]\n            Groups of columns that must be selected as a unit\n            e.g. [0, 0, 1, 2] specifies the first two columns are part of a group.\n        '
        try:
            self._fit(X, y, groups=groups)
        except constants.NanError:
            if self.verbose:
                print('Loss was nan, trying with Doubles')
            torch.set_default_tensor_type(torch.DoubleTensor)
            self._fit(X, y, groups=groups)
        return self

    def get_selected_features(self):
        if False:
            for i in range(10):
                print('nop')
        return self.selected_features_

    def _prepare_data(self, X, y, n_classes=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns a PrepareData object.\n        '
        return PrepareData(X=X.values if isinstance(X, pd.DataFrame) else X, y=y.values if isinstance(y, pd.Series) else y, data_format=constants.DataFormat.NUMPY, classification=int(self.classification), ordinal=self.ordinal, balanced=self.balanced, preprocess=self.preprocess, verbose=self.verbose, device=self.device, n_classes=n_classes)

    def _fit(self, X, y, groups=None):
        if False:
            while True:
                i = 10
        '\n        Private function for fit to enable trying floats before doubles.\n        '
        data_train = self._prepare_data(X, y)
        (batch_size, _, accum_steps, max_iter) = self._set_batch_size(data_train)
        rng = None
        debug = 0
        dn_logs = None
        path_save = None
        (m, solver) = _train(data_train, batch_size, self.order, self.penalty, rng, self.learning_rate, debug, max_iter, self.max_time, self.init, self.dftol_stop, self.freltol_stop, dn_logs, accum_steps, path_save, self.shuffle, device=self.device, verbose=self.verbose, groups=groups if not self.soft_grouping else None, soft_groups=groups if self.soft_grouping else None)
        self._process_results(m, solver, X, groups=groups)
        return self

    def _process_torch_scores(self, scores):
        if False:
            while True:
                i = 10
        '\n        Convert scores into flat numpy arrays.\n        '
        if constants.Device.CUDA in scores.device.type:
            scores = scores.cpu()
        return scores.numpy().ravel()

    def _set_batch_size(self, data_train):
        if False:
            i = 10
            return i + 15
        '\n        Ensures that batch_size is less than the number of rows.\n        '
        batch_size = min(self.batch_size, data_train.N)
        target_batch_size = min(max(self.batch_size, self.target_batch_size), data_train.N)
        accum_steps = max(int(np.ceil(target_batch_size / self.batch_size)), 1)
        max_iter = self.n_epochs * (data_train.N // batch_size)
        return (batch_size, target_batch_size, accum_steps, max_iter)

    def _process_results(self, m, solver, X, groups=None):
        if False:
            print('Hello World!')
        '\n        Process the results of a run into something suitable for transform().\n        '
        self.scores_ = self._process_torch_scores(torch.sigmoid(m[constants.Checkpoint.MODEL]['x'] * 2))
        if self.max_features:
            self.max_features = min([self.max_features, self.scores_.shape[0]])
            n_features = self._recommend_number_features(solver)
            self.set_n_features(n_features, groups=groups)
        elif self.n_features:
            self.set_n_features(self.n_features, groups=groups)
        else:
            self.selected_features_ = m['feats']
        self.max_time -= m['t']
        self.model_ = m
        return self

    def transform(self, X):
        if False:
            return 10
        '\n        Returns selected features from X.\n\n        Paramters\n        ---------\n        X: array-like\n            Shape = [n_samples, n_features]\n            The training input samples.\n        '
        self._get_support_mask()
        if self.selected_features_.shape[0] == 0:
            raise ValueError('No Features selected, consider lowering the penalty or specifying n_features')
        return X.iloc[:, self.selected_features_] if isinstance(X, pd.DataFrame) else X[:, self.selected_features_]

    def get_support(self, indices=False):
        if False:
            return 10
        '\n        Get a mask, or integer index, of the features selected.\n\n        Parameters\n        ----------\n        indices : bool\n            Default False\n            If True, the return value will be an array of integers, rather than a boolean mask.\n\n        Returns\n        -------\n        list :\n            returns support: An index that selects the retained features from a feature vector.\n            If indices is False, this is a boolean array of shape [# input features],\n            in which an element is True iff its corresponding feature is selected for retention.\n            If indices is True, this is an integer array of shape [# output features] whose values\n            are indices into the input feature vector.\n        '
        self._get_support_mask()
        if indices:
            return self.selected_features_
        mask = np.zeros_like(self.scores_, dtype=bool)
        mask[self.selected_features_] = True
        return mask

    def inverse_transform(self, X):
        if False:
            while True:
                i = 10
        '\n        Returns transformed X to the original number of column.\n        This operation is lossy and all columns not in the transformed data\n        will be returned as columns of 0s.\n        '
        self._get_support_mask()
        X_new = np.zeros((X.shape[0], self.scores_.shape[0]))
        X_new[self.selected_features_] = X
        return X_new

    def get_params(self, deep=True):
        if False:
            print('Hello World!')
        '\n        Get parameters for this estimator.\n        '
        params = self.__dict__
        params = {key: val for (key, val) in params.items() if not key.endswith('_')}
        return params

    def set_params(self, **params):
        if False:
            while True:
                i = 10
        '\n        Set the parameters of this estimator.\n        '
        for param in params:
            if hasattr(self, param):
                setattr(self, param, params[param])
        return self

    def fit_transform(self, X, y):
        if False:
            print('Hello World!')
        '\n        Select features and then return X with the selected features.\n\n        Parameters\n        ----------\n        X : array-like\n            Shape = [n_samples, n_features]\n            The training input samples.\n        y : array-like\n            Shape = [n_samples]\n            The target values (class labels in classification, real numbers in\n            regression).\n        '
        self.fit(X, y)
        return self.transform(X)

    def _get_support_mask(self):
        if False:
            print('Hello World!')
        '\n        Check if it is fitted.\n        '
        check_is_fitted(self, 'scores_')

    def _generate_scores(self, solver, xsub, ysub, step_size, feature_order):
        if False:
            while True:
                i = 10
        '\n        Generate forward passes to determine the number of features when max_features is set.\n        '
        scores = []
        for i in np.arange(1, self.max_features + 1, step_size):
            i = int(np.ceil(i))
            score = solver.f_train(torch.tensor(np.ones(i), dtype=torch.get_default_dtype()).unsqueeze(1).to(self.device), xsub[:, feature_order[:i]], ysub)
            if constants.Device.CUDA in score.device.type:
                score = score.cpu()
            scores.append(score)
        return scores

    def set_n_features(self, n, groups=None):
        if False:
            return 10
        '\n        Set the number of features to return after fitting.\n        '
        self._get_support_mask()
        self.n_features = n
        return self._set_top_features(groups=groups)

    def _set_top_features(self, groups=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the selected features after a run.\n\n        With groups, ensures that if any member of a group is selected, all members are selected\n        '
        self._get_support_mask()
        assert self.n_features <= self.scores_.shape[0], 'n_features must be less than or equal to the number of columns in X'
        self.selected_features_ = np.argpartition(self.scores_, -self.n_features)[-self.n_features:]
        if groups is not None and (not self.soft_grouping):
            selected_feature_set = set(self.selected_features_.tolist())
            for _ in np.unique(groups):
                group_members = np.where(groups == groups)[0].tolist()
                if selected_feature_set.intersection(group_members):
                    selected_feature_set.update(group_members)
            self.selected_features_ = np.array(list(selected_feature_set))
        self.selected_features_ = np.sort(self.selected_features_)
        return self

    def set_top_percentile(self, percentile, groups=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the percentile of features to return after fitting.\n        '
        self._get_support_mask()
        assert percentile <= 1 and percentile >= 0, 'percentile must between 0 and 1 inclusive'
        self.n_features = int(self.scores_.shape[0] * percentile)
        return self._set_top_features(groups=groups)

    def _recommend_number_features(self, solver, max_time=None):
        if False:
            i = 10
            return i + 15
        '\n        Get the recommended number of features by doing forward passes when max_features is set.\n        '
        max_time = max_time if max_time else self.max_time
        if max_time < 0:
            max_time = 60
        MAX_FORWARD_PASS = 200
        MAX_FULL_BATCHES = 3
        accum_steps = solver.accum_steps
        step_size = max(self.max_features / MAX_FORWARD_PASS, 1)
        feature_order = np.argsort(-self.scores_)
        t = time.time()
        dataloader_iterator = iter(solver.ds_train)
        full_scores = []
        with torch.no_grad():
            for _ in range(accum_steps * MAX_FULL_BATCHES):
                scores = []
                try:
                    (xsub, ysub) = next(dataloader_iterator)
                except StopIteration:
                    break
                except Exception as e:
                    print(e)
                    break
                if max_time and time.time() - t > max_time:
                    if self.verbose:
                        print('Stoppinn forward passes because they reached max_time: ', max_time)
                    if not full_scores:
                        return self.max_features // 2
                    break
                if solver.multiclass:
                    for target_class in range(solver.n_classes):
                        ysub_binary = solver.transform_y_into_binary(ysub, target_class)
                        scaling_value = solver._get_scaling_value(ysub, target_class)
                        if not solver._skip_y_forward(ysub_binary):
                            scores = self._generate_scores(solver, xsub, ysub_binary, step_size, feature_order)
                            full_scores.append([score * scaling_value for score in scores])
                elif not solver._skip_y_forward(ysub):
                    scores = self._generate_scores(solver, xsub, ysub, step_size, feature_order)
                    full_scores.append(scores)
        best_index = FeatureGradientSelector._find_best_index_elbow(full_scores)
        if self.verbose:
            print('Forward passes took: ', time.time() - t)
        return int(np.ceil(np.arange(1, self.max_features + 1, step_size))[best_index])

    @staticmethod
    def _find_best_index_elbow(full_scores):
        if False:
            i = 10
            return i + 15
        '\n        Finds the point on the curve that maximizes distance from the line determined by the endpoints.\n        '
        scores = pd.DataFrame(full_scores).mean(0).values.tolist()
        first_point = np.array([0, scores[0]])
        last_point = np.array([len(scores) - 1, scores[-1]])
        elbow_metric = []
        for i in range(len(scores)):
            elbow_metric.append(FeatureGradientSelector._distance_to_line(first_point, last_point, np.array([i, scores[i]])))
        return np.argmax(elbow_metric)

    @staticmethod
    def _distance_to_line(start_point, end_point, new_point):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates the shortest distance from new_point to the line determined by start_point and end_point.\n        '
        return np.cross(new_point - start_point, end_point - start_point) / np.linalg.norm(end_point - start_point)

    def _reset(self):
        if False:
            print('Hello World!')
        '\n        Reset the estimator by deleting all private and fit parameters.\n        '
        params = self.__dict__
        for (key, _) in params.items():
            if key.endswith('_') or key.startswith('_'):
                delattr(self, key)
        return self