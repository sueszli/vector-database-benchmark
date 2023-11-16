"""Scikit-learn wrapper interface for LightGBM."""
import copy
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse
from .basic import Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _LGBM_BoosterBestScoreType, _LGBM_CategoricalFeatureConfiguration, _LGBM_EvalFunctionResultType, _LGBM_FeatureNameConfiguration, _LGBM_GroupType, _LGBM_InitScoreType, _LGBM_LabelType, _LGBM_WeightType, _log_warning
from .callback import _EvalResultDict, record_evaluation
from .compat import SKLEARN_INSTALLED, LGBMNotFittedError, _LGBMAssertAllFinite, _LGBMCheckArray, _LGBMCheckClassificationTargets, _LGBMCheckSampleWeight, _LGBMCheckXY, _LGBMClassifierBase, _LGBMComputeSampleWeight, _LGBMCpuCount, _LGBMLabelEncoder, _LGBMModelBase, _LGBMRegressorBase, dt_DataTable, np_random_Generator, pd_DataFrame
from .engine import train
__all__ = ['LGBMClassifier', 'LGBMModel', 'LGBMRanker', 'LGBMRegressor']
_LGBM_ScikitMatrixLike = Union[dt_DataTable, List[Union[List[float], List[int]]], np.ndarray, pd_DataFrame, scipy.sparse.spmatrix]
_LGBM_ScikitCustomObjectiveFunction = Union[Callable[[Optional[np.ndarray], np.ndarray], Tuple[np.ndarray, np.ndarray]], Callable[[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]], Callable[[Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]]]
_LGBM_ScikitCustomEvalFunction = Union[Callable[[Optional[np.ndarray], np.ndarray], _LGBM_EvalFunctionResultType], Callable[[Optional[np.ndarray], np.ndarray], List[_LGBM_EvalFunctionResultType]], Callable[[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]], _LGBM_EvalFunctionResultType], Callable[[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]], List[_LGBM_EvalFunctionResultType]], Callable[[Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]], _LGBM_EvalFunctionResultType], Callable[[Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]], List[_LGBM_EvalFunctionResultType]]]
_LGBM_ScikitEvalMetricType = Union[str, _LGBM_ScikitCustomEvalFunction, List[Union[str, _LGBM_ScikitCustomEvalFunction]]]
_LGBM_ScikitValidSet = Tuple[_LGBM_ScikitMatrixLike, _LGBM_LabelType]

def _get_group_from_constructed_dataset(dataset: Dataset) -> Optional[np.ndarray]:
    if False:
        while True:
            i = 10
    group = dataset.get_group()
    error_msg = "Estimators in lightgbm.sklearn should only retrieve query groups from a constructed Dataset. If you're seeing this message, it's a bug in lightgbm. Please report it at https://github.com/microsoft/LightGBM/issues."
    assert group is None or isinstance(group, np.ndarray), error_msg
    return group

def _get_label_from_constructed_dataset(dataset: Dataset) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    label = dataset.get_label()
    error_msg = "Estimators in lightgbm.sklearn should only retrieve labels from a constructed Dataset. If you're seeing this message, it's a bug in lightgbm. Please report it at https://github.com/microsoft/LightGBM/issues."
    assert isinstance(label, np.ndarray), error_msg
    return label

def _get_weight_from_constructed_dataset(dataset: Dataset) -> Optional[np.ndarray]:
    if False:
        i = 10
        return i + 15
    weight = dataset.get_weight()
    error_msg = "Estimators in lightgbm.sklearn should only retrieve weights from a constructed Dataset. If you're seeing this message, it's a bug in lightgbm. Please report it at https://github.com/microsoft/LightGBM/issues."
    assert weight is None or isinstance(weight, np.ndarray), error_msg
    return weight

class _ObjectiveFunctionWrapper:
    """Proxy class for objective function."""

    def __init__(self, func: _LGBM_ScikitCustomObjectiveFunction):
        if False:
            i = 10
            return i + 15
        'Construct a proxy class.\n\n        This class transforms objective function to match objective function with signature ``new_func(preds, dataset)``\n        as expected by ``lightgbm.engine.train``.\n\n        Parameters\n        ----------\n        func : callable\n            Expects a callable with following signatures:\n            ``func(y_true, y_pred)``,\n            ``func(y_true, y_pred, weight)``\n            or ``func(y_true, y_pred, weight, group)``\n            and returns (grad, hess):\n\n                y_true : numpy 1-D array of shape = [n_samples]\n                    The target values.\n                y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)\n                    The predicted values.\n                    Predicted values are returned before any transformation,\n                    e.g. they are raw margin instead of probability of positive class for binary task.\n                weight : numpy 1-D array of shape = [n_samples]\n                    The weight of samples. Weights should be non-negative.\n                group : numpy 1-D array\n                    Group/query data.\n                    Only used in the learning-to-rank task.\n                    sum(group) = n_samples.\n                    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,\n                    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.\n                grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape [n_samples, n_classes] (for multi-class task)\n                    The value of the first order derivative (gradient) of the loss\n                    with respect to the elements of y_pred for each sample point.\n                hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)\n                    The value of the second order derivative (Hessian) of the loss\n                    with respect to the elements of y_pred for each sample point.\n\n        .. note::\n\n            For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],\n            and grad and hess should be returned in the same format.\n        '
        self.func = func

    def __call__(self, preds: np.ndarray, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        'Call passed function with appropriate arguments.\n\n        Parameters\n        ----------\n        preds : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)\n            The predicted values.\n        dataset : Dataset\n            The training dataset.\n\n        Returns\n        -------\n        grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)\n            The value of the first order derivative (gradient) of the loss\n            with respect to the elements of preds for each sample point.\n        hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)\n            The value of the second order derivative (Hessian) of the loss\n            with respect to the elements of preds for each sample point.\n        '
        labels = _get_label_from_constructed_dataset(dataset)
        argc = len(signature(self.func).parameters)
        if argc == 2:
            (grad, hess) = self.func(labels, preds)
            return (grad, hess)
        weight = _get_weight_from_constructed_dataset(dataset)
        if argc == 3:
            (grad, hess) = self.func(labels, preds, weight)
            return (grad, hess)
        if argc == 4:
            group = _get_group_from_constructed_dataset(dataset)
            return self.func(labels, preds, weight, group)
        raise TypeError(f'Self-defined objective function should have 2, 3 or 4 arguments, got {argc}')

class _EvalFunctionWrapper:
    """Proxy class for evaluation function."""

    def __init__(self, func: _LGBM_ScikitCustomEvalFunction):
        if False:
            print('Hello World!')
        'Construct a proxy class.\n\n        This class transforms evaluation function to match evaluation function with signature ``new_func(preds, dataset)``\n        as expected by ``lightgbm.engine.train``.\n\n        Parameters\n        ----------\n        func : callable\n            Expects a callable with following signatures:\n            ``func(y_true, y_pred)``,\n            ``func(y_true, y_pred, weight)``\n            or ``func(y_true, y_pred, weight, group)``\n            and returns (eval_name, eval_result, is_higher_better) or\n            list of (eval_name, eval_result, is_higher_better):\n\n                y_true : numpy 1-D array of shape = [n_samples]\n                    The target values.\n                y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array shape = [n_samples, n_classes] (for multi-class task)\n                    The predicted values.\n                    In case of custom ``objective``, predicted values are returned before any transformation,\n                    e.g. they are raw margin instead of probability of positive class for binary task in this case.\n                weight : numpy 1-D array of shape = [n_samples]\n                    The weight of samples. Weights should be non-negative.\n                group : numpy 1-D array\n                    Group/query data.\n                    Only used in the learning-to-rank task.\n                    sum(group) = n_samples.\n                    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,\n                    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.\n                eval_name : str\n                    The name of evaluation function (without whitespace).\n                eval_result : float\n                    The eval result.\n                is_higher_better : bool\n                    Is eval result higher better, e.g. AUC is ``is_higher_better``.\n        '
        self.func = func

    def __call__(self, preds: np.ndarray, dataset: Dataset) -> Union[_LGBM_EvalFunctionResultType, List[_LGBM_EvalFunctionResultType]]:
        if False:
            print('Hello World!')
        'Call passed function with appropriate arguments.\n\n        Parameters\n        ----------\n        preds : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)\n            The predicted values.\n        dataset : Dataset\n            The training dataset.\n\n        Returns\n        -------\n        eval_name : str\n            The name of evaluation function (without whitespace).\n        eval_result : float\n            The eval result.\n        is_higher_better : bool\n            Is eval result higher better, e.g. AUC is ``is_higher_better``.\n        '
        labels = _get_label_from_constructed_dataset(dataset)
        argc = len(signature(self.func).parameters)
        if argc == 2:
            return self.func(labels, preds)
        weight = _get_weight_from_constructed_dataset(dataset)
        if argc == 3:
            return self.func(labels, preds, weight)
        if argc == 4:
            group = _get_group_from_constructed_dataset(dataset)
            return self.func(labels, preds, weight, group)
        raise TypeError(f'Self-defined eval function should have 2, 3 or 4 arguments, got {argc}')
_lgbmmodel_doc_fit = "\n    Build a gradient boosting model from the training set (X, y).\n\n    Parameters\n    ----------\n    X : {X_shape}\n        Input feature matrix.\n    y : {y_shape}\n        The target values (class labels in classification, real numbers in regression).\n    sample_weight : {sample_weight_shape}\n        Weights of training data. Weights should be non-negative.\n    init_score : {init_score_shape}\n        Init score of training data.\n    group : {group_shape}\n        Group/query data.\n        Only used in the learning-to-rank task.\n        sum(group) = n_samples.\n        For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,\n        where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.\n    eval_set : list or None, optional (default=None)\n        A list of (X, y) tuple pairs to use as validation sets.\n    eval_names : list of str, or None, optional (default=None)\n        Names of eval_set.\n    eval_sample_weight : {eval_sample_weight_shape}\n        Weights of eval data. Weights should be non-negative.\n    eval_class_weight : list or None, optional (default=None)\n        Class weights of eval data.\n    eval_init_score : {eval_init_score_shape}\n        Init score of eval data.\n    eval_group : {eval_group_shape}\n        Group data of eval data.\n    eval_metric : str, callable, list or None, optional (default=None)\n        If str, it should be a built-in evaluation metric to use.\n        If callable, it should be a custom evaluation metric, see note below for more details.\n        If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.\n        In either case, the ``metric`` from the model parameters will be evaluated and used as well.\n        Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.\n    feature_name : list of str, or 'auto', optional (default='auto')\n        Feature names.\n        If 'auto' and data is pandas DataFrame, data columns names are used.\n    categorical_feature : list of str or int, or 'auto', optional (default='auto')\n        Categorical features.\n        If list of int, interpreted as indices.\n        If list of str, interpreted as feature names (need to specify ``feature_name`` as well).\n        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.\n        All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).\n        Large values could be memory consuming. Consider using consecutive integers starting from zero.\n        All negative values in categorical features will be treated as missing values.\n        The output cannot be monotonically constrained with respect to a categorical feature.\n        Floating point numbers in categorical features will be rounded towards 0.\n    callbacks : list of callable, or None, optional (default=None)\n        List of callback functions that are applied at each iteration.\n        See Callbacks in Python API for more information.\n    init_model : str, pathlib.Path, Booster, LGBMModel or None, optional (default=None)\n        Filename of LightGBM model, Booster instance or LGBMModel instance used for continue training.\n\n    Returns\n    -------\n    self : LGBMModel\n        Returns self.\n    "
_lgbmmodel_doc_custom_eval_note = '\n    Note\n    ----\n    Custom eval function expects a callable with following signatures:\n    ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or\n    ``func(y_true, y_pred, weight, group)``\n    and returns (eval_name, eval_result, is_higher_better) or\n    list of (eval_name, eval_result, is_higher_better):\n\n        y_true : numpy 1-D array of shape = [n_samples]\n            The target values.\n        y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)\n            The predicted values.\n            In case of custom ``objective``, predicted values are returned before any transformation,\n            e.g. they are raw margin instead of probability of positive class for binary task in this case.\n        weight : numpy 1-D array of shape = [n_samples]\n            The weight of samples. Weights should be non-negative.\n        group : numpy 1-D array\n            Group/query data.\n            Only used in the learning-to-rank task.\n            sum(group) = n_samples.\n            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,\n            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.\n        eval_name : str\n            The name of evaluation function (without whitespace).\n        eval_result : float\n            The eval result.\n        is_higher_better : bool\n            Is eval result higher better, e.g. AUC is ``is_higher_better``.\n'
_lgbmmodel_doc_predict = "\n    {description}\n\n    Parameters\n    ----------\n    X : {X_shape}\n        Input features matrix.\n    raw_score : bool, optional (default=False)\n        Whether to predict raw scores.\n    start_iteration : int, optional (default=0)\n        Start index of the iteration to predict.\n        If <= 0, starts from the first iteration.\n    num_iteration : int or None, optional (default=None)\n        Total number of iterations used in the prediction.\n        If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;\n        otherwise, all iterations from ``start_iteration`` are used (no limits).\n        If <= 0, all iterations from ``start_iteration`` are used (no limits).\n    pred_leaf : bool, optional (default=False)\n        Whether to predict leaf index.\n    pred_contrib : bool, optional (default=False)\n        Whether to predict feature contributions.\n\n        .. note::\n\n            If you want to get more explanations for your model's predictions using SHAP values,\n            like SHAP interaction values,\n            you can install the shap package (https://github.com/slundberg/shap).\n            Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra\n            column, where the last column is the expected value.\n\n    validate_features : bool, optional (default=False)\n        If True, ensure that the features used to predict match the ones used to train.\n        Used only if data is pandas DataFrame.\n    **kwargs\n        Other parameters for the prediction.\n\n    Returns\n    -------\n    {output_name} : {predicted_result_shape}\n        The predicted values.\n    X_leaves : {X_leaves_shape}\n        If ``pred_leaf=True``, the predicted leaf of every tree for each sample.\n    X_SHAP_values : {X_SHAP_values_shape}\n        If ``pred_contrib=True``, the feature contributions for each sample.\n    "

class LGBMModel(_LGBMModelBase):
    """Implementation of the scikit-learn API for LightGBM."""

    def __init__(self, boosting_type: str='gbdt', num_leaves: int=31, max_depth: int=-1, learning_rate: float=0.1, n_estimators: int=100, subsample_for_bin: int=200000, objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]]=None, class_weight: Optional[Union[Dict, str]]=None, min_split_gain: float=0.0, min_child_weight: float=0.001, min_child_samples: int=20, subsample: float=1.0, subsample_freq: int=0, colsample_bytree: float=1.0, reg_alpha: float=0.0, reg_lambda: float=0.0, random_state: Optional[Union[int, np.random.RandomState, 'np.random.Generator']]=None, n_jobs: Optional[int]=None, importance_type: str='split', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Construct a gradient boosting model.\n\n        Parameters\n        ----------\n        boosting_type : str, optional (default='gbdt')\n            'gbdt', traditional Gradient Boosting Decision Tree.\n            'dart', Dropouts meet Multiple Additive Regression Trees.\n            'rf', Random Forest.\n        num_leaves : int, optional (default=31)\n            Maximum tree leaves for base learners.\n        max_depth : int, optional (default=-1)\n            Maximum tree depth for base learners, <=0 means no limit.\n        learning_rate : float, optional (default=0.1)\n            Boosting learning rate.\n            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate\n            in training using ``reset_parameter`` callback.\n            Note, that this will ignore the ``learning_rate`` argument in training.\n        n_estimators : int, optional (default=100)\n            Number of boosted trees to fit.\n        subsample_for_bin : int, optional (default=200000)\n            Number of samples for constructing bins.\n        objective : str, callable or None, optional (default=None)\n            Specify the learning task and the corresponding learning objective or\n            a custom objective function to be used (see note below).\n            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.\n        class_weight : dict, 'balanced' or None, optional (default=None)\n            Weights associated with classes in the form ``{class_label: weight}``.\n            Use this parameter only for multi-class classification task;\n            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.\n            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.\n            You may want to consider performing probability calibration\n            (https://scikit-learn.org/stable/modules/calibration.html) of your model.\n            The 'balanced' mode uses the values of y to automatically adjust weights\n            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.\n            If None, all classes are supposed to have weight one.\n            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)\n            if ``sample_weight`` is specified.\n        min_split_gain : float, optional (default=0.)\n            Minimum loss reduction required to make a further partition on a leaf node of the tree.\n        min_child_weight : float, optional (default=1e-3)\n            Minimum sum of instance weight (Hessian) needed in a child (leaf).\n        min_child_samples : int, optional (default=20)\n            Minimum number of data needed in a child (leaf).\n        subsample : float, optional (default=1.)\n            Subsample ratio of the training instance.\n        subsample_freq : int, optional (default=0)\n            Frequency of subsample, <=0 means no enable.\n        colsample_bytree : float, optional (default=1.)\n            Subsample ratio of columns when constructing each tree.\n        reg_alpha : float, optional (default=0.)\n            L1 regularization term on weights.\n        reg_lambda : float, optional (default=0.)\n            L2 regularization term on weights.\n        random_state : int, RandomState object or None, optional (default=None)\n            Random number seed.\n            If int, this number is used to seed the C++ code.\n            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.\n            If None, default seeds in C++ code are used.\n        n_jobs : int or None, optional (default=None)\n            Number of parallel threads to use for training (can be changed at prediction time by\n            passing it as an extra keyword argument).\n\n            For better performance, it is recommended to set this to the number of physical cores\n            in the CPU.\n\n            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like\n            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of\n            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds\n            to using the number of physical cores in the system (its correct detection requires\n            either the ``joblib`` or the ``psutil`` util libraries to be installed).\n\n            .. versionchanged:: 4.0.0\n\n        importance_type : str, optional (default='split')\n            The type of feature importance to be filled into ``feature_importances_``.\n            If 'split', result contains numbers of times the feature is used in a model.\n            If 'gain', result contains total gains of splits which use the feature.\n        **kwargs\n            Other parameters for the model.\n            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.\n\n            .. warning::\n\n                \\*\\*kwargs is not supported in sklearn, it may cause unexpected issues.\n\n        Note\n        ----\n        A custom objective function can be provided for the ``objective`` parameter.\n        In this case, it should have the signature\n        ``objective(y_true, y_pred) -> grad, hess``,\n        ``objective(y_true, y_pred, weight) -> grad, hess``\n        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:\n\n            y_true : numpy 1-D array of shape = [n_samples]\n                The target values.\n            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)\n                The predicted values.\n                Predicted values are returned before any transformation,\n                e.g. they are raw margin instead of probability of positive class for binary task.\n            weight : numpy 1-D array of shape = [n_samples]\n                The weight of samples. Weights should be non-negative.\n            group : numpy 1-D array\n                Group/query data.\n                Only used in the learning-to-rank task.\n                sum(group) = n_samples.\n                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,\n                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.\n            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)\n                The value of the first order derivative (gradient) of the loss\n                with respect to the elements of y_pred for each sample point.\n            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)\n                The value of the second order derivative (Hessian) of the loss\n                with respect to the elements of y_pred for each sample point.\n\n        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],\n        and grad and hess should be returned in the same format.\n        "
        if not SKLEARN_INSTALLED:
            raise LightGBMError('scikit-learn is required for lightgbm.sklearn. You must install scikit-learn and restart your session to use this module.')
        self.boosting_type = boosting_type
        self.objective = objective
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_type = importance_type
        self._Booster: Optional[Booster] = None
        self._evals_result: _EvalResultDict = {}
        self._best_score: _LGBM_BoosterBestScoreType = {}
        self._best_iteration: int = -1
        self._other_params: Dict[str, Any] = {}
        self._objective = objective
        self.class_weight = class_weight
        self._class_weight: Optional[Union[Dict, str]] = None
        self._class_map: Optional[Dict[int, int]] = None
        self._n_features: int = -1
        self._n_features_in: int = -1
        self._classes: Optional[np.ndarray] = None
        self._n_classes: int = -1
        self.set_params(**kwargs)

    def _more_tags(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return {'allow_nan': True, 'X_types': ['2darray', 'sparse', '1dlabels'], '_xfail_checks': {'check_no_attributes_set_in_init': 'scikit-learn incorrectly asserts that private attributes cannot be set in __init__: (see https://github.com/microsoft/LightGBM/issues/2628)'}}

    def __sklearn_is_fitted__(self) -> bool:
        if False:
            print('Hello World!')
        return getattr(self, 'fitted_', False)

    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Get parameters for this estimator.\n\n        Parameters\n        ----------\n        deep : bool, optional (default=True)\n            If True, will return the parameters for this estimator and\n            contained subobjects that are estimators.\n\n        Returns\n        -------\n        params : dict\n            Parameter names mapped to their values.\n        '
        params = super().get_params(deep=deep)
        params.update(self._other_params)
        return params

    def set_params(self, **params: Any) -> 'LGBMModel':
        if False:
            for i in range(10):
                print('nop')
        'Set the parameters of this estimator.\n\n        Parameters\n        ----------\n        **params\n            Parameter names with their new values.\n\n        Returns\n        -------\n        self : object\n            Returns self.\n        '
        for (key, value) in params.items():
            setattr(self, key, value)
            if hasattr(self, f'_{key}'):
                setattr(self, f'_{key}', value)
            self._other_params[key] = value
        return self

    def _process_params(self, stage: str) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Process the parameters of this estimator based on its type, parameter aliases, etc.\n\n        Parameters\n        ----------\n        stage : str\n            Name of the stage (can be ``fit`` or ``predict``) this method is called from.\n\n        Returns\n        -------\n        processed_params : dict\n            Processed parameter names mapped to their values.\n        '
        assert stage in {'fit', 'predict'}
        params = self.get_params()
        params.pop('objective', None)
        for alias in _ConfigAliases.get('objective'):
            if alias in params:
                obj = params.pop(alias)
                _log_warning(f"Found '{alias}' in params. Will use it instead of 'objective' argument")
                if stage == 'fit':
                    self._objective = obj
        if stage == 'fit':
            if self._objective is None:
                if isinstance(self, LGBMRegressor):
                    self._objective = 'regression'
                elif isinstance(self, LGBMClassifier):
                    if self._n_classes > 2:
                        self._objective = 'multiclass'
                    else:
                        self._objective = 'binary'
                elif isinstance(self, LGBMRanker):
                    self._objective = 'lambdarank'
                else:
                    raise ValueError('Unknown LGBMModel type.')
        if callable(self._objective):
            if stage == 'fit':
                params['objective'] = _ObjectiveFunctionWrapper(self._objective)
            else:
                params['objective'] = 'None'
        else:
            params['objective'] = self._objective
        params.pop('importance_type', None)
        params.pop('n_estimators', None)
        params.pop('class_weight', None)
        if isinstance(params['random_state'], np.random.RandomState):
            params['random_state'] = params['random_state'].randint(np.iinfo(np.int32).max)
        elif isinstance(params['random_state'], np_random_Generator):
            params['random_state'] = int(params['random_state'].integers(np.iinfo(np.int32).max))
        if self._n_classes > 2:
            for alias in _ConfigAliases.get('num_class'):
                params.pop(alias, None)
            params['num_class'] = self._n_classes
        if hasattr(self, '_eval_at'):
            eval_at = self._eval_at
            for alias in _ConfigAliases.get('eval_at'):
                if alias in params:
                    _log_warning(f"Found '{alias}' in params. Will use it instead of 'eval_at' argument")
                    eval_at = params.pop(alias)
            params['eval_at'] = eval_at
        original_metric = self._objective if isinstance(self._objective, str) else None
        if original_metric is None:
            if isinstance(self, LGBMRegressor):
                original_metric = 'l2'
            elif isinstance(self, LGBMClassifier):
                original_metric = 'multi_logloss' if self._n_classes > 2 else 'binary_logloss'
            elif isinstance(self, LGBMRanker):
                original_metric = 'ndcg'
        params = _choose_param_value('metric', params, original_metric)
        if stage == 'fit':
            params = _choose_param_value('num_threads', params, self.n_jobs)
            params['num_threads'] = self._process_n_jobs(params['num_threads'])
        return params

    def _process_n_jobs(self, n_jobs: Optional[int]) -> int:
        if False:
            for i in range(10):
                print('nop')
        "Convert special values of n_jobs to their actual values according to the formulas that apply.\n\n        Parameters\n        ----------\n        n_jobs : int or None\n            The original value of n_jobs, potentially having special values such as 'None' or\n            negative integers.\n\n        Returns\n        -------\n        n_jobs : int\n            The value of n_jobs with special values converted to actual number of threads.\n        "
        if n_jobs is None:
            n_jobs = _LGBMCpuCount(only_physical_cores=True)
        elif n_jobs < 0:
            n_jobs = max(_LGBMCpuCount(only_physical_cores=False) + 1 + n_jobs, 1)
        return n_jobs

    def fit(self, X: _LGBM_ScikitMatrixLike, y: _LGBM_LabelType, sample_weight: Optional[_LGBM_WeightType]=None, init_score: Optional[_LGBM_InitScoreType]=None, group: Optional[_LGBM_GroupType]=None, eval_set: Optional[List[_LGBM_ScikitValidSet]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_LGBM_WeightType]]=None, eval_class_weight: Optional[List[float]]=None, eval_init_score: Optional[List[_LGBM_InitScoreType]]=None, eval_group: Optional[List[_LGBM_GroupType]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, feature_name: _LGBM_FeatureNameConfiguration='auto', categorical_feature: _LGBM_CategoricalFeatureConfiguration='auto', callbacks: Optional[List[Callable]]=None, init_model: Optional[Union[str, Path, Booster, 'LGBMModel']]=None) -> 'LGBMModel':
        if False:
            for i in range(10):
                print('nop')
        'Docstring is set after definition, using a template.'
        params = self._process_params(stage='fit')
        eval_metric_list: List[Union[str, _LGBM_ScikitCustomEvalFunction]]
        if eval_metric is None:
            eval_metric_list = []
        elif isinstance(eval_metric, list):
            eval_metric_list = copy.deepcopy(eval_metric)
        else:
            eval_metric_list = [copy.deepcopy(eval_metric)]
        eval_metrics_callable = [_EvalFunctionWrapper(f) for f in eval_metric_list if callable(f)]
        eval_metrics_builtin = [m for m in eval_metric_list if isinstance(m, str)]
        params['metric'] = [params['metric']] if isinstance(params['metric'], (str, type(None))) else params['metric']
        params['metric'] = [e for e in eval_metrics_builtin if e not in params['metric']] + params['metric']
        params['metric'] = [metric for metric in params['metric'] if metric is not None]
        if not isinstance(X, (pd_DataFrame, dt_DataTable)):
            (_X, _y) = _LGBMCheckXY(X, y, accept_sparse=True, force_all_finite=False, ensure_min_samples=2)
            if sample_weight is not None:
                sample_weight = _LGBMCheckSampleWeight(sample_weight, _X)
        else:
            (_X, _y) = (X, y)
        if self._class_weight is None:
            self._class_weight = self.class_weight
        if self._class_weight is not None:
            class_sample_weight = _LGBMComputeSampleWeight(self._class_weight, y)
            if sample_weight is None or len(sample_weight) == 0:
                sample_weight = class_sample_weight
            else:
                sample_weight = np.multiply(sample_weight, class_sample_weight)
        self._n_features = _X.shape[1]
        self._n_features_in = self._n_features
        train_set = Dataset(data=_X, label=_y, weight=sample_weight, group=group, init_score=init_score, categorical_feature=categorical_feature, params=params)
        valid_sets: List[Dataset] = []
        if eval_set is not None:

            def _get_meta_data(collection, name, i):
                if False:
                    print('Hello World!')
                if collection is None:
                    return None
                elif isinstance(collection, list):
                    return collection[i] if len(collection) > i else None
                elif isinstance(collection, dict):
                    return collection.get(i, None)
                else:
                    raise TypeError(f'{name} should be dict or list')
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for (i, valid_data) in enumerate(eval_set):
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_weight = _get_meta_data(eval_sample_weight, 'eval_sample_weight', i)
                    valid_class_weight = _get_meta_data(eval_class_weight, 'eval_class_weight', i)
                    if valid_class_weight is not None:
                        if isinstance(valid_class_weight, dict) and self._class_map is not None:
                            valid_class_weight = {self._class_map[k]: v for (k, v) in valid_class_weight.items()}
                        valid_class_sample_weight = _LGBMComputeSampleWeight(valid_class_weight, valid_data[1])
                        if valid_weight is None or len(valid_weight) == 0:
                            valid_weight = valid_class_sample_weight
                        else:
                            valid_weight = np.multiply(valid_weight, valid_class_sample_weight)
                    valid_init_score = _get_meta_data(eval_init_score, 'eval_init_score', i)
                    valid_group = _get_meta_data(eval_group, 'eval_group', i)
                    valid_set = Dataset(data=valid_data[0], label=valid_data[1], weight=valid_weight, group=valid_group, init_score=valid_init_score, categorical_feature='auto', params=params)
                valid_sets.append(valid_set)
        if isinstance(init_model, LGBMModel):
            init_model = init_model.booster_
        if callbacks is None:
            callbacks = []
        else:
            callbacks = copy.copy(callbacks)
        evals_result: _EvalResultDict = {}
        callbacks.append(record_evaluation(evals_result))
        self._Booster = train(params=params, train_set=train_set, num_boost_round=self.n_estimators, valid_sets=valid_sets, valid_names=eval_names, feval=eval_metrics_callable, init_model=init_model, feature_name=feature_name, callbacks=callbacks)
        self._evals_result = evals_result
        self._best_iteration = self._Booster.best_iteration
        self._best_score = self._Booster.best_score
        self.fitted_ = True
        self._Booster.free_dataset()
        del train_set, valid_sets
        return self
    fit.__doc__ = _lgbmmodel_doc_fit.format(X_shape="numpy array, pandas DataFrame, H2O DataTable's Frame , scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]", y_shape='numpy array, pandas DataFrame, pandas Series, list of int or float of shape = [n_samples]', sample_weight_shape='numpy array, pandas Series, list of int or float of shape = [n_samples] or None, optional (default=None)', init_score_shape='numpy array, pandas DataFrame, pandas Series, list of int or float of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task) or shape = [n_samples, n_classes] (for multi-class task) or None, optional (default=None)', group_shape='numpy array, pandas Series, list of int or float, or None, optional (default=None)', eval_sample_weight_shape='list of array (same types as ``sample_weight`` supports), or None, optional (default=None)', eval_init_score_shape='list of array (same types as ``init_score`` supports), or None, optional (default=None)', eval_group_shape='list of array (same types as ``group`` supports), or None, optional (default=None)') + '\n\n' + _lgbmmodel_doc_custom_eval_note

    def predict(self, X: _LGBM_ScikitMatrixLike, raw_score: bool=False, start_iteration: int=0, num_iteration: Optional[int]=None, pred_leaf: bool=False, pred_contrib: bool=False, validate_features: bool=False, **kwargs: Any):
        if False:
            print('Hello World!')
        'Docstring is set after definition, using a template.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('Estimator not fitted, call fit before exploiting the model.')
        if not isinstance(X, (pd_DataFrame, dt_DataTable)):
            X = _LGBMCheckArray(X, accept_sparse=True, force_all_finite=False)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError(f'Number of features of the model must match the input. Model n_features_ is {self._n_features} and input n_features is {n_features}')
        predict_params = self._process_params(stage='predict')
        for alias in _ConfigAliases.get_by_alias('data', 'X', 'raw_score', 'start_iteration', 'num_iteration', 'pred_leaf', 'pred_contrib', *kwargs.keys()):
            predict_params.pop(alias, None)
        predict_params.update(kwargs)
        predict_params = _choose_param_value('num_threads', predict_params, self.n_jobs)
        predict_params['num_threads'] = self._process_n_jobs(predict_params['num_threads'])
        return self._Booster.predict(X, raw_score=raw_score, start_iteration=start_iteration, num_iteration=num_iteration, pred_leaf=pred_leaf, pred_contrib=pred_contrib, validate_features=validate_features, **predict_params)
    predict.__doc__ = _lgbmmodel_doc_predict.format(description='Return the predicted value for each sample.', X_shape="numpy array, pandas DataFrame, H2O DataTable's Frame , scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]", output_name='predicted_result', predicted_result_shape='array-like of shape = [n_samples] or shape = [n_samples, n_classes]', X_leaves_shape='array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]', X_SHAP_values_shape='array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects')

    @property
    def n_features_(self) -> int:
        if False:
            while True:
                i = 10
        ':obj:`int`: The number of features of fitted model.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No n_features found. Need to call fit beforehand.')
        return self._n_features

    @property
    def n_features_in_(self) -> int:
        if False:
            return 10
        ':obj:`int`: The number of features of fitted model.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No n_features_in found. Need to call fit beforehand.')
        return self._n_features_in

    @property
    def best_score_(self) -> _LGBM_BoosterBestScoreType:
        if False:
            for i in range(10):
                print('nop')
        ':obj:`dict`: The best score of fitted model.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No best_score found. Need to call fit beforehand.')
        return self._best_score

    @property
    def best_iteration_(self) -> int:
        if False:
            i = 10
            return i + 15
        ':obj:`int`: The best iteration of fitted model if ``early_stopping()`` callback has been specified.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No best_iteration found. Need to call fit with early_stopping callback beforehand.')
        return self._best_iteration

    @property
    def objective_(self) -> Union[str, _LGBM_ScikitCustomObjectiveFunction]:
        if False:
            for i in range(10):
                print('nop')
        ':obj:`str` or :obj:`callable`: The concrete objective used while fitting this model.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No objective found. Need to call fit beforehand.')
        return self._objective

    @property
    def n_estimators_(self) -> int:
        if False:
            i = 10
            return i + 15
        ':obj:`int`: True number of boosting iterations performed.\n\n        This might be less than parameter ``n_estimators`` if early stopping was enabled or\n        if boosting stopped early due to limits on complexity like ``min_gain_to_split``.\n        \n        .. versionadded:: 4.0.0\n        '
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No n_estimators found. Need to call fit beforehand.')
        return self._Booster.current_iteration()

    @property
    def n_iter_(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        ':obj:`int`: True number of boosting iterations performed.\n\n        This might be less than parameter ``n_estimators`` if early stopping was enabled or\n        if boosting stopped early due to limits on complexity like ``min_gain_to_split``.\n        \n        .. versionadded:: 4.0.0\n        '
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No n_iter found. Need to call fit beforehand.')
        return self._Booster.current_iteration()

    @property
    def booster_(self) -> Booster:
        if False:
            print('Hello World!')
        'Booster: The underlying Booster of this model.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No booster found. Need to call fit beforehand.')
        return self._Booster

    @property
    def evals_result_(self) -> _EvalResultDict:
        if False:
            print('Hello World!')
        ':obj:`dict`: The evaluation results if validation sets have been specified.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No results found. Need to call fit with eval_set beforehand.')
        return self._evals_result

    @property
    def feature_importances_(self) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        ':obj:`array` of shape = [n_features]: The feature importances (the higher, the more important).\n\n        .. note::\n\n            ``importance_type`` attribute is passed to the function\n            to configure the type of importance values to be extracted.\n        '
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No feature_importances found. Need to call fit beforehand.')
        return self._Booster.feature_importance(importance_type=self.importance_type)

    @property
    def feature_name_(self) -> List[str]:
        if False:
            while True:
                i = 10
        ':obj:`list` of shape = [n_features]: The names of features.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No feature_name found. Need to call fit beforehand.')
        return self._Booster.feature_name()

class LGBMRegressor(_LGBMRegressorBase, LGBMModel):
    """LightGBM regressor."""

    def fit(self, X: _LGBM_ScikitMatrixLike, y: _LGBM_LabelType, sample_weight: Optional[_LGBM_WeightType]=None, init_score: Optional[_LGBM_InitScoreType]=None, eval_set: Optional[List[_LGBM_ScikitValidSet]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_LGBM_WeightType]]=None, eval_init_score: Optional[List[_LGBM_InitScoreType]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, feature_name: _LGBM_FeatureNameConfiguration='auto', categorical_feature: _LGBM_CategoricalFeatureConfiguration='auto', callbacks: Optional[List[Callable]]=None, init_model: Optional[Union[str, Path, Booster, LGBMModel]]=None) -> 'LGBMRegressor':
        if False:
            while True:
                i = 10
        'Docstring is inherited from the LGBMModel.'
        super().fit(X, y, sample_weight=sample_weight, init_score=init_score, eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight, eval_init_score=eval_init_score, eval_metric=eval_metric, feature_name=feature_name, categorical_feature=categorical_feature, callbacks=callbacks, init_model=init_model)
        return self
    _base_doc = LGBMModel.fit.__doc__.replace('self : LGBMModel', 'self : LGBMRegressor')
    _base_doc = _base_doc[:_base_doc.find('group :')] + _base_doc[_base_doc.find('eval_set :'):]
    _base_doc = _base_doc[:_base_doc.find('eval_class_weight :')] + _base_doc[_base_doc.find('eval_init_score :'):]
    fit.__doc__ = _base_doc[:_base_doc.find('eval_group :')] + _base_doc[_base_doc.find('eval_metric :'):]

class LGBMClassifier(_LGBMClassifierBase, LGBMModel):
    """LightGBM classifier."""

    def fit(self, X: _LGBM_ScikitMatrixLike, y: _LGBM_LabelType, sample_weight: Optional[_LGBM_WeightType]=None, init_score: Optional[_LGBM_InitScoreType]=None, eval_set: Optional[List[_LGBM_ScikitValidSet]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_LGBM_WeightType]]=None, eval_class_weight: Optional[List[float]]=None, eval_init_score: Optional[List[_LGBM_InitScoreType]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, feature_name: _LGBM_FeatureNameConfiguration='auto', categorical_feature: _LGBM_CategoricalFeatureConfiguration='auto', callbacks: Optional[List[Callable]]=None, init_model: Optional[Union[str, Path, Booster, LGBMModel]]=None) -> 'LGBMClassifier':
        if False:
            for i in range(10):
                print('nop')
        'Docstring is inherited from the LGBMModel.'
        _LGBMAssertAllFinite(y)
        _LGBMCheckClassificationTargets(y)
        self._le = _LGBMLabelEncoder().fit(y)
        _y = self._le.transform(y)
        self._class_map = dict(zip(self._le.classes_, self._le.transform(self._le.classes_)))
        if isinstance(self.class_weight, dict):
            self._class_weight = {self._class_map[k]: v for (k, v) in self.class_weight.items()}
        self._classes = self._le.classes_
        self._n_classes = len(self._classes)
        if self.objective is None:
            self._objective = None
        if not callable(eval_metric):
            if isinstance(eval_metric, list):
                eval_metric_list = eval_metric
            elif isinstance(eval_metric, str):
                eval_metric_list = [eval_metric]
            else:
                eval_metric_list = []
            if self._n_classes > 2:
                for (index, metric) in enumerate(eval_metric_list):
                    if metric in {'logloss', 'binary_logloss'}:
                        eval_metric_list[index] = 'multi_logloss'
                    elif metric in {'error', 'binary_error'}:
                        eval_metric_list[index] = 'multi_error'
            else:
                for (index, metric) in enumerate(eval_metric_list):
                    if metric in {'logloss', 'multi_logloss'}:
                        eval_metric_list[index] = 'binary_logloss'
                    elif metric in {'error', 'multi_error'}:
                        eval_metric_list[index] = 'binary_error'
            eval_metric = eval_metric_list
        valid_sets: Optional[List[_LGBM_ScikitValidSet]] = None
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            valid_sets = []
            for (valid_x, valid_y) in eval_set:
                if valid_x is X and valid_y is y:
                    valid_sets.append((valid_x, _y))
                else:
                    valid_sets.append((valid_x, self._le.transform(valid_y)))
        super().fit(X, _y, sample_weight=sample_weight, init_score=init_score, eval_set=valid_sets, eval_names=eval_names, eval_sample_weight=eval_sample_weight, eval_class_weight=eval_class_weight, eval_init_score=eval_init_score, eval_metric=eval_metric, feature_name=feature_name, categorical_feature=categorical_feature, callbacks=callbacks, init_model=init_model)
        return self
    _base_doc = LGBMModel.fit.__doc__.replace('self : LGBMModel', 'self : LGBMClassifier')
    _base_doc = _base_doc[:_base_doc.find('group :')] + _base_doc[_base_doc.find('eval_set :'):]
    fit.__doc__ = _base_doc[:_base_doc.find('eval_group :')] + _base_doc[_base_doc.find('eval_metric :'):]

    def predict(self, X: _LGBM_ScikitMatrixLike, raw_score: bool=False, start_iteration: int=0, num_iteration: Optional[int]=None, pred_leaf: bool=False, pred_contrib: bool=False, validate_features: bool=False, **kwargs: Any):
        if False:
            print('Hello World!')
        'Docstring is inherited from the LGBMModel.'
        result = self.predict_proba(X=X, raw_score=raw_score, start_iteration=start_iteration, num_iteration=num_iteration, pred_leaf=pred_leaf, pred_contrib=pred_contrib, validate_features=validate_features, **kwargs)
        if callable(self._objective) or raw_score or pred_leaf or pred_contrib:
            return result
        else:
            class_index = np.argmax(result, axis=1)
            return self._le.inverse_transform(class_index)
    predict.__doc__ = LGBMModel.predict.__doc__

    def predict_proba(self, X: _LGBM_ScikitMatrixLike, raw_score: bool=False, start_iteration: int=0, num_iteration: Optional[int]=None, pred_leaf: bool=False, pred_contrib: bool=False, validate_features: bool=False, **kwargs: Any):
        if False:
            while True:
                i = 10
        'Docstring is set after definition, using a template.'
        result = super().predict(X=X, raw_score=raw_score, start_iteration=start_iteration, num_iteration=num_iteration, pred_leaf=pred_leaf, pred_contrib=pred_contrib, validate_features=validate_features, **kwargs)
        if callable(self._objective) and (not (raw_score or pred_leaf or pred_contrib)):
            _log_warning('Cannot compute class probabilities or labels due to the usage of customized objective function.\nReturning raw scores instead.')
            return result
        elif self._n_classes > 2 or raw_score or pred_leaf or pred_contrib:
            return result
        else:
            return np.vstack((1.0 - result, result)).transpose()
    predict_proba.__doc__ = _lgbmmodel_doc_predict.format(description='Return the predicted probability for each class for each sample.', X_shape="numpy array, pandas DataFrame, H2O DataTable's Frame , scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]", output_name='predicted_probability', predicted_result_shape='array-like of shape = [n_samples] or shape = [n_samples, n_classes]', X_leaves_shape='array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]', X_SHAP_values_shape='array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects')

    @property
    def classes_(self) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        ':obj:`array` of shape = [n_classes]: The class label array.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No classes found. Need to call fit beforehand.')
        return self._classes

    @property
    def n_classes_(self) -> int:
        if False:
            i = 10
            return i + 15
        ':obj:`int`: The number of classes.'
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError('No classes found. Need to call fit beforehand.')
        return self._n_classes

class LGBMRanker(LGBMModel):
    """LightGBM ranker.

    .. warning::

        scikit-learn doesn't support ranking applications yet,
        therefore this class is not really compatible with the sklearn ecosystem.
        Please use this class mainly for training and applying ranking models in common sklearnish way.
    """

    def fit(self, X: _LGBM_ScikitMatrixLike, y: _LGBM_LabelType, sample_weight: Optional[_LGBM_WeightType]=None, init_score: Optional[_LGBM_InitScoreType]=None, group: Optional[_LGBM_GroupType]=None, eval_set: Optional[List[_LGBM_ScikitValidSet]]=None, eval_names: Optional[List[str]]=None, eval_sample_weight: Optional[List[_LGBM_WeightType]]=None, eval_init_score: Optional[List[_LGBM_InitScoreType]]=None, eval_group: Optional[List[_LGBM_GroupType]]=None, eval_metric: Optional[_LGBM_ScikitEvalMetricType]=None, eval_at: Union[List[int], Tuple[int, ...]]=(1, 2, 3, 4, 5), feature_name: _LGBM_FeatureNameConfiguration='auto', categorical_feature: _LGBM_CategoricalFeatureConfiguration='auto', callbacks: Optional[List[Callable]]=None, init_model: Optional[Union[str, Path, Booster, LGBMModel]]=None) -> 'LGBMRanker':
        if False:
            return 10
        'Docstring is inherited from the LGBMModel.'
        if group is None:
            raise ValueError('Should set group for ranking task')
        if eval_set is not None:
            if eval_group is None:
                raise ValueError('Eval_group cannot be None when eval_set is not None')
            elif len(eval_group) != len(eval_set):
                raise ValueError('Length of eval_group should be equal to eval_set')
            elif isinstance(eval_group, dict) and any((i not in eval_group or eval_group[i] is None for i in range(len(eval_group)))) or (isinstance(eval_group, list) and any((group is None for group in eval_group))):
                raise ValueError('Should set group for all eval datasets for ranking task; if you use dict, the index should start from 0')
        self._eval_at = eval_at
        super().fit(X, y, sample_weight=sample_weight, init_score=init_score, group=group, eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight, eval_init_score=eval_init_score, eval_group=eval_group, eval_metric=eval_metric, feature_name=feature_name, categorical_feature=categorical_feature, callbacks=callbacks, init_model=init_model)
        return self
    _base_doc = LGBMModel.fit.__doc__.replace('self : LGBMModel', 'self : LGBMRanker')
    fit.__doc__ = _base_doc[:_base_doc.find('eval_class_weight :')] + _base_doc[_base_doc.find('eval_init_score :'):]
    _base_doc = fit.__doc__
    (_before_feature_name, _feature_name, _after_feature_name) = _base_doc.partition('feature_name :')
    fit.__doc__ = f'{_before_feature_name}eval_at : list or tuple of int, optional (default=(1, 2, 3, 4, 5))\n        The evaluation positions of the specified metric.\n    {_feature_name}{_after_feature_name}'