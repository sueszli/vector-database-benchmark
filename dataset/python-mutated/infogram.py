import ast
import json
import warnings
import h2o
from h2o.utils.shared_utils import can_use_numpy
from h2o.utils.typechecks import is_type
from h2o.plot import get_matplotlib_pyplot, decorate_plot_result, get_polycollection
from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OInfogram(H2OEstimator):
    """
    Information Diagram

    The infogram is a graphical information-theoretic interpretability tool which allows the user to quickly spot the core, decision-making variables 
    that uniquely and safely drive the response, in supervised classification problems. The infogram can significantly cut down the number of predictors needed to build 
    a model by identifying only the most valuable, admissible features. When protected variables such as race or gender are present in the data, the admissibility 
    of a variable is determined by a safety and relevancy index, and thus serves as a diagnostic tool for fairness. The safety of each feature can be quantified and 
    variables that are unsafe will be considered inadmissible. Models built using only admissible features will naturally be more interpretable, given the reduced 
    feature set.  Admissible models are also less susceptible to overfitting and train faster, while providing similar accuracy as models built using all available features.
    """
    algo = 'infogram'
    supervised_learning = True

    def __init__(self, model_id=None, training_frame=None, validation_frame=None, seed=-1, keep_cross_validation_models=True, keep_cross_validation_predictions=False, keep_cross_validation_fold_assignment=False, nfolds=0, fold_assignment='auto', fold_column=None, response_column=None, ignored_columns=None, ignore_const_cols=True, score_each_iteration=False, offset_column=None, weights_column=None, standardize=False, distribution='auto', plug_values=None, max_iterations=0, stopping_rounds=0, stopping_metric='auto', stopping_tolerance=0.001, balance_classes=False, class_sampling_factors=None, max_after_balance_size=5.0, max_runtime_secs=0.0, custom_metric_func=None, auc_type='auto', algorithm='auto', algorithm_params=None, protected_columns=None, total_information_threshold=-1.0, net_information_threshold=-1.0, relevance_index_threshold=-1.0, safety_index_threshold=-1.0, data_fraction=1.0, top_n_features=50):
        if False:
            i = 10
            return i + 15
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param validation_frame: Id of the validation data frame.\n               Defaults to ``None``.\n        :type validation_frame: Union[None, str, H2OFrame], optional\n        :param seed: Seed for pseudo random number generator (if applicable).\n               Defaults to ``-1``.\n        :type seed: int\n        :param keep_cross_validation_models: Whether to keep the cross-validation models.\n               Defaults to ``True``.\n        :type keep_cross_validation_models: bool\n        :param keep_cross_validation_predictions: Whether to keep the predictions of the cross-validation models.\n               Defaults to ``False``.\n        :type keep_cross_validation_predictions: bool\n        :param keep_cross_validation_fold_assignment: Whether to keep the cross-validation fold assignment.\n               Defaults to ``False``.\n        :type keep_cross_validation_fold_assignment: bool\n        :param nfolds: Number of folds for K-fold cross-validation (0 to disable or >= 2).\n               Defaults to ``0``.\n        :type nfolds: int\n        :param fold_assignment: Cross-validation fold assignment scheme, if fold_column is not specified. The\n               \'Stratified\' option will stratify the folds based on the response variable, for classification problems.\n               Defaults to ``"auto"``.\n        :type fold_assignment: Literal["auto", "random", "modulo", "stratified"]\n        :param fold_column: Column with cross-validation fold index assignment per observation.\n               Defaults to ``None``.\n        :type fold_column: str, optional\n        :param response_column: Response variable column.\n               Defaults to ``None``.\n        :type response_column: str, optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param ignore_const_cols: Ignore constant columns.\n               Defaults to ``True``.\n        :type ignore_const_cols: bool\n        :param score_each_iteration: Whether to score during each iteration of model training.\n               Defaults to ``False``.\n        :type score_each_iteration: bool\n        :param offset_column: Offset column. This will be added to the combination of columns before applying the link\n               function.\n               Defaults to ``None``.\n        :type offset_column: str, optional\n        :param weights_column: Column with observation weights. Giving some observation a weight of zero is equivalent\n               to excluding it from the dataset; giving an observation a relative weight of 2 is equivalent to repeating\n               that row twice. Negative weights are not allowed. Note: Weights are per-row observation weights and do\n               not increase the size of the data frame. This is typically the number of times a row is repeated, but\n               non-integer values are supported as well. During training, rows with higher weights matter more, due to\n               the larger loss function pre-factor. If you set weight = 0 for a row, the returned prediction frame at\n               that row is zero and this is incorrect. To get an accurate prediction, remove all rows with weight == 0.\n               Defaults to ``None``.\n        :type weights_column: str, optional\n        :param standardize: Standardize numeric columns to have zero mean and unit variance.\n               Defaults to ``False``.\n        :type standardize: bool\n        :param distribution: Distribution function\n               Defaults to ``"auto"``.\n        :type distribution: Literal["auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace",\n               "quantile", "huber"]\n        :param plug_values: Plug Values (a single row frame containing values that will be used to impute missing values\n               of the training/validation frame, use with conjunction missing_values_handling = PlugValues).\n               Defaults to ``None``.\n        :type plug_values: Union[None, str, H2OFrame], optional\n        :param max_iterations: Maximum number of iterations.\n               Defaults to ``0``.\n        :type max_iterations: int\n        :param stopping_rounds: Early stopping based on convergence of stopping_metric. Stop if simple moving average of\n               length k of the stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)\n               Defaults to ``0``.\n        :type stopping_rounds: int\n        :param stopping_metric: Metric to use for early stopping (AUTO: logloss for classification, deviance for\n               regression and anomaly_score for Isolation Forest). Note that custom and custom_increasing can only be\n               used in GBM and DRF with the Python client.\n               Defaults to ``"auto"``.\n        :type stopping_metric: Literal["auto", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "lift_top_group",\n               "misclassification", "mean_per_class_error", "custom", "custom_increasing"]\n        :param stopping_tolerance: Relative tolerance for metric-based stopping criterion (stop if relative improvement\n               is not at least this much)\n               Defaults to ``0.001``.\n        :type stopping_tolerance: float\n        :param balance_classes: Balance training data class counts via over/under-sampling (for imbalanced data).\n               Defaults to ``False``.\n        :type balance_classes: bool\n        :param class_sampling_factors: Desired over/under-sampling ratios per class (in lexicographic order). If not\n               specified, sampling factors will be automatically computed to obtain class balance during training.\n               Requires balance_classes.\n               Defaults to ``None``.\n        :type class_sampling_factors: List[float], optional\n        :param max_after_balance_size: Maximum relative size of the training data after balancing class counts (can be\n               less than 1.0). Requires balance_classes.\n               Defaults to ``5.0``.\n        :type max_after_balance_size: float\n        :param max_runtime_secs: Maximum allowed runtime in seconds for model training. Use 0 to disable.\n               Defaults to ``0.0``.\n        :type max_runtime_secs: float\n        :param custom_metric_func: Reference to custom evaluation function, format: `language:keyName=funcName`\n               Defaults to ``None``.\n        :type custom_metric_func: str, optional\n        :param auc_type: Set default multinomial AUC type.\n               Defaults to ``"auto"``.\n        :type auc_type: Literal["auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"]\n        :param algorithm: Type of machine learning algorithm used to build the infogram. Options include \'AUTO\' (gbm),\n               \'deeplearning\' (Deep Learning with default parameters), \'drf\' (Random Forest with default parameters),\n               \'gbm\' (GBM with default parameters), \'glm\' (GLM with default parameters), or \'xgboost\' (if available,\n               XGBoost with default parameters).\n               Defaults to ``"auto"``.\n        :type algorithm: Literal["auto", "deeplearning", "drf", "gbm", "glm", "xgboost"]\n        :param algorithm_params: Customized parameters for the machine learning algorithm specified in the algorithm\n               parameter.\n               Defaults to ``None``.\n        :type algorithm_params: dict, optional\n        :param protected_columns: Columns that contain features that are sensitive and need to be protected (legally, or\n               otherwise), if applicable. These features (e.g. race, gender, etc) should not drive the prediction of the\n               response.\n               Defaults to ``None``.\n        :type protected_columns: List[str], optional\n        :param total_information_threshold: A number between 0 and 1 representing a threshold for total information,\n               defaulting to 0.1. For a specific feature, if the total information is higher than this threshold, and\n               the corresponding net information is also higher than the threshold ``net_information_threshold``, that\n               feature will be considered admissible. The total information is the x-axis of the Core Infogram. Default\n               is -1 which gets set to 0.1.\n               Defaults to ``-1.0``.\n        :type total_information_threshold: float\n        :param net_information_threshold: A number between 0 and 1 representing a threshold for net information,\n               defaulting to 0.1.  For a specific feature, if the net information is higher than this threshold, and the\n               corresponding total information is also higher than the total_information_threshold, that feature will be\n               considered admissible. The net information is the y-axis of the Core Infogram. Default is -1 which gets\n               set to 0.1.\n               Defaults to ``-1.0``.\n        :type net_information_threshold: float\n        :param relevance_index_threshold: A number between 0 and 1 representing a threshold for the relevance index,\n               defaulting to 0.1.  This is only used when ``protected_columns`` is set by the user.  For a specific\n               feature, if the relevance index value is higher than this threshold, and the corresponding safety index\n               is also higher than the safety_index_threshold``, that feature will be considered admissible.  The\n               relevance index is the x-axis of the Fair Infogram. Default is -1 which gets set to 0.1.\n               Defaults to ``-1.0``.\n        :type relevance_index_threshold: float\n        :param safety_index_threshold: A number between 0 and 1 representing a threshold for the safety index,\n               defaulting to 0.1.  This is only used when protected_columns is set by the user.  For a specific feature,\n               if the safety index value is higher than this threshold, and the corresponding relevance index is also\n               higher than the relevance_index_threshold, that feature will be considered admissible.  The safety index\n               is the y-axis of the Fair Infogram. Default is -1 which gets set to 0.1.\n               Defaults to ``-1.0``.\n        :type safety_index_threshold: float\n        :param data_fraction: The fraction of training frame to use to build the infogram model. Defaults to 1.0, and\n               any value greater than 0 and less than or equal to 1.0 is acceptable.\n               Defaults to ``1.0``.\n        :type data_fraction: float\n        :param top_n_features: An integer specifying the number of columns to evaluate in the infogram.  The columns are\n               ranked by variable importance, and the top N are evaluated.  Defaults to 50.\n               Defaults to ``50``.\n        :type top_n_features: int\n        '
        super(H2OInfogram, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.validation_frame = validation_frame
        self.seed = seed
        self.keep_cross_validation_models = keep_cross_validation_models
        self.keep_cross_validation_predictions = keep_cross_validation_predictions
        self.keep_cross_validation_fold_assignment = keep_cross_validation_fold_assignment
        self.nfolds = nfolds
        self.fold_assignment = fold_assignment
        self.fold_column = fold_column
        self.response_column = response_column
        self.ignored_columns = ignored_columns
        self.ignore_const_cols = ignore_const_cols
        self.score_each_iteration = score_each_iteration
        self.offset_column = offset_column
        self.weights_column = weights_column
        self.standardize = standardize
        self.distribution = distribution
        self.plug_values = plug_values
        self.max_iterations = max_iterations
        self.stopping_rounds = stopping_rounds
        self.stopping_metric = stopping_metric
        self.stopping_tolerance = stopping_tolerance
        self.balance_classes = balance_classes
        self.class_sampling_factors = class_sampling_factors
        self.max_after_balance_size = max_after_balance_size
        self.max_runtime_secs = max_runtime_secs
        self.custom_metric_func = custom_metric_func
        self.auc_type = auc_type
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params
        self.protected_columns = protected_columns
        self.total_information_threshold = total_information_threshold
        self.net_information_threshold = net_information_threshold
        self.relevance_index_threshold = relevance_index_threshold
        self.safety_index_threshold = safety_index_threshold
        self.data_fraction = data_fraction
        self.top_n_features = top_n_features
        self._parms['_rest_version'] = 3

    @property
    def training_frame(self):
        if False:
            while True:
                i = 10
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('training_frame')

    @training_frame.setter
    def training_frame(self, training_frame):
        if False:
            return 10
        self._parms['training_frame'] = H2OFrame._validate(training_frame, 'training_frame')

    @property
    def validation_frame(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Id of the validation data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('validation_frame')

    @validation_frame.setter
    def validation_frame(self, validation_frame):
        if False:
            while True:
                i = 10
        self._parms['validation_frame'] = H2OFrame._validate(validation_frame, 'validation_frame')

    @property
    def seed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Seed for pseudo random number generator (if applicable).\n\n        Type: ``int``, defaults to ``-1``.\n        '
        return self._parms.get('seed')

    @seed.setter
    def seed(self, seed):
        if False:
            while True:
                i = 10
        assert_is_type(seed, None, int)
        self._parms['seed'] = seed

    @property
    def keep_cross_validation_models(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether to keep the cross-validation models.\n\n        Type: ``bool``, defaults to ``True``.\n        '
        return self._parms.get('keep_cross_validation_models')

    @keep_cross_validation_models.setter
    def keep_cross_validation_models(self, keep_cross_validation_models):
        if False:
            print('Hello World!')
        assert_is_type(keep_cross_validation_models, None, bool)
        self._parms['keep_cross_validation_models'] = keep_cross_validation_models

    @property
    def keep_cross_validation_predictions(self):
        if False:
            while True:
                i = 10
        '\n        Whether to keep the predictions of the cross-validation models.\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('keep_cross_validation_predictions')

    @keep_cross_validation_predictions.setter
    def keep_cross_validation_predictions(self, keep_cross_validation_predictions):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(keep_cross_validation_predictions, None, bool)
        self._parms['keep_cross_validation_predictions'] = keep_cross_validation_predictions

    @property
    def keep_cross_validation_fold_assignment(self):
        if False:
            while True:
                i = 10
        '\n        Whether to keep the cross-validation fold assignment.\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('keep_cross_validation_fold_assignment')

    @keep_cross_validation_fold_assignment.setter
    def keep_cross_validation_fold_assignment(self, keep_cross_validation_fold_assignment):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(keep_cross_validation_fold_assignment, None, bool)
        self._parms['keep_cross_validation_fold_assignment'] = keep_cross_validation_fold_assignment

    @property
    def nfolds(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Number of folds for K-fold cross-validation (0 to disable or >= 2).\n\n        Type: ``int``, defaults to ``0``.\n        '
        return self._parms.get('nfolds')

    @nfolds.setter
    def nfolds(self, nfolds):
        if False:
            i = 10
            return i + 15
        assert_is_type(nfolds, None, int)
        self._parms['nfolds'] = nfolds

    @property
    def fold_assignment(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Cross-validation fold assignment scheme, if fold_column is not specified. The \'Stratified\' option will stratify\n        the folds based on the response variable, for classification problems.\n\n        Type: ``Literal["auto", "random", "modulo", "stratified"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('fold_assignment')

    @fold_assignment.setter
    def fold_assignment(self, fold_assignment):
        if False:
            i = 10
            return i + 15
        assert_is_type(fold_assignment, None, Enum('auto', 'random', 'modulo', 'stratified'))
        self._parms['fold_assignment'] = fold_assignment

    @property
    def fold_column(self):
        if False:
            print('Hello World!')
        '\n        Column with cross-validation fold index assignment per observation.\n\n        Type: ``str``.\n        '
        return self._parms.get('fold_column')

    @fold_column.setter
    def fold_column(self, fold_column):
        if False:
            i = 10
            return i + 15
        assert_is_type(fold_column, None, str)
        self._parms['fold_column'] = fold_column

    @property
    def response_column(self):
        if False:
            return 10
        '\n        Response variable column.\n\n        Type: ``str``.\n        '
        return self._parms.get('response_column')

    @response_column.setter
    def response_column(self, response_column):
        if False:
            while True:
                i = 10
        assert_is_type(response_column, None, str)
        self._parms['response_column'] = response_column

    @property
    def ignored_columns(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Names of columns to ignore for training.\n\n        Type: ``List[str]``.\n        '
        return self._parms.get('ignored_columns')

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        if False:
            return 10
        assert_is_type(ignored_columns, None, [str])
        self._parms['ignored_columns'] = ignored_columns

    @property
    def ignore_const_cols(self):
        if False:
            i = 10
            return i + 15
        '\n        Ignore constant columns.\n\n        Type: ``bool``, defaults to ``True``.\n        '
        return self._parms.get('ignore_const_cols')

    @ignore_const_cols.setter
    def ignore_const_cols(self, ignore_const_cols):
        if False:
            print('Hello World!')
        assert_is_type(ignore_const_cols, None, bool)
        self._parms['ignore_const_cols'] = ignore_const_cols

    @property
    def score_each_iteration(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether to score during each iteration of model training.\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('score_each_iteration')

    @score_each_iteration.setter
    def score_each_iteration(self, score_each_iteration):
        if False:
            print('Hello World!')
        assert_is_type(score_each_iteration, None, bool)
        self._parms['score_each_iteration'] = score_each_iteration

    @property
    def offset_column(self):
        if False:
            print('Hello World!')
        '\n        Offset column. This will be added to the combination of columns before applying the link function.\n\n        Type: ``str``.\n        '
        return self._parms.get('offset_column')

    @offset_column.setter
    def offset_column(self, offset_column):
        if False:
            print('Hello World!')
        assert_is_type(offset_column, None, str)
        self._parms['offset_column'] = offset_column

    @property
    def weights_column(self):
        if False:
            while True:
                i = 10
        '\n        Column with observation weights. Giving some observation a weight of zero is equivalent to excluding it from the\n        dataset; giving an observation a relative weight of 2 is equivalent to repeating that row twice. Negative\n        weights are not allowed. Note: Weights are per-row observation weights and do not increase the size of the data\n        frame. This is typically the number of times a row is repeated, but non-integer values are supported as well.\n        During training, rows with higher weights matter more, due to the larger loss function pre-factor. If you set\n        weight = 0 for a row, the returned prediction frame at that row is zero and this is incorrect. To get an\n        accurate prediction, remove all rows with weight == 0.\n\n        Type: ``str``.\n        '
        return self._parms.get('weights_column')

    @weights_column.setter
    def weights_column(self, weights_column):
        if False:
            i = 10
            return i + 15
        assert_is_type(weights_column, None, str)
        self._parms['weights_column'] = weights_column

    @property
    def standardize(self):
        if False:
            print('Hello World!')
        '\n        Standardize numeric columns to have zero mean and unit variance.\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('standardize')

    @standardize.setter
    def standardize(self, standardize):
        if False:
            i = 10
            return i + 15
        assert_is_type(standardize, None, bool)
        self._parms['standardize'] = standardize

    @property
    def distribution(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Distribution function\n\n        Type: ``Literal["auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace",\n        "quantile", "huber"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('distribution')

    @distribution.setter
    def distribution(self, distribution):
        if False:
            while True:
                i = 10
        assert_is_type(distribution, None, Enum('auto', 'bernoulli', 'multinomial', 'gaussian', 'poisson', 'gamma', 'tweedie', 'laplace', 'quantile', 'huber'))
        self._parms['distribution'] = distribution

    @property
    def plug_values(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Plug Values (a single row frame containing values that will be used to impute missing values of the\n        training/validation frame, use with conjunction missing_values_handling = PlugValues).\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('plug_values')

    @plug_values.setter
    def plug_values(self, plug_values):
        if False:
            return 10
        self._parms['plug_values'] = H2OFrame._validate(plug_values, 'plug_values')

    @property
    def max_iterations(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Maximum number of iterations.\n\n        Type: ``int``, defaults to ``0``.\n        '
        return self._parms.get('max_iterations')

    @max_iterations.setter
    def max_iterations(self, max_iterations):
        if False:
            return 10
        assert_is_type(max_iterations, None, int)
        self._parms['max_iterations'] = max_iterations

    @property
    def stopping_rounds(self):
        if False:
            while True:
                i = 10
        '\n        Early stopping based on convergence of stopping_metric. Stop if simple moving average of length k of the\n        stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)\n\n        Type: ``int``, defaults to ``0``.\n        '
        return self._parms.get('stopping_rounds')

    @stopping_rounds.setter
    def stopping_rounds(self, stopping_rounds):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(stopping_rounds, None, int)
        self._parms['stopping_rounds'] = stopping_rounds

    @property
    def stopping_metric(self):
        if False:
            while True:
                i = 10
        '\n        Metric to use for early stopping (AUTO: logloss for classification, deviance for regression and anomaly_score\n        for Isolation Forest). Note that custom and custom_increasing can only be used in GBM and DRF with the Python\n        client.\n\n        Type: ``Literal["auto", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "lift_top_group",\n        "misclassification", "mean_per_class_error", "custom", "custom_increasing"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('stopping_metric')

    @stopping_metric.setter
    def stopping_metric(self, stopping_metric):
        if False:
            print('Hello World!')
        assert_is_type(stopping_metric, None, Enum('auto', 'deviance', 'logloss', 'mse', 'rmse', 'mae', 'rmsle', 'auc', 'aucpr', 'lift_top_group', 'misclassification', 'mean_per_class_error', 'custom', 'custom_increasing'))
        self._parms['stopping_metric'] = stopping_metric

    @property
    def stopping_tolerance(self):
        if False:
            while True:
                i = 10
        '\n        Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at least this much)\n\n        Type: ``float``, defaults to ``0.001``.\n        '
        return self._parms.get('stopping_tolerance')

    @stopping_tolerance.setter
    def stopping_tolerance(self, stopping_tolerance):
        if False:
            return 10
        assert_is_type(stopping_tolerance, None, numeric)
        self._parms['stopping_tolerance'] = stopping_tolerance

    @property
    def balance_classes(self):
        if False:
            return 10
        '\n        Balance training data class counts via over/under-sampling (for imbalanced data).\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('balance_classes')

    @balance_classes.setter
    def balance_classes(self, balance_classes):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(balance_classes, None, bool)
        self._parms['balance_classes'] = balance_classes

    @property
    def class_sampling_factors(self):
        if False:
            i = 10
            return i + 15
        '\n        Desired over/under-sampling ratios per class (in lexicographic order). If not specified, sampling factors will\n        be automatically computed to obtain class balance during training. Requires balance_classes.\n\n        Type: ``List[float]``.\n        '
        return self._parms.get('class_sampling_factors')

    @class_sampling_factors.setter
    def class_sampling_factors(self, class_sampling_factors):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(class_sampling_factors, None, [float])
        self._parms['class_sampling_factors'] = class_sampling_factors

    @property
    def max_after_balance_size(self):
        if False:
            return 10
        '\n        Maximum relative size of the training data after balancing class counts (can be less than 1.0). Requires\n        balance_classes.\n\n        Type: ``float``, defaults to ``5.0``.\n        '
        return self._parms.get('max_after_balance_size')

    @max_after_balance_size.setter
    def max_after_balance_size(self, max_after_balance_size):
        if False:
            while True:
                i = 10
        assert_is_type(max_after_balance_size, None, float)
        self._parms['max_after_balance_size'] = max_after_balance_size

    @property
    def max_runtime_secs(self):
        if False:
            return 10
        '\n        Maximum allowed runtime in seconds for model training. Use 0 to disable.\n\n        Type: ``float``, defaults to ``0.0``.\n        '
        return self._parms.get('max_runtime_secs')

    @max_runtime_secs.setter
    def max_runtime_secs(self, max_runtime_secs):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(max_runtime_secs, None, numeric)
        self._parms['max_runtime_secs'] = max_runtime_secs

    @property
    def custom_metric_func(self):
        if False:
            return 10
        '\n        Reference to custom evaluation function, format: `language:keyName=funcName`\n\n        Type: ``str``.\n        '
        return self._parms.get('custom_metric_func')

    @custom_metric_func.setter
    def custom_metric_func(self, custom_metric_func):
        if False:
            i = 10
            return i + 15
        assert_is_type(custom_metric_func, None, str)
        self._parms['custom_metric_func'] = custom_metric_func

    @property
    def auc_type(self):
        if False:
            print('Hello World!')
        '\n        Set default multinomial AUC type.\n\n        Type: ``Literal["auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"]``, defaults to\n        ``"auto"``.\n        '
        return self._parms.get('auc_type')

    @auc_type.setter
    def auc_type(self, auc_type):
        if False:
            return 10
        assert_is_type(auc_type, None, Enum('auto', 'none', 'macro_ovr', 'weighted_ovr', 'macro_ovo', 'weighted_ovo'))
        self._parms['auc_type'] = auc_type

    @property
    def algorithm(self):
        if False:
            return 10
        '\n        Type of machine learning algorithm used to build the infogram. Options include \'AUTO\' (gbm), \'deeplearning\'\n        (Deep Learning with default parameters), \'drf\' (Random Forest with default parameters), \'gbm\' (GBM with default\n        parameters), \'glm\' (GLM with default parameters), or \'xgboost\' (if available, XGBoost with default parameters).\n\n        Type: ``Literal["auto", "deeplearning", "drf", "gbm", "glm", "xgboost"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('algorithm')

    @algorithm.setter
    def algorithm(self, algorithm):
        if False:
            i = 10
            return i + 15
        assert_is_type(algorithm, None, Enum('auto', 'deeplearning', 'drf', 'gbm', 'glm', 'xgboost'))
        self._parms['algorithm'] = algorithm

    @property
    def algorithm_params(self):
        if False:
            print('Hello World!')
        '\n        Customized parameters for the machine learning algorithm specified in the algorithm parameter.\n\n        Type: ``dict``.\n        '
        if self._parms.get('algorithm_params') != None:
            algorithm_params_dict = ast.literal_eval(self._parms.get('algorithm_params'))
            for k in algorithm_params_dict:
                if len(algorithm_params_dict[k]) == 1:
                    algorithm_params_dict[k] = algorithm_params_dict[k][0]
            return algorithm_params_dict
        else:
            return self._parms.get('algorithm_params')

    @algorithm_params.setter
    def algorithm_params(self, algorithm_params):
        if False:
            print('Hello World!')
        assert_is_type(algorithm_params, None, dict)
        if algorithm_params is not None and algorithm_params != '':
            for k in algorithm_params:
                if ('[' and ']') not in str(algorithm_params[k]):
                    algorithm_params[k] = [algorithm_params[k]]
            self._parms['algorithm_params'] = str(json.dumps(algorithm_params))
        else:
            self._parms['algorithm_params'] = None

    @property
    def protected_columns(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Columns that contain features that are sensitive and need to be protected (legally, or otherwise), if\n        applicable. These features (e.g. race, gender, etc) should not drive the prediction of the response.\n\n        Type: ``List[str]``.\n        '
        return self._parms.get('protected_columns')

    @protected_columns.setter
    def protected_columns(self, protected_columns):
        if False:
            while True:
                i = 10
        assert_is_type(protected_columns, None, [str])
        self._parms['protected_columns'] = protected_columns

    @property
    def total_information_threshold(self):
        if False:
            return 10
        '\n        A number between 0 and 1 representing a threshold for total information, defaulting to 0.1. For a specific\n        feature, if the total information is higher than this threshold, and the corresponding net information is also\n        higher than the threshold ``net_information_threshold``, that feature will be considered admissible. The total\n        information is the x-axis of the Core Infogram. Default is -1 which gets set to 0.1.\n\n        Type: ``float``, defaults to ``-1.0``.\n        '
        return self._parms.get('total_information_threshold')

    @total_information_threshold.setter
    def total_information_threshold(self, total_information_threshold):
        if False:
            print('Hello World!')
        if total_information_threshold <= -1:
            if self._parms['protected_columns'] is None:
                self._parms['total_information_threshold'] = 0.1
        elif self._parms['protected_columns'] is not None:
            warnings.warn('Should not set total_information_threshold for fair infogram runs.  Set relevance_index_threshold instead.  Using default of 0.1 if not set', RuntimeWarning)
        else:
            self._parms['total_information_threshold'] = total_information_threshold

    @property
    def net_information_threshold(self):
        if False:
            i = 10
            return i + 15
        '\n        A number between 0 and 1 representing a threshold for net information, defaulting to 0.1.  For a specific\n        feature, if the net information is higher than this threshold, and the corresponding total information is also\n        higher than the total_information_threshold, that feature will be considered admissible. The net information is\n        the y-axis of the Core Infogram. Default is -1 which gets set to 0.1.\n\n        Type: ``float``, defaults to ``-1.0``.\n        '
        return self._parms.get('net_information_threshold')

    @net_information_threshold.setter
    def net_information_threshold(self, net_information_threshold):
        if False:
            return 10
        if net_information_threshold <= -1:
            if self._parms['protected_columns'] is None:
                self._parms['net_information_threshold'] = 0.1
        elif self._parms['protected_columns'] is not None:
            warnings.warn('Should not set net_information_threshold for fair infogram runs.  Set safety_index_threshold instead.  Using default of 0.1 if not set', RuntimeWarning)
        else:
            self._parms['net_information_threshold'] = net_information_threshold

    @property
    def relevance_index_threshold(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A number between 0 and 1 representing a threshold for the relevance index, defaulting to 0.1.  This is only used\n        when ``protected_columns`` is set by the user.  For a specific feature, if the relevance index value is higher\n        than this threshold, and the corresponding safety index is also higher than the safety_index_threshold``, that\n        feature will be considered admissible.  The relevance index is the x-axis of the Fair Infogram. Default is -1\n        which gets set to 0.1.\n\n        Type: ``float``, defaults to ``-1.0``.\n        '
        return self._parms.get('relevance_index_threshold')

    @relevance_index_threshold.setter
    def relevance_index_threshold(self, relevance_index_threshold):
        if False:
            while True:
                i = 10
        if relevance_index_threshold <= -1:
            if self._parms['protected_columns'] is not None:
                self._parms['relevance_index_threshold'] = 0.1
        elif self._parms['protected_columns'] is not None:
            self._parms['relevance_index_threshold'] = relevance_index_threshold
        else:
            warnings.warn('Should not set relevance_index_threshold for core infogram runs.  Set total_information_threshold instead.  Using default of 0.1 if not set', RuntimeWarning)

    @property
    def safety_index_threshold(self):
        if False:
            while True:
                i = 10
        '\n        A number between 0 and 1 representing a threshold for the safety index, defaulting to 0.1.  This is only used\n        when protected_columns is set by the user.  For a specific feature, if the safety index value is higher than\n        this threshold, and the corresponding relevance index is also higher than the relevance_index_threshold, that\n        feature will be considered admissible.  The safety index is the y-axis of the Fair Infogram. Default is -1 which\n        gets set to 0.1.\n\n        Type: ``float``, defaults to ``-1.0``.\n        '
        return self._parms.get('safety_index_threshold')

    @safety_index_threshold.setter
    def safety_index_threshold(self, safety_index_threshold):
        if False:
            print('Hello World!')
        if safety_index_threshold <= -1:
            if self._parms['protected_columns'] is not None:
                self._parms['safety_index_threshold'] = 0.1
        elif self._parms['protected_columns'] is not None:
            self._parms['safety_index_threshold'] = safety_index_threshold
        else:
            warnings.warn('Should not set safety_index_threshold for core infogram runs.  Set net_information_threshold instead.  Using default of 0.1 if not set', RuntimeWarning)

    @property
    def data_fraction(self):
        if False:
            i = 10
            return i + 15
        '\n        The fraction of training frame to use to build the infogram model. Defaults to 1.0, and any value greater than 0\n        and less than or equal to 1.0 is acceptable.\n\n        Type: ``float``, defaults to ``1.0``.\n        '
        return self._parms.get('data_fraction')

    @data_fraction.setter
    def data_fraction(self, data_fraction):
        if False:
            print('Hello World!')
        assert_is_type(data_fraction, None, numeric)
        self._parms['data_fraction'] = data_fraction

    @property
    def top_n_features(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An integer specifying the number of columns to evaluate in the infogram.  The columns are ranked by variable\n        importance, and the top N are evaluated.  Defaults to 50.\n\n        Type: ``int``, defaults to ``50``.\n        '
        return self._parms.get('top_n_features')

    @top_n_features.setter
    def top_n_features(self, top_n_features):
        if False:
            while True:
                i = 10
        assert_is_type(top_n_features, None, int)
        self._parms['top_n_features'] = top_n_features

    def _extract_x_from_model(self):
        if False:
            while True:
                i = 10
        '\n        extract admissible features from an Infogram model.\n\n        :return: List of predictors that are considered admissible\n        '
        features = self._model_json.get('output', {}).get('admissible_features')
        if features is None:
            raise ValueError("model %s doesn't have any admissible features" % self.key)
        return set(features)

    def plot(self, train=True, valid=False, xval=False, figsize=(10, 10), title='Infogram', legend_on=False, server=False):
        if False:
            while True:
                i = 10
        '\n        Plot the infogram.  By default, it will plot the infogram calculated from training dataset.  \n        Note that the frame rel_cmi_frame contains the following columns:\n        - 0: predictor names\n        - 1: admissible \n        - 2: admissible index\n        - 3: relevance-index or total information\n        - 4: safety-index or net information, normalized from 0 to 1\n        - 5: safety-index or net information not normalized\n\n        :param train: True if infogram is generated from training dataset\n        :param valid: True if infogram is generated from validation dataset\n        :param xval: True if infogram is generated from cross-validation holdout dataset\n        :param figsize: size of infogram plot\n        :param title: string to denote title of the plot\n        :param legend_on: legend text is included if True\n        :param server: True will not generate plot, False will produce plot\n        :return: infogram plot if server=True or None if server=False\n        '
        plt = get_matplotlib_pyplot(server, raise_if_not_available=True)
        polycoll = get_polycollection(server, raise_if_not_available=True)
        if not can_use_numpy():
            raise ImportError('numpy is required for Infogram.')
        import numpy as np
        if train:
            rel_cmi_frame = self.get_admissible_score_frame()
            if rel_cmi_frame is None:
                raise H2OValueError('Cannot locate the H2OFrame containing the infogram data from training dataset.')
        if valid:
            rel_cmi_frame_valid = self.get_admissible_score_frame(valid=True)
            if rel_cmi_frame_valid is None:
                raise H2OValueError('Cannot locate the H2OFrame containing the infogram data from validation dataset.')
        if xval:
            rel_cmi_frame_xval = self.get_admissible_score_frame(xval=True)
            if rel_cmi_frame_xval is None:
                raise H2OValueError('Cannot locate the H2OFrame containing the infogram data from xval holdout dataset.')
        rel_cmi_frame_names = rel_cmi_frame.names
        x_label = rel_cmi_frame_names[3]
        y_label = rel_cmi_frame_names[4]
        ig_x_column = 3
        ig_y_column = 4
        index_of_admissible = 1
        features_column = 0
        if self.actual_params['protected_columns'] == None:
            x_thresh = self.actual_params['total_information_threshold']
            y_thresh = self.actual_params['net_information_threshold']
        else:
            x_thresh = self.actual_params['relevance_index_threshold']
            y_thresh = self.actual_params['safety_index_threshold']
        xmax = 1.1
        ymax = 1.1
        X = np.array(rel_cmi_frame[ig_x_column].as_data_frame(header=False, use_pandas=False)).astype(float).reshape((-1,))
        Y = np.array(rel_cmi_frame[ig_y_column].as_data_frame(header=False, use_pandas=False)).astype(float).reshape((-1,))
        features = np.array(rel_cmi_frame[features_column].as_data_frame(header=False, use_pandas=False)).reshape((-1,))
        admissible = np.array(rel_cmi_frame[index_of_admissible].as_data_frame(header=False, use_pandas=False)).astype(float).reshape((-1,))
        mask = admissible > 0
        if valid:
            X_valid = np.array(rel_cmi_frame_valid[ig_x_column].as_data_frame(header=False, use_pandas=False)).astype(float).reshape((-1,))
            Y_valid = np.array(rel_cmi_frame_valid[ig_y_column].as_data_frame(header=False, use_pandas=False)).astype(float).reshape((-1,))
            features_valid = np.array(rel_cmi_frame_valid[features_column].as_data_frame(header=False, use_pandas=False)).reshape((-1,))
            admissible_valid = np.array(rel_cmi_frame_valid[index_of_admissible].as_data_frame(header=False, use_pandas=False)).astype(float).reshape((-1,))
            mask_valid = admissible_valid > 0
        if xval:
            X_xval = np.array(rel_cmi_frame_xval[ig_x_column].as_data_frame(header=False, use_pandas=False)).astype(float).reshape((-1,))
            Y_xval = np.array(rel_cmi_frame_xval[ig_y_column].as_data_frame(header=False, use_pandas=False)).astype(float).reshape((-1,))
            features_xval = np.array(rel_cmi_frame_xval[features_column].as_data_frame(header=False, use_pandas=False)).reshape((-1,))
            admissible_xval = np.array(rel_cmi_frame_xval[index_of_admissible].as_data_frame(header=False, use_pandas=False)).astype(float).reshape((-1,))
            mask_xval = admissible_xval > 0
        plt.figure(figsize=figsize)
        plt.grid(True)
        plt.scatter(X, Y, zorder=10, c=np.where(mask, 'black', 'gray'), label='training data')
        if valid:
            plt.scatter(X_valid, Y_valid, zorder=10, marker=',', c=np.where(mask_valid, 'black', 'gray'), label='validation data')
        if xval:
            plt.scatter(X_xval, Y_xval, zorder=10, marker='v', c=np.where(mask_xval, 'black', 'gray'), label='xval holdout data')
        if legend_on:
            plt.legend(loc=2, fancybox=True, framealpha=0.5)
        plt.hlines(y_thresh, xmin=x_thresh, xmax=xmax, colors='red', linestyle='dashed')
        plt.vlines(x_thresh, ymin=y_thresh, ymax=ymax, colors='red', linestyle='dashed')
        plt.gca().add_collection(polycoll(verts=[[(0, 0), (0, ymax), (x_thresh, ymax), (x_thresh, y_thresh), (xmax, y_thresh), (xmax, 0)]], color='#CC663E', alpha=0.1, zorder=5))
        for i in mask.nonzero()[0]:
            plt.annotate(features[i], (X[i], Y[i]), xytext=(0, -10), textcoords='offset points', horizontalalignment='center', verticalalignment='top', color='blue')
        if valid:
            for i in mask_valid.nonzero()[0]:
                plt.annotate(features_valid[i], (X_valid[i], Y_valid[i]), xytext=(0, -10), textcoords='offset points', horizontalalignment='center', verticalalignment='top', color='magenta')
        if xval:
            for i in mask_xval.nonzero()[0]:
                plt.annotate(features_xval[i], (X_xval[i], Y_xval[i]), xytext=(0, -10), textcoords='offset points', horizontalalignment='center', verticalalignment='top', color='green')
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        fig = plt.gcf()
        if not server:
            plt.show()
        return decorate_plot_result(figure=fig)

    def get_admissible_score_frame(self, valid=False, xval=False):
        if False:
            return 10
        '\n        Retreive admissible score frame which includes relevance and CMI information in an H2OFrame for training dataset by default\n        :param valid: return infogram info on validation dataset if True\n        :param xval: return infogram info on cross-validation hold outs if True\n        :return: H2OFrame\n        '
        keyString = self._model_json['output']['admissible_score_key']
        if valid:
            keyString = self._model_json['output']['admissible_score_key_valid']
        elif xval:
            keyString = self._model_json['output']['admissible_score_key_xval']
        if keyString is None:
            return None
        else:
            return h2o.get_frame(keyString['name'])

    def get_admissible_features(self):
        if False:
            i = 10
            return i + 15
        '\n        :return: a list of predictor that are considered admissible\n        '
        if self._model_json['output']['admissible_features'] is None:
            return None
        else:
            return self._model_json['output']['admissible_features']

    def get_admissible_relevance(self):
        if False:
            i = 10
            return i + 15
        '\n        :return: a list of relevance (variable importance) for admissible attributes\n        '
        if self._model_json['output']['admissible_relevance'] is None:
            return None
        else:
            return self._model_json['output']['admissible_relevance']

    def get_admissible_cmi(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :return: a list of the normalized CMI of admissible attributes\n        '
        if self._model_json['output']['admissible_cmi'] is None:
            return None
        else:
            return self._model_json['output']['admissible_cmi']

    def get_admissible_cmi_raw(self):
        if False:
            print('Hello World!')
        '\n        :return: a list of raw cmi of admissible attributes \n        '
        if self._model_json['output']['admissible_cmi_raw'] is None:
            return None
        else:
            return self._model_json['output']['admissible_cmi_raw']

    def get_all_predictor_relevance(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get relevance of all predictors\n        :return: two tuples, first one is predictor names and second one is relevance\n        '
        if self._model_json['output']['all_predictor_names'] is None:
            return None
        else:
            return (self._model_json['output']['all_predictor_names'], self._model_json['output']['relevance'])

    def get_all_predictor_cmi(self):
        if False:
            i = 10
            return i + 15
        '\n        Get normalized CMI of all predictors.\n        :return: two tuples, first one is predictor names and second one is cmi\n        '
        if self._model_json['output']['all_predictor_names'] is None:
            return None
        else:
            return (self._model_json['output']['all_predictor_names'], self._model_json['output']['cmi'])

    def get_all_predictor_cmi_raw(self):
        if False:
            return 10
        '\n        Get raw CMI of all predictors.\n        :return: two tuples, first one is predictor names and second one is cmi\n        '
        if self._model_json['output']['all_predictor_names'] is None:
            return None
        else:
            return (self._model_json['output']['all_predictor_names'], self._model_json['output']['cmi_raw'])

    def train(self, x=None, y=None, training_frame=None, verbose=False, **kwargs):
        if False:
            return 10
        sup = super(self.__class__, self)

        def extend_parms(parms):
            if False:
                return 10
            if parms['data_fraction'] is not None:
                assert_is_type(parms['data_fraction'], numeric)
                assert parms['data_fraction'] > 0 and parms['data_fraction'] <= 1, 'data_fraction should exceed 0 and <= 1.'
        parms = sup._make_parms(x, y, training_frame, extend_parms_fn=extend_parms, **kwargs)
        sup._train(parms, verbose=verbose)
        return self

    @staticmethod
    def _train_and_get_models(model_class, x, y, train, **kwargs):
        if False:
            i = 10
            return i + 15
        from h2o.automl import H2OAutoML
        from h2o.grid import H2OGridSearch
        model = model_class(**kwargs)
        model.train(x, y, train)
        if model_class is H2OAutoML:
            return [h2o.get_model(m[0]) for m in model.leaderboard['model_id'].as_data_frame(False, False)]
        elif model_class is H2OGridSearch:
            return [h2o.get_model(m) for m in model.model_ids]
        else:
            return [model]

    def train_subset_models(self, model_class, y, training_frame, test_frame, protected_columns=None, reference=None, favorable_class=None, feature_selection_metrics=None, metric='euclidean', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Train models using different feature subsets selected by infogram.\n\n        :param model_class: H2O Estimator class, H2OAutoML, or H2OGridSearch\n        :param y: response column\n        :param training_frame: training frame\n        :param test_frame: test frame\n        :param protected_columns: List of categorical columns that contain sensitive information\n                                  such as race, gender, age etc.\n        :param reference: List of values corresponding to a reference for each protected columns.\n                          If set to ``None``, it will use the biggest group as the reference.\n        :param favorable_class: Positive/favorable outcome class of the response.\n        :param feature_selection_metrics: column names from infogram\'s admissible score frame that are used\n                                          for the feature subset selection. Defaults to ``safety_index`` for fair infogram\n                                          and ``admissible_index`` for the core infogram.\n        :param metric: metric to combine information from the columns specified in feature_selection_metrics. Can be one\n                       of "euclidean", "manhattan", "maximum", or a function with that takes the admissible score frame\n                       and feature_selection_metrics and produces a single column.\n        :param kwargs: Arguments passed to the constructor of the model_class\n        :return: H2OFrame\n\n        :examples:\n        >>> from h2o.estimators import H2OGradientBoostingEstimator, H2OInfogram\n        >>> data = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/admissibleml_test/taiwan_credit_card_uci.csv")\n        >>> x = [\'LIMIT_BAL\', \'AGE\', \'PAY_0\', \'PAY_2\', \'PAY_3\', \'PAY_4\', \'PAY_5\', \'PAY_6\', \'BILL_AMT1\', \'BILL_AMT2\', \'BILL_AMT3\',\n        >>>      \'BILL_AMT4\', \'BILL_AMT5\', \'BILL_AMT6\', \'PAY_AMT1\', \'PAY_AMT2\', \'PAY_AMT3\', \'PAY_AMT4\', \'PAY_AMT5\', \'PAY_AMT6\']\n        >>> y = "default payment next month"\n        >>> protected_columns = [\'SEX\', \'EDUCATION\']\n        >>>\n        >>> for c in [y] + protected_columns:\n        >>>     data[c] = data[c].asfactor()\n        >>>\n        >>> train, test = data.split_frame([0.8])\n        >>>\n        >>> reference = ["1", "2"]  # university educated single man\n        >>> favorable_class = "0"  # no default next month\n        >>>\n        >>> ig = H2OInfogram(protected_columns=protected_columns)\n        >>> ig.train(x, y, training_frame=train)\n        >>>\n        >>> ig.train_subset_models(H2OGradientBoostingEstimator, y, train, test, protected_columns, reference, favorable_class)\n        '
        from h2o import H2OFrame, make_leaderboard
        from h2o.explanation import disparate_analysis
        from h2o.utils.typechecks import assert_is_type
        assert hasattr(model_class, 'train')
        assert_is_type(y, str)
        assert_is_type(training_frame, H2OFrame)
        score = self.get_admissible_score_frame()
        if feature_selection_metrics is None:
            if 'safety_index' in score.columns:
                feature_selection_metrics = ['safety_index']
            else:
                feature_selection_metrics = ['admissible_index']
        for fs_col in feature_selection_metrics:
            if fs_col not in score.columns:
                raise ValueError("Column '{}' is not present in the admissible score frame.".format(fs_col))
        metrics = dict(euclidean=lambda fr, fs_metrics: (fr[:, fs_metrics] ** 2).sum(axis=1).sqrt(), manhattan=lambda fr, fs_metrics: fr[:, fs_metrics].abs().sum(axis=1), maximum=lambda fr, fs_metrics: fr[:, fs_metrics].apply(lambda row: row.max(), axis=1))
        metric_fn = metric
        if not callable(metric) and metric.lower() not in metrics.keys():
            raise ValueError("Metric '{}' is not supported!".format(metric.lower()))
        if not callable(metric):
            metric_fn = metrics.get(metric.lower())
        if len(feature_selection_metrics) == 1:
            score['sort_metric'] = score[:, feature_selection_metrics]
        else:
            score['sort_metric'] = metric_fn(score, feature_selection_metrics)
        score = score.sort('sort_metric', False)
        cols = [x[0] for x in score['column'].as_data_frame(False, False)]
        subsets = [cols[0:i] for i in range(1, len(cols) + 1)]
        models = []
        for x in subsets:
            models.extend(self._train_and_get_models(model_class, x, y, training_frame, **kwargs))
        if protected_columns is None or len(protected_columns) == 0:
            return make_leaderboard(models, leaderboard_frame=test_frame)
        return disparate_analysis(models, test_frame, protected_columns, reference, favorable_class)