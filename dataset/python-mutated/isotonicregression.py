from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OIsotonicRegressionEstimator(H2OEstimator):
    """
    Isotonic Regression

    """
    algo = 'isotonicregression'
    supervised_learning = True

    def __init__(self, model_id=None, training_frame=None, validation_frame=None, response_column=None, ignored_columns=None, weights_column=None, out_of_bounds='na', custom_metric_func=None, nfolds=0, keep_cross_validation_models=True, keep_cross_validation_predictions=False, keep_cross_validation_fold_assignment=False, fold_assignment='auto', fold_column=None):
        if False:
            print('Hello World!')
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param validation_frame: Id of the validation data frame.\n               Defaults to ``None``.\n        :type validation_frame: Union[None, str, H2OFrame], optional\n        :param response_column: Response variable column.\n               Defaults to ``None``.\n        :type response_column: str, optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param weights_column: Column with observation weights. Giving some observation a weight of zero is equivalent\n               to excluding it from the dataset; giving an observation a relative weight of 2 is equivalent to repeating\n               that row twice. Negative weights are not allowed. Note: Weights are per-row observation weights and do\n               not increase the size of the data frame. This is typically the number of times a row is repeated, but\n               non-integer values are supported as well. During training, rows with higher weights matter more, due to\n               the larger loss function pre-factor. If you set weight = 0 for a row, the returned prediction frame at\n               that row is zero and this is incorrect. To get an accurate prediction, remove all rows with weight == 0.\n               Defaults to ``None``.\n        :type weights_column: str, optional\n        :param out_of_bounds: Method of handling values of X predictor that are outside of the bounds seen in training.\n               Defaults to ``"na"``.\n        :type out_of_bounds: Literal["na", "clip"]\n        :param custom_metric_func: Reference to custom evaluation function, format: `language:keyName=funcName`\n               Defaults to ``None``.\n        :type custom_metric_func: str, optional\n        :param nfolds: Number of folds for K-fold cross-validation (0 to disable or >= 2).\n               Defaults to ``0``.\n        :type nfolds: int\n        :param keep_cross_validation_models: Whether to keep the cross-validation models.\n               Defaults to ``True``.\n        :type keep_cross_validation_models: bool\n        :param keep_cross_validation_predictions: Whether to keep the predictions of the cross-validation models.\n               Defaults to ``False``.\n        :type keep_cross_validation_predictions: bool\n        :param keep_cross_validation_fold_assignment: Whether to keep the cross-validation fold assignment.\n               Defaults to ``False``.\n        :type keep_cross_validation_fold_assignment: bool\n        :param fold_assignment: Cross-validation fold assignment scheme, if fold_column is not specified. The\n               \'Stratified\' option will stratify the folds based on the response variable, for classification problems.\n               Defaults to ``"auto"``.\n        :type fold_assignment: Literal["auto", "random", "modulo", "stratified"]\n        :param fold_column: Column with cross-validation fold index assignment per observation.\n               Defaults to ``None``.\n        :type fold_column: str, optional\n        '
        super(H2OIsotonicRegressionEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.validation_frame = validation_frame
        self.response_column = response_column
        self.ignored_columns = ignored_columns
        self.weights_column = weights_column
        self.out_of_bounds = out_of_bounds
        self.custom_metric_func = custom_metric_func
        self.nfolds = nfolds
        self.keep_cross_validation_models = keep_cross_validation_models
        self.keep_cross_validation_predictions = keep_cross_validation_predictions
        self.keep_cross_validation_fold_assignment = keep_cross_validation_fold_assignment
        self.fold_assignment = fold_assignment
        self.fold_column = fold_column

    @property
    def training_frame(self):
        if False:
            return 10
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('training_frame')

    @training_frame.setter
    def training_frame(self, training_frame):
        if False:
            print('Hello World!')
        self._parms['training_frame'] = H2OFrame._validate(training_frame, 'training_frame')

    @property
    def validation_frame(self):
        if False:
            while True:
                i = 10
        '\n        Id of the validation data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('validation_frame')

    @validation_frame.setter
    def validation_frame(self, validation_frame):
        if False:
            return 10
        self._parms['validation_frame'] = H2OFrame._validate(validation_frame, 'validation_frame')

    @property
    def response_column(self):
        if False:
            while True:
                i = 10
        '\n        Response variable column.\n\n        Type: ``str``.\n        '
        return self._parms.get('response_column')

    @response_column.setter
    def response_column(self, response_column):
        if False:
            print('Hello World!')
        assert_is_type(response_column, None, str)
        self._parms['response_column'] = response_column

    @property
    def ignored_columns(self):
        if False:
            print('Hello World!')
        '\n        Names of columns to ignore for training.\n\n        Type: ``List[str]``.\n        '
        return self._parms.get('ignored_columns')

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        if False:
            return 10
        assert_is_type(ignored_columns, None, [str])
        self._parms['ignored_columns'] = ignored_columns

    @property
    def weights_column(self):
        if False:
            print('Hello World!')
        '\n        Column with observation weights. Giving some observation a weight of zero is equivalent to excluding it from the\n        dataset; giving an observation a relative weight of 2 is equivalent to repeating that row twice. Negative\n        weights are not allowed. Note: Weights are per-row observation weights and do not increase the size of the data\n        frame. This is typically the number of times a row is repeated, but non-integer values are supported as well.\n        During training, rows with higher weights matter more, due to the larger loss function pre-factor. If you set\n        weight = 0 for a row, the returned prediction frame at that row is zero and this is incorrect. To get an\n        accurate prediction, remove all rows with weight == 0.\n\n        Type: ``str``.\n        '
        return self._parms.get('weights_column')

    @weights_column.setter
    def weights_column(self, weights_column):
        if False:
            print('Hello World!')
        assert_is_type(weights_column, None, str)
        self._parms['weights_column'] = weights_column

    @property
    def out_of_bounds(self):
        if False:
            while True:
                i = 10
        '\n        Method of handling values of X predictor that are outside of the bounds seen in training.\n\n        Type: ``Literal["na", "clip"]``, defaults to ``"na"``.\n        '
        return self._parms.get('out_of_bounds')

    @out_of_bounds.setter
    def out_of_bounds(self, out_of_bounds):
        if False:
            while True:
                i = 10
        assert_is_type(out_of_bounds, None, Enum('na', 'clip'))
        self._parms['out_of_bounds'] = out_of_bounds

    @property
    def custom_metric_func(self):
        if False:
            return 10
        '\n        Reference to custom evaluation function, format: `language:keyName=funcName`\n\n        Type: ``str``.\n        '
        return self._parms.get('custom_metric_func')

    @custom_metric_func.setter
    def custom_metric_func(self, custom_metric_func):
        if False:
            print('Hello World!')
        assert_is_type(custom_metric_func, None, str)
        self._parms['custom_metric_func'] = custom_metric_func

    @property
    def nfolds(self):
        if False:
            i = 10
            return i + 15
        '\n        Number of folds for K-fold cross-validation (0 to disable or >= 2).\n\n        Type: ``int``, defaults to ``0``.\n        '
        return self._parms.get('nfolds')

    @nfolds.setter
    def nfolds(self, nfolds):
        if False:
            return 10
        assert_is_type(nfolds, None, int)
        self._parms['nfolds'] = nfolds

    @property
    def keep_cross_validation_models(self):
        if False:
            while True:
                i = 10
        '\n        Whether to keep the cross-validation models.\n\n        Type: ``bool``, defaults to ``True``.\n        '
        return self._parms.get('keep_cross_validation_models')

    @keep_cross_validation_models.setter
    def keep_cross_validation_models(self, keep_cross_validation_models):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(keep_cross_validation_models, None, bool)
        self._parms['keep_cross_validation_models'] = keep_cross_validation_models

    @property
    def keep_cross_validation_predictions(self):
        if False:
            print('Hello World!')
        '\n        Whether to keep the predictions of the cross-validation models.\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('keep_cross_validation_predictions')

    @keep_cross_validation_predictions.setter
    def keep_cross_validation_predictions(self, keep_cross_validation_predictions):
        if False:
            return 10
        assert_is_type(keep_cross_validation_predictions, None, bool)
        self._parms['keep_cross_validation_predictions'] = keep_cross_validation_predictions

    @property
    def keep_cross_validation_fold_assignment(self):
        if False:
            print('Hello World!')
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
    def fold_assignment(self):
        if False:
            i = 10
            return i + 15
        '\n        Cross-validation fold assignment scheme, if fold_column is not specified. The \'Stratified\' option will stratify\n        the folds based on the response variable, for classification problems.\n\n        Type: ``Literal["auto", "random", "modulo", "stratified"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('fold_assignment')

    @fold_assignment.setter
    def fold_assignment(self, fold_assignment):
        if False:
            while True:
                i = 10
        assert_is_type(fold_assignment, None, Enum('auto', 'random', 'modulo', 'stratified'))
        self._parms['fold_assignment'] = fold_assignment

    @property
    def fold_column(self):
        if False:
            return 10
        '\n        Column with cross-validation fold index assignment per observation.\n\n        Type: ``str``.\n        '
        return self._parms.get('fold_column')

    @fold_column.setter
    def fold_column(self, fold_column):
        if False:
            i = 10
            return i + 15
        assert_is_type(fold_column, None, str)
        self._parms['fold_column'] = fold_column