import ast
import json
import warnings
import h2o
from h2o.base import Keyed
from h2o.exceptions import H2OResponseError, H2ODeprecationWarning
from h2o.grid import H2OGridSearch
from h2o.job import H2OJob
from h2o.utils.shared_utils import quoted
from h2o.utils.typechecks import is_type
from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OStackedEnsembleEstimator(H2OEstimator):
    """
    Stacked Ensemble

    Builds a stacked ensemble (aka "super learner") machine learning method that uses two
    or more H2O learning algorithms to improve predictive performance. It is a loss-based
    supervised learning method that finds the optimal combination of a collection of prediction
    algorithms.This method supports regression and binary classification.

    :examples:

    >>> import h2o
    >>> h2o.init()
    >>> from h2o.estimators.random_forest import H2ORandomForestEstimator
    >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator
    >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
    >>> col_types = ["numeric", "numeric", "numeric", "enum",
    ...              "enum", "numeric", "numeric", "numeric", "numeric"]
    >>> data = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv", col_types=col_types)
    >>> train, test = data.split_frame(ratios=[.8], seed=1)
    >>> x = ["CAPSULE","GLEASON","RACE","DPROS","DCAPS","PSA","VOL"]
    >>> y = "AGE"
    >>> nfolds = 5
    >>> gbm = H2OGradientBoostingEstimator(nfolds=nfolds,
    ...                                    fold_assignment="Modulo",
    ...                                    keep_cross_validation_predictions=True)
    >>> gbm.train(x=x, y=y, training_frame=train)
    >>> rf = H2ORandomForestEstimator(nfolds=nfolds,
    ...                               fold_assignment="Modulo",
    ...                               keep_cross_validation_predictions=True)
    >>> rf.train(x=x, y=y, training_frame=train)
    >>> stack = H2OStackedEnsembleEstimator(model_id="ensemble",
    ...                                     training_frame=train,
    ...                                     validation_frame=test,
    ...                                     base_models=[gbm.model_id, rf.model_id])
    >>> stack.train(x=x, y=y, training_frame=train, validation_frame=test)
    >>> stack.model_performance()
    """
    algo = 'stackedensemble'
    supervised_learning = True
    _options_ = {'model_extensions': ['h2o.model.extensions.Fairness', 'h2o.model.extensions.Contributions']}

    def __init__(self, model_id=None, training_frame=None, response_column=None, validation_frame=None, blending_frame=None, base_models=[], metalearner_algorithm='auto', metalearner_nfolds=0, metalearner_fold_assignment=None, metalearner_fold_column=None, metalearner_params=None, metalearner_transform='none', max_runtime_secs=0.0, weights_column=None, offset_column=None, custom_metric_func=None, seed=-1, score_training_samples=10000, keep_levelone_frame=False, export_checkpoints_dir=None, auc_type='auto'):
        if False:
            while True:
                i = 10
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param response_column: Response variable column.\n               Defaults to ``None``.\n        :type response_column: str, optional\n        :param validation_frame: Id of the validation data frame.\n               Defaults to ``None``.\n        :type validation_frame: Union[None, str, H2OFrame], optional\n        :param blending_frame: Frame used to compute the predictions that serve as the training frame for the\n               metalearner (triggers blending mode if provided)\n               Defaults to ``None``.\n        :type blending_frame: Union[None, str, H2OFrame], optional\n        :param base_models: List of models or grids (or their ids) to ensemble/stack together. Grids are expanded to\n               individual models. If not using blending frame, then models must have been cross-validated using nfolds >\n               1, and folds must be identical across models.\n               Defaults to ``[]``.\n        :type base_models: List[str]\n        :param metalearner_algorithm: Type of algorithm to use as the metalearner. Options include \'AUTO\' (GLM with non\n               negative weights; if validation_frame is present, a lambda search is performed), \'deeplearning\' (Deep\n               Learning with default parameters), \'drf\' (Random Forest with default parameters), \'gbm\' (GBM with default\n               parameters), \'glm\' (GLM with default parameters), \'naivebayes\' (NaiveBayes with default parameters), or\n               \'xgboost\' (if available, XGBoost with default parameters).\n               Defaults to ``"auto"``.\n        :type metalearner_algorithm: Literal["auto", "deeplearning", "drf", "gbm", "glm", "naivebayes", "xgboost"]\n        :param metalearner_nfolds: Number of folds for K-fold cross-validation of the metalearner algorithm (0 to\n               disable or >= 2).\n               Defaults to ``0``.\n        :type metalearner_nfolds: int\n        :param metalearner_fold_assignment: Cross-validation fold assignment scheme for metalearner cross-validation.\n               Defaults to AUTO (which is currently set to Random). The \'Stratified\' option will stratify the folds\n               based on the response variable, for classification problems.\n               Defaults to ``None``.\n        :type metalearner_fold_assignment: Literal["auto", "random", "modulo", "stratified"], optional\n        :param metalearner_fold_column: Column with cross-validation fold index assignment per observation for cross-\n               validation of the metalearner.\n               Defaults to ``None``.\n        :type metalearner_fold_column: str, optional\n        :param metalearner_params: Parameters for metalearner algorithm\n               Defaults to ``None``.\n        :type metalearner_params: dict, optional\n        :param metalearner_transform: Transformation used for the level one frame.\n               Defaults to ``"none"``.\n        :type metalearner_transform: Literal["none", "logit"]\n        :param max_runtime_secs: Maximum allowed runtime in seconds for model training. Use 0 to disable.\n               Defaults to ``0.0``.\n        :type max_runtime_secs: float\n        :param weights_column: Column with observation weights. Giving some observation a weight of zero is equivalent\n               to excluding it from the dataset; giving an observation a relative weight of 2 is equivalent to repeating\n               that row twice. Negative weights are not allowed. Note: Weights are per-row observation weights and do\n               not increase the size of the data frame. This is typically the number of times a row is repeated, but\n               non-integer values are supported as well. During training, rows with higher weights matter more, due to\n               the larger loss function pre-factor. If you set weight = 0 for a row, the returned prediction frame at\n               that row is zero and this is incorrect. To get an accurate prediction, remove all rows with weight == 0.\n               Defaults to ``None``.\n        :type weights_column: str, optional\n        :param offset_column: Offset column. This will be added to the combination of columns before applying the link\n               function.\n               Defaults to ``None``.\n        :type offset_column: str, optional\n        :param custom_metric_func: Reference to custom evaluation function, format: `language:keyName=funcName`\n               Defaults to ``None``.\n        :type custom_metric_func: str, optional\n        :param seed: Seed for random numbers; passed through to the metalearner algorithm. Defaults to -1 (time-based\n               random number)\n               Defaults to ``-1``.\n        :type seed: int\n        :param score_training_samples: Specify the number of training set samples for scoring. The value must be >= 0.\n               To use all training samples, enter 0.\n               Defaults to ``10000``.\n        :type score_training_samples: int\n        :param keep_levelone_frame: Keep level one frame used for metalearner training.\n               Defaults to ``False``.\n        :type keep_levelone_frame: bool\n        :param export_checkpoints_dir: Automatically export generated models to this directory.\n               Defaults to ``None``.\n        :type export_checkpoints_dir: str, optional\n        :param auc_type: Set default multinomial AUC type.\n               Defaults to ``"auto"``.\n        :type auc_type: Literal["auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"]\n        '
        super(H2OStackedEnsembleEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.response_column = response_column
        self.validation_frame = validation_frame
        self.blending_frame = blending_frame
        self.base_models = base_models
        self.metalearner_algorithm = metalearner_algorithm
        self.metalearner_nfolds = metalearner_nfolds
        self.metalearner_fold_assignment = metalearner_fold_assignment
        self.metalearner_fold_column = metalearner_fold_column
        self.metalearner_params = metalearner_params
        self.metalearner_transform = metalearner_transform
        self.max_runtime_secs = max_runtime_secs
        self.weights_column = weights_column
        self.offset_column = offset_column
        self.custom_metric_func = custom_metric_func
        self.seed = seed
        self.score_training_samples = score_training_samples
        self.keep_levelone_frame = keep_levelone_frame
        self.export_checkpoints_dir = export_checkpoints_dir
        self.auc_type = auc_type
        self._parms['_rest_version'] = 99

    @property
    def training_frame(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, valid = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=1,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1,\n        ...                                           metalearner_fold_assignment="Random")\n        >>> stack_blend.train(x=x, y=y, training_frame=train, validation_frame=valid)\n        >>> stack_blend.model_performance(blend).auc()\n        '
        return self._parms.get('training_frame')

    @training_frame.setter
    def training_frame(self, training_frame):
        if False:
            while True:
                i = 10
        self._parms['training_frame'] = H2OFrame._validate(training_frame, 'training_frame')

    @property
    def response_column(self):
        if False:
            return 10
        '\n        Response variable column.\n\n        Type: ``str``.\n        '
        return self._parms.get('response_column')

    @response_column.setter
    def response_column(self, response_column):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(response_column, None, str)
        self._parms['response_column'] = response_column

    @property
    def validation_frame(self):
        if False:
            return 10
        '\n        Id of the validation data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, valid = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3 \n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=1,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1,\n        ...                                           metalearner_fold_assignment="Random")\n        >>> stack_blend.train(x=x, y=y, training_frame=train, validation_frame=valid)\n        >>> stack_blend.model_performance(blend).auc()\n        '
        return self._parms.get('validation_frame')

    @validation_frame.setter
    def validation_frame(self, validation_frame):
        if False:
            i = 10
            return i + 15
        self._parms['validation_frame'] = H2OFrame._validate(validation_frame, 'validation_frame')

    @property
    def blending_frame(self):
        if False:
            i = 10
            return i + 15
        '\n        Frame used to compute the predictions that serve as the training frame for the metalearner (triggers blending\n        mode if provided)\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, blend = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=10,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1)\n        >>> stack_blend.train(x=x, y=y, training_frame=train, blending_frame=blend)\n        >>> stack_blend.model_performance(blend).auc()\n        '
        return self._parms.get('blending_frame')

    @blending_frame.setter
    def blending_frame(self, blending_frame):
        if False:
            i = 10
            return i + 15
        self._parms['blending_frame'] = H2OFrame._validate(blending_frame, 'blending_frame')

    @property
    def base_models(self):
        if False:
            i = 10
            return i + 15
        '\n        List of models or grids (or their ids) to ensemble/stack together. Grids are expanded to individual models. If\n        not using blending frame, then models must have been cross-validated using nfolds > 1, and folds must be\n        identical across models.\n\n        Type: ``List[str]``, defaults to ``[]``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> col_types = ["numeric", "numeric", "numeric", "enum",\n        ...              "enum", "numeric", "numeric", "numeric", "numeric"]\n        >>> data = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv", col_types=col_types)\n        >>> train, test = data.split_frame(ratios=[.8], seed=1)\n        >>> x = ["CAPSULE","GLEASON","RACE","DPROS","DCAPS","PSA","VOL"]\n        >>> y = "AGE"\n        >>> nfolds = 5\n        >>> gbm = H2OGradientBoostingEstimator(nfolds=nfolds,\n        ...                                    fold_assignment="Modulo",\n        ...                                    keep_cross_validation_predictions=True)\n        >>> gbm.train(x=x, y=y, training_frame=train)\n        >>> rf = H2ORandomForestEstimator(nfolds=nfolds,\n        ...                               fold_assignment="Modulo",\n        ...                               keep_cross_validation_predictions=True)\n        >>> rf.train(x=x, y=y, training_frame=train)\n        >>> stack = H2OStackedEnsembleEstimator(model_id="ensemble",\n        ...                                     training_frame=train,\n        ...                                     validation_frame=test,\n        ...                                     base_models=[gbm.model_id, rf.model_id])\n        >>> stack.train(x=x, y=y, training_frame=train, validation_frame=test)\n        >>> stack.model_performance()\n        '
        base_models = self.actual_params.get('base_models', [])
        base_models = [base_model['name'] for base_model in base_models]
        if len(base_models) == 0:
            base_models = self._parms.get('base_models')
        return base_models

    @base_models.setter
    def base_models(self, base_models):
        if False:
            return 10

        def _get_id(something):
            if False:
                print('Hello World!')
            if isinstance(something, Keyed):
                return something.key
            return something
        if not is_type(base_models, list):
            base_models = [base_models]
        if is_type(base_models, [H2OEstimator, H2OGridSearch, str]):
            base_models = [_get_id(b) for b in base_models]
            self._parms['base_models'] = base_models
        else:
            assert_is_type(base_models, None)

    @property
    def metalearner_algorithm(self):
        if False:
            return 10
        '\n        Type of algorithm to use as the metalearner. Options include \'AUTO\' (GLM with non negative weights; if\n        validation_frame is present, a lambda search is performed), \'deeplearning\' (Deep Learning with default\n        parameters), \'drf\' (Random Forest with default parameters), \'gbm\' (GBM with default parameters), \'glm\' (GLM with\n        default parameters), \'naivebayes\' (NaiveBayes with default parameters), or \'xgboost\' (if available, XGBoost with\n        default parameters).\n\n        Type: ``Literal["auto", "deeplearning", "drf", "gbm", "glm", "naivebayes", "xgboost"]``, defaults to ``"auto"``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, blend = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=1,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1,\n        ...                                           metalearner_algorithm="gbm")\n        >>> stack_blend.train(x=x, y=y, training_frame=train, blending_frame=blend)\n        >>> stack_blend.model_performance(blend).auc()\n        '
        return self._parms.get('metalearner_algorithm')

    @metalearner_algorithm.setter
    def metalearner_algorithm(self, metalearner_algorithm):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(metalearner_algorithm, None, Enum('auto', 'deeplearning', 'drf', 'gbm', 'glm', 'naivebayes', 'xgboost'))
        self._parms['metalearner_algorithm'] = metalearner_algorithm

    @property
    def metalearner_nfolds(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Number of folds for K-fold cross-validation of the metalearner algorithm (0 to disable or >= 2).\n\n        Type: ``int``, defaults to ``0``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, blend = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=1,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1,\n        ...                                           metalearner_nfolds=3)\n        >>> stack_blend.train(x=x, y=y, training_frame=train, blending_frame=blend)\n        >>> stack_blend.model_performance(blend).auc()\n        '
        return self._parms.get('metalearner_nfolds')

    @metalearner_nfolds.setter
    def metalearner_nfolds(self, metalearner_nfolds):
        if False:
            while True:
                i = 10
        assert_is_type(metalearner_nfolds, None, int)
        self._parms['metalearner_nfolds'] = metalearner_nfolds

    @property
    def metalearner_fold_assignment(self):
        if False:
            i = 10
            return i + 15
        '\n        Cross-validation fold assignment scheme for metalearner cross-validation.  Defaults to AUTO (which is currently\n        set to Random). The \'Stratified\' option will stratify the folds based on the response variable, for\n        classification problems.\n\n        Type: ``Literal["auto", "random", "modulo", "stratified"]``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, blend = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=1,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1,\n        ...                                           metalearner_fold_assignment="Random")\n        >>> stack_blend.train(x=x, y=y, training_frame=train, blending_frame=blend)\n        >>> stack_blend.model_performance(blend).auc()\n        '
        return self._parms.get('metalearner_fold_assignment')

    @metalearner_fold_assignment.setter
    def metalearner_fold_assignment(self, metalearner_fold_assignment):
        if False:
            return 10
        assert_is_type(metalearner_fold_assignment, None, Enum('auto', 'random', 'modulo', 'stratified'))
        self._parms['metalearner_fold_assignment'] = metalearner_fold_assignment

    @property
    def metalearner_fold_column(self):
        if False:
            return 10
        '\n        Column with cross-validation fold index assignment per observation for cross-validation of the metalearner.\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> train = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> test = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_test_5k.csv")\n        >>> fold_column = "fold_id"\n        >>> train[fold_column] = train.kfold_column(n_folds=3, seed=1)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> x.remove(fold_column)\n        >>> train[y] = train[y].asfactor()\n        >>> test[y] = test[y].asfactor()\n        >>> nfolds = 3\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=10,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                     metalearner_fold_column=fold_column,\n        ...                                     metalearner_params=dict(keep_cross_validation_models=True))\n        >>> stack.train(x=x, y=y, training_frame=train)\n        >>> stack.model_performance().auc()\n        '
        return self._parms.get('metalearner_fold_column')

    @metalearner_fold_column.setter
    def metalearner_fold_column(self, metalearner_fold_column):
        if False:
            while True:
                i = 10
        assert_is_type(metalearner_fold_column, None, str)
        self._parms['metalearner_fold_column'] = metalearner_fold_column

    @property
    def metalearner_params(self):
        if False:
            i = 10
            return i + 15
        '\n        Parameters for metalearner algorithm\n\n        Type: ``dict``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, blend = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> gbm_params = {"ntrees" : 100, "max_depth" : 6}\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=1,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           metalearner_algorithm="gbm",\n        ...                                           metalearner_params=gbm_params)\n        >>> stack_blend.train(x=x, y=y, training_frame=train, blending_frame=blend)\n        >>> stack_blend.model_performance(blend).auc()\n        '
        if self._parms.get('metalearner_params') != None:
            metalearner_params_dict = ast.literal_eval(self._parms.get('metalearner_params'))
            for k in metalearner_params_dict:
                if len(metalearner_params_dict[k]) == 1:
                    metalearner_params_dict[k] = metalearner_params_dict[k][0]
            return metalearner_params_dict
        else:
            return self._parms.get('metalearner_params')

    @metalearner_params.setter
    def metalearner_params(self, metalearner_params):
        if False:
            i = 10
            return i + 15
        assert_is_type(metalearner_params, None, dict)
        if metalearner_params is not None and metalearner_params != '':
            for k in metalearner_params:
                if ('[' and ']') not in str(metalearner_params[k]):
                    metalearner_params[k] = [metalearner_params[k]]
            self._parms['metalearner_params'] = str(json.dumps(metalearner_params))
        else:
            self._parms['metalearner_params'] = None

    @property
    def metalearner_transform(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transformation used for the level one frame.\n\n        Type: ``Literal["none", "logit"]``, defaults to ``"none"``.\n        '
        return self._parms.get('metalearner_transform')

    @metalearner_transform.setter
    def metalearner_transform(self, metalearner_transform):
        if False:
            i = 10
            return i + 15
        assert_is_type(metalearner_transform, None, Enum('none', 'logit'))
        self._parms['metalearner_transform'] = metalearner_transform

    @property
    def max_runtime_secs(self):
        if False:
            i = 10
            return i + 15
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
    def offset_column(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Offset column. This will be added to the combination of columns before applying the link function.\n\n        Type: ``str``.\n        '
        return self._parms.get('offset_column')

    @offset_column.setter
    def offset_column(self, offset_column):
        if False:
            i = 10
            return i + 15
        assert_is_type(offset_column, None, str)
        self._parms['offset_column'] = offset_column

    @property
    def custom_metric_func(self):
        if False:
            print('Hello World!')
        '\n        Reference to custom evaluation function, format: `language:keyName=funcName`\n\n        Type: ``str``.\n        '
        return self._parms.get('custom_metric_func')

    @custom_metric_func.setter
    def custom_metric_func(self, custom_metric_func):
        if False:
            return 10
        assert_is_type(custom_metric_func, None, str)
        self._parms['custom_metric_func'] = custom_metric_func

    @property
    def seed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Seed for random numbers; passed through to the metalearner algorithm. Defaults to -1 (time-based random number)\n\n        Type: ``int``, defaults to ``-1``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, blend = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=1,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1,\n        ...                                           metalearner_fold_assignment="Random")\n        >>> stack_blend.train(x=x, y=y, training_frame=train, blending_frame=blend)\n        >>> stack_blend.model_performance(blend).auc()\n        '
        return self._parms.get('seed')

    @seed.setter
    def seed(self, seed):
        if False:
            print('Hello World!')
        assert_is_type(seed, None, int)
        self._parms['seed'] = seed

    @property
    def score_training_samples(self):
        if False:
            i = 10
            return i + 15
        '\n        Specify the number of training set samples for scoring. The value must be >= 0. To use all training samples,\n        enter 0.\n\n        Type: ``int``, defaults to ``10000``.\n        '
        return self._parms.get('score_training_samples')

    @score_training_samples.setter
    def score_training_samples(self, score_training_samples):
        if False:
            print('Hello World!')
        assert_is_type(score_training_samples, None, int)
        self._parms['score_training_samples'] = score_training_samples

    @property
    def keep_levelone_frame(self):
        if False:
            print('Hello World!')
        '\n        Keep level one frame used for metalearner training.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, blend = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=1,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1,\n        ...                                           keep_levelone_frame=True)\n        >>> stack_blend.train(x=x, y=y, training_frame=train, blending_frame=blend)\n        >>> stack_blend.model_performance(blend).auc()\n        '
        return self._parms.get('keep_levelone_frame')

    @keep_levelone_frame.setter
    def keep_levelone_frame(self, keep_levelone_frame):
        if False:
            while True:
                i = 10
        assert_is_type(keep_levelone_frame, None, bool)
        self._parms['keep_levelone_frame'] = keep_levelone_frame

    @property
    def export_checkpoints_dir(self):
        if False:
            return 10
        '\n        Automatically export generated models to this directory.\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> import tempfile\n        >>> from os import listdir\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, blend = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> checkpoints_dir = tempfile.mkdtemp()\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=10,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1,\n        ...                                           export_checkpoints_dir=checkpoints_dir)\n        >>> stack_blend.train(x=x, y=y, training_frame=train, blending_frame=blend)\n        >>> len(listdir(checkpoints_dir))\n        '
        return self._parms.get('export_checkpoints_dir')

    @export_checkpoints_dir.setter
    def export_checkpoints_dir(self, export_checkpoints_dir):
        if False:
            while True:
                i = 10
        assert_is_type(export_checkpoints_dir, None, str)
        self._parms['export_checkpoints_dir'] = export_checkpoints_dir

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

    def metalearner(self):
        if False:
            i = 10
            return i + 15
        'Print the metalearner of an H2OStackedEnsembleEstimator.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, blend = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=10,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1,\n        ...                                           keep_levelone_frame=True)\n        >>> stack_blend.train(x=x, y=y, training_frame=train, blending_frame=blend)\n        >>> stack_blend.metalearner()\n        '

        def _get_item(self, key):
            if False:
                print('Hello World!')
            warnings.warn("The usage of stacked_ensemble.metalearner()['name'] will be deprecated. Metalearner now returns the metalearner object. If you need to get the 'name' please use stacked_ensemble.metalearner().model_id", H2ODeprecationWarning)
            if key == 'name':
                return self.model_id
            raise NotImplementedError
        model = self._model_json['output']
        if 'metalearner' in model and model['metalearner'] is not None:
            metalearner = h2o.get_model(model['metalearner']['name'])
            metalearner.__class__.__getitem__ = _get_item
            return metalearner
        print('No metalearner for this model')

    def levelone_frame_id(self):
        if False:
            return 10
        'Fetch the levelone_frame_id for an H2OStackedEnsembleEstimator.\n\n        :examples:\n\n        >>> from h2o.estimators.random_forest import H2ORandomForestEstimator\n        >>> from h2o.estimators.gbm import H2OGradientBoostingEstimator\n        >>> from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator\n        >>> higgs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")\n        >>> train, blend = higgs.split_frame(ratios = [.8], seed = 1234)\n        >>> x = train.columns\n        >>> y = "response"\n        >>> x.remove(y)\n        >>> train[y] = train[y].asfactor()\n        >>> blend[y] = blend[y].asfactor()\n        >>> nfolds = 3\n        >>> my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n        ...                                       ntrees=10,\n        ...                                       nfolds=nfolds,\n        ...                                       fold_assignment="Modulo",\n        ...                                       keep_cross_validation_predictions=True,\n        ...                                       seed=1)\n        >>> my_gbm.train(x=x, y=y, training_frame=train)\n        >>> my_rf = H2ORandomForestEstimator(ntrees=50,\n        ...                                  nfolds=nfolds,\n        ...                                  fold_assignment="Modulo",\n        ...                                  keep_cross_validation_predictions=True,\n        ...                                  seed=1)\n        >>> my_rf.train(x=x, y=y, training_frame=train)\n        >>> stack_blend = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf],\n        ...                                           seed=1,\n        ...                                           keep_levelone_frame=True)\n        >>> stack_blend.train(x=x, y=y, training_frame=train, blending_frame=blend)\n        >>> stack_blend.levelone_frame_id()\n        '
        model = self._model_json['output']
        if 'levelone_frame_id' in model and model['levelone_frame_id'] is not None:
            return model['levelone_frame_id']
        print('No levelone_frame_id for this model')

    def stacking_strategy(self):
        if False:
            for i in range(10):
                print('nop')
        model = self._model_json['output']
        if 'stacking_strategy' in model and model['stacking_strategy'] is not None:
            return model['stacking_strategy']
        print('No stacking strategy for this model')

    def train(self, x=None, y=None, training_frame=None, blending_frame=None, verbose=False, **kwargs):
        if False:
            i = 10
            return i + 15
        has_training_frame = training_frame is not None or self.training_frame is not None
        blending_frame = H2OFrame._validate(blending_frame, 'blending_frame', required=not has_training_frame)
        if not has_training_frame:
            training_frame = blending_frame
        sup = super(self.__class__, self)

        def extend_parms(parms):
            if False:
                while True:
                    i = 10
            if blending_frame is not None:
                parms['blending_frame'] = blending_frame
            if self.metalearner_fold_column is not None:
                parms['ignored_columns'].remove(quoted(self.metalearner_fold_column))
        parms = sup._make_parms(x, y, training_frame, extend_parms_fn=extend_parms, **kwargs)
        sup._train(parms, verbose=verbose)
        if self.metalearner() is None:
            raise H2OResponseError("Meta learner didn't get to be trained in time. Try increasing max_runtime_secs or setting it to 0 (unlimited).")
        return self