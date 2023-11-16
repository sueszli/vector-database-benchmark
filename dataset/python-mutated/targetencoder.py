from h2o.utils.metaclass import deprecated_params, deprecated_property
import h2o
import warnings
from h2o.exceptions import H2ODeprecationWarning
from h2o.utils.typechecks import U
from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OTargetEncoderEstimator(H2OEstimator):
    """
    TargetEncoder

    """
    algo = 'targetencoder'
    supervised_learning = True

    @deprecated_params({'k': 'inflection_point', 'f': 'smoothing', 'noise_level': 'noise'})
    def __init__(self, model_id=None, training_frame=None, fold_column=None, response_column=None, ignored_columns=None, columns_to_encode=None, keep_original_categorical_columns=True, blending=False, inflection_point=10.0, smoothing=20.0, data_leakage_handling='none', noise=0.01, seed=-1):
        if False:
            return 10
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param fold_column: Column with cross-validation fold index assignment per observation.\n               Defaults to ``None``.\n        :type fold_column: str, optional\n        :param response_column: Response variable column.\n               Defaults to ``None``.\n        :type response_column: str, optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param columns_to_encode: List of categorical columns or groups of categorical columns to encode. When groups of\n               columns are specified, each group is encoded as a single column (interactions are created internally).\n               Defaults to ``None``.\n        :type columns_to_encode: List[List[str]], optional\n        :param keep_original_categorical_columns: If true, the original non-encoded categorical features will remain in\n               the result frame.\n               Defaults to ``True``.\n        :type keep_original_categorical_columns: bool\n        :param blending: If true, enables blending of posterior probabilities (computed for a given categorical value)\n               with prior probabilities (computed on the entire set). This allows to mitigate the effect of categorical\n               values with small cardinality. The blending effect can be tuned using the `inflection_point` and\n               `smoothing` parameters.\n               Defaults to ``False``.\n        :type blending: bool\n        :param inflection_point: Inflection point of the sigmoid used to blend probabilities (see `blending` parameter).\n               For a given categorical value, if it appears less that `inflection_point` in a data sample, then the\n               influence of the posterior probability will be smaller than the prior.\n               Defaults to ``10.0``.\n        :type inflection_point: float\n        :param smoothing: Smoothing factor corresponds to the inverse of the slope at the inflection point on the\n               sigmoid used to blend probabilities (see `blending` parameter). If smoothing tends towards 0, then the\n               sigmoid used for blending turns into a Heaviside step function.\n               Defaults to ``20.0``.\n        :type smoothing: float\n        :param data_leakage_handling: Data leakage handling strategy used to generate the encoding. Supported options\n               are:\n               1) "none" (default) - no holdout, using the entire training frame.\n               2) "leave_one_out" - current row\'s response value is subtracted from the per-level frequencies pre-\n               calculated on the entire training frame.\n               3) "k_fold" - encodings for a fold are generated based on out-of-fold data.\n\n               Defaults to ``"none"``.\n        :type data_leakage_handling: Literal["leave_one_out", "k_fold", "none"]\n        :param noise: The amount of noise to add to the encoded column. Use 0 to disable noise, and -1 (=AUTO) to let\n               the algorithm determine a reasonable amount of noise.\n               Defaults to ``0.01``.\n        :type noise: float\n        :param seed: Seed used to generate the noise. By default, the seed is chosen randomly.\n               Defaults to ``-1``.\n        :type seed: int\n        '
        super(H2OTargetEncoderEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.fold_column = fold_column
        self.response_column = response_column
        self.ignored_columns = ignored_columns
        self.columns_to_encode = columns_to_encode
        self.keep_original_categorical_columns = keep_original_categorical_columns
        self.blending = blending
        self.inflection_point = inflection_point
        self.smoothing = smoothing
        self.data_leakage_handling = data_leakage_handling
        self.noise = noise
        self.seed = seed

    @property
    def training_frame(self):
        if False:
            return 10
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")\n        >>> predictors = ["home.dest", "cabin", "embarked"]\n        >>> response = "survived"\n        >>> titanic["survived"] = titanic["survived"].asfactor()\n        >>> fold_col = "kfold_column"\n        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)\n        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,\n        ...                                        smoothing=25,\n        ...                                        blending=True)\n        >>> titanic_te.train(x=predictors,\n        ...                  y=response,\n        ...                  training_frame=titanic)\n        >>> titanic_te\n        '
        return self._parms.get('training_frame')

    @training_frame.setter
    def training_frame(self, training_frame):
        if False:
            print('Hello World!')
        self._parms['training_frame'] = H2OFrame._validate(training_frame, 'training_frame')

    @property
    def fold_column(self):
        if False:
            return 10
        '\n        Column with cross-validation fold index assignment per observation.\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")\n        >>> predictors = ["home.dest", "cabin", "embarked"]\n        >>> response = "survived"\n        >>> titanic["survived"] = titanic["survived"].asfactor()\n        >>> fold_col = "kfold_column"\n        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)\n        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,\n        ...                                        smoothing=25,\n        ...                                        blending=True)\n        >>> titanic_te.train(x=predictors,\n        ...                  y=response,\n        ...                  training_frame=titanic)\n        >>> titanic_te\n        '
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
            i = 10
            return i + 15
        '\n        Response variable column.\n\n        Type: ``str``.\n        '
        return self._parms.get('response_column')

    @response_column.setter
    def response_column(self, response_column):
        if False:
            return 10
        assert_is_type(response_column, None, str)
        self._parms['response_column'] = response_column

    @property
    def ignored_columns(self):
        if False:
            return 10
        '\n        Names of columns to ignore for training.\n\n        Type: ``List[str]``.\n        '
        return self._parms.get('ignored_columns')

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        if False:
            i = 10
            return i + 15
        assert_is_type(ignored_columns, None, [str])
        self._parms['ignored_columns'] = ignored_columns

    @property
    def columns_to_encode(self):
        if False:
            while True:
                i = 10
        '\n        List of categorical columns or groups of categorical columns to encode. When groups of columns are specified,\n        each group is encoded as a single column (interactions are created internally).\n\n        Type: ``List[List[str]]``.\n        '
        return self._parms.get('columns_to_encode')

    @columns_to_encode.setter
    def columns_to_encode(self, columns_to_encode):
        if False:
            return 10
        assert_is_type(columns_to_encode, None, [U(str, [str])])
        if columns_to_encode:
            columns_to_encode = [[g] if isinstance(g, str) else g for g in columns_to_encode]
        self._parms['columns_to_encode'] = columns_to_encode

    @property
    def keep_original_categorical_columns(self):
        if False:
            while True:
                i = 10
        '\n        If true, the original non-encoded categorical features will remain in the result frame.\n\n        Type: ``bool``, defaults to ``True``.\n        '
        return self._parms.get('keep_original_categorical_columns')

    @keep_original_categorical_columns.setter
    def keep_original_categorical_columns(self, keep_original_categorical_columns):
        if False:
            while True:
                i = 10
        assert_is_type(keep_original_categorical_columns, None, bool)
        self._parms['keep_original_categorical_columns'] = keep_original_categorical_columns

    @property
    def blending(self):
        if False:
            print('Hello World!')
        '\n        If true, enables blending of posterior probabilities (computed for a given categorical value) with prior\n        probabilities (computed on the entire set). This allows to mitigate the effect of categorical values with small\n        cardinality. The blending effect can be tuned using the `inflection_point` and `smoothing` parameters.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")\n        >>> predictors = ["home.dest", "cabin", "embarked"]\n        >>> response = "survived"\n        >>> titanic["survived"] = titanic["survived"].asfactor()\n        >>> fold_col = "kfold_column"\n        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)\n        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,\n        ...                                        smoothing=25,\n        ...                                        blending=True)\n        >>> titanic_te.train(x=predictors,\n        ...                  y=response,\n        ...                  training_frame=titanic)\n        >>> titanic_te\n        '
        return self._parms.get('blending')

    @blending.setter
    def blending(self, blending):
        if False:
            while True:
                i = 10
        assert_is_type(blending, None, bool)
        self._parms['blending'] = blending

    @property
    def inflection_point(self):
        if False:
            print('Hello World!')
        '\n        Inflection point of the sigmoid used to blend probabilities (see `blending` parameter). For a given categorical\n        value, if it appears less that `inflection_point` in a data sample, then the influence of the posterior\n        probability will be smaller than the prior.\n\n        Type: ``float``, defaults to ``10.0``.\n\n        :examples:\n\n        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")\n        >>> predictors = ["home.dest", "cabin", "embarked"]\n        >>> response = "survived"\n        >>> titanic["survived"] = titanic["survived"].asfactor()\n        >>> fold_col = "kfold_column"\n        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)\n        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,\n        ...                                        smoothing=25,\n        ...                                        blending=True)\n        >>> titanic_te.train(x=predictors,\n        ...                  y=response,\n        ...                  training_frame=titanic)\n        >>> titanic_te\n        '
        return self._parms.get('inflection_point')

    @inflection_point.setter
    def inflection_point(self, inflection_point):
        if False:
            while True:
                i = 10
        assert_is_type(inflection_point, None, numeric)
        self._parms['inflection_point'] = inflection_point

    @property
    def smoothing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Smoothing factor corresponds to the inverse of the slope at the inflection point on the sigmoid used to blend\n        probabilities (see `blending` parameter). If smoothing tends towards 0, then the sigmoid used for blending turns\n        into a Heaviside step function.\n\n        Type: ``float``, defaults to ``20.0``.\n\n        :examples:\n\n        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")\n        >>> predictors = ["home.dest", "cabin", "embarked"]\n        >>> response = "survived"\n        >>> titanic["survived"] = titanic["survived"].asfactor()\n        >>> fold_col = "kfold_column"\n        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)\n        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,\n        ...                                        smoothing=25,\n        ...                                        blending=True)\n        >>> titanic_te.train(x=predictors,\n        ...                  y=response,\n        ...                  training_frame=titanic)\n        >>> titanic_te\n        '
        return self._parms.get('smoothing')

    @smoothing.setter
    def smoothing(self, smoothing):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(smoothing, None, numeric)
        self._parms['smoothing'] = smoothing

    @property
    def data_leakage_handling(self):
        if False:
            print('Hello World!')
        '\n        Data leakage handling strategy used to generate the encoding. Supported options are:\n        1) "none" (default) - no holdout, using the entire training frame.\n        2) "leave_one_out" - current row\'s response value is subtracted from the per-level frequencies pre-calculated on\n        the entire training frame.\n        3) "k_fold" - encodings for a fold are generated based on out-of-fold data.\n\n        Type: ``Literal["leave_one_out", "k_fold", "none"]``, defaults to ``"none"``.\n\n        :examples:\n\n        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")\n        >>> predictors = ["home.dest", "cabin", "embarked"]\n        >>> response = "survived"\n        >>> titanic["survived"] = titanic["survived"].asfactor()\n        >>> fold_col = "kfold_column"\n        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)\n        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,\n        ...                                        smoothing=25,\n        ...                                        data_leakage_handling="k_fold",\n        ...                                        blending=True)\n        >>> titanic_te.train(x=predictors,\n        ...                  y=response,\n        ...                  training_frame=titanic)\n        >>> titanic_te\n        '
        return self._parms.get('data_leakage_handling')

    @data_leakage_handling.setter
    def data_leakage_handling(self, data_leakage_handling):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(data_leakage_handling, None, Enum('leave_one_out', 'k_fold', 'none'))
        self._parms['data_leakage_handling'] = data_leakage_handling

    @property
    def noise(self):
        if False:
            return 10
        '\n        The amount of noise to add to the encoded column. Use 0 to disable noise, and -1 (=AUTO) to let the algorithm\n        determine a reasonable amount of noise.\n\n        Type: ``float``, defaults to ``0.01``.\n        '
        return self._parms.get('noise')

    @noise.setter
    def noise(self, noise):
        if False:
            return 10
        assert_is_type(noise, None, numeric)
        self._parms['noise'] = noise

    @property
    def seed(self):
        if False:
            i = 10
            return i + 15
        '\n        Seed used to generate the noise. By default, the seed is chosen randomly.\n\n        Type: ``int``, defaults to ``-1``.\n        '
        return self._parms.get('seed')

    @seed.setter
    def seed(self, seed):
        if False:
            while True:
                i = 10
        assert_is_type(seed, None, int)
        self._parms['seed'] = seed
    k = deprecated_property('k', inflection_point)
    f = deprecated_property('f', smoothing)
    noise_level = deprecated_property('noise_level', noise)

    def transform(self, frame, blending=None, inflection_point=None, smoothing=None, noise=None, as_training=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply transformation to `te_columns` based on the encoding maps generated during `train()` method call.\n\n        :param H2OFrame frame: the frame on which to apply the target encoding transformations.\n        :param boolean blending: If provided, this overrides the `blending` parameter on the model.\n        :param float inflection_point: If provided, this overrides the `inflection_point` parameter on the model.\n        :param float smoothing: If provided, this overrides the `smoothing` parameter on the model.\n        :param float noise: If provided, this overrides the amount of random noise added to the target encoding defined on the model, this helps prevent overfitting.\n        :param boolean as_training: Must be set to True when encoding the training frame. Defaults to False.\n\n        :examples:\n\n        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")\n        >>> predictors = ["home.dest", "cabin", "embarked"]\n        >>> response = "survived"\n        >>> titanic[response] = titanic[response].asfactor()\n        >>> fold_col = "kfold_column"\n        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)\n        >>> titanic_te = H2OTargetEncoderEstimator(data_leakage_handling="leave_one_out",\n        ...                                        inflection_point=35,\n        ...                                        smoothing=25,\n        ...                                        blending=True,\n        ...                                        seed=1234)\n        >>> titanic_te.train(x=predictors,\n        ...                  y=response,\n        ...                  training_frame=titanic)\n        >>> transformed = titanic_te.transform(frame=titanic)\n        '
        for k in kwargs:
            if k in ['seed', 'data_leakage_handling']:
                warnings.warn('`%s` is deprecated in `transform` method and will be ignored. Instead, please ensure that it was set before training on the H2OTargetEncoderEstimator model.' % k, H2ODeprecationWarning)
            else:
                raise TypeError("transform() got an unexpected keyword argument '%s'" % k)
        if 'data_leakage_handling' in kwargs:
            dlh = kwargs['data_leakage_handling']
            assert_is_type(dlh, None, Enum('leave_one_out', 'k_fold', 'none'))
            if dlh is not None and dlh.lower() != 'none':
                warnings.warn('Deprecated `data_leakage_handling=%s` is replaced by `as_training=True`. Please update your code.' % dlh, H2ODeprecationWarning)
                as_training = True
        params = dict(model=self.model_id, frame=frame.key, blending=blending if blending is not None else self.blending, inflection_point=inflection_point, smoothing=smoothing, noise=noise, as_training=as_training)
        output = h2o.api('GET /3/TargetEncoderTransform', data=params)
        return h2o.get_frame(output['name'])