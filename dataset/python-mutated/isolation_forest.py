from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OIsolationForestEstimator(H2OEstimator):
    """
    Isolation Forest

    Builds an Isolation Forest model. Isolation Forest algorithm samples the training frame
    and in each iteration builds a tree that partitions the space of the sample observations until
    it isolates each observation. Length of the path from root to a leaf node of the resulting tree
    is used to calculate the anomaly score. Anomalies are easier to isolate and their average
    tree path is expected to be shorter than paths of regular observations.
    """
    algo = 'isolationforest'
    supervised_learning = False
    _options_ = {'model_extensions': ['h2o.model.extensions.Trees']}

    def __init__(self, model_id=None, training_frame=None, score_each_iteration=False, score_tree_interval=0, ignored_columns=None, ignore_const_cols=True, ntrees=50, max_depth=8, min_rows=1.0, max_runtime_secs=0.0, seed=-1, build_tree_one_node=False, mtries=-1, sample_size=256, sample_rate=-1.0, col_sample_rate_change_per_level=1.0, col_sample_rate_per_tree=1.0, categorical_encoding='auto', stopping_rounds=0, stopping_metric='auto', stopping_tolerance=0.01, export_checkpoints_dir=None, contamination=-1.0, validation_frame=None, validation_response_column=None):
        if False:
            i = 10
            return i + 15
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param score_each_iteration: Whether to score during each iteration of model training.\n               Defaults to ``False``.\n        :type score_each_iteration: bool\n        :param score_tree_interval: Score the model after every so many trees. Disabled if set to 0.\n               Defaults to ``0``.\n        :type score_tree_interval: int\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param ignore_const_cols: Ignore constant columns.\n               Defaults to ``True``.\n        :type ignore_const_cols: bool\n        :param ntrees: Number of trees.\n               Defaults to ``50``.\n        :type ntrees: int\n        :param max_depth: Maximum tree depth (0 for unlimited).\n               Defaults to ``8``.\n        :type max_depth: int\n        :param min_rows: Fewest allowed (weighted) observations in a leaf.\n               Defaults to ``1.0``.\n        :type min_rows: float\n        :param max_runtime_secs: Maximum allowed runtime in seconds for model training. Use 0 to disable.\n               Defaults to ``0.0``.\n        :type max_runtime_secs: float\n        :param seed: Seed for pseudo random number generator (if applicable)\n               Defaults to ``-1``.\n        :type seed: int\n        :param build_tree_one_node: Run on one node only; no network overhead but fewer cpus used. Suitable for small\n               datasets.\n               Defaults to ``False``.\n        :type build_tree_one_node: bool\n        :param mtries: Number of variables randomly sampled as candidates at each split. If set to -1, defaults (number\n               of predictors)/3.\n               Defaults to ``-1``.\n        :type mtries: int\n        :param sample_size: Number of randomly sampled observations used to train each Isolation Forest tree. Only one\n               of parameters sample_size and sample_rate should be defined. If sample_rate is defined, sample_size will\n               be ignored.\n               Defaults to ``256``.\n        :type sample_size: int\n        :param sample_rate: Rate of randomly sampled observations used to train each Isolation Forest tree. Needs to be\n               in range from 0.0 to 1.0. If set to -1, sample_rate is disabled and sample_size will be used instead.\n               Defaults to ``-1.0``.\n        :type sample_rate: float\n        :param col_sample_rate_change_per_level: Relative change of the column sampling rate for every level (must be >\n               0.0 and <= 2.0)\n               Defaults to ``1.0``.\n        :type col_sample_rate_change_per_level: float\n        :param col_sample_rate_per_tree: Column sample rate per tree (from 0.0 to 1.0)\n               Defaults to ``1.0``.\n        :type col_sample_rate_per_tree: float\n        :param categorical_encoding: Encoding scheme for categorical features\n               Defaults to ``"auto"``.\n        :type categorical_encoding: Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",\n               "sort_by_response", "enum_limited"]\n        :param stopping_rounds: Early stopping based on convergence of stopping_metric. Stop if simple moving average of\n               length k of the stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)\n               Defaults to ``0``.\n        :type stopping_rounds: int\n        :param stopping_metric: Metric to use for early stopping (AUTO: logloss for classification, deviance for\n               regression and anomaly_score for Isolation Forest). Note that custom and custom_increasing can only be\n               used in GBM and DRF with the Python client.\n               Defaults to ``"auto"``.\n        :type stopping_metric: Literal["auto", "anomaly_score", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr",\n               "misclassification", "mean_per_class_error"]\n        :param stopping_tolerance: Relative tolerance for metric-based stopping criterion (stop if relative improvement\n               is not at least this much)\n               Defaults to ``0.01``.\n        :type stopping_tolerance: float\n        :param export_checkpoints_dir: Automatically export generated models to this directory.\n               Defaults to ``None``.\n        :type export_checkpoints_dir: str, optional\n        :param contamination: Contamination ratio - the proportion of anomalies in the input dataset. If undefined (-1)\n               the predict function will not mark observations as anomalies and only anomaly score will be returned.\n               Defaults to -1 (undefined).\n               Defaults to ``-1.0``.\n        :type contamination: float\n        :param validation_frame: Id of the validation data frame.\n               Defaults to ``None``.\n        :type validation_frame: Union[None, str, H2OFrame], optional\n        :param validation_response_column: (experimental) Name of the response column in the validation frame. Response\n               column should be binary and indicate not anomaly/anomaly.\n               Defaults to ``None``.\n        :type validation_response_column: str, optional\n        '
        super(H2OIsolationForestEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.score_each_iteration = score_each_iteration
        self.score_tree_interval = score_tree_interval
        self.ignored_columns = ignored_columns
        self.ignore_const_cols = ignore_const_cols
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.min_rows = min_rows
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.build_tree_one_node = build_tree_one_node
        self.mtries = mtries
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.col_sample_rate_change_per_level = col_sample_rate_change_per_level
        self.col_sample_rate_per_tree = col_sample_rate_per_tree
        self.categorical_encoding = categorical_encoding
        self.stopping_rounds = stopping_rounds
        self.stopping_metric = stopping_metric
        self.stopping_tolerance = stopping_tolerance
        self.export_checkpoints_dir = export_checkpoints_dir
        self.contamination = contamination
        self.validation_frame = validation_frame
        self.validation_response_column = validation_response_column

    @property
    def training_frame(self):
        if False:
            i = 10
            return i + 15
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> cars_if = H2OIsolationForestEstimator(seed=1234)\n        >>> cars_if.train(x=predictors,\n        ...               training_frame=cars)\n        >>> cars_if.model_performance()\n        '
        return self._parms.get('training_frame')

    @training_frame.setter
    def training_frame(self, training_frame):
        if False:
            i = 10
            return i + 15
        self._parms['training_frame'] = H2OFrame._validate(training_frame, 'training_frame')

    @property
    def score_each_iteration(self):
        if False:
            print('Hello World!')
        '\n        Whether to score during each iteration of model training.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> cars_if = H2OIsolationForestEstimator(score_each_iteration=True,\n        ...                                       ntrees=55,\n        ...                                       seed=1234)\n        >>> cars_if.train(x=predictors,\n        ...               training_frame=cars)\n        >>> cars_if.model_performance()\n        '
        return self._parms.get('score_each_iteration')

    @score_each_iteration.setter
    def score_each_iteration(self, score_each_iteration):
        if False:
            return 10
        assert_is_type(score_each_iteration, None, bool)
        self._parms['score_each_iteration'] = score_each_iteration

    @property
    def score_tree_interval(self):
        if False:
            print('Hello World!')
        '\n        Score the model after every so many trees. Disabled if set to 0.\n\n        Type: ``int``, defaults to ``0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> cars_if = H2OIsolationForestEstimator(score_tree_interval=5,\n        ...                                       seed=1234)\n        >>> cars_if.train(x=predictors,\n        ...               training_frame=cars)\n        >>> cars_if.model_performance()\n        '
        return self._parms.get('score_tree_interval')

    @score_tree_interval.setter
    def score_tree_interval(self, score_tree_interval):
        if False:
            i = 10
            return i + 15
        assert_is_type(score_tree_interval, None, int)
        self._parms['score_tree_interval'] = score_tree_interval

    @property
    def ignored_columns(self):
        if False:
            print('Hello World!')
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
    def ignore_const_cols(self):
        if False:
            while True:
                i = 10
        '\n        Ignore constant columns.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> predictors = ["displacement","power","weight","acceleration","year","const_1","const_2"]\n        >>> cars["const_1"] = 6\n        >>> cars["const_2"] = 7\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_if = H2OIsolationForestEstimator(seed=1234,\n        ...                                       ignore_const_cols=True)\n        >>> cars_if.train(x=predictors,\n        ...               training_frame=cars)\n        >>> cars_if.model_performance()\n        '
        return self._parms.get('ignore_const_cols')

    @ignore_const_cols.setter
    def ignore_const_cols(self, ignore_const_cols):
        if False:
            i = 10
            return i + 15
        assert_is_type(ignore_const_cols, None, bool)
        self._parms['ignore_const_cols'] = ignore_const_cols

    @property
    def ntrees(self):
        if False:
            while True:
                i = 10
        '\n        Number of trees.\n\n        Type: ``int``, defaults to ``50``.\n\n        :examples:\n\n        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")\n        >>> predictors = titanic.columns\n        >>> tree_num = [20, 50, 80, 110, 140, 170, 200]\n        >>> label = ["20", "50", "80", "110", "140", "170", "200"]\n        >>> for key, num in enumerate(tree_num):\n        ...     titanic_if = H2OIsolationForestEstimator(ntrees=num,\n        ...                                              seed=1234)\n        ...     titanic_if.train(x=predictors,\n        ...                      training_frame=titanic) \n        ...     print(label[key], \'training score\', titanic_if.mse(train=True))\n        '
        return self._parms.get('ntrees')

    @ntrees.setter
    def ntrees(self, ntrees):
        if False:
            while True:
                i = 10
        assert_is_type(ntrees, None, int)
        self._parms['ntrees'] = ntrees

    @property
    def max_depth(self):
        if False:
            return 10
        '\n        Maximum tree depth (0 for unlimited).\n\n        Type: ``int``, defaults to ``8``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> cars_if = H2OIsolationForestEstimator(max_depth=2,\n        ...                                       seed=1234)\n        >>> cars_if.train(x=predictors,\n        ...               training_frame=cars)\n        >>> cars_if.model_performance()\n        '
        return self._parms.get('max_depth')

    @max_depth.setter
    def max_depth(self, max_depth):
        if False:
            i = 10
            return i + 15
        assert_is_type(max_depth, None, int)
        self._parms['max_depth'] = max_depth

    @property
    def min_rows(self):
        if False:
            print('Hello World!')
        '\n        Fewest allowed (weighted) observations in a leaf.\n\n        Type: ``float``, defaults to ``1.0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> cars_if = H2OIsolationForestEstimator(min_rows=16,\n        ...                                       seed=1234)\n        >>> cars_if.train(x=predictors,\n        ...               training_frame=cars)\n        >>> cars_if.model_performance()\n        '
        return self._parms.get('min_rows')

    @min_rows.setter
    def min_rows(self, min_rows):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(min_rows, None, numeric)
        self._parms['min_rows'] = min_rows

    @property
    def max_runtime_secs(self):
        if False:
            while True:
                i = 10
        '\n        Maximum allowed runtime in seconds for model training. Use 0 to disable.\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> cars_if = H2OIsolationForestEstimator(max_runtime_secs=10,\n        ...                                       ntrees=10000,\n        ...                                       max_depth=10,\n        ...                                       seed=1234)\n        >>> cars_if.train(x=predictors,\n        ...               training_frame=cars)\n        >>> cars_if.model_performance()\n        '
        return self._parms.get('max_runtime_secs')

    @max_runtime_secs.setter
    def max_runtime_secs(self, max_runtime_secs):
        if False:
            i = 10
            return i + 15
        assert_is_type(max_runtime_secs, None, numeric)
        self._parms['max_runtime_secs'] = max_runtime_secs

    @property
    def seed(self):
        if False:
            while True:
                i = 10
        '\n        Seed for pseudo random number generator (if applicable)\n\n        Type: ``int``, defaults to ``-1``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> isofor_w_seed = H2OIsolationForestEstimator(seed=1234) \n        >>> isofor_w_seed.train(x=predictors,\n        ...                     training_frame=airlines)\n        >>> isofor_wo_seed = H2OIsolationForestEstimator()\n        >>> isofor_wo_seed.train(x=predictors,\n        ...                      training_frame=airlines)\n        >>> isofor_w_seed.model_performance()\n        >>> isofor_wo_seed.model_performance()\n        '
        return self._parms.get('seed')

    @seed.setter
    def seed(self, seed):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(seed, None, int)
        self._parms['seed'] = seed

    @property
    def build_tree_one_node(self):
        if False:
            while True:
                i = 10
        '\n        Run on one node only; no network overhead but fewer cpus used. Suitable for small datasets.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> cars_if = H2OIsolationForestEstimator(build_tree_one_node=True,\n        ...                                       seed=1234)\n        >>> cars_if.train(x=predictors,\n        ...               training_frame=cars)\n        >>> cars_if.model_performance()\n        '
        return self._parms.get('build_tree_one_node')

    @build_tree_one_node.setter
    def build_tree_one_node(self, build_tree_one_node):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(build_tree_one_node, None, bool)
        self._parms['build_tree_one_node'] = build_tree_one_node

    @property
    def mtries(self):
        if False:
            print('Hello World!')
        '\n        Number of variables randomly sampled as candidates at each split. If set to -1, defaults (number of\n        predictors)/3.\n\n        Type: ``int``, defaults to ``-1``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> predictors = covtype.columns[0:54]\n        >>> cov_if = H2OIsolationForestEstimator(mtries=30, seed=1234)\n        >>> cov_if.train(x=predictors,\n        ...              training_frame=covtype)\n        >>> cov_if.model_performance()\n        '
        return self._parms.get('mtries')

    @mtries.setter
    def mtries(self, mtries):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(mtries, None, int)
        self._parms['mtries'] = mtries

    @property
    def sample_size(self):
        if False:
            print('Hello World!')
        '\n        Number of randomly sampled observations used to train each Isolation Forest tree. Only one of parameters\n        sample_size and sample_rate should be defined. If sample_rate is defined, sample_size will be ignored.\n\n        Type: ``int``, defaults to ``256``.\n\n        :examples:\n\n        >>> train = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/anomaly/ecg_discord_train.csv")\n        >>> isofor_model = H2OIsolationForestEstimator(sample_size=5,\n        ...                                            ntrees=7)\n        >>> isofor_model.train(training_frame=train)\n        >>> isofor_model.model_performance()\n        '
        return self._parms.get('sample_size')

    @sample_size.setter
    def sample_size(self, sample_size):
        if False:
            print('Hello World!')
        assert_is_type(sample_size, None, int)
        self._parms['sample_size'] = sample_size

    @property
    def sample_rate(self):
        if False:
            while True:
                i = 10
        '\n        Rate of randomly sampled observations used to train each Isolation Forest tree. Needs to be in range from 0.0 to\n        1.0. If set to -1, sample_rate is disabled and sample_size will be used instead.\n\n        Type: ``float``, defaults to ``-1.0``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> airlines_if = H2OIsolationForestEstimator(sample_rate=.7,\n        ...                                           seed=1234)\n        >>> airlines_if.train(x=predictors,\n        ...                   training_frame=airlines)\n        >>> airlines_if.model_performance()\n        '
        return self._parms.get('sample_rate')

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(sample_rate, None, numeric)
        self._parms['sample_rate'] = sample_rate

    @property
    def col_sample_rate_change_per_level(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Relative change of the column sampling rate for every level (must be > 0.0 and <= 2.0)\n\n        Type: ``float``, defaults to ``1.0``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> airlines_if = H2OIsolationForestEstimator(col_sample_rate_change_per_level=.9,\n        ...                                           seed=1234)\n        >>> airlines_if.train(x=predictors,\n        ...                   training_frame=airlines)\n        >>> airlines_if.model_performance()\n        '
        return self._parms.get('col_sample_rate_change_per_level')

    @col_sample_rate_change_per_level.setter
    def col_sample_rate_change_per_level(self, col_sample_rate_change_per_level):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(col_sample_rate_change_per_level, None, numeric)
        self._parms['col_sample_rate_change_per_level'] = col_sample_rate_change_per_level

    @property
    def col_sample_rate_per_tree(self):
        if False:
            while True:
                i = 10
        '\n        Column sample rate per tree (from 0.0 to 1.0)\n\n        Type: ``float``, defaults to ``1.0``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> airlines_if = H2OIsolationForestEstimator(col_sample_rate_per_tree=.7,\n        ...                                           seed=1234)\n        >>> airlines_if.train(x=predictors,\n        ...                   training_frame=airlines)\n        >>> airlines_if.model_performance()\n        '
        return self._parms.get('col_sample_rate_per_tree')

    @col_sample_rate_per_tree.setter
    def col_sample_rate_per_tree(self, col_sample_rate_per_tree):
        if False:
            print('Hello World!')
        assert_is_type(col_sample_rate_per_tree, None, numeric)
        self._parms['col_sample_rate_per_tree'] = col_sample_rate_per_tree

    @property
    def categorical_encoding(self):
        if False:
            print('Hello World!')
        '\n        Encoding scheme for categorical features\n\n        Type: ``Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",\n        "sort_by_response", "enum_limited"]``, defaults to ``"auto"``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> encoding = "one_hot_explicit"\n        >>> airlines_if = H2OIsolationForestEstimator(categorical_encoding=encoding,\n        ...                                           seed=1234)\n        >>> airlines_if.train(x=predictors,\n        ...                   training_frame=airlines)\n        >>> airlines_if.model_performance()\n        '
        return self._parms.get('categorical_encoding')

    @categorical_encoding.setter
    def categorical_encoding(self, categorical_encoding):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(categorical_encoding, None, Enum('auto', 'enum', 'one_hot_internal', 'one_hot_explicit', 'binary', 'eigen', 'label_encoder', 'sort_by_response', 'enum_limited'))
        self._parms['categorical_encoding'] = categorical_encoding

    @property
    def stopping_rounds(self):
        if False:
            print('Hello World!')
        '\n        Early stopping based on convergence of stopping_metric. Stop if simple moving average of length k of the\n        stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)\n\n        Type: ``int``, defaults to ``0``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> airlines_if = H2OIsolationForestEstimator(stopping_metric="auto",\n        ...                                           stopping_rounds=3,\n        ...                                           stopping_tolerance=1e-2,\n        ...                                           seed=1234)\n        >>> airlines_if.train(x=predictors,\n        ...                   training_frame=airlines)\n        >>> airlines_if.model_performance()\n        '
        return self._parms.get('stopping_rounds')

    @stopping_rounds.setter
    def stopping_rounds(self, stopping_rounds):
        if False:
            return 10
        assert_is_type(stopping_rounds, None, int)
        self._parms['stopping_rounds'] = stopping_rounds

    @property
    def stopping_metric(self):
        if False:
            while True:
                i = 10
        '\n        Metric to use for early stopping (AUTO: logloss for classification, deviance for regression and anomaly_score\n        for Isolation Forest). Note that custom and custom_increasing can only be used in GBM and DRF with the Python\n        client.\n\n        Type: ``Literal["auto", "anomaly_score", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr",\n        "misclassification", "mean_per_class_error"]``, defaults to ``"auto"``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> airlines_if = H2OIsolationForestEstimator(stopping_metric="auto",\n        ...                                           stopping_rounds=3,\n        ...                                           stopping_tolerance=1e-2,\n        ...                                           seed=1234)\n        >>> airlines_if.train(x=predictors,\n        ...                   training_frame=airlines)\n        >>> airlines_if.model_performance()\n        '
        return self._parms.get('stopping_metric')

    @stopping_metric.setter
    def stopping_metric(self, stopping_metric):
        if False:
            return 10
        assert_is_type(stopping_metric, None, Enum('auto', 'anomaly_score', 'deviance', 'logloss', 'mse', 'rmse', 'mae', 'rmsle', 'auc', 'aucpr', 'misclassification', 'mean_per_class_error'))
        self._parms['stopping_metric'] = stopping_metric

    @property
    def stopping_tolerance(self):
        if False:
            print('Hello World!')
        '\n        Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at least this much)\n\n        Type: ``float``, defaults to ``0.01``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> airlines_if = H2OIsolationForestEstimator(stopping_metric="auto",\n        ...                                           stopping_rounds=3,\n        ...                                           stopping_tolerance=1e-2,\n        ...                                           seed=1234)\n        >>> airlines_if.train(x=predictors,\n        ...                   training_frame=airlines)\n        >>> airlines_if.model_performance()\n        '
        return self._parms.get('stopping_tolerance')

    @stopping_tolerance.setter
    def stopping_tolerance(self, stopping_tolerance):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(stopping_tolerance, None, numeric)
        self._parms['stopping_tolerance'] = stopping_tolerance

    @property
    def export_checkpoints_dir(self):
        if False:
            while True:
                i = 10
        '\n        Automatically export generated models to this directory.\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> import tempfile\n        >>> from os import listdir\n        >>> airlines = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip", destination_frame="air.hex")\n        >>> predictors = ["DayofMonth", "DayOfWeek"]\n        >>> checkpoints_dir = tempfile.mkdtemp()\n        >>> air_if = H2OIsolationForestEstimator(max_depth=3,\n        ...                                      seed=1234,\n        ...                                      export_checkpoints_dir=checkpoints_dir)\n        >>> air_if.train(x=predictors,\n        ...              training_frame=airlines)\n        >>> len(listdir(checkpoints_dir))\n        '
        return self._parms.get('export_checkpoints_dir')

    @export_checkpoints_dir.setter
    def export_checkpoints_dir(self, export_checkpoints_dir):
        if False:
            return 10
        assert_is_type(export_checkpoints_dir, None, str)
        self._parms['export_checkpoints_dir'] = export_checkpoints_dir

    @property
    def contamination(self):
        if False:
            while True:
                i = 10
        '\n        Contamination ratio - the proportion of anomalies in the input dataset. If undefined (-1) the predict function\n        will not mark observations as anomalies and only anomaly score will be returned. Defaults to -1 (undefined).\n\n        Type: ``float``, defaults to ``-1.0``.\n        '
        return self._parms.get('contamination')

    @contamination.setter
    def contamination(self, contamination):
        if False:
            while True:
                i = 10
        assert_is_type(contamination, None, numeric)
        self._parms['contamination'] = contamination

    @property
    def validation_frame(self):
        if False:
            return 10
        '\n        Id of the validation data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('validation_frame')

    @validation_frame.setter
    def validation_frame(self, validation_frame):
        if False:
            return 10
        self._parms['validation_frame'] = H2OFrame._validate(validation_frame, 'validation_frame')

    @property
    def validation_response_column(self):
        if False:
            return 10
        '\n        (experimental) Name of the response column in the validation frame. Response column should be binary and\n        indicate not anomaly/anomaly.\n\n        Type: ``str``.\n        '
        return self._parms.get('validation_response_column')

    @validation_response_column.setter
    def validation_response_column(self, validation_response_column):
        if False:
            print('Hello World!')
        assert_is_type(validation_response_column, None, str)
        self._parms['validation_response_column'] = validation_response_column