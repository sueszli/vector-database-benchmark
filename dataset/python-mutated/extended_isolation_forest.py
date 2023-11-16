from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OExtendedIsolationForestEstimator(H2OEstimator):
    """
    Extended Isolation Forest

    Builds an Extended Isolation Forest model. Extended Isolation Forest generalizes its predecessor algorithm, 
    Isolation Forest. The original Isolation Forest algorithm suffers from bias due to tree branching. Extension of the 
    algorithm mitigates the bias by adjusting the branching, and the original algorithm becomes just a special case.
    Extended Isolation Forest's attribute "extension_level" allows leveraging the generalization. The minimum value is 0 and
    means the Isolation Forest's behavior. Maximum value is (numCols - 1) and stands for full extension. The rest of the 
    algorithm is analogical to the Isolation Forest algorithm. Each iteration builds a tree that partitions the sample 
    observations' space until it isolates observation. The length of the path from root to a leaf node of the resulting tree
    is used to calculate the anomaly score. Anomalies are easier to isolate, and their average
    tree path is expected to be shorter than paths of regular observations. Anomaly score is a number between 0 and 1. 
    A number closer to 0 is a normal point, and a number closer to 1 is a more anomalous point.
    """
    algo = 'extendedisolationforest'
    supervised_learning = False

    def __init__(self, model_id=None, training_frame=None, ignored_columns=None, ignore_const_cols=True, categorical_encoding='auto', ntrees=100, sample_size=256, extension_level=0, seed=-1):
        if False:
            i = 10
            return i + 15
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param ignore_const_cols: Ignore constant columns.\n               Defaults to ``True``.\n        :type ignore_const_cols: bool\n        :param categorical_encoding: Encoding scheme for categorical features\n               Defaults to ``"auto"``.\n        :type categorical_encoding: Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",\n               "sort_by_response", "enum_limited"]\n        :param ntrees: Number of Extended Isolation Forest trees.\n               Defaults to ``100``.\n        :type ntrees: int\n        :param sample_size: Number of randomly sampled observations used to train each Extended Isolation Forest tree.\n               Defaults to ``256``.\n        :type sample_size: int\n        :param extension_level: Maximum is N - 1 (N = numCols). Minimum is 0. Extended Isolation Forest with\n               extension_Level = 0 behaves like Isolation Forest.\n               Defaults to ``0``.\n        :type extension_level: int\n        :param seed: Seed for pseudo random number generator (if applicable)\n               Defaults to ``-1``.\n        :type seed: int\n        '
        super(H2OExtendedIsolationForestEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.ignored_columns = ignored_columns
        self.ignore_const_cols = ignore_const_cols
        self.categorical_encoding = categorical_encoding
        self.ntrees = ntrees
        self.sample_size = sample_size
        self.extension_level = extension_level
        self.seed = seed

    @property
    def training_frame(self):
        if False:
            return 10
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> cars_eif = H2OExtendedIsolationForestEstimator(seed = 1234, \n        ...                                                sample_size = 256, \n        ...                                                extension_level = cars.dim[1] - 1)\n        >>> cars_eif.train(x = predictors,\n        ...                training_frame = cars)\n        >>> print(cars_eif)\n        '
        return self._parms.get('training_frame')

    @training_frame.setter
    def training_frame(self, training_frame):
        if False:
            print('Hello World!')
        self._parms['training_frame'] = H2OFrame._validate(training_frame, 'training_frame')

    @property
    def ignored_columns(self):
        if False:
            while True:
                i = 10
        '\n        Names of columns to ignore for training.\n\n        Type: ``List[str]``.\n        '
        return self._parms.get('ignored_columns')

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        if False:
            while True:
                i = 10
        assert_is_type(ignored_columns, None, [str])
        self._parms['ignored_columns'] = ignored_columns

    @property
    def ignore_const_cols(self):
        if False:
            while True:
                i = 10
        '\n        Ignore constant columns.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> predictors = ["displacement","power","weight","acceleration","year","const_1","const_2"]\n        >>> cars["const_1"] = 6\n        >>> cars["const_2"] = 7\n        >>> train, valid = cars.split_frame(ratios = [.8], seed = 1234)\n        >>> cars_eif = H2OExtendedIsolationForestEstimator(seed = 1234,\n        ...                                                ignore_const_cols = True)\n        >>> cars_eif.train(x = predictors,\n        ...               training_frame = cars)\n        >>> cars_eif.model_performance()\n        '
        return self._parms.get('ignore_const_cols')

    @ignore_const_cols.setter
    def ignore_const_cols(self, ignore_const_cols):
        if False:
            return 10
        assert_is_type(ignore_const_cols, None, bool)
        self._parms['ignore_const_cols'] = ignore_const_cols

    @property
    def categorical_encoding(self):
        if False:
            while True:
                i = 10
        '\n        Encoding scheme for categorical features\n\n        Type: ``Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",\n        "sort_by_response", "enum_limited"]``, defaults to ``"auto"``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> encoding = "one_hot_explicit"\n        >>> airlines_eif = H2OExtendedIsolationForestEstimator(categorical_encoding = encoding,\n        ...                                                    seed = 1234)\n        >>> airlines_eif.train(x = predictors,\n        ...                   training_frame = airlines)\n        >>> airlines_eif.model_performance()\n        '
        return self._parms.get('categorical_encoding')

    @categorical_encoding.setter
    def categorical_encoding(self, categorical_encoding):
        if False:
            while True:
                i = 10
        assert_is_type(categorical_encoding, None, Enum('auto', 'enum', 'one_hot_internal', 'one_hot_explicit', 'binary', 'eigen', 'label_encoder', 'sort_by_response', 'enum_limited'))
        self._parms['categorical_encoding'] = categorical_encoding

    @property
    def ntrees(self):
        if False:
            print('Hello World!')
        '\n        Number of Extended Isolation Forest trees.\n\n        Type: ``int``, defaults to ``100``.\n\n        :examples:\n\n        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")\n        >>> predictors = titanic.columns\n        >>> tree_num = [20, 50, 80, 110, 140, 170, 200]\n        >>> label = ["20", "50", "80", "110", "140", "170", "200"]\n        >>> for key, num in enumerate(tree_num):\n        ...     titanic_eif = H2OExtendedIsolationForestEstimator(ntrees = num,\n        ...                                                       seed = 1234,\n        ...                                                       extension_level = titanic.dim[1] - 1)\n        ...     titanic_eif.train(x = predictors,\n        ...                      training_frame = titanic) \n        '
        return self._parms.get('ntrees')

    @ntrees.setter
    def ntrees(self, ntrees):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(ntrees, None, int)
        self._parms['ntrees'] = ntrees

    @property
    def sample_size(self):
        if False:
            i = 10
            return i + 15
        '\n        Number of randomly sampled observations used to train each Extended Isolation Forest tree.\n\n        Type: ``int``, defaults to ``256``.\n\n        :examples:\n\n        >>> train = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/anomaly/ecg_discord_train.csv")\n        >>> eif_model = H2OExtendedIsolationForestEstimator(sample_size = 5,\n        ...                                                 ntrees=7)\n        >>> eif_model.train(training_frame = train)\n        >>> print(eif_model)\n        '
        return self._parms.get('sample_size')

    @sample_size.setter
    def sample_size(self, sample_size):
        if False:
            return 10
        assert_is_type(sample_size, None, int)
        self._parms['sample_size'] = sample_size

    @property
    def extension_level(self):
        if False:
            print('Hello World!')
        '\n        Maximum is N - 1 (N = numCols). Minimum is 0. Extended Isolation Forest with extension_Level = 0 behaves like\n        Isolation Forest.\n\n        Type: ``int``, defaults to ``0``.\n\n        :examples:\n\n        >>> train = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/anomaly/single_blob.csv")\n        >>> eif_model = H2OExtendedIsolationForestEstimator(extension_level = 1,\n        ...                                                 ntrees=7)\n        >>> eif_model.train(training_frame = train)\n        >>> print(eif_model)\n        '
        return self._parms.get('extension_level')

    @extension_level.setter
    def extension_level(self, extension_level):
        if False:
            return 10
        assert_is_type(extension_level, None, int)
        self._parms['extension_level'] = extension_level

    @property
    def seed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Seed for pseudo random number generator (if applicable)\n\n        Type: ``int``, defaults to ``-1``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> eif_w_seed = H2OExtendedIsolationForestEstimator(seed = 1234) \n        >>> eif_w_seed.train(x = predictors,\n        ...                        training_frame = airlines)\n        >>> eif_wo_seed = H2OExtendedIsolationForestEstimator()\n        >>> eif_wo_seed.train(x = predictors,\n        ...                         training_frame = airlines)\n        >>> print(eif_w_seed)\n        >>> print(eif_wo_seed)\n        '
        return self._parms.get('seed')

    @seed.setter
    def seed(self, seed):
        if False:
            while True:
                i = 10
        assert_is_type(seed, None, int)
        self._parms['seed'] = seed