from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OUpliftRandomForestEstimator(H2OEstimator):
    """
    Uplift Distributed Random Forest

    """
    algo = 'upliftdrf'
    supervised_learning = True

    def __init__(self, model_id=None, training_frame=None, validation_frame=None, score_each_iteration=False, score_tree_interval=0, response_column=None, ignored_columns=None, ignore_const_cols=True, ntrees=50, max_depth=20, min_rows=1.0, nbins=20, nbins_top_level=1024, nbins_cats=1024, max_runtime_secs=0.0, seed=-1, mtries=-2, sample_rate=0.632, sample_rate_per_class=None, col_sample_rate_change_per_level=1.0, col_sample_rate_per_tree=1.0, histogram_type='auto', categorical_encoding='auto', distribution='auto', check_constant_response=True, custom_metric_func=None, treatment_column='treatment', uplift_metric='auto', auuc_type='auto', auuc_nbins=-1):
        if False:
            i = 10
            return i + 15
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param validation_frame: Id of the validation data frame.\n               Defaults to ``None``.\n        :type validation_frame: Union[None, str, H2OFrame], optional\n        :param score_each_iteration: Whether to score during each iteration of model training.\n               Defaults to ``False``.\n        :type score_each_iteration: bool\n        :param score_tree_interval: Score the model after every so many trees. Disabled if set to 0.\n               Defaults to ``0``.\n        :type score_tree_interval: int\n        :param response_column: Response variable column.\n               Defaults to ``None``.\n        :type response_column: str, optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param ignore_const_cols: Ignore constant columns.\n               Defaults to ``True``.\n        :type ignore_const_cols: bool\n        :param ntrees: Number of trees.\n               Defaults to ``50``.\n        :type ntrees: int\n        :param max_depth: Maximum tree depth (0 for unlimited).\n               Defaults to ``20``.\n        :type max_depth: int\n        :param min_rows: Fewest allowed (weighted) observations in a leaf.\n               Defaults to ``1.0``.\n        :type min_rows: float\n        :param nbins: For numerical columns (real/int), build a histogram of (at least) this many bins, then split at\n               the best point\n               Defaults to ``20``.\n        :type nbins: int\n        :param nbins_top_level: For numerical columns (real/int), build a histogram of (at most) this many bins at the\n               root level, then decrease by factor of two per level\n               Defaults to ``1024``.\n        :type nbins_top_level: int\n        :param nbins_cats: For categorical columns (factors), build a histogram of this many bins, then split at the\n               best point. Higher values can lead to more overfitting.\n               Defaults to ``1024``.\n        :type nbins_cats: int\n        :param max_runtime_secs: Maximum allowed runtime in seconds for model training. Use 0 to disable.\n               Defaults to ``0.0``.\n        :type max_runtime_secs: float\n        :param seed: Seed for pseudo random number generator (if applicable)\n               Defaults to ``-1``.\n        :type seed: int\n        :param mtries: Number of variables randomly sampled as candidates at each split. If set to -1, defaults to\n               sqrt{p} for classification and p/3 for regression (where p is the # of predictors\n               Defaults to ``-2``.\n        :type mtries: int\n        :param sample_rate: Row sample rate per tree (from 0.0 to 1.0)\n               Defaults to ``0.632``.\n        :type sample_rate: float\n        :param sample_rate_per_class: A list of row sample rates per class (relative fraction for each class, from 0.0\n               to 1.0), for each tree\n               Defaults to ``None``.\n        :type sample_rate_per_class: List[float], optional\n        :param col_sample_rate_change_per_level: Relative change of the column sampling rate for every level (must be >\n               0.0 and <= 2.0)\n               Defaults to ``1.0``.\n        :type col_sample_rate_change_per_level: float\n        :param col_sample_rate_per_tree: Column sample rate per tree (from 0.0 to 1.0)\n               Defaults to ``1.0``.\n        :type col_sample_rate_per_tree: float\n        :param histogram_type: What type of histogram to use for finding optimal split points\n               Defaults to ``"auto"``.\n        :type histogram_type: Literal["auto", "uniform_adaptive", "random", "quantiles_global", "round_robin", "uniform_robust"]\n        :param categorical_encoding: Encoding scheme for categorical features\n               Defaults to ``"auto"``.\n        :type categorical_encoding: Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",\n               "sort_by_response", "enum_limited"]\n        :param distribution: Distribution function\n               Defaults to ``"auto"``.\n        :type distribution: Literal["auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace",\n               "quantile", "huber"]\n        :param check_constant_response: Check if response column is constant. If enabled, then an exception is thrown if\n               the response column is a constant value.If disabled, then model will train regardless of the response\n               column being a constant value or not.\n               Defaults to ``True``.\n        :type check_constant_response: bool\n        :param custom_metric_func: Reference to custom evaluation function, format: `language:keyName=funcName`\n               Defaults to ``None``.\n        :type custom_metric_func: str, optional\n        :param treatment_column: Define the column which will be used for computing uplift gain to select best split for\n               a tree. The column has to divide the dataset into treatment (value 1) and control (value 0) groups.\n               Defaults to ``"treatment"``.\n        :type treatment_column: str\n        :param uplift_metric: Divergence metric used to find best split when building an uplift tree.\n               Defaults to ``"auto"``.\n        :type uplift_metric: Literal["auto", "kl", "euclidean", "chi_squared"]\n        :param auuc_type: Metric used to calculate Area Under Uplift Curve.\n               Defaults to ``"auto"``.\n        :type auuc_type: Literal["auto", "qini", "lift", "gain"]\n        :param auuc_nbins: Number of bins to calculate Area Under Uplift Curve.\n               Defaults to ``-1``.\n        :type auuc_nbins: int\n        '
        super(H2OUpliftRandomForestEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.validation_frame = validation_frame
        self.score_each_iteration = score_each_iteration
        self.score_tree_interval = score_tree_interval
        self.response_column = response_column
        self.ignored_columns = ignored_columns
        self.ignore_const_cols = ignore_const_cols
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.min_rows = min_rows
        self.nbins = nbins
        self.nbins_top_level = nbins_top_level
        self.nbins_cats = nbins_cats
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.mtries = mtries
        self.sample_rate = sample_rate
        self.sample_rate_per_class = sample_rate_per_class
        self.col_sample_rate_change_per_level = col_sample_rate_change_per_level
        self.col_sample_rate_per_tree = col_sample_rate_per_tree
        self.histogram_type = histogram_type
        self.categorical_encoding = categorical_encoding
        self.distribution = distribution
        self.check_constant_response = check_constant_response
        self.custom_metric_func = custom_metric_func
        self.treatment_column = treatment_column
        self.uplift_metric = uplift_metric
        self.auuc_type = auuc_type
        self.auuc_nbins = auuc_nbins

    @property
    def training_frame(self):
        if False:
            i = 10
            return i + 15
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('training_frame')

    @training_frame.setter
    def training_frame(self, training_frame):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        self._parms['validation_frame'] = H2OFrame._validate(validation_frame, 'validation_frame')

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
            for i in range(10):
                print('nop')
        assert_is_type(score_each_iteration, None, bool)
        self._parms['score_each_iteration'] = score_each_iteration

    @property
    def score_tree_interval(self):
        if False:
            while True:
                i = 10
        '\n        Score the model after every so many trees. Disabled if set to 0.\n\n        Type: ``int``, defaults to ``0``.\n        '
        return self._parms.get('score_tree_interval')

    @score_tree_interval.setter
    def score_tree_interval(self, score_tree_interval):
        if False:
            return 10
        assert_is_type(score_tree_interval, None, int)
        self._parms['score_tree_interval'] = score_tree_interval

    @property
    def response_column(self):
        if False:
            for i in range(10):
                print('nop')
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
    def ignore_const_cols(self):
        if False:
            print('Hello World!')
        '\n        Ignore constant columns.\n\n        Type: ``bool``, defaults to ``True``.\n        '
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
            for i in range(10):
                print('nop')
        '\n        Number of trees.\n\n        Type: ``int``, defaults to ``50``.\n        '
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
        '\n        Maximum tree depth (0 for unlimited).\n\n        Type: ``int``, defaults to ``20``.\n        '
        return self._parms.get('max_depth')

    @max_depth.setter
    def max_depth(self, max_depth):
        if False:
            print('Hello World!')
        assert_is_type(max_depth, None, int)
        self._parms['max_depth'] = max_depth

    @property
    def min_rows(self):
        if False:
            print('Hello World!')
        '\n        Fewest allowed (weighted) observations in a leaf.\n\n        Type: ``float``, defaults to ``1.0``.\n        '
        return self._parms.get('min_rows')

    @min_rows.setter
    def min_rows(self, min_rows):
        if False:
            return 10
        assert_is_type(min_rows, None, numeric)
        self._parms['min_rows'] = min_rows

    @property
    def nbins(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        For numerical columns (real/int), build a histogram of (at least) this many bins, then split at the best point\n\n        Type: ``int``, defaults to ``20``.\n        '
        return self._parms.get('nbins')

    @nbins.setter
    def nbins(self, nbins):
        if False:
            return 10
        assert_is_type(nbins, None, int)
        self._parms['nbins'] = nbins

    @property
    def nbins_top_level(self):
        if False:
            while True:
                i = 10
        '\n        For numerical columns (real/int), build a histogram of (at most) this many bins at the root level, then decrease\n        by factor of two per level\n\n        Type: ``int``, defaults to ``1024``.\n        '
        return self._parms.get('nbins_top_level')

    @nbins_top_level.setter
    def nbins_top_level(self, nbins_top_level):
        if False:
            print('Hello World!')
        assert_is_type(nbins_top_level, None, int)
        self._parms['nbins_top_level'] = nbins_top_level

    @property
    def nbins_cats(self):
        if False:
            i = 10
            return i + 15
        '\n        For categorical columns (factors), build a histogram of this many bins, then split at the best point. Higher\n        values can lead to more overfitting.\n\n        Type: ``int``, defaults to ``1024``.\n        '
        return self._parms.get('nbins_cats')

    @nbins_cats.setter
    def nbins_cats(self, nbins_cats):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(nbins_cats, None, int)
        self._parms['nbins_cats'] = nbins_cats

    @property
    def max_runtime_secs(self):
        if False:
            while True:
                i = 10
        '\n        Maximum allowed runtime in seconds for model training. Use 0 to disable.\n\n        Type: ``float``, defaults to ``0.0``.\n        '
        return self._parms.get('max_runtime_secs')

    @max_runtime_secs.setter
    def max_runtime_secs(self, max_runtime_secs):
        if False:
            print('Hello World!')
        assert_is_type(max_runtime_secs, None, numeric)
        self._parms['max_runtime_secs'] = max_runtime_secs

    @property
    def seed(self):
        if False:
            i = 10
            return i + 15
        '\n        Seed for pseudo random number generator (if applicable)\n\n        Type: ``int``, defaults to ``-1``.\n        '
        return self._parms.get('seed')

    @seed.setter
    def seed(self, seed):
        if False:
            while True:
                i = 10
        assert_is_type(seed, None, int)
        self._parms['seed'] = seed

    @property
    def mtries(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Number of variables randomly sampled as candidates at each split. If set to -1, defaults to sqrt{p} for\n        classification and p/3 for regression (where p is the # of predictors\n\n        Type: ``int``, defaults to ``-2``.\n        '
        return self._parms.get('mtries')

    @mtries.setter
    def mtries(self, mtries):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(mtries, None, int)
        self._parms['mtries'] = mtries

    @property
    def sample_rate(self):
        if False:
            while True:
                i = 10
        '\n        Row sample rate per tree (from 0.0 to 1.0)\n\n        Type: ``float``, defaults to ``0.632``.\n        '
        return self._parms.get('sample_rate')

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(sample_rate, None, numeric)
        self._parms['sample_rate'] = sample_rate

    @property
    def sample_rate_per_class(self):
        if False:
            print('Hello World!')
        '\n        A list of row sample rates per class (relative fraction for each class, from 0.0 to 1.0), for each tree\n\n        Type: ``List[float]``.\n        '
        return self._parms.get('sample_rate_per_class')

    @sample_rate_per_class.setter
    def sample_rate_per_class(self, sample_rate_per_class):
        if False:
            while True:
                i = 10
        assert_is_type(sample_rate_per_class, None, [numeric])
        self._parms['sample_rate_per_class'] = sample_rate_per_class

    @property
    def col_sample_rate_change_per_level(self):
        if False:
            print('Hello World!')
        '\n        Relative change of the column sampling rate for every level (must be > 0.0 and <= 2.0)\n\n        Type: ``float``, defaults to ``1.0``.\n        '
        return self._parms.get('col_sample_rate_change_per_level')

    @col_sample_rate_change_per_level.setter
    def col_sample_rate_change_per_level(self, col_sample_rate_change_per_level):
        if False:
            while True:
                i = 10
        assert_is_type(col_sample_rate_change_per_level, None, numeric)
        self._parms['col_sample_rate_change_per_level'] = col_sample_rate_change_per_level

    @property
    def col_sample_rate_per_tree(self):
        if False:
            return 10
        '\n        Column sample rate per tree (from 0.0 to 1.0)\n\n        Type: ``float``, defaults to ``1.0``.\n        '
        return self._parms.get('col_sample_rate_per_tree')

    @col_sample_rate_per_tree.setter
    def col_sample_rate_per_tree(self, col_sample_rate_per_tree):
        if False:
            while True:
                i = 10
        assert_is_type(col_sample_rate_per_tree, None, numeric)
        self._parms['col_sample_rate_per_tree'] = col_sample_rate_per_tree

    @property
    def histogram_type(self):
        if False:
            print('Hello World!')
        '\n        What type of histogram to use for finding optimal split points\n\n        Type: ``Literal["auto", "uniform_adaptive", "random", "quantiles_global", "round_robin", "uniform_robust"]``,\n        defaults to ``"auto"``.\n        '
        return self._parms.get('histogram_type')

    @histogram_type.setter
    def histogram_type(self, histogram_type):
        if False:
            while True:
                i = 10
        assert_is_type(histogram_type, None, Enum('auto', 'uniform_adaptive', 'random', 'quantiles_global', 'round_robin', 'uniform_robust'))
        self._parms['histogram_type'] = histogram_type

    @property
    def categorical_encoding(self):
        if False:
            i = 10
            return i + 15
        '\n        Encoding scheme for categorical features\n\n        Type: ``Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",\n        "sort_by_response", "enum_limited"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('categorical_encoding')

    @categorical_encoding.setter
    def categorical_encoding(self, categorical_encoding):
        if False:
            print('Hello World!')
        assert_is_type(categorical_encoding, None, Enum('auto', 'enum', 'one_hot_internal', 'one_hot_explicit', 'binary', 'eigen', 'label_encoder', 'sort_by_response', 'enum_limited'))
        self._parms['categorical_encoding'] = categorical_encoding

    @property
    def distribution(self):
        if False:
            i = 10
            return i + 15
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
    def check_constant_response(self):
        if False:
            i = 10
            return i + 15
        '\n        Check if response column is constant. If enabled, then an exception is thrown if the response column is a\n        constant value.If disabled, then model will train regardless of the response column being a constant value or\n        not.\n\n        Type: ``bool``, defaults to ``True``.\n        '
        return self._parms.get('check_constant_response')

    @check_constant_response.setter
    def check_constant_response(self, check_constant_response):
        if False:
            print('Hello World!')
        assert_is_type(check_constant_response, None, bool)
        self._parms['check_constant_response'] = check_constant_response

    @property
    def custom_metric_func(self):
        if False:
            return 10
        '\n        Reference to custom evaluation function, format: `language:keyName=funcName`\n\n        Type: ``str``.\n        '
        return self._parms.get('custom_metric_func')

    @custom_metric_func.setter
    def custom_metric_func(self, custom_metric_func):
        if False:
            while True:
                i = 10
        assert_is_type(custom_metric_func, None, str)
        self._parms['custom_metric_func'] = custom_metric_func

    @property
    def treatment_column(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Define the column which will be used for computing uplift gain to select best split for a tree. The column has\n        to divide the dataset into treatment (value 1) and control (value 0) groups.\n\n        Type: ``str``, defaults to ``"treatment"``.\n        '
        return self._parms.get('treatment_column')

    @treatment_column.setter
    def treatment_column(self, treatment_column):
        if False:
            return 10
        assert_is_type(treatment_column, None, str)
        self._parms['treatment_column'] = treatment_column

    @property
    def uplift_metric(self):
        if False:
            return 10
        '\n        Divergence metric used to find best split when building an uplift tree.\n\n        Type: ``Literal["auto", "kl", "euclidean", "chi_squared"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('uplift_metric')

    @uplift_metric.setter
    def uplift_metric(self, uplift_metric):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(uplift_metric, None, Enum('auto', 'kl', 'euclidean', 'chi_squared'))
        self._parms['uplift_metric'] = uplift_metric

    @property
    def auuc_type(self):
        if False:
            while True:
                i = 10
        '\n        Metric used to calculate Area Under Uplift Curve.\n\n        Type: ``Literal["auto", "qini", "lift", "gain"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('auuc_type')

    @auuc_type.setter
    def auuc_type(self, auuc_type):
        if False:
            print('Hello World!')
        assert_is_type(auuc_type, None, Enum('auto', 'qini', 'lift', 'gain'))
        self._parms['auuc_type'] = auuc_type

    @property
    def auuc_nbins(self):
        if False:
            while True:
                i = 10
        '\n        Number of bins to calculate Area Under Uplift Curve.\n\n        Type: ``int``, defaults to ``-1``.\n        '
        return self._parms.get('auuc_nbins')

    @auuc_nbins.setter
    def auuc_nbins(self, auuc_nbins):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(auuc_nbins, None, int)
        self._parms['auuc_nbins'] = auuc_nbins