from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OSupportVectorMachineEstimator(H2OEstimator):
    """
    PSVM

    """
    algo = 'psvm'
    supervised_learning = True

    def __init__(self, model_id=None, training_frame=None, validation_frame=None, response_column=None, ignored_columns=None, ignore_const_cols=True, hyper_param=1.0, kernel_type='gaussian', gamma=-1.0, rank_ratio=-1.0, positive_weight=1.0, negative_weight=1.0, disable_training_metrics=True, sv_threshold=0.0001, fact_threshold=1e-05, feasible_threshold=0.001, surrogate_gap_threshold=0.001, mu_factor=10.0, max_iterations=200, seed=-1):
        if False:
            print('Hello World!')
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param validation_frame: Id of the validation data frame.\n               Defaults to ``None``.\n        :type validation_frame: Union[None, str, H2OFrame], optional\n        :param response_column: Response variable column.\n               Defaults to ``None``.\n        :type response_column: str, optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param ignore_const_cols: Ignore constant columns.\n               Defaults to ``True``.\n        :type ignore_const_cols: bool\n        :param hyper_param: Penalty parameter C of the error term\n               Defaults to ``1.0``.\n        :type hyper_param: float\n        :param kernel_type: Type of used kernel\n               Defaults to ``"gaussian"``.\n        :type kernel_type: Literal["gaussian"]\n        :param gamma: Coefficient of the kernel (currently RBF gamma for gaussian kernel, -1 means 1/#features)\n               Defaults to ``-1.0``.\n        :type gamma: float\n        :param rank_ratio: Desired rank of the ICF matrix expressed as an ration of number of input rows (-1 means use\n               sqrt(#rows)).\n               Defaults to ``-1.0``.\n        :type rank_ratio: float\n        :param positive_weight: Weight of positive (+1) class of observations\n               Defaults to ``1.0``.\n        :type positive_weight: float\n        :param negative_weight: Weight of positive (-1) class of observations\n               Defaults to ``1.0``.\n        :type negative_weight: float\n        :param disable_training_metrics: Disable calculating training metrics (expensive on large datasets)\n               Defaults to ``True``.\n        :type disable_training_metrics: bool\n        :param sv_threshold: Threshold for accepting a candidate observation into the set of support vectors\n               Defaults to ``0.0001``.\n        :type sv_threshold: float\n        :param fact_threshold: Convergence threshold of the Incomplete Cholesky Factorization (ICF)\n               Defaults to ``1e-05``.\n        :type fact_threshold: float\n        :param feasible_threshold: Convergence threshold for primal-dual residuals in the IPM iteration\n               Defaults to ``0.001``.\n        :type feasible_threshold: float\n        :param surrogate_gap_threshold: Feasibility criterion of the surrogate duality gap (eta)\n               Defaults to ``0.001``.\n        :type surrogate_gap_threshold: float\n        :param mu_factor: Increasing factor mu\n               Defaults to ``10.0``.\n        :type mu_factor: float\n        :param max_iterations: Maximum number of iteration of the algorithm\n               Defaults to ``200``.\n        :type max_iterations: int\n        :param seed: Seed for pseudo random number generator (if applicable)\n               Defaults to ``-1``.\n        :type seed: int\n        '
        super(H2OSupportVectorMachineEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.validation_frame = validation_frame
        self.response_column = response_column
        self.ignored_columns = ignored_columns
        self.ignore_const_cols = ignore_const_cols
        self.hyper_param = hyper_param
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.rank_ratio = rank_ratio
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.disable_training_metrics = disable_training_metrics
        self.sv_threshold = sv_threshold
        self.fact_threshold = fact_threshold
        self.feasible_threshold = feasible_threshold
        self.surrogate_gap_threshold = surrogate_gap_threshold
        self.mu_factor = mu_factor
        self.max_iterations = max_iterations
        self.seed = seed

    @property
    def training_frame(self):
        if False:
            i = 10
            return i + 15
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> train, valid = splice.split_frame(ratios=[0.8])\n        >>> svm = H2OSupportVectorMachineEstimator(disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=train)\n        >>> svm.mse()\n        '
        return self._parms.get('training_frame')

    @training_frame.setter
    def training_frame(self, training_frame):
        if False:
            print('Hello World!')
        self._parms['training_frame'] = H2OFrame._validate(training_frame, 'training_frame')

    @property
    def validation_frame(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Id of the validation data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> train, valid = splice.split_frame(ratios=[0.8])\n        >>> svm = H2OSupportVectorMachineEstimator(disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=train, validation_frame=valid)\n        >>> svm.mse()\n        '
        return self._parms.get('validation_frame')

    @validation_frame.setter
    def validation_frame(self, validation_frame):
        if False:
            return 10
        self._parms['validation_frame'] = H2OFrame._validate(validation_frame, 'validation_frame')

    @property
    def response_column(self):
        if False:
            print('Hello World!')
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
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
        '\n        Ignore constant columns.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.01,\n        ...                                        rank_ratio=0.1,\n        ...                                        ignore_const_cols=False,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice)\n        >>> svm.mse()\n        '
        return self._parms.get('ignore_const_cols')

    @ignore_const_cols.setter
    def ignore_const_cols(self, ignore_const_cols):
        if False:
            return 10
        assert_is_type(ignore_const_cols, None, bool)
        self._parms['ignore_const_cols'] = ignore_const_cols

    @property
    def hyper_param(self):
        if False:
            i = 10
            return i + 15
        '\n        Penalty parameter C of the error term\n\n        Type: ``float``, defaults to ``1.0``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.01,\n        ...                                        rank_ratio=0.1,\n        ...                                        hyper_param=0.01,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice)\n        >>> svm.mse()\n        '
        return self._parms.get('hyper_param')

    @hyper_param.setter
    def hyper_param(self, hyper_param):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(hyper_param, None, numeric)
        self._parms['hyper_param'] = hyper_param

    @property
    def kernel_type(self):
        if False:
            print('Hello World!')
        '\n        Type of used kernel\n\n        Type: ``Literal["gaussian"]``, defaults to ``"gaussian"``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.1,\n        ...                                        rank_ratio=0.1,\n        ...                                        hyper_param=0.01,\n        ...                                        kernel_type="gaussian",\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice) \n        >>> svm.mse()\n        '
        return self._parms.get('kernel_type')

    @kernel_type.setter
    def kernel_type(self, kernel_type):
        if False:
            while True:
                i = 10
        assert_is_type(kernel_type, None, Enum('gaussian'))
        self._parms['kernel_type'] = kernel_type

    @property
    def gamma(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Coefficient of the kernel (currently RBF gamma for gaussian kernel, -1 means 1/#features)\n\n        Type: ``float``, defaults to ``-1.0``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.01,\n        ...                                        rank_ratio=0.1,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice)\n        >>> svm.mse()\n        '
        return self._parms.get('gamma')

    @gamma.setter
    def gamma(self, gamma):
        if False:
            while True:
                i = 10
        assert_is_type(gamma, None, numeric)
        self._parms['gamma'] = gamma

    @property
    def rank_ratio(self):
        if False:
            i = 10
            return i + 15
        '\n        Desired rank of the ICF matrix expressed as an ration of number of input rows (-1 means use sqrt(#rows)).\n\n        Type: ``float``, defaults to ``-1.0``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.01,\n        ...                                        rank_ratio=0.1,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice)\n        >>> svm.mse()\n        '
        return self._parms.get('rank_ratio')

    @rank_ratio.setter
    def rank_ratio(self, rank_ratio):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(rank_ratio, None, numeric)
        self._parms['rank_ratio'] = rank_ratio

    @property
    def positive_weight(self):
        if False:
            while True:
                i = 10
        '\n        Weight of positive (+1) class of observations\n\n        Type: ``float``, defaults to ``1.0``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.1,\n        ...                                        rank_ratio=0.1,\n        ...                                        positive_weight=0.1,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice)   \n        >>> svm.mse()\n        '
        return self._parms.get('positive_weight')

    @positive_weight.setter
    def positive_weight(self, positive_weight):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(positive_weight, None, numeric)
        self._parms['positive_weight'] = positive_weight

    @property
    def negative_weight(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Weight of positive (-1) class of observations\n\n        Type: ``float``, defaults to ``1.0``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.1,\n        ...                                        rank_ratio=0.1,\n        ...                                        negative_weight=10,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice)  \n        >>> svm.mse()\n        '
        return self._parms.get('negative_weight')

    @negative_weight.setter
    def negative_weight(self, negative_weight):
        if False:
            print('Hello World!')
        assert_is_type(negative_weight, None, numeric)
        self._parms['negative_weight'] = negative_weight

    @property
    def disable_training_metrics(self):
        if False:
            return 10
        '\n        Disable calculating training metrics (expensive on large datasets)\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> from h2o.estimators import H2OSupportVectorMachineEstimator\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.01,\n        ...                                        rank_ratio=0.1,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice)\n        >>> svm.mse()\n        '
        return self._parms.get('disable_training_metrics')

    @disable_training_metrics.setter
    def disable_training_metrics(self, disable_training_metrics):
        if False:
            while True:
                i = 10
        assert_is_type(disable_training_metrics, None, bool)
        self._parms['disable_training_metrics'] = disable_training_metrics

    @property
    def sv_threshold(self):
        if False:
            return 10
        '\n        Threshold for accepting a candidate observation into the set of support vectors\n\n        Type: ``float``, defaults to ``0.0001``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.01,\n        ...                                        rank_ratio=0.1,\n        ...                                        sv_threshold=0.01,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice) \n        >>> svm.mse()\n        '
        return self._parms.get('sv_threshold')

    @sv_threshold.setter
    def sv_threshold(self, sv_threshold):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(sv_threshold, None, numeric)
        self._parms['sv_threshold'] = sv_threshold

    @property
    def fact_threshold(self):
        if False:
            i = 10
            return i + 15
        '\n        Convergence threshold of the Incomplete Cholesky Factorization (ICF)\n\n        Type: ``float``, defaults to ``1e-05``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(disable_training_metrics=False,\n        ...                                        fact_threshold=1e-7)\n        >>> svm.train(y="C1", training_frame=splice)\n        >>> svm.mse()\n        '
        return self._parms.get('fact_threshold')

    @fact_threshold.setter
    def fact_threshold(self, fact_threshold):
        if False:
            return 10
        assert_is_type(fact_threshold, None, numeric)
        self._parms['fact_threshold'] = fact_threshold

    @property
    def feasible_threshold(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convergence threshold for primal-dual residuals in the IPM iteration\n\n        Type: ``float``, defaults to ``0.001``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(disable_training_metrics=False,\n        ...                                        fact_threshold=1e-7)\n        >>> svm.train(y="C1", training_frame=splice)\n        >>> svm.mse()\n        '
        return self._parms.get('feasible_threshold')

    @feasible_threshold.setter
    def feasible_threshold(self, feasible_threshold):
        if False:
            while True:
                i = 10
        assert_is_type(feasible_threshold, None, numeric)
        self._parms['feasible_threshold'] = feasible_threshold

    @property
    def surrogate_gap_threshold(self):
        if False:
            print('Hello World!')
        '\n        Feasibility criterion of the surrogate duality gap (eta)\n\n        Type: ``float``, defaults to ``0.001``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.01,\n        ...                                        rank_ratio=0.1,\n        ...                                        surrogate_gap_threshold=0.1,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice) \n        >>> svm.mse()\n        '
        return self._parms.get('surrogate_gap_threshold')

    @surrogate_gap_threshold.setter
    def surrogate_gap_threshold(self, surrogate_gap_threshold):
        if False:
            print('Hello World!')
        assert_is_type(surrogate_gap_threshold, None, numeric)
        self._parms['surrogate_gap_threshold'] = surrogate_gap_threshold

    @property
    def mu_factor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Increasing factor mu\n\n        Type: ``float``, defaults to ``10.0``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.1,\n        ...                                        mu_factor=100.5,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice) \n        >>> svm.mse()\n        '
        return self._parms.get('mu_factor')

    @mu_factor.setter
    def mu_factor(self, mu_factor):
        if False:
            return 10
        assert_is_type(mu_factor, None, numeric)
        self._parms['mu_factor'] = mu_factor

    @property
    def max_iterations(self):
        if False:
            i = 10
            return i + 15
        '\n        Maximum number of iteration of the algorithm\n\n        Type: ``int``, defaults to ``200``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.1,\n        ...                                        rank_ratio=0.1,\n        ...                                        hyper_param=0.01,\n        ...                                        max_iterations=20,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice)  \n        >>> svm.mse()\n        '
        return self._parms.get('max_iterations')

    @max_iterations.setter
    def max_iterations(self, max_iterations):
        if False:
            return 10
        assert_is_type(max_iterations, None, int)
        self._parms['max_iterations'] = max_iterations

    @property
    def seed(self):
        if False:
            i = 10
            return i + 15
        '\n        Seed for pseudo random number generator (if applicable)\n\n        Type: ``int``, defaults to ``-1``.\n\n        :examples:\n\n        >>> splice = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/splice/splice.svm")\n        >>> svm = H2OSupportVectorMachineEstimator(gamma=0.1,\n        ...                                        rank_ratio=0.1,\n        ...                                        seed=1234,\n        ...                                        disable_training_metrics=False)\n        >>> svm.train(y="C1", training_frame=splice)\n        >>> svm.model_performance\n        '
        return self._parms.get('seed')

    @seed.setter
    def seed(self, seed):
        if False:
            print('Hello World!')
        assert_is_type(seed, None, int)
        self._parms['seed'] = seed