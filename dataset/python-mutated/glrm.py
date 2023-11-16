from h2o.base import Keyed
from h2o.frame import H2OFrame
from h2o.expr import ExprNode
from h2o.expr import ASTId
from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OGeneralizedLowRankEstimator(H2OEstimator):
    """
    Generalized Low Rank Modeling

    Builds a generalized low rank model of a H2O dataset.
    """
    algo = 'glrm'
    supervised_learning = False

    def __init__(self, model_id=None, training_frame=None, validation_frame=None, ignored_columns=None, ignore_const_cols=True, score_each_iteration=False, representation_name=None, loading_name=None, transform='none', k=1, loss='quadratic', loss_by_col=None, loss_by_col_idx=None, multi_loss='categorical', period=1, regularization_x='none', regularization_y='none', gamma_x=0.0, gamma_y=0.0, max_iterations=1000, max_updates=2000, init_step_size=1.0, min_step_size=0.0001, seed=-1, init='plus_plus', svd_method='randomized', user_y=None, user_x=None, expand_user_y=True, impute_original=False, recover_svd=False, max_runtime_secs=0.0, export_checkpoints_dir=None):
        if False:
            i = 10
            return i + 15
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param validation_frame: Id of the validation data frame.\n               Defaults to ``None``.\n        :type validation_frame: Union[None, str, H2OFrame], optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param ignore_const_cols: Ignore constant columns.\n               Defaults to ``True``.\n        :type ignore_const_cols: bool\n        :param score_each_iteration: Whether to score during each iteration of model training.\n               Defaults to ``False``.\n        :type score_each_iteration: bool\n        :param representation_name: Frame key to save resulting X\n               Defaults to ``None``.\n        :type representation_name: str, optional\n        :param loading_name: [Deprecated] Use representation_name instead.  Frame key to save resulting X.\n               Defaults to ``None``.\n        :type loading_name: str, optional\n        :param transform: Transformation of training data\n               Defaults to ``"none"``.\n        :type transform: Literal["none", "standardize", "normalize", "demean", "descale"]\n        :param k: Rank of matrix approximation\n               Defaults to ``1``.\n        :type k: int\n        :param loss: Numeric loss function\n               Defaults to ``"quadratic"``.\n        :type loss: Literal["quadratic", "absolute", "huber", "poisson", "hinge", "logistic", "periodic"]\n        :param loss_by_col: Loss function by column (override)\n               Defaults to ``None``.\n        :type loss_by_col: List[Literal["quadratic", "absolute", "huber", "poisson", "hinge", "logistic", "periodic", "categorical",\n               "ordinal"]], optional\n        :param loss_by_col_idx: Loss function by column index (override)\n               Defaults to ``None``.\n        :type loss_by_col_idx: List[int], optional\n        :param multi_loss: Categorical loss function\n               Defaults to ``"categorical"``.\n        :type multi_loss: Literal["categorical", "ordinal"]\n        :param period: Length of period (only used with periodic loss function)\n               Defaults to ``1``.\n        :type period: int\n        :param regularization_x: Regularization function for X matrix\n               Defaults to ``"none"``.\n        :type regularization_x: Literal["none", "quadratic", "l2", "l1", "non_negative", "one_sparse", "unit_one_sparse", "simplex"]\n        :param regularization_y: Regularization function for Y matrix\n               Defaults to ``"none"``.\n        :type regularization_y: Literal["none", "quadratic", "l2", "l1", "non_negative", "one_sparse", "unit_one_sparse", "simplex"]\n        :param gamma_x: Regularization weight on X matrix\n               Defaults to ``0.0``.\n        :type gamma_x: float\n        :param gamma_y: Regularization weight on Y matrix\n               Defaults to ``0.0``.\n        :type gamma_y: float\n        :param max_iterations: Maximum number of iterations\n               Defaults to ``1000``.\n        :type max_iterations: int\n        :param max_updates: Maximum number of updates, defaults to 2*max_iterations\n               Defaults to ``2000``.\n        :type max_updates: int\n        :param init_step_size: Initial step size\n               Defaults to ``1.0``.\n        :type init_step_size: float\n        :param min_step_size: Minimum step size\n               Defaults to ``0.0001``.\n        :type min_step_size: float\n        :param seed: RNG seed for initialization\n               Defaults to ``-1``.\n        :type seed: int\n        :param init: Initialization mode\n               Defaults to ``"plus_plus"``.\n        :type init: Literal["random", "svd", "plus_plus", "user"]\n        :param svd_method: Method for computing SVD during initialization (Caution: Randomized is currently experimental\n               and unstable)\n               Defaults to ``"randomized"``.\n        :type svd_method: Literal["gram_s_v_d", "power", "randomized"]\n        :param user_y: User-specified initial Y\n               Defaults to ``None``.\n        :type user_y: Union[None, str, H2OFrame], optional\n        :param user_x: User-specified initial X\n               Defaults to ``None``.\n        :type user_x: Union[None, str, H2OFrame], optional\n        :param expand_user_y: Expand categorical columns in user-specified initial Y\n               Defaults to ``True``.\n        :type expand_user_y: bool\n        :param impute_original: Reconstruct original training data by reversing transform\n               Defaults to ``False``.\n        :type impute_original: bool\n        :param recover_svd: Recover singular values and eigenvectors of XY\n               Defaults to ``False``.\n        :type recover_svd: bool\n        :param max_runtime_secs: Maximum allowed runtime in seconds for model training. Use 0 to disable.\n               Defaults to ``0.0``.\n        :type max_runtime_secs: float\n        :param export_checkpoints_dir: Automatically export generated models to this directory.\n               Defaults to ``None``.\n        :type export_checkpoints_dir: str, optional\n        '
        super(H2OGeneralizedLowRankEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.validation_frame = validation_frame
        self.ignored_columns = ignored_columns
        self.ignore_const_cols = ignore_const_cols
        self.score_each_iteration = score_each_iteration
        self.representation_name = representation_name
        self.loading_name = loading_name
        self.transform = transform
        self.k = k
        self.loss = loss
        self.loss_by_col = loss_by_col
        self.loss_by_col_idx = loss_by_col_idx
        self.multi_loss = multi_loss
        self.period = period
        self.regularization_x = regularization_x
        self.regularization_y = regularization_y
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.max_iterations = max_iterations
        self.max_updates = max_updates
        self.init_step_size = init_step_size
        self.min_step_size = min_step_size
        self.seed = seed
        self.init = init
        self.svd_method = svd_method
        self.user_y = user_y
        self.user_x = user_x
        self.expand_user_y = expand_user_y
        self.impute_original = impute_original
        self.recover_svd = recover_svd
        self.max_runtime_secs = max_runtime_secs
        self.export_checkpoints_dir = export_checkpoints_dir

    @property
    def training_frame(self):
        if False:
            return 10
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> prostate = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate_cat.csv")\n        >>> prostate[0] = prostate[0].asnumeric()\n        >>> prostate[4] = prostate[4].asnumeric()\n        >>> pros_glrm = H2OGeneralizedLowRankEstimator(k=5,\n        ...                                            seed=1234)\n        >>> pros_glrm.train(x=prostate.names, training_frame=prostate)\n        >>> pros_glrm.show()\n        '
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
            print('Hello World!')
        '\n        Id of the validation data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/iris/iris_wheader.csv")\n        >>> iris_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                            loss="quadratic",\n        ...                                            gamma_x=0.5,\n        ...                                            gamma_y=0.5,\n        ...                                            transform="standardize")\n        >>> iris_glrm.train(x=iris.names,\n        ...                 training_frame=iris,\n        ...                 validation_frame=iris)\n        >>> iris_glrm.show()\n        '
        return self._parms.get('validation_frame')

    @validation_frame.setter
    def validation_frame(self, validation_frame):
        if False:
            print('Hello World!')
        self._parms['validation_frame'] = H2OFrame._validate(validation_frame, 'validation_frame')

    @property
    def ignored_columns(self):
        if False:
            i = 10
            return i + 15
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
            i = 10
            return i + 15
        '\n        Ignore constant columns.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")\n        >>> iris_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                            ignore_const_cols=False,\n        ...                                            seed=1234)\n        >>> iris_glrm.train(x=iris.names, training_frame=iris)\n        >>> iris_glrm.show()\n        '
        return self._parms.get('ignore_const_cols')

    @ignore_const_cols.setter
    def ignore_const_cols(self, ignore_const_cols):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(ignore_const_cols, None, bool)
        self._parms['ignore_const_cols'] = ignore_const_cols

    @property
    def score_each_iteration(self):
        if False:
            i = 10
            return i + 15
        '\n        Whether to score during each iteration of model training.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> prostate = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate_cat.csv")\n        >>> prostate[0] = prostate[0].asnumeric()\n        >>> prostate[4] = prostate[4].asnumeric()\n        >>> loss_all = ["Hinge", "Quadratic", "Categorical", "Categorical",\n        ...             "Hinge", "Quadratic", "Quadratic", "Quadratic"]\n        >>> pros_glrm = H2OGeneralizedLowRankEstimator(k=5,\n        ...                                            loss_by_col=loss_all,\n        ...                                            score_each_iteration=True,\n        ...                                            transform="standardize",\n        ...                                            seed=12345)\n        >>> pros_glrm.train(x=prostate.names, training_frame=prostate)\n        >>> pros_glrm.show()\n        '
        return self._parms.get('score_each_iteration')

    @score_each_iteration.setter
    def score_each_iteration(self, score_each_iteration):
        if False:
            print('Hello World!')
        assert_is_type(score_each_iteration, None, bool)
        self._parms['score_each_iteration'] = score_each_iteration

    @property
    def representation_name(self):
        if False:
            return 10
        '\n        Frame key to save resulting X\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> acs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/census/ACS_13_5YR_DP02_cleaned.zip")\n        >>> acs_fill = acs.drop("ZCTA5")\n        >>> acs_glrm = H2OGeneralizedLowRankEstimator(k=10,\n        ...                                           transform="standardize",\n        ...                                           loss="quadratic",\n        ...                                           regularization_x="quadratic",\n        ...                                           regularization_y="L1",\n        ...                                           gamma_x=0.25,\n        ...                                           gamma_y=0.5,\n        ...                                           max_iterations=1,\n        ...                                           representation_name="acs_full")\n        >>> acs_glrm.train(x=acs_fill.names, training_frame=acs)\n        >>> acs_glrm.loading_name\n        >>> acs_glrm.show()\n        '
        return self._parms.get('representation_name')

    @representation_name.setter
    def representation_name(self, representation_name):
        if False:
            i = 10
            return i + 15
        assert_is_type(representation_name, None, str)
        self._parms['representation_name'] = representation_name

    @property
    def loading_name(self):
        if False:
            print('Hello World!')
        '\n        [Deprecated] Use representation_name instead.  Frame key to save resulting X.\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> # loading_name will be deprecated.  Use representation_name instead.    \n        >>> acs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/census/ACS_13_5YR_DP02_cleaned.zip")\n        >>> acs_fill = acs.drop("ZCTA5")\n        >>> acs_glrm = H2OGeneralizedLowRankEstimator(k=10,\n        ...                                           transform="standardize",\n        ...                                           loss="quadratic",\n        ...                                           regularization_x="quadratic",\n        ...                                           regularization_y="L1",\n        ...                                           gamma_x=0.25,\n        ...                                           gamma_y=0.5,\n        ...                                           max_iterations=1,\n        ...                                           loading_name="acs_full")\n        >>> acs_glrm.train(x=acs_fill.names, training_frame=acs)\n        >>> acs_glrm.loading_name\n        >>> acs_glrm.show()\n        '
        return self._parms.get('loading_name')

    @loading_name.setter
    def loading_name(self, loading_name):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(loading_name, None, str)
        self._parms['loading_name'] = loading_name

    @property
    def transform(self):
        if False:
            i = 10
            return i + 15
        '\n        Transformation of training data\n\n        Type: ``Literal["none", "standardize", "normalize", "demean", "descale"]``, defaults to ``"none"``.\n\n        :examples:\n\n        >>> prostate = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate_cat.csv")\n        >>> prostate[0] = prostate[0].asnumeric()\n        >>> prostate[4] = prostate[4].asnumeric()\n        >>> pros_glrm = H2OGeneralizedLowRankEstimator(k=5,\n        ...                                            score_each_iteration=True,\n        ...                                            transform="standardize",\n        ...                                            seed=12345)\n        >>> pros_glrm.train(x=prostate.names, training_frame=prostate)\n        >>> pros_glrm.show()\n        '
        return self._parms.get('transform')

    @transform.setter
    def transform(self, transform):
        if False:
            i = 10
            return i + 15
        assert_is_type(transform, None, Enum('none', 'standardize', 'normalize', 'demean', 'descale'))
        self._parms['transform'] = transform

    @property
    def k(self):
        if False:
            return 10
        '\n        Rank of matrix approximation\n\n        Type: ``int``, defaults to ``1``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")\n        >>> iris_glrm = H2OGeneralizedLowRankEstimator(k=3)\n        >>> iris_glrm.train(x=iris.names, training_frame=iris)\n        >>> iris_glrm.show()\n        '
        return self._parms.get('k')

    @k.setter
    def k(self, k):
        if False:
            print('Hello World!')
        assert_is_type(k, None, int)
        self._parms['k'] = k

    @property
    def loss(self):
        if False:
            i = 10
            return i + 15
        '\n        Numeric loss function\n\n        Type: ``Literal["quadratic", "absolute", "huber", "poisson", "hinge", "logistic", "periodic"]``, defaults to\n        ``"quadratic"``.\n\n        :examples:\n\n        >>> acs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/census/ACS_13_5YR_DP02_cleaned.zip")\n        >>> acs_fill = acs.drop("ZCTA5")\n        >>> acs_glrm = H2OGeneralizedLowRankEstimator(k=10,\n        ...                                           transform="standardize",\n        ...                                           loss="absolute",\n        ...                                           regularization_x="quadratic",\n        ...                                           regularization_y="L1",\n        ...                                           gamma_x=0.25,\n        ...                                           gamma_y=0.5,\n        ...                                           max_iterations=700)\n        >>> acs_glrm.train(x=acs_fill.names, training_frame=acs)\n        >>> acs_glrm.show()\n        '
        return self._parms.get('loss')

    @loss.setter
    def loss(self, loss):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(loss, None, Enum('quadratic', 'absolute', 'huber', 'poisson', 'hinge', 'logistic', 'periodic'))
        self._parms['loss'] = loss

    @property
    def loss_by_col(self):
        if False:
            while True:
                i = 10
        '\n        Loss function by column (override)\n\n        Type: ``List[Literal["quadratic", "absolute", "huber", "poisson", "hinge", "logistic", "periodic",\n        "categorical", "ordinal"]]``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/pca_test/USArrests.csv")\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                               loss="quadratic",\n        ...                                               loss_by_col=["absolute","huber"],\n        ...                                               loss_by_col_idx=[0,3],\n        ...                                               regularization_x="quadratic",\n        ...                                               regularization_y="l1")\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('loss_by_col')

    @loss_by_col.setter
    def loss_by_col(self, loss_by_col):
        if False:
            while True:
                i = 10
        assert_is_type(loss_by_col, None, [Enum('quadratic', 'absolute', 'huber', 'poisson', 'hinge', 'logistic', 'periodic', 'categorical', 'ordinal')])
        self._parms['loss_by_col'] = loss_by_col

    @property
    def loss_by_col_idx(self):
        if False:
            print('Hello World!')
        '\n        Loss function by column index (override)\n\n        Type: ``List[int]``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/pca_test/USArrests.csv")\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                               loss="quadratic",\n        ...                                               loss_by_col=["absolute","huber"],\n        ...                                               loss_by_col_idx=[0,3],\n        ...                                               regularization_x="quadratic",\n        ...                                               regularization_y="l1")\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('loss_by_col_idx')

    @loss_by_col_idx.setter
    def loss_by_col_idx(self, loss_by_col_idx):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(loss_by_col_idx, None, [int])
        self._parms['loss_by_col_idx'] = loss_by_col_idx

    @property
    def multi_loss(self):
        if False:
            return 10
        '\n        Categorical loss function\n\n        Type: ``Literal["categorical", "ordinal"]``, defaults to ``"categorical"``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/pca_test/USArrests.csv")\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                               loss="quadratic",\n        ...                                               loss_by_col=["absolute","huber"],\n        ...                                               loss_by_col_idx=[0,3],\n        ...                                               regularization_x="quadratic",\n        ...                                               regularization_y="l1"\n        ...                                               multi_loss="ordinal")\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('multi_loss')

    @multi_loss.setter
    def multi_loss(self, multi_loss):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(multi_loss, None, Enum('categorical', 'ordinal'))
        self._parms['multi_loss'] = multi_loss

    @property
    def period(self):
        if False:
            while True:
                i = 10
        '\n        Length of period (only used with periodic loss function)\n\n        Type: ``int``, defaults to ``1``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/pca_test/USArrests.csv")\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                               max_runtime_secs=15,\n        ...                                               max_iterations=500,\n        ...                                               max_updates=900,\n        ...                                               min_step_size=0.005,\n        ...                                               period=5)\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('period')

    @period.setter
    def period(self, period):
        if False:
            print('Hello World!')
        assert_is_type(period, None, int)
        self._parms['period'] = period

    @property
    def regularization_x(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Regularization function for X matrix\n\n        Type: ``Literal["none", "quadratic", "l2", "l1", "non_negative", "one_sparse", "unit_one_sparse", "simplex"]``,\n        defaults to ``"none"``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/pca_test/USArrests.csv")\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                               loss="quadratic",\n        ...                                               loss_by_col=["absolute","huber"],\n        ...                                               loss_by_col_idx=[0,3],\n        ...                                               regularization_x="quadratic",\n        ...                                               regularization_y="l1")\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('regularization_x')

    @regularization_x.setter
    def regularization_x(self, regularization_x):
        if False:
            print('Hello World!')
        assert_is_type(regularization_x, None, Enum('none', 'quadratic', 'l2', 'l1', 'non_negative', 'one_sparse', 'unit_one_sparse', 'simplex'))
        self._parms['regularization_x'] = regularization_x

    @property
    def regularization_y(self):
        if False:
            print('Hello World!')
        '\n        Regularization function for Y matrix\n\n        Type: ``Literal["none", "quadratic", "l2", "l1", "non_negative", "one_sparse", "unit_one_sparse", "simplex"]``,\n        defaults to ``"none"``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/pca_test/USArrests.csv")\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                               loss="quadratic",\n        ...                                               loss_by_col=["absolute","huber"],\n        ...                                               loss_by_col_idx=[0,3],\n        ...                                               regularization_x="quadratic",\n        ...                                               regularization_y="l1")\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('regularization_y')

    @regularization_y.setter
    def regularization_y(self, regularization_y):
        if False:
            while True:
                i = 10
        assert_is_type(regularization_y, None, Enum('none', 'quadratic', 'l2', 'l1', 'non_negative', 'one_sparse', 'unit_one_sparse', 'simplex'))
        self._parms['regularization_y'] = regularization_y

    @property
    def gamma_x(self):
        if False:
            return 10
        '\n        Regularization weight on X matrix\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")\n        >>> rank = 3\n        >>> gx = 0.5\n        >>> gy = 0.5\n        >>> trans = "standardize"\n        >>> iris_glrm = H2OGeneralizedLowRankEstimator(k=rank,\n        ...                                            loss="Quadratic",\n        ...                                            gamma_x=gx,\n        ...                                            gamma_y=gy,\n        ...                                            transform=trans)\n        >>> iris_glrm.train(x=iris.names, training_frame=iris)\n        >>> iris_glrm.show()\n        '
        return self._parms.get('gamma_x')

    @gamma_x.setter
    def gamma_x(self, gamma_x):
        if False:
            while True:
                i = 10
        assert_is_type(gamma_x, None, numeric)
        self._parms['gamma_x'] = gamma_x

    @property
    def gamma_y(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Regularization weight on Y matrix\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")\n        >>> rank = 3\n        >>> gx = 0.5\n        >>> gy = 0.5\n        >>> trans = "standardize"\n        >>> iris_glrm = H2OGeneralizedLowRankEstimator(k=rank,\n        ...                                            loss="Quadratic",\n        ...                                            gamma_x=gx,\n        ...                                            gamma_y=gy,\n        ...                                            transform=trans)\n        >>> iris_glrm.train(x=iris.names, training_frame=iris)\n        >>> iris_glrm.show()\n        '
        return self._parms.get('gamma_y')

    @gamma_y.setter
    def gamma_y(self, gamma_y):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(gamma_y, None, numeric)
        self._parms['gamma_y'] = gamma_y

    @property
    def max_iterations(self):
        if False:
            i = 10
            return i + 15
        '\n        Maximum number of iterations\n\n        Type: ``int``, defaults to ``1000``.\n\n        :examples:\n\n        >>> acs = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/census/ACS_13_5YR_DP02_cleaned.zip")\n        >>> acs_fill = acs.drop("ZCTA5")\n        >>> acs_glrm = H2OGeneralizedLowRankEstimator(k=10,\n        ...                                           transform="standardize",\n        ...                                           loss="quadratic",\n        ...                                           regularization_x="quadratic",\n        ...                                           regularization_y="L1",\n        ...                                           gamma_x=0.25,\n        ...                                           gamma_y=0.5,\n        ...                                           max_iterations=700)\n        >>> acs_glrm.train(x=acs_fill.names, training_frame=acs)\n        >>> acs_glrm.show()\n        '
        return self._parms.get('max_iterations')

    @max_iterations.setter
    def max_iterations(self, max_iterations):
        if False:
            print('Hello World!')
        assert_is_type(max_iterations, None, int)
        self._parms['max_iterations'] = max_iterations

    @property
    def max_updates(self):
        if False:
            i = 10
            return i + 15
        '\n        Maximum number of updates, defaults to 2*max_iterations\n\n        Type: ``int``, defaults to ``2000``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/pca_test/USArrests.csv")\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                               max_runtime_secs=15,\n        ...                                               max_iterations=500,\n        ...                                               max_updates=900,\n        ...                                               min_step_size=0.005)\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('max_updates')

    @max_updates.setter
    def max_updates(self, max_updates):
        if False:
            print('Hello World!')
        assert_is_type(max_updates, None, int)
        self._parms['max_updates'] = max_updates

    @property
    def init_step_size(self):
        if False:
            return 10
        '\n        Initial step size\n\n        Type: ``float``, defaults to ``1.0``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")\n        >>> iris_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                            init_step_size=2.5,\n        ...                                            seed=1234) \n        >>> iris_glrm.train(x=iris.names, training_frame=iris)\n        >>> iris_glrm.show()\n        '
        return self._parms.get('init_step_size')

    @init_step_size.setter
    def init_step_size(self, init_step_size):
        if False:
            i = 10
            return i + 15
        assert_is_type(init_step_size, None, numeric)
        self._parms['init_step_size'] = init_step_size

    @property
    def min_step_size(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Minimum step size\n\n        Type: ``float``, defaults to ``0.0001``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/pca_test/USArrests.csv")\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                               max_runtime_secs=15,\n        ...                                               max_iterations=500,\n        ...                                               max_updates=900,\n        ...                                               min_step_size=0.005)\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('min_step_size')

    @min_step_size.setter
    def min_step_size(self, min_step_size):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(min_step_size, None, numeric)
        self._parms['min_step_size'] = min_step_size

    @property
    def seed(self):
        if False:
            print('Hello World!')
        '\n        RNG seed for initialization\n\n        Type: ``int``, defaults to ``-1``.\n\n        :examples:\n\n        >>> prostate = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate_cat.csv")\n        >>> prostate[0] = prostate[0].asnumeric()\n        >>> prostate[4] = prostate[4].asnumeric()\n        >>> glrm_w_seed = H2OGeneralizedLowRankEstimator(k=5, seed=12345) \n        >>> glrm_w_seed.train(x=prostate.names, training_frame=prostate)\n        >>> glrm_wo_seed = H2OGeneralizedLowRankEstimator(k=5, \n        >>> glrm_wo_seed.train(x=prostate.names, training_frame=prostate)\n        >>> glrm_w_seed.show()\n        >>> glrm_wo_seed.show()\n        '
        return self._parms.get('seed')

    @seed.setter
    def seed(self, seed):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(seed, None, int)
        self._parms['seed'] = seed

    @property
    def init(self):
        if False:
            while True:
                i = 10
        '\n        Initialization mode\n\n        Type: ``Literal["random", "svd", "plus_plus", "user"]``, defaults to ``"plus_plus"``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")\n        >>> iris_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                            init="svd",\n        ...                                            seed=1234) \n        >>> iris_glrm.train(x=iris.names, training_frame=iris)\n        >>> iris_glrm.show()\n        '
        return self._parms.get('init')

    @init.setter
    def init(self, init):
        if False:
            print('Hello World!')
        assert_is_type(init, None, Enum('random', 'svd', 'plus_plus', 'user'))
        self._parms['init'] = init

    @property
    def svd_method(self):
        if False:
            while True:
                i = 10
        '\n        Method for computing SVD during initialization (Caution: Randomized is currently experimental and unstable)\n\n        Type: ``Literal["gram_s_v_d", "power", "randomized"]``, defaults to ``"randomized"``.\n\n        :examples:\n\n        >>> prostate = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate_cat.csv")\n        >>> prostate[0] = prostate[0].asnumeric()\n        >>> prostate[4] = prostate[4].asnumeric()\n        >>> pros_glrm = H2OGeneralizedLowRankEstimator(k=5,\n        ...                                            svd_method="power",\n        ...                                            seed=1234)\n        >>> pros_glrm.train(x=prostate.names, training_frame=prostate)\n        >>> pros_glrm.show()\n        '
        return self._parms.get('svd_method')

    @svd_method.setter
    def svd_method(self, svd_method):
        if False:
            i = 10
            return i + 15
        assert_is_type(svd_method, None, Enum('gram_s_v_d', 'power', 'randomized'))
        self._parms['svd_method'] = svd_method

    @property
    def user_y(self):
        if False:
            return 10
        '\n        User-specified initial Y\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/pca_test/USArrests.csv")\n        >>> initial_y = [[5.412,  65.24,  -7.54, -0.032],\n        ...              [2.212,  92.24, -17.54, 23.268],\n        ...              [0.312, 123.24,  14.46,  9.768],\n        ...              [1.012,  19.24, -15.54, -1.732]]\n        >>> initial_y_h2o = h2o.H2OFrame(list(zip(*initial_y)))\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=4,\n        ...                                               transform="demean",\n        ...                                               loss="quadratic",\n        ...                                               gamma_x=0.5,\n        ...                                               gamma_y=0.3,\n        ...                                               init="user",\n        ...                                               user_y=initial_y_h2o,\n        ...                                               recover_svd=True)\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('user_y')

    @user_y.setter
    def user_y(self, user_y):
        if False:
            print('Hello World!')
        self._parms['user_y'] = H2OFrame._validate(user_y, 'user_y')

    @property
    def user_x(self):
        if False:
            i = 10
            return i + 15
        '\n        User-specified initial X\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/pca_test/USArrests.csv")\n        >>> initial_x = ([[5.412, 65.24, -7.54, -0.032, 2.212, 92.24, -17.54, 23.268, 0.312,\n        ...                123.24, 14.46, 9.768, 1.012, 19.24, -15.54, -1.732, 5.412, 65.24,\n        ...                -7.54, -0.032, 2.212, 92.24, -17.54, 23.268, 0.312, 123.24, 14.46,\n        ...                9.76, 1.012, 19.24, -15.54, -1.732, 5.412, 65.24, -7.54, -0.032,\n        ...                2.212, 92.24, -17.54, 23.268, 0.312, 123.24, 14.46, 9.768, 1.012,\n        ...                19.24, -15.54, -1.732, 5.412, 65.24]]*4)\n        >>> initial_x_h2o = h2o.H2OFrame(list(zip(*initial_x)))\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=4,\n        ...                                               transform="demean",\n        ...                                               loss="quadratic",\n        ...                                               gamma_x=0.5,\n        ...                                               gamma_y=0.3,\n        ...                                               init="user",\n        ...                                               user_x=initial_x_h2o,\n        ...                                               recover_svd=True)\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('user_x')

    @user_x.setter
    def user_x(self, user_x):
        if False:
            i = 10
            return i + 15
        self._parms['user_x'] = H2OFrame._validate(user_x, 'user_x')

    @property
    def expand_user_y(self):
        if False:
            i = 10
            return i + 15
        '\n        Expand categorical columns in user-specified initial Y\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")\n        >>> rank = 3\n        >>> gx = 0.5\n        >>> gy = 0.5\n        >>> trans = "standardize"\n        >>> iris_glrm = H2OGeneralizedLowRankEstimator(k=rank,\n        ...                                            loss="Quadratic",\n        ...                                            gamma_x=gx,\n        ...                                            gamma_y=gy,\n        ...                                            transform=trans,\n        ...                                            expand_user_y=False)\n        >>> iris_glrm.train(x=iris.names, training_frame=iris)\n        >>> iris_glrm.show()\n        '
        return self._parms.get('expand_user_y')

    @expand_user_y.setter
    def expand_user_y(self, expand_user_y):
        if False:
            print('Hello World!')
        assert_is_type(expand_user_y, None, bool)
        self._parms['expand_user_y'] = expand_user_y

    @property
    def impute_original(self):
        if False:
            while True:
                i = 10
        '\n        Reconstruct original training data by reversing transform\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")\n        >>> rank = 3\n        >>> gx = 0.5\n        >>> gy = 0.5\n        >>> trans = "standardize"\n        >>> iris_glrm = H2OGeneralizedLowRankEstimator(k=rank,\n        ...                                            loss="Quadratic",\n        ...                                            gamma_x=gx,\n        ...                                            gamma_y=gy,\n        ...                                            transform=trans\n        ...                                            impute_original=True)\n        >>> iris_glrm.train(x=iris.names, training_frame=iris)\n        >>> iris_glrm.show()\n        '
        return self._parms.get('impute_original')

    @impute_original.setter
    def impute_original(self, impute_original):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(impute_original, None, bool)
        self._parms['impute_original'] = impute_original

    @property
    def recover_svd(self):
        if False:
            return 10
        '\n        Recover singular values and eigenvectors of XY\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> prostate = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate_cat.csv")\n        >>> prostate[0] = prostate[0].asnumeric()\n        >>> prostate[4] = prostate[4].asnumeric()\n        >>> loss_all = ["Hinge", "Quadratic", "Categorical", "Categorical",\n        ...             "Hinge", "Quadratic", "Quadratic", "Quadratic"]\n        >>> pros_glrm = H2OGeneralizedLowRankEstimator(k=5,\n        ...                                            loss_by_col=loss_all,\n        ...                                            recover_svd=True,\n        ...                                            transform="standardize",\n        ...                                            seed=12345)\n        >>> pros_glrm.train(x=prostate.names, training_frame=prostate)\n        >>> pros_glrm.show()\n        '
        return self._parms.get('recover_svd')

    @recover_svd.setter
    def recover_svd(self, recover_svd):
        if False:
            while True:
                i = 10
        assert_is_type(recover_svd, None, bool)
        self._parms['recover_svd'] = recover_svd

    @property
    def max_runtime_secs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Maximum allowed runtime in seconds for model training. Use 0 to disable.\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> arrestsH2O = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/pca_test/USArrests.csv")\n        >>> arrests_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                               max_runtime_secs=15,\n        ...                                               max_iterations=500,\n        ...                                               max_updates=900,\n        ...                                               min_step_size=0.005)\n        >>> arrests_glrm.train(x=arrestsH2O.names, training_frame=arrestsH2O)\n        >>> arrests_glrm.show()\n        '
        return self._parms.get('max_runtime_secs')

    @max_runtime_secs.setter
    def max_runtime_secs(self, max_runtime_secs):
        if False:
            while True:
                i = 10
        assert_is_type(max_runtime_secs, None, numeric)
        self._parms['max_runtime_secs'] = max_runtime_secs

    @property
    def export_checkpoints_dir(self):
        if False:
            return 10
        '\n        Automatically export generated models to this directory.\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> import tempfile\n        >>> from os import listdir\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv")\n        >>> checkpoints_dir = tempfile.mkdtemp()\n        >>> iris_glrm = H2OGeneralizedLowRankEstimator(k=3,\n        ...                                            export_checkpoints_dir=checkpoints_dir,\n        ...                                            seed=1234)\n        >>> iris_glrm.train(x=iris.names, training_frame=iris)\n        >>> len(listdir(checkpoints_dir))\n        '
        return self._parms.get('export_checkpoints_dir')

    @export_checkpoints_dir.setter
    def export_checkpoints_dir(self, export_checkpoints_dir):
        if False:
            i = 10
            return i + 15
        assert_is_type(export_checkpoints_dir, None, str)
        self._parms['export_checkpoints_dir'] = export_checkpoints_dir

    def transform_frame(self, fr):
        if False:
            for i in range(10):
                print('nop')
        '\n        GLRM performs A=X*Y during training.  When a new dataset is given, GLRM will perform Anew = Xnew*Y.  When\n        predict is called, Xnew*Y is returned.  When transform_frame is called, Xnew is returned instead.\n        :return: an H2OFrame that contains Xnew.\n        '
        return H2OFrame._expr(expr=ExprNode('transform', ASTId(self.key), ASTId(fr.key)))._frame(fill_cache=True)