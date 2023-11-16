import h2o
from h2o.base import Keyed
from h2o.frame import H2OFrame
from h2o.expr import ExprNode
from h2o.expr import ASTId
from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OANOVAGLMEstimator(H2OEstimator):
    """
    ANOVA for Generalized Linear Model

    H2O ANOVAGLM is used to calculate Type III SS which is used to evaluate the contributions of individual predictors 
    and their interactions to a model.  Predictors or interactions with negligible contributions to the model will have 
    high p-values while those with more contributions will have low p-values. 
    """
    algo = 'anovaglm'
    supervised_learning = True

    def __init__(self, model_id=None, training_frame=None, seed=-1, response_column=None, ignored_columns=None, ignore_const_cols=True, score_each_iteration=False, offset_column=None, weights_column=None, family='auto', tweedie_variance_power=0.0, tweedie_link_power=1.0, theta=0.0, solver='irlsm', missing_values_handling='mean_imputation', plug_values=None, compute_p_values=True, standardize=True, non_negative=False, max_iterations=0, link='family_default', prior=0.0, alpha=None, lambda_=[0.0], lambda_search=False, stopping_rounds=0, stopping_metric='auto', early_stopping=False, stopping_tolerance=0.001, balance_classes=False, class_sampling_factors=None, max_after_balance_size=5.0, max_runtime_secs=0.0, save_transformed_framekeys=False, highest_interaction_term=0, nparallelism=4, type=0):
        if False:
            i = 10
            return i + 15
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param seed: Seed for pseudo random number generator (if applicable)\n               Defaults to ``-1``.\n        :type seed: int\n        :param response_column: Response variable column.\n               Defaults to ``None``.\n        :type response_column: str, optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param ignore_const_cols: Ignore constant columns.\n               Defaults to ``True``.\n        :type ignore_const_cols: bool\n        :param score_each_iteration: Whether to score during each iteration of model training.\n               Defaults to ``False``.\n        :type score_each_iteration: bool\n        :param offset_column: Offset column. This will be added to the combination of columns before applying the link\n               function.\n               Defaults to ``None``.\n        :type offset_column: str, optional\n        :param weights_column: Column with observation weights. Giving some observation a weight of zero is equivalent\n               to excluding it from the dataset; giving an observation a relative weight of 2 is equivalent to repeating\n               that row twice. Negative weights are not allowed. Note: Weights are per-row observation weights and do\n               not increase the size of the data frame. This is typically the number of times a row is repeated, but\n               non-integer values are supported as well. During training, rows with higher weights matter more, due to\n               the larger loss function pre-factor. If you set weight = 0 for a row, the returned prediction frame at\n               that row is zero and this is incorrect. To get an accurate prediction, remove all rows with weight == 0.\n               Defaults to ``None``.\n        :type weights_column: str, optional\n        :param family: Family. Use binomial for classification with logistic regression, others are for regression\n               problems.\n               Defaults to ``"auto"``.\n        :type family: Literal["auto", "gaussian", "binomial", "fractionalbinomial", "quasibinomial", "poisson", "gamma",\n               "tweedie", "negativebinomial"]\n        :param tweedie_variance_power: Tweedie variance power\n               Defaults to ``0.0``.\n        :type tweedie_variance_power: float\n        :param tweedie_link_power: Tweedie link power\n               Defaults to ``1.0``.\n        :type tweedie_link_power: float\n        :param theta: Theta\n               Defaults to ``0.0``.\n        :type theta: float\n        :param solver: AUTO will set the solver based on given data and the other parameters. IRLSM is fast on on\n               problems with small number of predictors and for lambda-search with L1 penalty, L_BFGS scales better for\n               datasets with many columns.\n               Defaults to ``"irlsm"``.\n        :type solver: Literal["auto", "irlsm", "l_bfgs", "coordinate_descent_naive", "coordinate_descent",\n               "gradient_descent_lh", "gradient_descent_sqerr"]\n        :param missing_values_handling: Handling of missing values. Either MeanImputation, Skip or PlugValues.\n               Defaults to ``"mean_imputation"``.\n        :type missing_values_handling: Literal["mean_imputation", "skip", "plug_values"]\n        :param plug_values: Plug Values (a single row frame containing values that will be used to impute missing values\n               of the training/validation frame, use with conjunction missing_values_handling = PlugValues)\n               Defaults to ``None``.\n        :type plug_values: Union[None, str, H2OFrame], optional\n        :param compute_p_values: Request p-values computation, p-values work only with IRLSM solver and no\n               regularization\n               Defaults to ``True``.\n        :type compute_p_values: bool\n        :param standardize: Standardize numeric columns to have zero mean and unit variance\n               Defaults to ``True``.\n        :type standardize: bool\n        :param non_negative: Restrict coefficients (not intercept) to be non-negative\n               Defaults to ``False``.\n        :type non_negative: bool\n        :param max_iterations: Maximum number of iterations\n               Defaults to ``0``.\n        :type max_iterations: int\n        :param link: Link function.\n               Defaults to ``"family_default"``.\n        :type link: Literal["family_default", "identity", "logit", "log", "inverse", "tweedie", "ologit"]\n        :param prior: Prior probability for y==1. To be used only for logistic regression iff the data has been sampled\n               and the mean of response does not reflect reality.\n               Defaults to ``0.0``.\n        :type prior: float\n        :param alpha: Distribution of regularization between the L1 (Lasso) and L2 (Ridge) penalties. A value of 1 for\n               alpha represents Lasso regression, a value of 0 produces Ridge regression, and anything in between\n               specifies the amount of mixing between the two. Default value of alpha is 0 when SOLVER = \'L-BFGS\'; 0.5\n               otherwise.\n               Defaults to ``None``.\n        :type alpha: List[float], optional\n        :param lambda_: Regularization strength\n               Defaults to ``[0.0]``.\n        :type lambda_: List[float]\n        :param lambda_search: Use lambda search starting at lambda max, given lambda is then interpreted as lambda min\n               Defaults to ``False``.\n        :type lambda_search: bool\n        :param stopping_rounds: Early stopping based on convergence of stopping_metric. Stop if simple moving average of\n               length k of the stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)\n               Defaults to ``0``.\n        :type stopping_rounds: int\n        :param stopping_metric: Metric to use for early stopping (AUTO: logloss for classification, deviance for\n               regression and anomaly_score for Isolation Forest). Note that custom and custom_increasing can only be\n               used in GBM and DRF with the Python client.\n               Defaults to ``"auto"``.\n        :type stopping_metric: Literal["auto", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "lift_top_group",\n               "misclassification", "mean_per_class_error", "custom", "custom_increasing"]\n        :param early_stopping: Stop early when there is no more relative improvement on train or validation (if\n               provided).\n               Defaults to ``False``.\n        :type early_stopping: bool\n        :param stopping_tolerance: Relative tolerance for metric-based stopping criterion (stop if relative improvement\n               is not at least this much)\n               Defaults to ``0.001``.\n        :type stopping_tolerance: float\n        :param balance_classes: Balance training data class counts via over/under-sampling (for imbalanced data).\n               Defaults to ``False``.\n        :type balance_classes: bool\n        :param class_sampling_factors: Desired over/under-sampling ratios per class (in lexicographic order). If not\n               specified, sampling factors will be automatically computed to obtain class balance during training.\n               Requires balance_classes.\n               Defaults to ``None``.\n        :type class_sampling_factors: List[float], optional\n        :param max_after_balance_size: Maximum relative size of the training data after balancing class counts (can be\n               less than 1.0). Requires balance_classes.\n               Defaults to ``5.0``.\n        :type max_after_balance_size: float\n        :param max_runtime_secs: Maximum allowed runtime in seconds for model training. Use 0 to disable.\n               Defaults to ``0.0``.\n        :type max_runtime_secs: float\n        :param save_transformed_framekeys: true to save the keys of transformed predictors and interaction column.\n               Defaults to ``False``.\n        :type save_transformed_framekeys: bool\n        :param highest_interaction_term: Limit the number of interaction terms, if 2 means interaction between 2 columns\n               only, 3 for three columns and so on...  Default to 2.\n               Defaults to ``0``.\n        :type highest_interaction_term: int\n        :param nparallelism: Number of models to build in parallel.  Default to 4.  Adjust according to your system.\n               Defaults to ``4``.\n        :type nparallelism: int\n        :param type: Refer to the SS type 1, 2, 3, or 4.  We are currently only supporting 3\n               Defaults to ``0``.\n        :type type: int\n        '
        super(H2OANOVAGLMEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.seed = seed
        self.response_column = response_column
        self.ignored_columns = ignored_columns
        self.ignore_const_cols = ignore_const_cols
        self.score_each_iteration = score_each_iteration
        self.offset_column = offset_column
        self.weights_column = weights_column
        self.family = family
        self.tweedie_variance_power = tweedie_variance_power
        self.tweedie_link_power = tweedie_link_power
        self.theta = theta
        self.solver = solver
        self.missing_values_handling = missing_values_handling
        self.plug_values = plug_values
        self.compute_p_values = compute_p_values
        self.standardize = standardize
        self.non_negative = non_negative
        self.max_iterations = max_iterations
        self.link = link
        self.prior = prior
        self.alpha = alpha
        self.lambda_ = lambda_
        self.lambda_search = lambda_search
        self.stopping_rounds = stopping_rounds
        self.stopping_metric = stopping_metric
        self.early_stopping = early_stopping
        self.stopping_tolerance = stopping_tolerance
        self.balance_classes = balance_classes
        self.class_sampling_factors = class_sampling_factors
        self.max_after_balance_size = max_after_balance_size
        self.max_runtime_secs = max_runtime_secs
        self.save_transformed_framekeys = save_transformed_framekeys
        self.highest_interaction_term = highest_interaction_term
        self.nparallelism = nparallelism
        self.type = type
        self._parms['_rest_version'] = 3

    @property
    def training_frame(self):
        if False:
            return 10
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('training_frame')

    @training_frame.setter
    def training_frame(self, training_frame):
        if False:
            i = 10
            return i + 15
        self._parms['training_frame'] = H2OFrame._validate(training_frame, 'training_frame')

    @property
    def seed(self):
        if False:
            return 10
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
    def response_column(self):
        if False:
            while True:
                i = 10
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
    def ignored_columns(self):
        if False:
            i = 10
            return i + 15
        '\n        Names of columns to ignore for training.\n\n        Type: ``List[str]``.\n        '
        return self._parms.get('ignored_columns')

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
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
            return 10
        assert_is_type(score_each_iteration, None, bool)
        self._parms['score_each_iteration'] = score_each_iteration

    @property
    def offset_column(self):
        if False:
            while True:
                i = 10
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
    def family(self):
        if False:
            print('Hello World!')
        '\n        Family. Use binomial for classification with logistic regression, others are for regression problems.\n\n        Type: ``Literal["auto", "gaussian", "binomial", "fractionalbinomial", "quasibinomial", "poisson", "gamma",\n        "tweedie", "negativebinomial"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('family')

    @family.setter
    def family(self, family):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(family, None, Enum('auto', 'gaussian', 'binomial', 'fractionalbinomial', 'quasibinomial', 'poisson', 'gamma', 'tweedie', 'negativebinomial'))
        self._parms['family'] = family

    @property
    def tweedie_variance_power(self):
        if False:
            print('Hello World!')
        '\n        Tweedie variance power\n\n        Type: ``float``, defaults to ``0.0``.\n        '
        return self._parms.get('tweedie_variance_power')

    @tweedie_variance_power.setter
    def tweedie_variance_power(self, tweedie_variance_power):
        if False:
            while True:
                i = 10
        assert_is_type(tweedie_variance_power, None, numeric)
        self._parms['tweedie_variance_power'] = tweedie_variance_power

    @property
    def tweedie_link_power(self):
        if False:
            print('Hello World!')
        '\n        Tweedie link power\n\n        Type: ``float``, defaults to ``1.0``.\n        '
        return self._parms.get('tweedie_link_power')

    @tweedie_link_power.setter
    def tweedie_link_power(self, tweedie_link_power):
        if False:
            return 10
        assert_is_type(tweedie_link_power, None, numeric)
        self._parms['tweedie_link_power'] = tweedie_link_power

    @property
    def theta(self):
        if False:
            while True:
                i = 10
        '\n        Theta\n\n        Type: ``float``, defaults to ``0.0``.\n        '
        return self._parms.get('theta')

    @theta.setter
    def theta(self, theta):
        if False:
            i = 10
            return i + 15
        assert_is_type(theta, None, numeric)
        self._parms['theta'] = theta

    @property
    def solver(self):
        if False:
            return 10
        '\n        AUTO will set the solver based on given data and the other parameters. IRLSM is fast on on problems with small\n        number of predictors and for lambda-search with L1 penalty, L_BFGS scales better for datasets with many columns.\n\n        Type: ``Literal["auto", "irlsm", "l_bfgs", "coordinate_descent_naive", "coordinate_descent",\n        "gradient_descent_lh", "gradient_descent_sqerr"]``, defaults to ``"irlsm"``.\n        '
        return self._parms.get('solver')

    @solver.setter
    def solver(self, solver):
        if False:
            while True:
                i = 10
        assert_is_type(solver, None, Enum('auto', 'irlsm', 'l_bfgs', 'coordinate_descent_naive', 'coordinate_descent', 'gradient_descent_lh', 'gradient_descent_sqerr'))
        self._parms['solver'] = solver

    @property
    def missing_values_handling(self):
        if False:
            i = 10
            return i + 15
        '\n        Handling of missing values. Either MeanImputation, Skip or PlugValues.\n\n        Type: ``Literal["mean_imputation", "skip", "plug_values"]``, defaults to ``"mean_imputation"``.\n        '
        return self._parms.get('missing_values_handling')

    @missing_values_handling.setter
    def missing_values_handling(self, missing_values_handling):
        if False:
            return 10
        assert_is_type(missing_values_handling, None, Enum('mean_imputation', 'skip', 'plug_values'))
        self._parms['missing_values_handling'] = missing_values_handling

    @property
    def plug_values(self):
        if False:
            print('Hello World!')
        '\n        Plug Values (a single row frame containing values that will be used to impute missing values of the\n        training/validation frame, use with conjunction missing_values_handling = PlugValues)\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('plug_values')

    @plug_values.setter
    def plug_values(self, plug_values):
        if False:
            while True:
                i = 10
        self._parms['plug_values'] = H2OFrame._validate(plug_values, 'plug_values')

    @property
    def compute_p_values(self):
        if False:
            return 10
        '\n        Request p-values computation, p-values work only with IRLSM solver and no regularization\n\n        Type: ``bool``, defaults to ``True``.\n        '
        return self._parms.get('compute_p_values')

    @compute_p_values.setter
    def compute_p_values(self, compute_p_values):
        if False:
            print('Hello World!')
        assert_is_type(compute_p_values, None, bool)
        self._parms['compute_p_values'] = compute_p_values

    @property
    def standardize(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Standardize numeric columns to have zero mean and unit variance\n\n        Type: ``bool``, defaults to ``True``.\n        '
        return self._parms.get('standardize')

    @standardize.setter
    def standardize(self, standardize):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(standardize, None, bool)
        self._parms['standardize'] = standardize

    @property
    def non_negative(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Restrict coefficients (not intercept) to be non-negative\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('non_negative')

    @non_negative.setter
    def non_negative(self, non_negative):
        if False:
            i = 10
            return i + 15
        assert_is_type(non_negative, None, bool)
        self._parms['non_negative'] = non_negative

    @property
    def max_iterations(self):
        if False:
            print('Hello World!')
        '\n        Maximum number of iterations\n\n        Type: ``int``, defaults to ``0``.\n        '
        return self._parms.get('max_iterations')

    @max_iterations.setter
    def max_iterations(self, max_iterations):
        if False:
            i = 10
            return i + 15
        assert_is_type(max_iterations, None, int)
        self._parms['max_iterations'] = max_iterations

    @property
    def link(self):
        if False:
            return 10
        '\n        Link function.\n\n        Type: ``Literal["family_default", "identity", "logit", "log", "inverse", "tweedie", "ologit"]``, defaults to\n        ``"family_default"``.\n        '
        return self._parms.get('link')

    @link.setter
    def link(self, link):
        if False:
            i = 10
            return i + 15
        assert_is_type(link, None, Enum('family_default', 'identity', 'logit', 'log', 'inverse', 'tweedie', 'ologit'))
        self._parms['link'] = link

    @property
    def prior(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prior probability for y==1. To be used only for logistic regression iff the data has been sampled and the mean\n        of response does not reflect reality.\n\n        Type: ``float``, defaults to ``0.0``.\n        '
        return self._parms.get('prior')

    @prior.setter
    def prior(self, prior):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(prior, None, numeric)
        self._parms['prior'] = prior

    @property
    def alpha(self):
        if False:
            return 10
        "\n        Distribution of regularization between the L1 (Lasso) and L2 (Ridge) penalties. A value of 1 for alpha\n        represents Lasso regression, a value of 0 produces Ridge regression, and anything in between specifies the\n        amount of mixing between the two. Default value of alpha is 0 when SOLVER = 'L-BFGS'; 0.5 otherwise.\n\n        Type: ``List[float]``.\n        "
        return self._parms.get('alpha')

    @alpha.setter
    def alpha(self, alpha):
        if False:
            return 10
        assert_is_type(alpha, None, numeric, [numeric])
        self._parms['alpha'] = alpha

    @property
    def lambda_(self):
        if False:
            i = 10
            return i + 15
        '\n        Regularization strength\n\n        Type: ``List[float]``, defaults to ``[0.0]``.\n        '
        return self._parms.get('lambda')

    @lambda_.setter
    def lambda_(self, lambda_):
        if False:
            return 10
        assert_is_type(lambda_, None, numeric, [numeric])
        self._parms['lambda'] = lambda_

    @property
    def lambda_search(self):
        if False:
            i = 10
            return i + 15
        '\n        Use lambda search starting at lambda max, given lambda is then interpreted as lambda min\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('lambda_search')

    @lambda_search.setter
    def lambda_search(self, lambda_search):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(lambda_search, None, bool)
        self._parms['lambda_search'] = lambda_search

    @property
    def stopping_rounds(self):
        if False:
            print('Hello World!')
        '\n        Early stopping based on convergence of stopping_metric. Stop if simple moving average of length k of the\n        stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)\n\n        Type: ``int``, defaults to ``0``.\n        '
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
        '\n        Metric to use for early stopping (AUTO: logloss for classification, deviance for regression and anomaly_score\n        for Isolation Forest). Note that custom and custom_increasing can only be used in GBM and DRF with the Python\n        client.\n\n        Type: ``Literal["auto", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "lift_top_group",\n        "misclassification", "mean_per_class_error", "custom", "custom_increasing"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('stopping_metric')

    @stopping_metric.setter
    def stopping_metric(self, stopping_metric):
        if False:
            i = 10
            return i + 15
        assert_is_type(stopping_metric, None, Enum('auto', 'deviance', 'logloss', 'mse', 'rmse', 'mae', 'rmsle', 'auc', 'aucpr', 'lift_top_group', 'misclassification', 'mean_per_class_error', 'custom', 'custom_increasing'))
        self._parms['stopping_metric'] = stopping_metric

    @property
    def early_stopping(self):
        if False:
            print('Hello World!')
        '\n        Stop early when there is no more relative improvement on train or validation (if provided).\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('early_stopping')

    @early_stopping.setter
    def early_stopping(self, early_stopping):
        if False:
            while True:
                i = 10
        assert_is_type(early_stopping, None, bool)
        self._parms['early_stopping'] = early_stopping

    @property
    def stopping_tolerance(self):
        if False:
            print('Hello World!')
        '\n        Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at least this much)\n\n        Type: ``float``, defaults to ``0.001``.\n        '
        return self._parms.get('stopping_tolerance')

    @stopping_tolerance.setter
    def stopping_tolerance(self, stopping_tolerance):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(stopping_tolerance, None, numeric)
        self._parms['stopping_tolerance'] = stopping_tolerance

    @property
    def balance_classes(self):
        if False:
            print('Hello World!')
        '\n        Balance training data class counts via over/under-sampling (for imbalanced data).\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('balance_classes')

    @balance_classes.setter
    def balance_classes(self, balance_classes):
        if False:
            while True:
                i = 10
        assert_is_type(balance_classes, None, bool)
        self._parms['balance_classes'] = balance_classes

    @property
    def class_sampling_factors(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Desired over/under-sampling ratios per class (in lexicographic order). If not specified, sampling factors will\n        be automatically computed to obtain class balance during training. Requires balance_classes.\n\n        Type: ``List[float]``.\n        '
        return self._parms.get('class_sampling_factors')

    @class_sampling_factors.setter
    def class_sampling_factors(self, class_sampling_factors):
        if False:
            print('Hello World!')
        assert_is_type(class_sampling_factors, None, [float])
        self._parms['class_sampling_factors'] = class_sampling_factors

    @property
    def max_after_balance_size(self):
        if False:
            while True:
                i = 10
        '\n        Maximum relative size of the training data after balancing class counts (can be less than 1.0). Requires\n        balance_classes.\n\n        Type: ``float``, defaults to ``5.0``.\n        '
        return self._parms.get('max_after_balance_size')

    @max_after_balance_size.setter
    def max_after_balance_size(self, max_after_balance_size):
        if False:
            return 10
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
            print('Hello World!')
        assert_is_type(max_runtime_secs, None, numeric)
        self._parms['max_runtime_secs'] = max_runtime_secs

    @property
    def save_transformed_framekeys(self):
        if False:
            return 10
        '\n        true to save the keys of transformed predictors and interaction column.\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('save_transformed_framekeys')

    @save_transformed_framekeys.setter
    def save_transformed_framekeys(self, save_transformed_framekeys):
        if False:
            print('Hello World!')
        assert_is_type(save_transformed_framekeys, None, bool)
        self._parms['save_transformed_framekeys'] = save_transformed_framekeys

    @property
    def highest_interaction_term(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Limit the number of interaction terms, if 2 means interaction between 2 columns only, 3 for three columns and so\n        on...  Default to 2.\n\n        Type: ``int``, defaults to ``0``.\n        '
        return self._parms.get('highest_interaction_term')

    @highest_interaction_term.setter
    def highest_interaction_term(self, highest_interaction_term):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(highest_interaction_term, None, int)
        self._parms['highest_interaction_term'] = highest_interaction_term

    @property
    def nparallelism(self):
        if False:
            return 10
        '\n        Number of models to build in parallel.  Default to 4.  Adjust according to your system.\n\n        Type: ``int``, defaults to ``4``.\n        '
        return self._parms.get('nparallelism')

    @nparallelism.setter
    def nparallelism(self, nparallelism):
        if False:
            return 10
        assert_is_type(nparallelism, None, int)
        self._parms['nparallelism'] = nparallelism

    @property
    def type(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Refer to the SS type 1, 2, 3, or 4.  We are currently only supporting 3\n\n        Type: ``int``, defaults to ``0``.\n        '
        return self._parms.get('type')

    @type.setter
    def type(self, type):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(type, None, int)
        self._parms['type'] = type

    @property
    def Lambda(self):
        if False:
            return 10
        'DEPRECATED. Use ``self.lambda_`` instead'
        return self._parms['lambda'] if 'lambda' in self._parms else None

    @Lambda.setter
    def Lambda(self, value):
        if False:
            i = 10
            return i + 15
        self._parms['lambda'] = value

    def result(self):
        if False:
            print('Hello World!')
        '\n        Get result frame that contains information about the model building process like for modelselection and anovaglm.\n        :return: the H2OFrame that contains information about the model building process like for modelselection and anovaglm.\n        '
        return H2OFrame._expr(expr=ExprNode('result', ASTId(self.key)))._frame(fill_cache=True)