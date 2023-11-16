from h2o.utils.metaclass import deprecated_params, deprecated_property
import h2o
from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2ORuleFitEstimator(H2OEstimator):
    """
    RuleFit

    Builds a RuleFit on a parsed dataset, for regression or 
    classification. 
    """
    algo = 'rulefit'
    supervised_learning = True

    @deprecated_params({'Lambda': 'lambda_'})
    def __init__(self, model_id=None, training_frame=None, validation_frame=None, seed=-1, response_column=None, ignored_columns=None, algorithm='auto', min_rule_length=3, max_rule_length=3, max_num_rules=-1, model_type='rules_and_linear', weights_column=None, distribution='auto', rule_generation_ntrees=50, auc_type='auto', remove_duplicates=True, lambda_=None, max_categorical_levels=10):
        if False:
            i = 10
            return i + 15
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param validation_frame: Id of the validation data frame.\n               Defaults to ``None``.\n        :type validation_frame: Union[None, str, H2OFrame], optional\n        :param seed: Seed for pseudo random number generator (if applicable).\n               Defaults to ``-1``.\n        :type seed: int\n        :param response_column: Response variable column.\n               Defaults to ``None``.\n        :type response_column: str, optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param algorithm: The algorithm to use to generate rules.\n               Defaults to ``"auto"``.\n        :type algorithm: Literal["auto", "drf", "gbm"]\n        :param min_rule_length: Minimum length of rules. Defaults to 3.\n               Defaults to ``3``.\n        :type min_rule_length: int\n        :param max_rule_length: Maximum length of rules. Defaults to 3.\n               Defaults to ``3``.\n        :type max_rule_length: int\n        :param max_num_rules: The maximum number of rules to return. defaults to -1 which means the number of rules is\n               selected\n               by diminishing returns in model deviance.\n               Defaults to ``-1``.\n        :type max_num_rules: int\n        :param model_type: Specifies type of base learners in the ensemble.\n               Defaults to ``"rules_and_linear"``.\n        :type model_type: Literal["rules_and_linear", "rules", "linear"]\n        :param weights_column: Column with observation weights. Giving some observation a weight of zero is equivalent\n               to excluding it from the dataset; giving an observation a relative weight of 2 is equivalent to repeating\n               that row twice. Negative weights are not allowed. Note: Weights are per-row observation weights and do\n               not increase the size of the data frame. This is typically the number of times a row is repeated, but\n               non-integer values are supported as well. During training, rows with higher weights matter more, due to\n               the larger loss function pre-factor. If you set weight = 0 for a row, the returned prediction frame at\n               that row is zero and this is incorrect. To get an accurate prediction, remove all rows with weight == 0.\n               Defaults to ``None``.\n        :type weights_column: str, optional\n        :param distribution: Distribution function\n               Defaults to ``"auto"``.\n        :type distribution: Literal["auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace",\n               "quantile", "huber"]\n        :param rule_generation_ntrees: Specifies the number of trees to build in the tree model. Defaults to 50.\n               Defaults to ``50``.\n        :type rule_generation_ntrees: int\n        :param auc_type: Set default multinomial AUC type.\n               Defaults to ``"auto"``.\n        :type auc_type: Literal["auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"]\n        :param remove_duplicates: Whether to remove rules which are identical to an earlier rule. Defaults to true.\n               Defaults to ``True``.\n        :type remove_duplicates: bool\n        :param lambda_: Lambda for LASSO regressor.\n               Defaults to ``None``.\n        :type lambda_: List[float], optional\n        :param max_categorical_levels: For every categorical feature, only use this many most frequent categorical\n               levels for model training. Only used for categorical_encoding == EnumLimited.\n               Defaults to ``10``.\n        :type max_categorical_levels: int\n        '
        super(H2ORuleFitEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.validation_frame = validation_frame
        self.seed = seed
        self.response_column = response_column
        self.ignored_columns = ignored_columns
        self.algorithm = algorithm
        self.min_rule_length = min_rule_length
        self.max_rule_length = max_rule_length
        self.max_num_rules = max_num_rules
        self.model_type = model_type
        self.weights_column = weights_column
        self.distribution = distribution
        self.rule_generation_ntrees = rule_generation_ntrees
        self.auc_type = auc_type
        self.remove_duplicates = remove_duplicates
        self.lambda_ = lambda_
        self.max_categorical_levels = max_categorical_levels

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
    def validation_frame(self):
        if False:
            print('Hello World!')
        '\n        Id of the validation data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('validation_frame')

    @validation_frame.setter
    def validation_frame(self, validation_frame):
        if False:
            i = 10
            return i + 15
        self._parms['validation_frame'] = H2OFrame._validate(validation_frame, 'validation_frame')

    @property
    def seed(self):
        if False:
            print('Hello World!')
        '\n        Seed for pseudo random number generator (if applicable).\n\n        Type: ``int``, defaults to ``-1``.\n        '
        return self._parms.get('seed')

    @seed.setter
    def seed(self, seed):
        if False:
            print('Hello World!')
        assert_is_type(seed, None, int)
        self._parms['seed'] = seed

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
            i = 10
            return i + 15
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
            while True:
                i = 10
        assert_is_type(ignored_columns, None, [str])
        self._parms['ignored_columns'] = ignored_columns

    @property
    def algorithm(self):
        if False:
            return 10
        '\n        The algorithm to use to generate rules.\n\n        Type: ``Literal["auto", "drf", "gbm"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('algorithm')

    @algorithm.setter
    def algorithm(self, algorithm):
        if False:
            i = 10
            return i + 15
        assert_is_type(algorithm, None, Enum('auto', 'drf', 'gbm'))
        self._parms['algorithm'] = algorithm

    @property
    def min_rule_length(self):
        if False:
            print('Hello World!')
        '\n        Minimum length of rules. Defaults to 3.\n\n        Type: ``int``, defaults to ``3``.\n        '
        return self._parms.get('min_rule_length')

    @min_rule_length.setter
    def min_rule_length(self, min_rule_length):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(min_rule_length, None, int)
        self._parms['min_rule_length'] = min_rule_length

    @property
    def max_rule_length(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Maximum length of rules. Defaults to 3.\n\n        Type: ``int``, defaults to ``3``.\n        '
        return self._parms.get('max_rule_length')

    @max_rule_length.setter
    def max_rule_length(self, max_rule_length):
        if False:
            while True:
                i = 10
        assert_is_type(max_rule_length, None, int)
        self._parms['max_rule_length'] = max_rule_length

    @property
    def max_num_rules(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The maximum number of rules to return. defaults to -1 which means the number of rules is selected\n        by diminishing returns in model deviance.\n\n        Type: ``int``, defaults to ``-1``.\n        '
        return self._parms.get('max_num_rules')

    @max_num_rules.setter
    def max_num_rules(self, max_num_rules):
        if False:
            while True:
                i = 10
        assert_is_type(max_num_rules, None, int)
        self._parms['max_num_rules'] = max_num_rules

    @property
    def model_type(self):
        if False:
            return 10
        '\n        Specifies type of base learners in the ensemble.\n\n        Type: ``Literal["rules_and_linear", "rules", "linear"]``, defaults to ``"rules_and_linear"``.\n        '
        return self._parms.get('model_type')

    @model_type.setter
    def model_type(self, model_type):
        if False:
            while True:
                i = 10
        assert_is_type(model_type, None, Enum('rules_and_linear', 'rules', 'linear'))
        self._parms['model_type'] = model_type

    @property
    def weights_column(self):
        if False:
            print('Hello World!')
        '\n        Column with observation weights. Giving some observation a weight of zero is equivalent to excluding it from the\n        dataset; giving an observation a relative weight of 2 is equivalent to repeating that row twice. Negative\n        weights are not allowed. Note: Weights are per-row observation weights and do not increase the size of the data\n        frame. This is typically the number of times a row is repeated, but non-integer values are supported as well.\n        During training, rows with higher weights matter more, due to the larger loss function pre-factor. If you set\n        weight = 0 for a row, the returned prediction frame at that row is zero and this is incorrect. To get an\n        accurate prediction, remove all rows with weight == 0.\n\n        Type: ``str``.\n        '
        return self._parms.get('weights_column')

    @weights_column.setter
    def weights_column(self, weights_column):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(weights_column, None, str)
        self._parms['weights_column'] = weights_column

    @property
    def distribution(self):
        if False:
            print('Hello World!')
        '\n        Distribution function\n\n        Type: ``Literal["auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace",\n        "quantile", "huber"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('distribution')

    @distribution.setter
    def distribution(self, distribution):
        if False:
            return 10
        assert_is_type(distribution, None, Enum('auto', 'bernoulli', 'multinomial', 'gaussian', 'poisson', 'gamma', 'tweedie', 'laplace', 'quantile', 'huber'))
        self._parms['distribution'] = distribution

    @property
    def rule_generation_ntrees(self):
        if False:
            print('Hello World!')
        '\n        Specifies the number of trees to build in the tree model. Defaults to 50.\n\n        Type: ``int``, defaults to ``50``.\n        '
        return self._parms.get('rule_generation_ntrees')

    @rule_generation_ntrees.setter
    def rule_generation_ntrees(self, rule_generation_ntrees):
        if False:
            while True:
                i = 10
        assert_is_type(rule_generation_ntrees, None, int)
        self._parms['rule_generation_ntrees'] = rule_generation_ntrees

    @property
    def auc_type(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set default multinomial AUC type.\n\n        Type: ``Literal["auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"]``, defaults to\n        ``"auto"``.\n        '
        return self._parms.get('auc_type')

    @auc_type.setter
    def auc_type(self, auc_type):
        if False:
            return 10
        assert_is_type(auc_type, None, Enum('auto', 'none', 'macro_ovr', 'weighted_ovr', 'macro_ovo', 'weighted_ovo'))
        self._parms['auc_type'] = auc_type

    @property
    def remove_duplicates(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether to remove rules which are identical to an earlier rule. Defaults to true.\n\n        Type: ``bool``, defaults to ``True``.\n        '
        return self._parms.get('remove_duplicates')

    @remove_duplicates.setter
    def remove_duplicates(self, remove_duplicates):
        if False:
            while True:
                i = 10
        assert_is_type(remove_duplicates, None, bool)
        self._parms['remove_duplicates'] = remove_duplicates

    @property
    def lambda_(self):
        if False:
            i = 10
            return i + 15
        '\n        Lambda for LASSO regressor.\n\n        Type: ``List[float]``.\n        '
        return self._parms.get('lambda')

    @lambda_.setter
    def lambda_(self, lambda_):
        if False:
            print('Hello World!')
        assert_is_type(lambda_, None, numeric, [numeric])
        self._parms['lambda'] = lambda_

    @property
    def max_categorical_levels(self):
        if False:
            i = 10
            return i + 15
        '\n        For every categorical feature, only use this many most frequent categorical levels for model training. Only used\n        for categorical_encoding == EnumLimited.\n\n        Type: ``int``, defaults to ``10``.\n        '
        return self._parms.get('max_categorical_levels')

    @max_categorical_levels.setter
    def max_categorical_levels(self, max_categorical_levels):
        if False:
            print('Hello World!')
        assert_is_type(max_categorical_levels, None, int)
        self._parms['max_categorical_levels'] = max_categorical_levels
    Lambda = deprecated_property('Lambda', lambda_)

    def rule_importance(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve rule importances for a Rulefit model\n\n        :return: H2OTwoDimTable\n        '
        if self._model_json['algo'] != 'rulefit':
            raise H2OValueError('This function is available for Rulefit models only')
        kwargs = {}
        kwargs['model_id'] = self.model_id
        json = h2o.api('POST /3/SignificantRules', data=kwargs)
        return json['significant_rules_table']

    def predict_rules(self, frame, rule_ids):
        if False:
            return 10
        '\n        Evaluates validity of the given rules on the given data. \n\n        :param frame: H2OFrame on which rule validity is to be evaluated\n        :param rule_ids: string array of rule ids to be evaluated against the frame\n        :return: H2OFrame with a column per each input ruleId, representing a flag whether given rule is applied to the observation or not.\n        '
        from h2o.frame import H2OFrame
        from h2o.utils.typechecks import assert_is_type
        from h2o.expr import ExprNode
        assert_is_type(frame, H2OFrame)
        return H2OFrame._expr(expr=ExprNode('rulefit.predict.rules', self, frame, rule_ids))