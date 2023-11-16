import functools as ft
from inspect import getdoc
import re
import h2o
from h2o.base import Keyed
from h2o.estimators import H2OEstimator
from h2o.exceptions import H2OResponseError, H2OValueError
from h2o.frame import H2OFrame
from h2o.job import H2OJob
from h2o.utils.compatibility import *
from h2o.utils.shared_utils import check_id
from h2o.utils.typechecks import assert_is_type, is_type, numeric
from ._base import H2OAutoMLBaseMixin, _fetch_state
_params_doc_ = dict()

def _extract_params_doc(docstr):
    if False:
        print('Hello World!')
    pat = re.compile('^:param (?P<type>.*? )?(?P<name>\\w+):\\s?(?P<doc>.*)')
    lines = docstr.splitlines()
    (param, ptype, pdoc) = (None, None, None)
    for l in lines:
        m = pat.match(l)
        if m:
            if param:
                fulldoc = '\n'.join(pdoc)
                if ptype:
                    fulldoc += '\n\nType: %s' % ptype
                _params_doc_[param] = fulldoc
            param = m.group('name')
            ptype = m.group('type')
            pdoc = [m.group('doc')]
        elif param:
            pdoc.append(l)

def _aml_property(param_path, name=None, types=None, validate_fn=None, freezable=False, set_input=True):
    if False:
        while True:
            i = 10
    path = param_path.split('.')
    name = name or path[-1]

    def attr_name(self, attr):
        if False:
            while True:
                i = 10
        return '_' + self.__class__.__name__ + attr if attr.startswith('__') and (not attr.endswith('__')) else attr

    def _fget(self):
        if False:
            print('Hello World!')
        _input = getattr(self, attr_name(self, '__input'))
        return _input.get(name)

    def _fset(self, value):
        if False:
            while True:
                i = 10
        if freezable and getattr(self, attr_name(self, '__frozen'), False):
            raise H2OValueError('Param ``%s`` can not be modified after the first call to ``train``.' % name, name)
        if types is not None:
            assert_is_type(value, *types)
        input_val = value
        if validate_fn:
            value = validate_fn(self, value)
        _input = getattr(self, attr_name(self, '__input'))
        _input[name] = input_val if set_input else value
        group = getattr(self, attr_name(self, path[0]))
        if group is None:
            group = {}
            setattr(self, attr_name(self, path[0]), group)
        obj = group
        for t in path[1:-1]:
            tmp = obj.get(t)
            if tmp is None:
                tmp = obj[t] = {}
            obj = tmp
        obj[path[-1]] = value
    return property(fget=_fget, fset=_fset, doc=_params_doc_.get(name, None))

class H2OAutoML(H2OAutoMLBaseMixin, Keyed):
    """
    Automatic Machine Learning

    The Automatic Machine Learning (AutoML) function automates the supervised machine learning model training process.
    It trains several models, cross-validated by default, by using the following available algorithms:
    
    - XGBoost
    - GBM (Gradient Boosting Machine)
    - GLM (Generalized Linear Model)
    - DRF (Distributed Random Forest)
    - XRT (eXtremely Randomized Trees)
    - DeepLearning (Fully Connected Deep Neural Network)
    
    It also applies HPO on the following algorithms:
    
    - XGBoost
    - GBM
    - DeepLearning
    
    In some cases, there will not be enough time to complete all the algorithms, so some may be missing from the
    leaderboard. 
    Finally, AutoML also trains several Stacked Ensemble models at various stages during the run.
    Mainly two kinds of Stacked Ensemble models are trained:
    
    - one of all available models at time t
    - one of only the best models of each kind at time t.
    
    Note that Stacked Ensemble models are trained only if there isn't another stacked ensemble with the same base models.

    :examples:
    
    >>> import h2o
    >>> from h2o.automl import H2OAutoML
    >>> h2o.init()
    >>> # Import a sample binary outcome train/test set into H2O
    >>> train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
    >>> test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")
    >>> # Identify the response and set of predictors
    >>> y = "response"
    >>> x = list(train.columns)  #if x is defined as all columns except the response, then x is not required
    >>> x.remove(y)
    >>> # For binary classification, response should be a factor
    >>> train[y] = train[y].asfactor()
    >>> test[y] = test[y].asfactor()
    >>> # Run AutoML for 30 seconds
    >>> aml = H2OAutoML(max_runtime_secs = 30)
    >>> aml.train(x = x, y = y, training_frame = train)
    >>> # Print Leaderboard (ranked by xval metrics)
    >>> aml.leaderboard
    >>> # (Optional) Evaluate performance on a test set
    >>> perf = aml.leader.model_performance(test)
    >>> perf.auc()
    """

    def __init__(self, nfolds=-1, balance_classes=False, class_sampling_factors=None, max_after_balance_size=5.0, max_runtime_secs=None, max_runtime_secs_per_model=None, max_models=None, distribution='AUTO', stopping_metric='AUTO', stopping_tolerance=None, stopping_rounds=3, seed=None, project_name=None, exclude_algos=None, include_algos=None, exploitation_ratio=-1, modeling_plan=None, preprocessing=None, monotone_constraints=None, keep_cross_validation_predictions=False, keep_cross_validation_models=False, keep_cross_validation_fold_assignment=False, sort_metric='AUTO', custom_metric_func=None, export_checkpoints_dir=None, verbosity='warn', **kwargs):
        if False:
            return 10
        '\n        Create a new H2OAutoML instance.\n\n        :param int nfolds: Specify a value >= 2 for the number of folds for k-fold cross-validation for the models in the AutoML or specify ``-1`` (default)\n            to let AutoML choose what it will do. If the data is big enough (depending on the cluster resources), it will create a blending frame\n            and will not do cross-validation. Otherwise, it will use 5 fold cross-validation.        \n        :param bool balance_classes: Specify whether to oversample the minority classes to balance the class distribution. This option can increase\n            the data frame size. This option is only applicable for classification. If the oversampled size of the dataset exceeds the maximum size\n            calculated using the ``max_after_balance_size`` parameter, then the majority classes will be undersampled to satisfy the size limit.\n            Defaults to ``False``.\n        :param class_sampling_factors: Desired over/under-sampling ratios per class (in lexicographic order).\n            If not specified, sampling factors will be automatically computed to obtain class balance during training. Requires ``balance_classes`` set to ``True``.\n        :param float max_after_balance_size: Maximum relative size of the training data after balancing class counts (can be less than 1.0).\n            Requires ``balance_classes``.\n            Defaults to ``5.0``.\n        :param int max_runtime_secs: Specify the maximum time that the AutoML process will run for.\n            If both ``max_runtime_secs`` and ``max_models`` are specified, then the AutoML run will stop as soon as it hits either of these limits.\n            If neither ``max_runtime_secs`` nor ``max_models`` are specified, then ``max_runtime_secs`` dynamically\n            defaults to 3600 seconds (1 hour). Otherwise, defaults to ``0`` (no limit).\n        :param int max_runtime_secs_per_model: Controls the max time the AutoML run will dedicate to each individual model.\n            Defaults to ``0`` (disabled: no time limit).\n            Note that models constrained by a time budget are not guaranteed reproducible.\n        :param int max_models: Specify the maximum number of models to build in an AutoML run, excluding the Stacked Ensemble models.\n            Defaults to ``None`` (disabled: no limitation).\n            Always set this parameter to ensure AutoML reproducibility: all models are then trained until convergence and none is constrained by a time budget.\n        :param Union[str, dict] distribution: Distribution function used by algorithms that support it; other algorithms\n            use their defaults.  Possible values: "AUTO", "bernoulli", "multinomial", "gaussian", "poisson", "gamma",\n            "tweedie", "laplace", "quantile", "huber", "custom", and for parameterized distributions dictionary form is\n            used to specify the parameter, e.g., ``dict(type="tweedie", tweedie_power=1.5)``.\n            Defaults to ``AUTO``.\n        :param str stopping_metric: Specifies the metric to use for early stopping. \n            The available options are:\n            \n                - ``"AUTO"`` (This defaults to ``"logloss"`` for classification, ``"deviance"`` for regression)\n                - ``"deviance"``\n                - ``"logloss"``\n                - ``"mse"``\n                - ``"rmse"``\n                - ``"mae"``\n                - ``"rmsle"``\n                - ``"auc"``\n                - ``aucpr``\n                - ``"lift_top_group"``\n                - ``"misclassification"``\n                - ``"mean_per_class_error"``\n                - ``"r2"``\n                \n            Defaults to ``"AUTO"``.\n        :param float stopping_tolerance: Specify the relative tolerance for the metric-based stopping criterion to stop a grid search and\n            the training of individual models within the AutoML run.\n            Defaults to ``0.001`` if the dataset is at least 1 million rows;\n            otherwise it defaults to a value determined by the size of the dataset and the non-NA-rate, in which case the value is computed as 1/sqrt(nrows * non-NA-rate).\n        :param int stopping_rounds: Stop training new models in the AutoML run when the option selected for\n            ``stopping_metric`` doesn\'t improve for the specified number of models, based on a simple moving average.\n            To disable this feature, set it to ``0``.\n            Defaults to ``3`` and must be an non-negative integer.\n        :param int seed: Set a seed for reproducibility. \n            AutoML can only guarantee reproducibility if ``max_models`` or early stopping is used because ``max_runtime_secs`` is resource limited, \n            meaning that if the resources are not the same between runs, AutoML may be able to train more models on one run vs another.\n            In addition, H2O Deep Learning models are not reproducible by default for performance reasons, so ``exclude_algos`` must contain ``DeepLearning``.\n            Defaults to ``None``.\n        :param str project_name: Character string to identify an AutoML project.\n            Defaults to ``None``, which means a project name will be auto-generated based on the training frame ID.\n            More models can be trained on an existing AutoML project by specifying the same project name in multiple calls to the AutoML function\n            (as long as the same training frame, or a sample, is used in subsequent runs).\n        :param exclude_algos: List the algorithms to skip during the model-building phase. \n            The full list of options is:\n            \n                - ``"DRF"`` (Random Forest and Extremely-Randomized Trees)\n                - ``"GLM"``\n                - ``"XGBoost"``\n                - ``"GBM"``\n                - ``"DeepLearning"``\n                - ``"StackedEnsemble"``\n                \n            Defaults to ``None``, which means that all appropriate H2O algorithms will be used, if the search stopping criteria allow. Optional.\n            Usage example::\n            \n                exclude_algos = ["GLM", "DeepLearning", "DRF"]\n                \n        :param include_algos: List the algorithms to restrict to during the model-building phase.\n            This can\'t be used in combination with ``exclude_algos`` param.\n            Defaults to ``None``, which means that all appropriate H2O algorithms will be used, if the search stopping criteria allow. Optional.\n            Usage example::\n\n                include_algos = ["GLM", "DeepLearning", "DRF"]\n                \n        :param exploitation_ratio: The budget ratio (between 0 and 1) dedicated to the exploitation (vs exploration) phase.\n            By default, the exploitation phase is ``0`` (disabled) as this is still experimental;\n            to activate it, it is recommended to try a ratio around 0.1.\n            Note that the current exploitation phase only tries to fine-tune the best XGBoost and the best GBM found during exploration.\n        :param modeling_plan: List of modeling steps to be used by the AutoML engine (they may not all get executed, depending on other constraints).\n            Defaults to ``None`` (Expert usage only).\n        :param preprocessing: List of preprocessing steps to run. Only ``["target_encoding"]`` is currently supported. Experimental.\n        :param monotone_constraints: A mapping that represents monotonic constraints.\n            Use ``+1`` to enforce an increasing constraint and ``-1`` to specify a decreasing constraint.\n        :param keep_cross_validation_predictions: Whether to keep the predictions of the cross-validation predictions.\n            This needs to be set to ``True`` if running the same AutoML object for repeated runs because CV predictions are required to build \n            additional Stacked Ensemble models in AutoML. \n            Defaults to ``False``.\n        :param keep_cross_validation_models: Whether to keep the cross-validated models.\n            Keeping cross-validation models may consume significantly more memory in the H2O cluster.\n            Defaults to ``False``.\n        :param keep_cross_validation_fold_assignment: Whether to keep fold assignments in the models.\n            Deleting them will save memory in the H2O cluster. \n            Defaults to ``False``.\n        :param sort_metric: Metric to sort the leaderboard by at the end of an AutoML run. \n            For binomial classification, select from the following options:\n            \n                - ``"auc"``\n                - ``"aucpr"``\n                - ``"logloss"``\n                - ``"mean_per_class_error"``\n                - ``"rmse"``\n                - ``"mse"``\n                \n            For multinomial classification, select from the following options:\n            \n                - ``"mean_per_class_error"``\n                - ``"logloss"``\n                - ``"rmse"``\n                - ``"mse"``\n                \n            For regression, select from the following options:\n\n                - ``"deviance"``\n                - ``"rmse"``\n                - ``"mse"``\n                - ``"mae"``\n                - ``"rmlse"``\n                \n            Defaults to ``"AUTO"`` (This translates to ``"auc"`` for binomial classification, ``"mean_per_class_error"`` for multinomial classification, ``"deviance"`` for regression).\n        :param custom_metric_func: Reference to custom evaluation function, format: `language:keyName=funcName`\n               Defaults to ``None``.\n        :type custom_metric_func: str, optional\n        :param export_checkpoints_dir: Path to a directory where every model will be stored in binary form.\n        :param verbosity: Verbosity of the backend messages printed during training.\n            Available options are ``None`` (live log disabled), ``"debug"``, ``"info"``, ``"warn"`` or ``"error"``.\n            Defaults to ``"warn"``.\n        '
        algo_parameters = {}
        for k in kwargs:
            if k == 'algo_parameters':
                algo_parameters = kwargs[k] or {}
            else:
                raise TypeError("H2OAutoML got an unexpected keyword argument '%s'" % k)
        try:
            h2o.api('GET /3/Metadata/schemas/AutoMLV99')
        except h2o.exceptions.H2OResponseError as e:
            print(e)
            print('*******************************************************************\n*Please verify that your H2O jar has the proper AutoML extensions.*\n*******************************************************************\n\nVerbose Error Message:')
        self._job = None
        self._leader_id = None
        self._leaderboard = None
        self._verbosity = verbosity
        self._event_log = None
        self._training_info = None
        self._state_json = None
        self._build_resp = None
        self.__frozen = False
        self.__input = dict()
        self.build_control = dict()
        self.build_models = dict()
        self.input_spec = dict()
        self.project_name = project_name
        self.nfolds = nfolds
        self.distribution = distribution
        self.custom_metric_func = custom_metric_func
        self.balance_classes = balance_classes
        self.class_sampling_factors = class_sampling_factors
        self.max_after_balance_size = max_after_balance_size
        self.keep_cross_validation_models = keep_cross_validation_models
        self.keep_cross_validation_fold_assignment = keep_cross_validation_fold_assignment
        self.keep_cross_validation_predictions = keep_cross_validation_predictions
        self.export_checkpoints_dir = export_checkpoints_dir
        self.max_runtime_secs = max_runtime_secs
        self.max_runtime_secs_per_model = max_runtime_secs_per_model
        self.max_models = max_models
        self.stopping_metric = stopping_metric
        self.stopping_tolerance = stopping_tolerance
        self.stopping_rounds = stopping_rounds
        self.seed = seed
        self.exclude_algos = exclude_algos
        self.include_algos = include_algos
        self.exploitation_ratio = exploitation_ratio
        self.modeling_plan = modeling_plan
        self.preprocessing = preprocessing
        if monotone_constraints is not None:
            algo_parameters['monotone_constraints'] = monotone_constraints
        self._algo_parameters = algo_parameters
        self.sort_metric = sort_metric

    def __validate_not_set(self, val, prop=None, message=None):
        if False:
            while True:
                i = 10
        assert val is None or getattr(self, prop, None) is None, message
        return val

    def __validate_project_name(self, project_name):
        if False:
            for i in range(10):
                print('nop')
        check_id(project_name, 'H2OAutoML')
        return project_name

    def __validate_nfolds(self, nfolds):
        if False:
            for i in range(10):
                print('nop')
        assert nfolds in (-1, 0) or nfolds > 1, 'nfolds set to %s; use nfolds >=2 if you want cross-validated metrics and Stacked Ensembles or use nfolds = 0 to disable or nfolds = -1 to let h2o choose automatically.' % nfolds
        return nfolds

    def __validate_modeling_plan(self, modeling_plan):
        if False:
            while True:
                i = 10
        if modeling_plan is None:
            return None
        supported_aliases = PList(['all', 'defaults', 'grids'])

        def assert_is_step_def(sd):
            if False:
                i = 10
                return i + 15
            assert 'name' in sd, "each definition must have a 'name' key"
            assert 0 < len(sd) < 3, 'each definition must have only 1 or 2 keys: name, name+alias or name+steps'
            assert len(sd) == 1 or 'alias' in sd or 'steps' in sd, 'steps definitions support only the following keys: name, alias, steps'
            assert 'alias' not in sd or sd['alias'] in supported_aliases, 'alias must be one of %s' % supported_aliases
            assert 'steps' not in sd or (is_type(sd['steps'], list) and all((assert_is_step(s) for s in sd['steps'])))

        def assert_is_step(s):
            if False:
                while True:
                    i = 10
            assert is_type(s, dict), "each step must be a dict with an 'id' key and optional keys among: weight, group"
            assert 'id' in s, "each step must have an 'id' key"
            assert len(s) == 1 or 'weight' in s or 'group' in s, 'steps support only the following keys: weight, group'
            assert 'weight' not in s or is_type(s['weight'], int), 'weight must be an integer'
            assert 'group' not in s or is_type(s['group'], int), 'group must be an integer'
            return True
        plan = []
        for step_def in modeling_plan:
            assert_is_type(step_def, dict, tuple, str)
            if is_type(step_def, dict):
                assert_is_step_def(step_def)
                plan.append(step_def)
            elif is_type(step_def, str):
                plan.append(dict(name=step_def))
            else:
                assert 0 < len(step_def) < 3
                assert_is_type(step_def[0], str)
                name = step_def[0]
                if len(step_def) == 1:
                    plan.append(dict(name=name))
                else:
                    assert_is_type(step_def[1], str, list)
                    ids = step_def[1]
                    if is_type(ids, str):
                        assert_is_type(ids, *supported_aliases)
                        plan.append(dict(name=name, alias=ids))
                    else:
                        plan.append(dict(name=name, steps=[dict(id=i) for i in ids]))
        return plan

    def __validate_preprocessing(self, preprocessing):
        if False:
            for i in range(10):
                print('nop')
        if preprocessing is None:
            return
        assert all((p in ['target_encoding'] for p in preprocessing))
        return [dict(type=p.replace('_', '')) for p in preprocessing]

    def __validate_monotone_constraints(self, monotone_constraints):
        if False:
            while True:
                i = 10
        if monotone_constraints is None:
            self._algo_parameters.pop('monotone_constraints', None)
        else:
            self._algo_parameters['monotone_constraints'] = monotone_constraints
        return self.__validate_algo_parameters(self._algo_parameters)

    def __validate_algo_parameters(self, algo_parameters):
        if False:
            while True:
                i = 10
        if algo_parameters is None:
            return None
        algo_parameters_json = []
        for (k, v) in algo_parameters.items():
            (scope, __, name) = k.partition('__')
            if len(name) == 0:
                (name, scope) = (scope, 'any')
            value = [dict(key=k, value=v) for (k, v) in v.items()] if isinstance(v, dict) else v
            algo_parameters_json.append(dict(scope=scope, name=name, value=value))
        return algo_parameters_json

    def __validate_frame(self, fr, name=None, required=False):
        if False:
            return 10
        return H2OFrame._validate(fr, name, required=required)

    def __validate_distribution(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        if is_type(distribution, str):
            distribution = distribution.lower()
            if distribution == 'custom':
                raise H2OValueError('Distribution "custom" has to be specified as a dictionary with their respective parameters, e.g., `dict(type = "custom", custom_distribution_func = "..."))`.')
            return distribution
        if is_type(distribution, dict):
            dist = distribution['type'].lower()
            allowed_distribution_parameters = dict(custom='custom_distribution_func', huber='huber_alpha', quantile='quantile_alpha', tweedie='tweedie_power')
            assert distribution.get(allowed_distribution_parameters.get(dist)) is not None or len(distribution) == 1, 'Distribution dictionary should contain distribution and a distribution parameter. For example `dict(type="{}", {}=...)`.'.format(dist, allowed_distribution_parameters[dist])
            if distribution['type'] == 'custom' and 'custom_distribution_func' not in distribution.keys():
                raise H2OValueError('Distribution "custom" has to be specified as a dictionary with their respective parameters, e.g., `dict(type = "custom", custom_distribution_func = "..."))`.')
            if allowed_distribution_parameters.get(dist) in distribution.keys():
                setattr(self, '_' + allowed_distribution_parameters[dist], distribution[allowed_distribution_parameters[dist]])
            return dist
    _extract_params_doc(getdoc(__init__))
    project_name = _aml_property('build_control.project_name', types=(None, str), freezable=True, validate_fn=__validate_project_name)
    nfolds = _aml_property('build_control.nfolds', types=(int,), freezable=True, validate_fn=__validate_nfolds)
    distribution = _aml_property('build_control.distribution', types=(str, dict), freezable=True, validate_fn=__validate_distribution)
    _custom_distribution_func = _aml_property('build_control.custom_distribution_func', types=(str,), freezable=True)
    _huber_alpha = _aml_property('build_control.huber_alpha', types=(numeric,), freezable=True)
    _tweedie_power = _aml_property('build_control.tweedie_power', types=(numeric,), freezable=True)
    _quantile_alpha = _aml_property('build_control.quantile_alpha', types=(numeric,), freezable=True)
    custom_metric_func = _aml_property('build_control.custom_metric_func', types=(str, None))
    balance_classes = _aml_property('build_control.balance_classes', types=(bool,), freezable=True)
    class_sampling_factors = _aml_property('build_control.class_sampling_factors', types=(None, [numeric]), freezable=True)
    max_after_balance_size = _aml_property('build_control.max_after_balance_size', types=(None, numeric), freezable=True)
    keep_cross_validation_models = _aml_property('build_control.keep_cross_validation_models', types=(bool,), freezable=True)
    keep_cross_validation_fold_assignment = _aml_property('build_control.keep_cross_validation_fold_assignment', types=(bool,), freezable=True)
    keep_cross_validation_predictions = _aml_property('build_control.keep_cross_validation_predictions', types=(bool,), freezable=True)
    export_checkpoints_dir = _aml_property('build_control.export_checkpoints_dir', types=(None, str), freezable=True)
    max_runtime_secs = _aml_property('build_control.stopping_criteria.max_runtime_secs', types=(None, int), freezable=True)
    max_runtime_secs_per_model = _aml_property('build_control.stopping_criteria.max_runtime_secs_per_model', types=(None, int), freezable=True)
    max_models = _aml_property('build_control.stopping_criteria.max_models', types=(None, int), freezable=True)
    stopping_metric = _aml_property('build_control.stopping_criteria.stopping_metric', types=(None, str), freezable=True)
    stopping_tolerance = _aml_property('build_control.stopping_criteria.stopping_tolerance', types=(None, numeric), freezable=True)
    stopping_rounds = _aml_property('build_control.stopping_criteria.stopping_rounds', types=(None, int), freezable=True)
    seed = _aml_property('build_control.stopping_criteria.seed', types=(None, int), freezable=True)
    exclude_algos = _aml_property('build_models.exclude_algos', types=(None, [str]), freezable=True, validate_fn=ft.partial(__validate_not_set, prop='include_algos', message='Use either `exclude_algos` or `include_algos`, not both.'))
    include_algos = _aml_property('build_models.include_algos', types=(None, [str]), freezable=True, validate_fn=ft.partial(__validate_not_set, prop='exclude_algos', message='Use either `exclude_algos` or `include_algos`, not both.'))
    exploitation_ratio = _aml_property('build_models.exploitation_ratio', types=(None, numeric), freezable=True)
    modeling_plan = _aml_property('build_models.modeling_plan', types=(None, list), freezable=True, validate_fn=__validate_modeling_plan)
    preprocessing = _aml_property('build_models.preprocessing', types=(None, [str]), freezable=True, validate_fn=__validate_preprocessing)
    monotone_constraints = _aml_property('build_models.algo_parameters', name='monotone_constraints', types=(None, dict), freezable=True, validate_fn=__validate_monotone_constraints)
    _algo_parameters = _aml_property('build_models.algo_parameters', types=(None, dict), freezable=True, validate_fn=__validate_algo_parameters)
    sort_metric = _aml_property('input_spec.sort_metric', types=(None, str))
    fold_column = _aml_property('input_spec.fold_column', types=(None, int, str))
    weights_column = _aml_property('input_spec.weights_column', types=(None, int, str))
    training_frame = _aml_property('input_spec.training_frame', set_input=False, validate_fn=ft.partial(__validate_frame, name='training_frame', required=True))
    validation_frame = _aml_property('input_spec.validation_frame', set_input=False, validate_fn=ft.partial(__validate_frame, name='validation_frame'))
    leaderboard_frame = _aml_property('input_spec.leaderboard_frame', set_input=False, validate_fn=ft.partial(__validate_frame, name='leaderboard_frame'))
    blending_frame = _aml_property('input_spec.blending_frame', set_input=False, validate_fn=ft.partial(__validate_frame, name='blending_frame'))
    response_column = _aml_property('input_spec.response_column', types=(str,))

    @property
    def key(self):
        if False:
            while True:
                i = 10
        return self._job.dest_key if self._job else self.project_name

    @property
    def leader(self):
        if False:
            print('Hello World!')
        return None if self._leader_id is None else h2o.get_model(self._leader_id)

    @property
    def leaderboard(self):
        if False:
            print('Hello World!')
        return H2OFrame([]) if self._leaderboard is None else self._leaderboard

    @property
    def event_log(self):
        if False:
            for i in range(10):
                print('nop')
        return H2OFrame([]) if self._event_log is None else self._event_log

    @property
    def training_info(self):
        if False:
            i = 10
            return i + 15
        return dict() if self._training_info is None else self._training_info

    @property
    def modeling_steps(self):
        if False:
            print('Hello World!')
        '\n        Expose the modeling steps effectively used by the AutoML run.\n        This executed plan can be directly reinjected as the `modeling_plan` property of a new AutoML instance\n        to improve reproducibility across AutoML versions.\n\n        :return: a list of dictionaries representing the effective modeling plan.\n        '
        return list(map(lambda sdef: dict(name=sdef['name'], steps=sdef['steps']), self._state_json['modeling_steps']))

    def train(self, x=None, y=None, training_frame=None, fold_column=None, weights_column=None, validation_frame=None, leaderboard_frame=None, blending_frame=None):
        if False:
            while True:
                i = 10
        "\n        Begins an AutoML task, a background task that automatically builds a number of models\n        with various algorithms and tracks their performance in a leaderboard. At any point \n        in the process you may use H2O's performance or prediction functions on the resulting \n        models.\n\n        :param x: A list of column names or indices indicating the predictor columns.\n        :param y: An index or a column name indicating the response column.\n        :param fold_column: The name or index of the column in training_frame that holds per-row fold\n            assignments.\n        :param weights_column: The name or index of the column in training_frame that holds per-row weights.\n        :param training_frame: The H2OFrame having the columns indicated by x and y (as well as any\n            additional columns specified by fold_column or weights_column).\n        :param validation_frame: H2OFrame with validation data. This argument is ignored unless the user sets \n            nfolds = 0. If cross-validation is turned off, then a validation frame can be specified and used \n            for early stopping of individual models and early stopping of the grid searches.  By default and \n            when nfolds > 1, cross-validation metrics will be used for early stopping and thus validation_frame will be ignored.\n        :param leaderboard_frame: H2OFrame with test data for scoring the leaderboard.  This is optional and\n            if this is set to None (the default), then cross-validation metrics will be used to generate the leaderboard \n            rankings instead.\n        :param blending_frame: H2OFrame used to train the the metalearning algorithm in Stacked Ensembles (instead of relying on cross-validated predicted values).\n            This is optional, but when provided, it is also recommended to disable cross validation \n            by setting `nfolds=0` and to provide a leaderboard frame for scoring purposes.\n\n        :returns: An H2OAutoML object.\n\n        :examples:\n        \n        >>> # Set up an H2OAutoML object\n        >>> aml = H2OAutoML(max_runtime_secs=30)\n        >>> # Launch an AutoML run\n        >>> aml.train(y=y, training_frame=train)\n        "
        self.training_frame = training_frame
        ncols = self.training_frame.ncols
        names = self.training_frame.names
        if y is None and self.response_column is None:
            raise H2OValueError('The response column (y) is not set; please set it to the name of the column that you are trying to predict in your data.')
        elif y is not None:
            assert_is_type(y, int, str)
            if is_type(y, int):
                if not -ncols <= y < ncols:
                    raise H2OValueError('Column %d does not exist in the training frame' % y)
                y = names[y]
            elif y not in names:
                raise H2OValueError('Column %s does not exist in the training frame' % y)
            self.response_column = y
        self.fold_column = fold_column
        self.weights_column = weights_column
        self.validation_frame = validation_frame
        self.leaderboard_frame = leaderboard_frame
        self.blending_frame = blending_frame
        if x is not None:
            assert_is_type(x, list)
            xset = set()
            if is_type(x, int, str):
                x = [x]
            for xi in x:
                if is_type(xi, int):
                    if not -ncols <= xi < ncols:
                        raise H2OValueError('Column %d does not exist in the training frame' % xi)
                    xset.add(names[xi])
                else:
                    if xi not in names:
                        raise H2OValueError('Column %s not in the training frame' % xi)
                    xset.add(xi)
            ignored_columns = set(names) - xset
            for col in [y, fold_column, weights_column]:
                if col is not None and col in ignored_columns:
                    ignored_columns.remove(col)
            if ignored_columns is not None:
                self.input_spec['ignored_columns'] = list(ignored_columns)

        def clean_params(params):
            if False:
                for i in range(10):
                    print('nop')
            return {k: clean_params(v) for (k, v) in params.items() if v is not None} if isinstance(params, dict) else H2OEstimator._keyify(params)
        automl_build_params = clean_params(dict(build_control=self.build_control, build_models=self.build_models, input_spec=self.input_spec))
        resp = self._build_resp = h2o.api('POST /99/AutoMLBuilder', json=automl_build_params)
        if 'job' not in resp:
            raise H2OResponseError('Backend failed to build the AutoML job: {}'.format(resp))
        if not self.project_name:
            self.project_name = resp['build_control']['project_name']
        self.__frozen = True
        self._job = H2OJob(resp['job'], 'AutoML')
        poll_updates = ft.partial(self._poll_training_updates, verbosity=self._verbosity, state={})
        try:
            self._job.poll(poll_updates=poll_updates)
        finally:
            poll_updates(self._job, 1)
        self._fetch()
        return self.leader

    def predict(self, test_data):
        if False:
            while True:
                i = 10
        leader = self.leader
        if leader is None:
            self._fetch()
            leader = self.leader
        if leader is not None:
            return leader.predict(test_data)
        print('No model built yet...')

    def detach(self):
        if False:
            print('Hello World!')
        self.__frozen = False
        self.project_name = None
        h2o.remove(self.leaderboard)
        h2o.remove(self.event_log)

    def _fetch(self):
        if False:
            print('Hello World!')
        state = _fetch_state(self.key)
        self._leader_id = state['leader_id']
        self._leaderboard = state['leaderboard']
        self._event_log = el = state['event_log']
        self._training_info = {r[0]: r[1] for r in el[el['name'] != '', ['name', 'value']].as_data_frame(use_pandas=False, header=False)}
        self._state_json = state['json']
        return self._leader_id is not None

    def _poll_training_updates(self, job, bar_progress=0, verbosity=None, state=None):
        if False:
            i = 10
            return i + 15
        '\n        the callback function used to print verbose info when polling AutoML job.\n        '
        levels = ['debug', 'info', 'warn', 'error']
        if verbosity is None or verbosity.lower() not in levels:
            return
        try:
            if job.progress > state.get('last_job_progress', 0):
                events_table = _fetch_state(job.dest_key, properties=[], verbosity=verbosity)['json']['event_log_table']
                last_nrows = state.get('last_events_nrows', 0)
                if len(events_table.cell_values) > last_nrows:
                    events = zip(*events_table[last_nrows:][['timestamp', 'message']])
                    print('')
                    for r in events:
                        print('{}: {}'.format(r[0], r[1]))
                    print('')
                    state['last_events_nrows'] = len(events_table.cell_values)
            state['last_job_progress'] = job.progress
        except Exception as e:
            print('Failed polling AutoML progress log: {}'.format(e))