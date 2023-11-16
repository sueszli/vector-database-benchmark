from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2ODeepLearningEstimator(H2OEstimator):
    """
    Deep Learning

    Build a Deep Neural Network model using CPUs
    Builds a feed-forward multilayer artificial neural network on an H2OFrame

    :examples:

    >>> from h2o.estimators.deeplearning import H2ODeepLearningEstimator
    >>> rows = [[1,2,3,4,0], [2,1,2,4,1], [2,1,4,2,1],
    ...         [0,1,2,34,1], [2,3,4,1,0]] * 50
    >>> fr = h2o.H2OFrame(rows)
    >>> fr[4] = fr[4].asfactor()
    >>> model = H2ODeepLearningEstimator()
    >>> model.train(x=range(4), y=4, training_frame=fr)
    >>> model.logloss()
    """
    algo = 'deeplearning'
    supervised_learning = True
    _options_ = {'model_extensions': ['h2o.model.extensions.ScoringHistoryDL', 'h2o.model.extensions.VariableImportance', 'h2o.model.extensions.Fairness', 'h2o.model.extensions.Contributions'], 'verbose': True}

    def __init__(self, model_id=None, training_frame=None, validation_frame=None, nfolds=0, keep_cross_validation_models=True, keep_cross_validation_predictions=False, keep_cross_validation_fold_assignment=False, fold_assignment='auto', fold_column=None, response_column=None, ignored_columns=None, ignore_const_cols=True, score_each_iteration=False, weights_column=None, offset_column=None, balance_classes=False, class_sampling_factors=None, max_after_balance_size=5.0, max_confusion_matrix_size=20, checkpoint=None, pretrained_autoencoder=None, overwrite_with_best_model=True, use_all_factor_levels=True, standardize=True, activation='rectifier', hidden=[200, 200], epochs=10.0, train_samples_per_iteration=-2, target_ratio_comm_to_comp=0.05, seed=-1, adaptive_rate=True, rho=0.99, epsilon=1e-08, rate=0.005, rate_annealing=1e-06, rate_decay=1.0, momentum_start=0.0, momentum_ramp=1000000.0, momentum_stable=0.0, nesterov_accelerated_gradient=True, input_dropout_ratio=0.0, hidden_dropout_ratios=None, l1=0.0, l2=0.0, max_w2=3.4028235e+38, initial_weight_distribution='uniform_adaptive', initial_weight_scale=1.0, initial_weights=None, initial_biases=None, loss='automatic', distribution='auto', quantile_alpha=0.5, tweedie_power=1.5, huber_alpha=0.9, score_interval=5.0, score_training_samples=10000, score_validation_samples=0, score_duty_cycle=0.1, classification_stop=0.0, regression_stop=1e-06, stopping_rounds=5, stopping_metric='auto', stopping_tolerance=0.0, max_runtime_secs=0.0, score_validation_sampling='uniform', diagnostics=True, fast_mode=True, force_load_balance=True, variable_importances=True, replicate_training_data=True, single_node_mode=False, shuffle_training_data=False, missing_values_handling='mean_imputation', quiet_mode=False, autoencoder=False, sparse=False, col_major=False, average_activation=0.0, sparsity_beta=0.0, max_categorical_features=2147483647, reproducible=False, export_weights_and_biases=False, mini_batch_size=1, categorical_encoding='auto', elastic_averaging=False, elastic_averaging_moving_rate=0.9, elastic_averaging_regularization=0.001, export_checkpoints_dir=None, auc_type='auto', custom_metric_func=None):
        if False:
            return 10
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param validation_frame: Id of the validation data frame.\n               Defaults to ``None``.\n        :type validation_frame: Union[None, str, H2OFrame], optional\n        :param nfolds: Number of folds for K-fold cross-validation (0 to disable or >= 2).\n               Defaults to ``0``.\n        :type nfolds: int\n        :param keep_cross_validation_models: Whether to keep the cross-validation models.\n               Defaults to ``True``.\n        :type keep_cross_validation_models: bool\n        :param keep_cross_validation_predictions: Whether to keep the predictions of the cross-validation models.\n               Defaults to ``False``.\n        :type keep_cross_validation_predictions: bool\n        :param keep_cross_validation_fold_assignment: Whether to keep the cross-validation fold assignment.\n               Defaults to ``False``.\n        :type keep_cross_validation_fold_assignment: bool\n        :param fold_assignment: Cross-validation fold assignment scheme, if fold_column is not specified. The\n               \'Stratified\' option will stratify the folds based on the response variable, for classification problems.\n               Defaults to ``"auto"``.\n        :type fold_assignment: Literal["auto", "random", "modulo", "stratified"]\n        :param fold_column: Column with cross-validation fold index assignment per observation.\n               Defaults to ``None``.\n        :type fold_column: str, optional\n        :param response_column: Response variable column.\n               Defaults to ``None``.\n        :type response_column: str, optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param ignore_const_cols: Ignore constant columns.\n               Defaults to ``True``.\n        :type ignore_const_cols: bool\n        :param score_each_iteration: Whether to score during each iteration of model training.\n               Defaults to ``False``.\n        :type score_each_iteration: bool\n        :param weights_column: Column with observation weights. Giving some observation a weight of zero is equivalent\n               to excluding it from the dataset; giving an observation a relative weight of 2 is equivalent to repeating\n               that row twice. Negative weights are not allowed. Note: Weights are per-row observation weights and do\n               not increase the size of the data frame. This is typically the number of times a row is repeated, but\n               non-integer values are supported as well. During training, rows with higher weights matter more, due to\n               the larger loss function pre-factor. If you set weight = 0 for a row, the returned prediction frame at\n               that row is zero and this is incorrect. To get an accurate prediction, remove all rows with weight == 0.\n               Defaults to ``None``.\n        :type weights_column: str, optional\n        :param offset_column: Offset column. This will be added to the combination of columns before applying the link\n               function.\n               Defaults to ``None``.\n        :type offset_column: str, optional\n        :param balance_classes: Balance training data class counts via over/under-sampling (for imbalanced data).\n               Defaults to ``False``.\n        :type balance_classes: bool\n        :param class_sampling_factors: Desired over/under-sampling ratios per class (in lexicographic order). If not\n               specified, sampling factors will be automatically computed to obtain class balance during training.\n               Requires balance_classes.\n               Defaults to ``None``.\n        :type class_sampling_factors: List[float], optional\n        :param max_after_balance_size: Maximum relative size of the training data after balancing class counts (can be\n               less than 1.0). Requires balance_classes.\n               Defaults to ``5.0``.\n        :type max_after_balance_size: float\n        :param max_confusion_matrix_size: [Deprecated] Maximum size (# classes) for confusion matrices to be printed in\n               the Logs.\n               Defaults to ``20``.\n        :type max_confusion_matrix_size: int\n        :param checkpoint: Model checkpoint to resume training with.\n               Defaults to ``None``.\n        :type checkpoint: Union[None, str, H2OEstimator], optional\n        :param pretrained_autoencoder: Pretrained autoencoder model to initialize this model with.\n               Defaults to ``None``.\n        :type pretrained_autoencoder: Union[None, str, H2OEstimator], optional\n        :param overwrite_with_best_model: If enabled, override the final model with the best model found during\n               training.\n               Defaults to ``True``.\n        :type overwrite_with_best_model: bool\n        :param use_all_factor_levels: Use all factor levels of categorical variables. Otherwise, the first factor level\n               is omitted (without loss of accuracy). Useful for variable importances and auto-enabled for autoencoder.\n               Defaults to ``True``.\n        :type use_all_factor_levels: bool\n        :param standardize: If enabled, automatically standardize the data. If disabled, the user must provide properly\n               scaled input data.\n               Defaults to ``True``.\n        :type standardize: bool\n        :param activation: Activation function.\n               Defaults to ``"rectifier"``.\n        :type activation: Literal["tanh", "tanh_with_dropout", "rectifier", "rectifier_with_dropout", "maxout",\n               "maxout_with_dropout"]\n        :param hidden: Hidden layer sizes (e.g. [100, 100]).\n               Defaults to ``[200, 200]``.\n        :type hidden: List[int]\n        :param epochs: How many times the dataset should be iterated (streamed), can be fractional.\n               Defaults to ``10.0``.\n        :type epochs: float\n        :param train_samples_per_iteration: Number of training samples (globally) per MapReduce iteration. Special\n               values are 0: one epoch, -1: all available data (e.g., replicated training data), -2: automatic.\n               Defaults to ``-2``.\n        :type train_samples_per_iteration: int\n        :param target_ratio_comm_to_comp: Target ratio of communication overhead to computation. Only for multi-node\n               operation and train_samples_per_iteration = -2 (auto-tuning).\n               Defaults to ``0.05``.\n        :type target_ratio_comm_to_comp: float\n        :param seed: Seed for random numbers (affects sampling) - Note: only reproducible when running single threaded.\n               Defaults to ``-1``.\n        :type seed: int\n        :param adaptive_rate: Adaptive learning rate.\n               Defaults to ``True``.\n        :type adaptive_rate: bool\n        :param rho: Adaptive learning rate time decay factor (similarity to prior updates).\n               Defaults to ``0.99``.\n        :type rho: float\n        :param epsilon: Adaptive learning rate smoothing factor (to avoid divisions by zero and allow progress).\n               Defaults to ``1e-08``.\n        :type epsilon: float\n        :param rate: Learning rate (higher => less stable, lower => slower convergence).\n               Defaults to ``0.005``.\n        :type rate: float\n        :param rate_annealing: Learning rate annealing: rate / (1 + rate_annealing * samples).\n               Defaults to ``1e-06``.\n        :type rate_annealing: float\n        :param rate_decay: Learning rate decay factor between layers (N-th layer: rate * rate_decay ^ (n - 1).\n               Defaults to ``1.0``.\n        :type rate_decay: float\n        :param momentum_start: Initial momentum at the beginning of training (try 0.5).\n               Defaults to ``0.0``.\n        :type momentum_start: float\n        :param momentum_ramp: Number of training samples for which momentum increases.\n               Defaults to ``1000000.0``.\n        :type momentum_ramp: float\n        :param momentum_stable: Final momentum after the ramp is over (try 0.99).\n               Defaults to ``0.0``.\n        :type momentum_stable: float\n        :param nesterov_accelerated_gradient: Use Nesterov accelerated gradient (recommended).\n               Defaults to ``True``.\n        :type nesterov_accelerated_gradient: bool\n        :param input_dropout_ratio: Input layer dropout ratio (can improve generalization, try 0.1 or 0.2).\n               Defaults to ``0.0``.\n        :type input_dropout_ratio: float\n        :param hidden_dropout_ratios: Hidden layer dropout ratios (can improve generalization), specify one value per\n               hidden layer, defaults to 0.5.\n               Defaults to ``None``.\n        :type hidden_dropout_ratios: List[float], optional\n        :param l1: L1 regularization (can add stability and improve generalization, causes many weights to become 0).\n               Defaults to ``0.0``.\n        :type l1: float\n        :param l2: L2 regularization (can add stability and improve generalization, causes many weights to be small.\n               Defaults to ``0.0``.\n        :type l2: float\n        :param max_w2: Constraint for squared sum of incoming weights per unit (e.g. for Rectifier).\n               Defaults to ``3.4028235e+38``.\n        :type max_w2: float\n        :param initial_weight_distribution: Initial weight distribution.\n               Defaults to ``"uniform_adaptive"``.\n        :type initial_weight_distribution: Literal["uniform_adaptive", "uniform", "normal"]\n        :param initial_weight_scale: Uniform: -value...value, Normal: stddev.\n               Defaults to ``1.0``.\n        :type initial_weight_scale: float\n        :param initial_weights: A list of H2OFrame ids to initialize the weight matrices of this model with.\n               Defaults to ``None``.\n        :type initial_weights: List[Union[None, str, H2OFrame]], optional\n        :param initial_biases: A list of H2OFrame ids to initialize the bias vectors of this model with.\n               Defaults to ``None``.\n        :type initial_biases: List[Union[None, str, H2OFrame]], optional\n        :param loss: Loss function.\n               Defaults to ``"automatic"``.\n        :type loss: Literal["automatic", "cross_entropy", "quadratic", "huber", "absolute", "quantile"]\n        :param distribution: Distribution function\n               Defaults to ``"auto"``.\n        :type distribution: Literal["auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace",\n               "quantile", "huber"]\n        :param quantile_alpha: Desired quantile for Quantile regression, must be between 0 and 1.\n               Defaults to ``0.5``.\n        :type quantile_alpha: float\n        :param tweedie_power: Tweedie power for Tweedie regression, must be between 1 and 2.\n               Defaults to ``1.5``.\n        :type tweedie_power: float\n        :param huber_alpha: Desired quantile for Huber/M-regression (threshold between quadratic and linear loss, must\n               be between 0 and 1).\n               Defaults to ``0.9``.\n        :type huber_alpha: float\n        :param score_interval: Shortest time interval (in seconds) between model scoring.\n               Defaults to ``5.0``.\n        :type score_interval: float\n        :param score_training_samples: Number of training set samples for scoring (0 for all).\n               Defaults to ``10000``.\n        :type score_training_samples: int\n        :param score_validation_samples: Number of validation set samples for scoring (0 for all).\n               Defaults to ``0``.\n        :type score_validation_samples: int\n        :param score_duty_cycle: Maximum duty cycle fraction for scoring (lower: more training, higher: more scoring).\n               Defaults to ``0.1``.\n        :type score_duty_cycle: float\n        :param classification_stop: Stopping criterion for classification error fraction on training data (-1 to\n               disable).\n               Defaults to ``0.0``.\n        :type classification_stop: float\n        :param regression_stop: Stopping criterion for regression error (MSE) on training data (-1 to disable).\n               Defaults to ``1e-06``.\n        :type regression_stop: float\n        :param stopping_rounds: Early stopping based on convergence of stopping_metric. Stop if simple moving average of\n               length k of the stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)\n               Defaults to ``5``.\n        :type stopping_rounds: int\n        :param stopping_metric: Metric to use for early stopping (AUTO: logloss for classification, deviance for\n               regression and anomaly_score for Isolation Forest). Note that custom and custom_increasing can only be\n               used in GBM and DRF with the Python client.\n               Defaults to ``"auto"``.\n        :type stopping_metric: Literal["auto", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "lift_top_group",\n               "misclassification", "mean_per_class_error", "custom", "custom_increasing"]\n        :param stopping_tolerance: Relative tolerance for metric-based stopping criterion (stop if relative improvement\n               is not at least this much)\n               Defaults to ``0.0``.\n        :type stopping_tolerance: float\n        :param max_runtime_secs: Maximum allowed runtime in seconds for model training. Use 0 to disable.\n               Defaults to ``0.0``.\n        :type max_runtime_secs: float\n        :param score_validation_sampling: Method used to sample validation dataset for scoring.\n               Defaults to ``"uniform"``.\n        :type score_validation_sampling: Literal["uniform", "stratified"]\n        :param diagnostics: Enable diagnostics for hidden layers.\n               Defaults to ``True``.\n        :type diagnostics: bool\n        :param fast_mode: Enable fast mode (minor approximation in back-propagation).\n               Defaults to ``True``.\n        :type fast_mode: bool\n        :param force_load_balance: Force extra load balancing to increase training speed for small datasets (to keep all\n               cores busy).\n               Defaults to ``True``.\n        :type force_load_balance: bool\n        :param variable_importances: Compute variable importances for input features (Gedeon method) - can be slow for\n               large networks.\n               Defaults to ``True``.\n        :type variable_importances: bool\n        :param replicate_training_data: Replicate the entire training dataset onto every node for faster training on\n               small datasets.\n               Defaults to ``True``.\n        :type replicate_training_data: bool\n        :param single_node_mode: Run on a single node for fine-tuning of model parameters.\n               Defaults to ``False``.\n        :type single_node_mode: bool\n        :param shuffle_training_data: Enable shuffling of training data (recommended if training data is replicated and\n               train_samples_per_iteration is close to #nodes x #rows, of if using balance_classes).\n               Defaults to ``False``.\n        :type shuffle_training_data: bool\n        :param missing_values_handling: Handling of missing values. Either MeanImputation or Skip.\n               Defaults to ``"mean_imputation"``.\n        :type missing_values_handling: Literal["mean_imputation", "skip"]\n        :param quiet_mode: Enable quiet mode for less output to standard output.\n               Defaults to ``False``.\n        :type quiet_mode: bool\n        :param autoencoder: Auto-Encoder.\n               Defaults to ``False``.\n        :type autoencoder: bool\n        :param sparse: Sparse data handling (more efficient for data with lots of 0 values).\n               Defaults to ``False``.\n        :type sparse: bool\n        :param col_major: #DEPRECATED Use a column major weight matrix for input layer. Can speed up forward\n               propagation, but might slow down backpropagation.\n               Defaults to ``False``.\n        :type col_major: bool\n        :param average_activation: Average activation for sparse auto-encoder. #Experimental\n               Defaults to ``0.0``.\n        :type average_activation: float\n        :param sparsity_beta: Sparsity regularization. #Experimental\n               Defaults to ``0.0``.\n        :type sparsity_beta: float\n        :param max_categorical_features: Max. number of categorical features, enforced via hashing. #Experimental\n               Defaults to ``2147483647``.\n        :type max_categorical_features: int\n        :param reproducible: Force reproducibility on small data (will be slow - only uses 1 thread).\n               Defaults to ``False``.\n        :type reproducible: bool\n        :param export_weights_and_biases: Whether to export Neural Network weights and biases to H2O Frames.\n               Defaults to ``False``.\n        :type export_weights_and_biases: bool\n        :param mini_batch_size: Mini-batch size (smaller leads to better fit, larger can speed up and generalize\n               better).\n               Defaults to ``1``.\n        :type mini_batch_size: int\n        :param categorical_encoding: Encoding scheme for categorical features\n               Defaults to ``"auto"``.\n        :type categorical_encoding: Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",\n               "sort_by_response", "enum_limited"]\n        :param elastic_averaging: Elastic averaging between compute nodes can improve distributed model convergence.\n               #Experimental\n               Defaults to ``False``.\n        :type elastic_averaging: bool\n        :param elastic_averaging_moving_rate: Elastic averaging moving rate (only if elastic averaging is enabled).\n               Defaults to ``0.9``.\n        :type elastic_averaging_moving_rate: float\n        :param elastic_averaging_regularization: Elastic averaging regularization strength (only if elastic averaging is\n               enabled).\n               Defaults to ``0.001``.\n        :type elastic_averaging_regularization: float\n        :param export_checkpoints_dir: Automatically export generated models to this directory.\n               Defaults to ``None``.\n        :type export_checkpoints_dir: str, optional\n        :param auc_type: Set default multinomial AUC type.\n               Defaults to ``"auto"``.\n        :type auc_type: Literal["auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"]\n        :param custom_metric_func: Reference to custom evaluation function, format: `language:keyName=funcName`\n               Defaults to ``None``.\n        :type custom_metric_func: str, optional\n        '
        super(H2ODeepLearningEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.validation_frame = validation_frame
        self.nfolds = nfolds
        self.keep_cross_validation_models = keep_cross_validation_models
        self.keep_cross_validation_predictions = keep_cross_validation_predictions
        self.keep_cross_validation_fold_assignment = keep_cross_validation_fold_assignment
        self.fold_assignment = fold_assignment
        self.fold_column = fold_column
        self.response_column = response_column
        self.ignored_columns = ignored_columns
        self.ignore_const_cols = ignore_const_cols
        self.score_each_iteration = score_each_iteration
        self.weights_column = weights_column
        self.offset_column = offset_column
        self.balance_classes = balance_classes
        self.class_sampling_factors = class_sampling_factors
        self.max_after_balance_size = max_after_balance_size
        self.max_confusion_matrix_size = max_confusion_matrix_size
        self.checkpoint = checkpoint
        self.pretrained_autoencoder = pretrained_autoencoder
        self.overwrite_with_best_model = overwrite_with_best_model
        self.use_all_factor_levels = use_all_factor_levels
        self.standardize = standardize
        self.activation = activation
        self.hidden = hidden
        self.epochs = epochs
        self.train_samples_per_iteration = train_samples_per_iteration
        self.target_ratio_comm_to_comp = target_ratio_comm_to_comp
        self.seed = seed
        self.adaptive_rate = adaptive_rate
        self.rho = rho
        self.epsilon = epsilon
        self.rate = rate
        self.rate_annealing = rate_annealing
        self.rate_decay = rate_decay
        self.momentum_start = momentum_start
        self.momentum_ramp = momentum_ramp
        self.momentum_stable = momentum_stable
        self.nesterov_accelerated_gradient = nesterov_accelerated_gradient
        self.input_dropout_ratio = input_dropout_ratio
        self.hidden_dropout_ratios = hidden_dropout_ratios
        self.l1 = l1
        self.l2 = l2
        self.max_w2 = max_w2
        self.initial_weight_distribution = initial_weight_distribution
        self.initial_weight_scale = initial_weight_scale
        self.initial_weights = initial_weights
        self.initial_biases = initial_biases
        self.loss = loss
        self.distribution = distribution
        self.quantile_alpha = quantile_alpha
        self.tweedie_power = tweedie_power
        self.huber_alpha = huber_alpha
        self.score_interval = score_interval
        self.score_training_samples = score_training_samples
        self.score_validation_samples = score_validation_samples
        self.score_duty_cycle = score_duty_cycle
        self.classification_stop = classification_stop
        self.regression_stop = regression_stop
        self.stopping_rounds = stopping_rounds
        self.stopping_metric = stopping_metric
        self.stopping_tolerance = stopping_tolerance
        self.max_runtime_secs = max_runtime_secs
        self.score_validation_sampling = score_validation_sampling
        self.diagnostics = diagnostics
        self.fast_mode = fast_mode
        self.force_load_balance = force_load_balance
        self.variable_importances = variable_importances
        self.replicate_training_data = replicate_training_data
        self.single_node_mode = single_node_mode
        self.shuffle_training_data = shuffle_training_data
        self.missing_values_handling = missing_values_handling
        self.quiet_mode = quiet_mode
        self.autoencoder = autoencoder
        self.sparse = sparse
        self.col_major = col_major
        self.average_activation = average_activation
        self.sparsity_beta = sparsity_beta
        self.max_categorical_features = max_categorical_features
        self.reproducible = reproducible
        self.export_weights_and_biases = export_weights_and_biases
        self.mini_batch_size = mini_batch_size
        self.categorical_encoding = categorical_encoding
        self.elastic_averaging = elastic_averaging
        self.elastic_averaging_moving_rate = elastic_averaging_moving_rate
        self.elastic_averaging_regularization = elastic_averaging_regularization
        self.export_checkpoints_dir = export_checkpoints_dir
        self.auc_type = auc_type
        self.custom_metric_func = custom_metric_func

    @property
    def training_frame(self):
        if False:
            i = 10
            return i + 15
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator()\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.auc()\n        '
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
        '\n        Id of the validation data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(standardize=True,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('validation_frame')

    @validation_frame.setter
    def validation_frame(self, validation_frame):
        if False:
            i = 10
            return i + 15
        self._parms['validation_frame'] = H2OFrame._validate(validation_frame, 'validation_frame')

    @property
    def nfolds(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Number of folds for K-fold cross-validation (0 to disable or >= 2).\n\n        Type: ``int``, defaults to ``0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars_dl = H2ODeepLearningEstimator(nfolds=5, seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('nfolds')

    @nfolds.setter
    def nfolds(self, nfolds):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(nfolds, None, int)
        self._parms['nfolds'] = nfolds

    @property
    def keep_cross_validation_models(self):
        if False:
            return 10
        '\n        Whether to keep the cross-validation models.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars_dl = H2ODeepLearningEstimator(keep_cross_validation_models=True,\n        ...                                    nfolds=5,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> print(cars_dl.cross_validation_models())\n        '
        return self._parms.get('keep_cross_validation_models')

    @keep_cross_validation_models.setter
    def keep_cross_validation_models(self, keep_cross_validation_models):
        if False:
            return 10
        assert_is_type(keep_cross_validation_models, None, bool)
        self._parms['keep_cross_validation_models'] = keep_cross_validation_models

    @property
    def keep_cross_validation_predictions(self):
        if False:
            print('Hello World!')
        '\n        Whether to keep the predictions of the cross-validation models.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars_dl = H2ODeepLearningEstimator(keep_cross_validation_predictions=True,\n        ...                                    nfolds=5,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> print(cars_dl.cross_validation_predictions())\n        '
        return self._parms.get('keep_cross_validation_predictions')

    @keep_cross_validation_predictions.setter
    def keep_cross_validation_predictions(self, keep_cross_validation_predictions):
        if False:
            return 10
        assert_is_type(keep_cross_validation_predictions, None, bool)
        self._parms['keep_cross_validation_predictions'] = keep_cross_validation_predictions

    @property
    def keep_cross_validation_fold_assignment(self):
        if False:
            print('Hello World!')
        '\n        Whether to keep the cross-validation fold assignment.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars_dl = H2ODeepLearningEstimator(keep_cross_validation_fold_assignment=True,\n        ...                                    nfolds=5,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> print(cars_dl.cross_validation_fold_assignment())\n        '
        return self._parms.get('keep_cross_validation_fold_assignment')

    @keep_cross_validation_fold_assignment.setter
    def keep_cross_validation_fold_assignment(self, keep_cross_validation_fold_assignment):
        if False:
            print('Hello World!')
        assert_is_type(keep_cross_validation_fold_assignment, None, bool)
        self._parms['keep_cross_validation_fold_assignment'] = keep_cross_validation_fold_assignment

    @property
    def fold_assignment(self):
        if False:
            i = 10
            return i + 15
        '\n        Cross-validation fold assignment scheme, if fold_column is not specified. The \'Stratified\' option will stratify\n        the folds based on the response variable, for classification problems.\n\n        Type: ``Literal["auto", "random", "modulo", "stratified"]``, defaults to ``"auto"``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(fold_assignment="Random",\n        ...                                    nfolds=5,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('fold_assignment')

    @fold_assignment.setter
    def fold_assignment(self, fold_assignment):
        if False:
            return 10
        assert_is_type(fold_assignment, None, Enum('auto', 'random', 'modulo', 'stratified'))
        self._parms['fold_assignment'] = fold_assignment

    @property
    def fold_column(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Column with cross-validation fold index assignment per observation.\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> fold_numbers = cars.kfold_column(n_folds=5, seed=1234)\n        >>> fold_numbers.set_names(["fold_numbers"])\n        >>> cars = cars.cbind(fold_numbers)\n        >>> print(cars[\'fold_numbers\'])\n        >>> cars_dl = H2ODeepLearningEstimator(seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars,\n        ...               fold_column="fold_numbers")\n        >>> cars_dl.mse()\n        '
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
    def ignored_columns(self):
        if False:
            return 10
        '\n        Names of columns to ignore for training.\n\n        Type: ``List[str]``.\n        '
        return self._parms.get('ignored_columns')

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(ignored_columns, None, [str])
        self._parms['ignored_columns'] = ignored_columns

    @property
    def ignore_const_cols(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ignore constant columns.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars["const_1"] = 6\n        >>> cars["const_2"] = 7\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(seed=1234,\n        ...                                    ignore_const_cols=True)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('ignore_const_cols')

    @ignore_const_cols.setter
    def ignore_const_cols(self, ignore_const_cols):
        if False:
            return 10
        assert_is_type(ignore_const_cols, None, bool)
        self._parms['ignore_const_cols'] = ignore_const_cols

    @property
    def score_each_iteration(self):
        if False:
            while True:
                i = 10
        '\n        Whether to score during each iteration of model training.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars_dl = H2ODeepLearningEstimator(score_each_iteration=True,\n        ...                                    seed=1234) \n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('score_each_iteration')

    @score_each_iteration.setter
    def score_each_iteration(self, score_each_iteration):
        if False:
            i = 10
            return i + 15
        assert_is_type(score_each_iteration, None, bool)
        self._parms['score_each_iteration'] = score_each_iteration

    @property
    def weights_column(self):
        if False:
            while True:
                i = 10
        '\n        Column with observation weights. Giving some observation a weight of zero is equivalent to excluding it from the\n        dataset; giving an observation a relative weight of 2 is equivalent to repeating that row twice. Negative\n        weights are not allowed. Note: Weights are per-row observation weights and do not increase the size of the data\n        frame. This is typically the number of times a row is repeated, but non-integer values are supported as well.\n        During training, rows with higher weights matter more, due to the larger loss function pre-factor. If you set\n        weight = 0 for a row, the returned prediction frame at that row is zero and this is incorrect. To get an\n        accurate prediction, remove all rows with weight == 0.\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('weights_column')

    @weights_column.setter
    def weights_column(self, weights_column):
        if False:
            print('Hello World!')
        assert_is_type(weights_column, None, str)
        self._parms['weights_column'] = weights_column

    @property
    def offset_column(self):
        if False:
            return 10
        '\n        Offset column. This will be added to the combination of columns before applying the link function.\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> boston = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/BostonHousing.csv")\n        >>> predictors = boston.columns[:-1]\n        >>> response = "medv"\n        >>> boston[\'chas\'] = boston[\'chas\'].asfactor()\n        >>> boston["offset"] = boston["medv"].log()\n        >>> train, valid = boston.split_frame(ratios=[.8], seed=1234)\n        >>> boston_dl = H2ODeepLearningEstimator(offset_column="offset",\n        ...                                      seed=1234)\n        >>> boston_dl.train(x=predictors,\n        ...                 y=response,\n        ...                 training_frame=train,\n        ...                 validation_frame=valid)\n        >>> boston_dl.mse()\n        '
        return self._parms.get('offset_column')

    @offset_column.setter
    def offset_column(self, offset_column):
        if False:
            i = 10
            return i + 15
        assert_is_type(offset_column, None, str)
        self._parms['offset_column'] = offset_column

    @property
    def balance_classes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Balance training data class counts via over/under-sampling (for imbalanced data).\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> predictors = covtype.columns[0:54]\n        >>> response = \'C55\'\n        >>> train, valid = covtype.split_frame(ratios=[.8], seed=1234)\n        >>> cov_dl = H2ODeepLearningEstimator(balance_classes=True,\n        ...                                   seed=1234)\n        >>> cov_dl.train(x=predictors,\n        ...              y=response,\n        ...              training_frame=train,\n        ...              validation_frame=valid)\n        >>> cov_dl.mse()\n        '
        return self._parms.get('balance_classes')

    @balance_classes.setter
    def balance_classes(self, balance_classes):
        if False:
            i = 10
            return i + 15
        assert_is_type(balance_classes, None, bool)
        self._parms['balance_classes'] = balance_classes

    @property
    def class_sampling_factors(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Desired over/under-sampling ratios per class (in lexicographic order). If not specified, sampling factors will\n        be automatically computed to obtain class balance during training. Requires balance_classes.\n\n        Type: ``List[float]``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> predictors = covtype.columns[0:54]\n        >>> response = \'C55\'\n        >>> train, valid = covtype.split_frame(ratios=[.8], seed=1234)\n        >>> sample_factors = [1., 0.5, 1., 1., 1., 1., 1.]\n        >>> cars_dl = H2ODeepLearningEstimator(balance_classes=True,\n        ...                                    class_sampling_factors=sample_factors,\n        ...                                    seed=1234)\n        >>> cov_dl.train(x=predictors,\n        ...              y=response,\n        ...              training_frame=train,\n        ...              validation_frame=valid)\n        >>> cov_dl.mse()\n        '
        return self._parms.get('class_sampling_factors')

    @class_sampling_factors.setter
    def class_sampling_factors(self, class_sampling_factors):
        if False:
            return 10
        assert_is_type(class_sampling_factors, None, [float])
        self._parms['class_sampling_factors'] = class_sampling_factors

    @property
    def max_after_balance_size(self):
        if False:
            while True:
                i = 10
        '\n        Maximum relative size of the training data after balancing class counts (can be less than 1.0). Requires\n        balance_classes.\n\n        Type: ``float``, defaults to ``5.0``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> predictors = covtype.columns[0:54]\n        >>> response = \'C55\'\n        >>> train, valid = covtype.split_frame(ratios=[.8], seed=1234)\n        >>> max = .85\n        >>> cov_dl = H2ODeepLearningEstimator(balance_classes=True,\n        ...                                   max_after_balance_size=max,\n        ...                                   seed=1234)\n        >>> cov_dl.train(x=predictors,\n        ...              y=response,\n        ...              training_frame=train,\n        ...              validation_frame=valid)\n        >>> cov_dl.logloss()\n        '
        return self._parms.get('max_after_balance_size')

    @max_after_balance_size.setter
    def max_after_balance_size(self, max_after_balance_size):
        if False:
            while True:
                i = 10
        assert_is_type(max_after_balance_size, None, float)
        self._parms['max_after_balance_size'] = max_after_balance_size

    @property
    def max_confusion_matrix_size(self):
        if False:
            i = 10
            return i + 15
        '\n        [Deprecated] Maximum size (# classes) for confusion matrices to be printed in the Logs.\n\n        Type: ``int``, defaults to ``20``.\n        '
        return self._parms.get('max_confusion_matrix_size')

    @max_confusion_matrix_size.setter
    def max_confusion_matrix_size(self, max_confusion_matrix_size):
        if False:
            return 10
        assert_is_type(max_confusion_matrix_size, None, int)
        self._parms['max_confusion_matrix_size'] = max_confusion_matrix_size

    @property
    def checkpoint(self):
        if False:
            print('Hello World!')
        '\n        Model checkpoint to resume training with.\n\n        Type: ``Union[None, str, H2OEstimator]``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(activation="tanh",\n        ...                                    autoencoder=True,\n        ...                                    seed=1234,\n        ...                                    model_id="cars_dl")\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        >>> cars_cont = H2ODeepLearningEstimator(checkpoint=cars_dl,\n        ...                                      seed=1234)\n        >>> cars_cont.train(x=predictors,\n        ...                 y=response,\n        ...                 training_frame=train,\n        ...                 validation_frame=valid)\n        >>> cars_cont.mse()\n        '
        return self._parms.get('checkpoint')

    @checkpoint.setter
    def checkpoint(self, checkpoint):
        if False:
            while True:
                i = 10
        assert_is_type(checkpoint, None, str, H2OEstimator)
        self._parms['checkpoint'] = checkpoint

    @property
    def pretrained_autoencoder(self):
        if False:
            while True:
                i = 10
        '\n        Pretrained autoencoder model to initialize this model with.\n\n        Type: ``Union[None, str, H2OEstimator]``.\n\n        :examples:\n\n        >>> from h2o.estimators.deeplearning import H2OAutoEncoderEstimator\n        >>> resp = 784\n        >>> nfeatures = 20\n        >>> train = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/train.csv.gz")\n        >>> test = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/test.csv.gz")\n        >>> train[resp] = train[resp].asfactor()\n        >>> test[resp] = test[resp].asfactor()\n        >>> sid = train[0].runif(0)\n        >>> train_unsupervised = train[sid>=0.5]\n        >>> train_unsupervised.pop(resp)\n        >>> train_supervised = train[sid<0.5]\n        >>> ae_model = H2OAutoEncoderEstimator(activation="Tanh",\n        ...                                    hidden=[nfeatures],\n        ...                                    model_id="ae_model",\n        ...                                    epochs=1,\n        ...                                    ignore_const_cols=False,\n        ...                                    reproducible=True,\n        ...                                    seed=1234)\n        >>> ae_model.train(list(range(resp)), training_frame=train_unsupervised)\n        >>> ae_model.mse()\n        >>> pretrained_model = H2ODeepLearningEstimator(activation="Tanh",\n        ...                                             hidden=[nfeatures],\n        ...                                             epochs=1,\n        ...                                             reproducible = True,\n        ...                                             seed=1234,\n        ...                                             ignore_const_cols=False,\n        ...                                             pretrained_autoencoder="ae_model")\n        >>> pretrained_model.train(list(range(resp)), resp,\n        ...                        training_frame=train_supervised,\n        ...                        validation_frame=test)\n        >>> pretrained_model.mse()\n        '
        return self._parms.get('pretrained_autoencoder')

    @pretrained_autoencoder.setter
    def pretrained_autoencoder(self, pretrained_autoencoder):
        if False:
            i = 10
            return i + 15
        assert_is_type(pretrained_autoencoder, None, str, H2OEstimator)
        self._parms['pretrained_autoencoder'] = pretrained_autoencoder

    @property
    def overwrite_with_best_model(self):
        if False:
            print('Hello World!')
        '\n        If enabled, override the final model with the best model found during training.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> boston = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/BostonHousing.csv")\n        >>> predictors = boston.columns[:-1]\n        >>> response = "medv"\n        >>> boston[\'chas\'] = boston[\'chas\'].asfactor()\n        >>> boston["offset"] = boston["medv"].log()\n        >>> train, valid = boston.split_frame(ratios=[.8], seed=1234)\n        >>> boston_dl = H2ODeepLearningEstimator(overwrite_with_best_model=True,\n        ...                                      seed=1234)\n        >>> boston_dl.train(x=predictors,\n        ...                 y=response,\n        ...                 training_frame=train,\n        ...                 validation_frame=valid)\n        >>> boston_dl.mse()\n        '
        return self._parms.get('overwrite_with_best_model')

    @overwrite_with_best_model.setter
    def overwrite_with_best_model(self, overwrite_with_best_model):
        if False:
            while True:
                i = 10
        assert_is_type(overwrite_with_best_model, None, bool)
        self._parms['overwrite_with_best_model'] = overwrite_with_best_model

    @property
    def use_all_factor_levels(self):
        if False:
            i = 10
            return i + 15
        '\n        Use all factor levels of categorical variables. Otherwise, the first factor level is omitted (without loss of\n        accuracy). Useful for variable importances and auto-enabled for autoencoder.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator(use_all_factor_levels=True,\n        ...                                        seed=1234)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.mse()\n        '
        return self._parms.get('use_all_factor_levels')

    @use_all_factor_levels.setter
    def use_all_factor_levels(self, use_all_factor_levels):
        if False:
            while True:
                i = 10
        assert_is_type(use_all_factor_levels, None, bool)
        self._parms['use_all_factor_levels'] = use_all_factor_levels

    @property
    def standardize(self):
        if False:
            print('Hello World!')
        '\n        If enabled, automatically standardize the data. If disabled, the user must provide properly scaled input data.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars_dl = H2ODeepLearningEstimator(standardize=True,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('standardize')

    @standardize.setter
    def standardize(self, standardize):
        if False:
            return 10
        assert_is_type(standardize, None, bool)
        self._parms['standardize'] = standardize

    @property
    def activation(self):
        if False:
            i = 10
            return i + 15
        '\n        Activation function.\n\n        Type: ``Literal["tanh", "tanh_with_dropout", "rectifier", "rectifier_with_dropout", "maxout",\n        "maxout_with_dropout"]``, defaults to ``"rectifier"``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> cars_dl = H2ODeepLearningEstimator(activation="tanh")\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('activation')

    @activation.setter
    def activation(self, activation):
        if False:
            i = 10
            return i + 15
        assert_is_type(activation, None, Enum('tanh', 'tanh_with_dropout', 'rectifier', 'rectifier_with_dropout', 'maxout', 'maxout_with_dropout'))
        self._parms['activation'] = activation

    @property
    def hidden(self):
        if False:
            i = 10
            return i + 15
        '\n        Hidden layer sizes (e.g. [100, 100]).\n\n        Type: ``List[int]``, defaults to ``[200, 200]``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(hidden=[100,100],\n        ...                                    seed=1234) \n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('hidden')

    @hidden.setter
    def hidden(self, hidden):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(hidden, None, [int])
        self._parms['hidden'] = hidden

    @property
    def epochs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        How many times the dataset should be iterated (streamed), can be fractional.\n\n        Type: ``float``, defaults to ``10.0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(epochs=15,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('epochs')

    @epochs.setter
    def epochs(self, epochs):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(epochs, None, numeric)
        self._parms['epochs'] = epochs

    @property
    def train_samples_per_iteration(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Number of training samples (globally) per MapReduce iteration. Special values are 0: one epoch, -1: all\n        available data (e.g., replicated training data), -2: automatic.\n\n        Type: ``int``, defaults to ``-2``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator(train_samples_per_iteration=-1,\n        ...                                        epochs=1,\n        ...                                        seed=1234)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.auc()\n        '
        return self._parms.get('train_samples_per_iteration')

    @train_samples_per_iteration.setter
    def train_samples_per_iteration(self, train_samples_per_iteration):
        if False:
            return 10
        assert_is_type(train_samples_per_iteration, None, int)
        self._parms['train_samples_per_iteration'] = train_samples_per_iteration

    @property
    def target_ratio_comm_to_comp(self):
        if False:
            i = 10
            return i + 15
        '\n        Target ratio of communication overhead to computation. Only for multi-node operation and\n        train_samples_per_iteration = -2 (auto-tuning).\n\n        Type: ``float``, defaults to ``0.05``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator(target_ratio_comm_to_comp=0.05,\n        ...                                        seed=1234)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.auc()\n        '
        return self._parms.get('target_ratio_comm_to_comp')

    @target_ratio_comm_to_comp.setter
    def target_ratio_comm_to_comp(self, target_ratio_comm_to_comp):
        if False:
            print('Hello World!')
        assert_is_type(target_ratio_comm_to_comp, None, numeric)
        self._parms['target_ratio_comm_to_comp'] = target_ratio_comm_to_comp

    @property
    def seed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Seed for random numbers (affects sampling) - Note: only reproducible when running single threaded.\n\n        Type: ``int``, defaults to ``-1``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('seed')

    @seed.setter
    def seed(self, seed):
        if False:
            while True:
                i = 10
        assert_is_type(seed, None, int)
        self._parms['seed'] = seed

    @property
    def adaptive_rate(self):
        if False:
            while True:
                i = 10
        '\n        Adaptive learning rate.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> cars_dl = H2ODeepLearningEstimator(adaptive_rate=True)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('adaptive_rate')

    @adaptive_rate.setter
    def adaptive_rate(self, adaptive_rate):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(adaptive_rate, None, bool)
        self._parms['adaptive_rate'] = adaptive_rate

    @property
    def rho(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adaptive learning rate time decay factor (similarity to prior updates).\n\n        Type: ``float``, defaults to ``0.99``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars_dl = H2ODeepLearningEstimator(rho=0.9,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('rho')

    @rho.setter
    def rho(self, rho):
        if False:
            i = 10
            return i + 15
        assert_is_type(rho, None, numeric)
        self._parms['rho'] = rho

    @property
    def epsilon(self):
        if False:
            while True:
                i = 10
        '\n        Adaptive learning rate smoothing factor (to avoid divisions by zero and allow progress).\n\n        Type: ``float``, defaults to ``1e-08``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(epsilon=1e-6,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('epsilon')

    @epsilon.setter
    def epsilon(self, epsilon):
        if False:
            while True:
                i = 10
        assert_is_type(epsilon, None, numeric)
        self._parms['epsilon'] = epsilon

    @property
    def rate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Learning rate (higher => less stable, lower => slower convergence).\n\n        Type: ``float``, defaults to ``0.005``.\n\n        :examples:\n\n        >>> train = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/train.csv.gz")\n        >>> test = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/test.csv.gz")\n        >>> predictors = list(range(0,784))\n        >>> resp = 784\n        >>> train[resp] = train[resp].asfactor()\n        >>> test[resp] = test[resp].asfactor()\n        >>> nclasses = train[resp].nlevels()[0]\n        >>> model = H2ODeepLearningEstimator(activation="RectifierWithDropout",\n        ...                                  adaptive_rate=False,\n        ...                                  rate=0.01,\n        ...                                  rate_decay=0.9,\n        ...                                  rate_annealing=1e-6,\n        ...                                  momentum_start=0.95,\n        ...                                  momentum_ramp=1e5,\n        ...                                  momentum_stable=0.99,\n        ...                                  nesterov_accelerated_gradient=False,\n        ...                                  input_dropout_ratio=0.2,\n        ...                                  train_samples_per_iteration=20000,\n        ...                                  classification_stop=-1,\n        ...                                  l1=1e-5)\n        >>> model.train (x=predictors,y=resp, training_frame=train, validation_frame=test)\n        >>> model.model_performance(valid=True)\n        '
        return self._parms.get('rate')

    @rate.setter
    def rate(self, rate):
        if False:
            print('Hello World!')
        assert_is_type(rate, None, numeric)
        self._parms['rate'] = rate

    @property
    def rate_annealing(self):
        if False:
            return 10
        '\n        Learning rate annealing: rate / (1 + rate_annealing * samples).\n\n        Type: ``float``, defaults to ``1e-06``.\n\n        :examples:\n\n        >>> train = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/train.csv.gz")\n        >>> test = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/test.csv.gz")\n        >>> predictors = list(range(0,784))\n        >>> resp = 784\n        >>> train[resp] = train[resp].asfactor()\n        >>> test[resp] = test[resp].asfactor()\n        >>> nclasses = train[resp].nlevels()[0]\n        >>> model = H2ODeepLearningEstimator(activation="RectifierWithDropout",\n        ...                                  adaptive_rate=False,\n        ...                                  rate=0.01,\n        ...                                  rate_decay=0.9,\n        ...                                  rate_annealing=1e-6,\n        ...                                  momentum_start=0.95,\n        ...                                  momentum_ramp=1e5,\n        ...                                  momentum_stable=0.99,\n        ...                                  nesterov_accelerated_gradient=False,\n        ...                                  input_dropout_ratio=0.2,\n        ...                                  train_samples_per_iteration=20000,\n        ...                                  classification_stop=-1,\n        ...                                  l1=1e-5)\n        >>> model.train (x=predictors,\n        ...              y=resp,\n        ...              training_frame=train,\n        ...              validation_frame=test)\n        >>> model.mse()\n        '
        return self._parms.get('rate_annealing')

    @rate_annealing.setter
    def rate_annealing(self, rate_annealing):
        if False:
            return 10
        assert_is_type(rate_annealing, None, numeric)
        self._parms['rate_annealing'] = rate_annealing

    @property
    def rate_decay(self):
        if False:
            i = 10
            return i + 15
        '\n        Learning rate decay factor between layers (N-th layer: rate * rate_decay ^ (n - 1).\n\n        Type: ``float``, defaults to ``1.0``.\n\n        :examples:\n\n        >>> train = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/train.csv.gz")\n        >>> test = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/test.csv.gz")\n        >>> predictors = list(range(0,784))\n        >>> resp = 784\n        >>> train[resp] = train[resp].asfactor()\n        >>> test[resp] = test[resp].asfactor()\n        >>> nclasses = train[resp].nlevels()[0]\n        >>> model = H2ODeepLearningEstimator(activation="RectifierWithDropout",\n        ...                                  adaptive_rate=False,\n        ...                                  rate=0.01,\n        ...                                  rate_decay=0.9,\n        ...                                  rate_annealing=1e-6,\n        ...                                  momentum_start=0.95,\n        ...                                  momentum_ramp=1e5,\n        ...                                  momentum_stable=0.99,\n        ...                                  nesterov_accelerated_gradient=False,\n        ...                                  input_dropout_ratio=0.2,\n        ...                                  train_samples_per_iteration=20000,\n        ...                                  classification_stop=-1,\n        ...                                  l1=1e-5)\n        >>> model.train (x=predictors,\n        ...              y=resp,\n        ...              training_frame=train,\n        ...              validation_frame=test)\n        >>> model.model_performance()\n        '
        return self._parms.get('rate_decay')

    @rate_decay.setter
    def rate_decay(self, rate_decay):
        if False:
            while True:
                i = 10
        assert_is_type(rate_decay, None, numeric)
        self._parms['rate_decay'] = rate_decay

    @property
    def momentum_start(self):
        if False:
            i = 10
            return i + 15
        '\n        Initial momentum at the beginning of training (try 0.5).\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime",\n        ...               "CRSArrTime","UniqueCarrier","FlightNum"]\n        >>> response_col = "IsDepDelayed"\n        >>> airlines_dl = H2ODeepLearningEstimator(hidden=[200,200],\n        ...                                        activation="Rectifier",\n        ...                                        input_dropout_ratio=0.0,\n        ...                                        momentum_start=0.9,\n        ...                                        momentum_stable=0.99,\n        ...                                        momentum_ramp=1e7,\n        ...                                        epochs=100,\n        ...                                        stopping_rounds=4,\n        ...                                        train_samples_per_iteration=30000,\n        ...                                        mini_batch_size=32,\n        ...                                        score_duty_cycle=0.25,\n        ...                                        score_interval=1)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response_col,\n        ...                   training_frame=airlines)\n        >>> airlines_dl.mse()\n        '
        return self._parms.get('momentum_start')

    @momentum_start.setter
    def momentum_start(self, momentum_start):
        if False:
            print('Hello World!')
        assert_is_type(momentum_start, None, numeric)
        self._parms['momentum_start'] = momentum_start

    @property
    def momentum_ramp(self):
        if False:
            print('Hello World!')
        '\n        Number of training samples for which momentum increases.\n\n        Type: ``float``, defaults to ``1000000.0``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime",\n        ...               "CRSArrTime","UniqueCarrier","FlightNum"]\n        >>> response_col = "IsDepDelayed"\n        >>> airlines_dl = H2ODeepLearningEstimator(hidden=[200,200],\n        ...                                        activation="Rectifier",\n        ...                                        input_dropout_ratio=0.0,\n        ...                                        momentum_start=0.9,\n        ...                                        momentum_stable=0.99,\n        ...                                        momentum_ramp=1e7,\n        ...                                        epochs=100,\n        ...                                        stopping_rounds=4,\n        ...                                        train_samples_per_iteration=30000,\n        ...                                        mini_batch_size=32,\n        ...                                        score_duty_cycle=0.25,\n        ...                                        score_interval=1)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response_col,\n        ...                   training_frame=airlines)\n        >>> airlines_dl.mse()\n        '
        return self._parms.get('momentum_ramp')

    @momentum_ramp.setter
    def momentum_ramp(self, momentum_ramp):
        if False:
            i = 10
            return i + 15
        assert_is_type(momentum_ramp, None, numeric)
        self._parms['momentum_ramp'] = momentum_ramp

    @property
    def momentum_stable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Final momentum after the ramp is over (try 0.99).\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> predictors = ["Year","Month","DayofMonth","DayOfWeek","CRSDepTime",\n        ...               "CRSArrTime","UniqueCarrier","FlightNum"]\n        >>> response_col = "IsDepDelayed"\n        >>> airlines_dl = H2ODeepLearningEstimator(hidden=[200,200],\n        ...                                        activation="Rectifier",\n        ...                                        input_dropout_ratio=0.0,\n        ...                                        momentum_start=0.9,\n        ...                                        momentum_stable=0.99,\n        ...                                        momentum_ramp=1e7,\n        ...                                        epochs=100,\n        ...                                        stopping_rounds=4,\n        ...                                        train_samples_per_iteration=30000,\n        ...                                        mini_batch_size=32,\n        ...                                        score_duty_cycle=0.25,\n        ...                                        score_interval=1)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response_col,\n        ...                   training_frame=airlines)\n        >>> airlines_dl.mse()\n        '
        return self._parms.get('momentum_stable')

    @momentum_stable.setter
    def momentum_stable(self, momentum_stable):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(momentum_stable, None, numeric)
        self._parms['momentum_stable'] = momentum_stable

    @property
    def nesterov_accelerated_gradient(self):
        if False:
            while True:
                i = 10
        '\n        Use Nesterov accelerated gradient (recommended).\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> train = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/train.csv.gz")\n        >>> test = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/test.csv.gz")\n        >>> predictors = list(range(0,784))\n        >>> resp = 784\n        >>> train[resp] = train[resp].asfactor()\n        >>> test[resp] = test[resp].asfactor()\n        >>> nclasses = train[resp].nlevels()[0]\n        >>> model = H2ODeepLearningEstimator(activation="RectifierWithDropout",\n        ...                                  adaptive_rate=False,\n        ...                                  rate=0.01,\n        ...                                  rate_decay=0.9,\n        ...                                  rate_annealing=1e-6,\n        ...                                  momentum_start=0.95,\n        ...                                  momentum_ramp=1e5,\n        ...                                  momentum_stable=0.99,\n        ...                                  nesterov_accelerated_gradient=False,\n        ...                                  input_dropout_ratio=0.2,\n        ...                                  train_samples_per_iteration=20000,\n        ...                                  classification_stop=-1,\n        ...                                  l1=1e-5) \n        >>> model.train (x=predictors,\n        ...              y=resp,\n        ...              training_frame=train,\n        ...              validation_frame=test)\n        >>> model.model_performance()\n        '
        return self._parms.get('nesterov_accelerated_gradient')

    @nesterov_accelerated_gradient.setter
    def nesterov_accelerated_gradient(self, nesterov_accelerated_gradient):
        if False:
            i = 10
            return i + 15
        assert_is_type(nesterov_accelerated_gradient, None, bool)
        self._parms['nesterov_accelerated_gradient'] = nesterov_accelerated_gradient

    @property
    def input_dropout_ratio(self):
        if False:
            print('Hello World!')
        '\n        Input layer dropout ratio (can improve generalization, try 0.1 or 0.2).\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(input_dropout_ratio=0.2,\n        ...                                    seed=1234) \n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('input_dropout_ratio')

    @input_dropout_ratio.setter
    def input_dropout_ratio(self, input_dropout_ratio):
        if False:
            i = 10
            return i + 15
        assert_is_type(input_dropout_ratio, None, numeric)
        self._parms['input_dropout_ratio'] = input_dropout_ratio

    @property
    def hidden_dropout_ratios(self):
        if False:
            i = 10
            return i + 15
        '\n        Hidden layer dropout ratios (can improve generalization), specify one value per hidden layer, defaults to 0.5.\n\n        Type: ``List[float]``.\n\n        :examples:\n\n        >>> train = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/train.csv.gz")\n        >>> valid = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/test.csv.gz")\n        >>> features = list(range(0,784))\n        >>> target = 784\n        >>> train[target] = train[target].asfactor()\n        >>> valid[target] = valid[target].asfactor()\n        >>> model = H2ODeepLearningEstimator(epochs=20,\n        ...                                  hidden=[200,200],\n        ...                                  hidden_dropout_ratios=[0.5,0.5],\n        ...                                  seed=1234,\n        ...                                  activation=\'tanhwithdropout\')\n        >>> model.train(x=features,\n        ...             y=target,\n        ...             training_frame=train,\n        ...             validation_frame=valid)\n        >>> model.mse()\n        '
        return self._parms.get('hidden_dropout_ratios')

    @hidden_dropout_ratios.setter
    def hidden_dropout_ratios(self, hidden_dropout_ratios):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(hidden_dropout_ratios, None, [numeric])
        self._parms['hidden_dropout_ratios'] = hidden_dropout_ratios

    @property
    def l1(self):
        if False:
            return 10
        '\n        L1 regularization (can add stability and improve generalization, causes many weights to become 0).\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> hh_imbalanced = H2ODeepLearningEstimator(l1=1e-5,\n        ...                                          activation="Rectifier",\n        ...                                          loss="CrossEntropy",\n        ...                                          hidden=[200,200],\n        ...                                          epochs=1,\n        ...                                          balance_classes=False,\n        ...                                          reproducible=True,\n        ...                                          seed=1234)\n        >>> hh_imbalanced.train(x=list(range(54)),y=54, training_frame=covtype)\n        >>> hh_imbalanced.mse()\n        '
        return self._parms.get('l1')

    @l1.setter
    def l1(self, l1):
        if False:
            while True:
                i = 10
        assert_is_type(l1, None, numeric)
        self._parms['l1'] = l1

    @property
    def l2(self):
        if False:
            print('Hello World!')
        '\n        L2 regularization (can add stability and improve generalization, causes many weights to be small.\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> hh_imbalanced = H2ODeepLearningEstimator(l2=1e-5,\n        ...                                          activation="Rectifier",\n        ...                                          loss="CrossEntropy",\n        ...                                          hidden=[200,200],\n        ...                                          epochs=1,\n        ...                                          balance_classes=False,\n        ...                                          reproducible=True,\n        ...                                          seed=1234)\n        >>> hh_imbalanced.train(x=list(range(54)),y=54, training_frame=covtype)\n        >>> hh_imbalanced.mse()\n        '
        return self._parms.get('l2')

    @l2.setter
    def l2(self, l2):
        if False:
            while True:
                i = 10
        assert_is_type(l2, None, numeric)
        self._parms['l2'] = l2

    @property
    def max_w2(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constraint for squared sum of incoming weights per unit (e.g. for Rectifier).\n\n        Type: ``float``, defaults to ``3.4028235e+38``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> predictors = covtype.columns[0:54]\n        >>> response = \'C55\'\n        >>> train, valid = covtype.split_frame(ratios=[.8], seed=1234)\n        >>> cov_dl = H2ODeepLearningEstimator(activation="RectifierWithDropout",\n        ...                                   hidden=[10,10],\n        ...                                   epochs=10,\n        ...                                   input_dropout_ratio=0.2,\n        ...                                   l1=1e-5,\n        ...                                   max_w2=10.5,\n        ...                                   stopping_rounds=0)\n        >>> cov_dl.train(x=predictors,\n        ...              y=response,\n        ...              training_frame=train,\n        ...              validation_frame=valid)\n        >>> cov_dl.mse()\n        '
        return self._parms.get('max_w2')

    @max_w2.setter
    def max_w2(self, max_w2):
        if False:
            print('Hello World!')
        assert_is_type(max_w2, None, float)
        self._parms['max_w2'] = max_w2

    @property
    def initial_weight_distribution(self):
        if False:
            while True:
                i = 10
        '\n        Initial weight distribution.\n\n        Type: ``Literal["uniform_adaptive", "uniform", "normal"]``, defaults to ``"uniform_adaptive"``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(initial_weight_distribution="Uniform",\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('initial_weight_distribution')

    @initial_weight_distribution.setter
    def initial_weight_distribution(self, initial_weight_distribution):
        if False:
            while True:
                i = 10
        assert_is_type(initial_weight_distribution, None, Enum('uniform_adaptive', 'uniform', 'normal'))
        self._parms['initial_weight_distribution'] = initial_weight_distribution

    @property
    def initial_weight_scale(self):
        if False:
            while True:
                i = 10
        '\n        Uniform: -value...value, Normal: stddev.\n\n        Type: ``float``, defaults to ``1.0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(initial_weight_scale=1.5,\n        ...                                    seed=1234) \n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('initial_weight_scale')

    @initial_weight_scale.setter
    def initial_weight_scale(self, initial_weight_scale):
        if False:
            return 10
        assert_is_type(initial_weight_scale, None, numeric)
        self._parms['initial_weight_scale'] = initial_weight_scale

    @property
    def initial_weights(self):
        if False:
            while True:
                i = 10
        '\n        A list of H2OFrame ids to initialize the weight matrices of this model with.\n\n        Type: ``List[Union[None, str, H2OFrame]]``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris.csv")\n        >>> dl1 = H2ODeepLearningEstimator(hidden=[10,10],\n        ...                                export_weights_and_biases=True)\n        >>> dl1.train(x=list(range(4)), y=4, training_frame=iris)\n        >>> p1 = dl1.model_performance(iris).logloss()\n        >>> ll1 = dl1.predict(iris)\n        >>> print(p1)\n        >>> w1 = dl1.weights(0)\n        >>> w2 = dl1.weights(1)\n        >>> w3 = dl1.weights(2)\n        >>> b1 = dl1.biases(0)\n        >>> b2 = dl1.biases(1)\n        >>> b3 = dl1.biases(2)\n        >>> dl2 = H2ODeepLearningEstimator(hidden=[10,10],\n        ...                                initial_weights=[w1, w2, w3],\n        ...                                initial_biases=[b1, b2, b3],\n        ...                                epochs=0)\n        >>> dl2.train(x=list(range(4)), y=4, training_frame=iris)\n        >>> dl2.initial_weights\n        '
        return self._parms.get('initial_weights')

    @initial_weights.setter
    def initial_weights(self, initial_weights):
        if False:
            i = 10
            return i + 15
        assert_is_type(initial_weights, None, [None, str, H2OFrame])
        self._parms['initial_weights'] = initial_weights

    @property
    def initial_biases(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A list of H2OFrame ids to initialize the bias vectors of this model with.\n\n        Type: ``List[Union[None, str, H2OFrame]]``.\n\n        :examples:\n\n        >>> iris = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris.csv")\n        >>> dl1 = H2ODeepLearningEstimator(hidden=[10,10],\n        ...                                export_weights_and_biases=True)\n        >>> dl1.train(x=list(range(4)), y=4, training_frame=iris)\n        >>> p1 = dl1.model_performance(iris).logloss()\n        >>> ll1 = dl1.predict(iris)\n        >>> print(p1)\n        >>> w1 = dl1.weights(0)\n        >>> w2 = dl1.weights(1)\n        >>> w3 = dl1.weights(2)\n        >>> b1 = dl1.biases(0)\n        >>> b2 = dl1.biases(1)\n        >>> b3 = dl1.biases(2)\n        >>> dl2 = H2ODeepLearningEstimator(hidden=[10,10],\n        ...                                initial_weights=[w1, w2, w3],\n        ...                                initial_biases=[b1, b2, b3],\n        ...                                epochs=0)\n        >>> dl2.train(x=list(range(4)), y=4, training_frame=iris)\n        >>> dl2.initial_biases\n        '
        return self._parms.get('initial_biases')

    @initial_biases.setter
    def initial_biases(self, initial_biases):
        if False:
            i = 10
            return i + 15
        assert_is_type(initial_biases, None, [None, str, H2OFrame])
        self._parms['initial_biases'] = initial_biases

    @property
    def loss(self):
        if False:
            print('Hello World!')
        '\n        Loss function.\n\n        Type: ``Literal["automatic", "cross_entropy", "quadratic", "huber", "absolute", "quantile"]``, defaults to\n        ``"automatic"``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> hh_imbalanced = H2ODeepLearningEstimator(l1=1e-5,\n        ...                                          activation="Rectifier",\n        ...                                          loss="CrossEntropy",\n        ...                                          hidden=[200,200],\n        ...                                          epochs=1,\n        ...                                          balance_classes=False,\n        ...                                          reproducible=True,\n        ...                                          seed=1234)\n        >>> hh_imbalanced.train(x=list(range(54)),y=54, training_frame=covtype)\n        >>> hh_imbalanced.mse()\n        '
        return self._parms.get('loss')

    @loss.setter
    def loss(self, loss):
        if False:
            print('Hello World!')
        assert_is_type(loss, None, Enum('automatic', 'cross_entropy', 'quadratic', 'huber', 'absolute', 'quantile'))
        self._parms['loss'] = loss

    @property
    def distribution(self):
        if False:
            while True:
                i = 10
        '\n        Distribution function\n\n        Type: ``Literal["auto", "bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace",\n        "quantile", "huber"]``, defaults to ``"auto"``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(distribution="poisson",\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('distribution')

    @distribution.setter
    def distribution(self, distribution):
        if False:
            print('Hello World!')
        assert_is_type(distribution, None, Enum('auto', 'bernoulli', 'multinomial', 'gaussian', 'poisson', 'gamma', 'tweedie', 'laplace', 'quantile', 'huber'))
        self._parms['distribution'] = distribution

    @property
    def quantile_alpha(self):
        if False:
            while True:
                i = 10
        '\n        Desired quantile for Quantile regression, must be between 0 and 1.\n\n        Type: ``float``, defaults to ``0.5``.\n\n        :examples:\n\n        >>> boston = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/BostonHousing.csv")\n        >>> predictors = boston.columns[:-1]\n        >>> response = "medv"\n        >>> boston[\'chas\'] = boston[\'chas\'].asfactor()\n        >>> train, valid = boston.split_frame(ratios=[.8], seed=1234)\n        >>> boston_dl = H2ODeepLearningEstimator(distribution="quantile",\n        ...                                      quantile_alpha=.8,\n        ...                                      seed=1234)\n        >>> boston_dl.train(x=predictors,\n        ...                 y=response,\n        ...                 training_frame=train,\n        ...                 validation_frame=valid)\n        >>> boston_dl.mse()\n        '
        return self._parms.get('quantile_alpha')

    @quantile_alpha.setter
    def quantile_alpha(self, quantile_alpha):
        if False:
            print('Hello World!')
        assert_is_type(quantile_alpha, None, numeric)
        self._parms['quantile_alpha'] = quantile_alpha

    @property
    def tweedie_power(self):
        if False:
            i = 10
            return i + 15
        '\n        Tweedie power for Tweedie regression, must be between 1 and 2.\n\n        Type: ``float``, defaults to ``1.5``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator(tweedie_power=1.5,\n        ...                                        seed=1234) \n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.auc()\n        '
        return self._parms.get('tweedie_power')

    @tweedie_power.setter
    def tweedie_power(self, tweedie_power):
        if False:
            i = 10
            return i + 15
        assert_is_type(tweedie_power, None, numeric)
        self._parms['tweedie_power'] = tweedie_power

    @property
    def huber_alpha(self):
        if False:
            print('Hello World!')
        '\n        Desired quantile for Huber/M-regression (threshold between quadratic and linear loss, must be between 0 and 1).\n\n        Type: ``float``, defaults to ``0.9``.\n\n        :examples:\n\n        >>> insurance = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/glm_test/insurance.csv")\n        >>> predictors = insurance.columns[0:4]\n        >>> response = \'Claims\'\n        >>> insurance[\'Group\'] = insurance[\'Group\'].asfactor()\n        >>> insurance[\'Age\'] = insurance[\'Age\'].asfactor()\n        >>> train, valid = insurance.split_frame(ratios=[.8], seed=1234)\n        >>> insurance_dl = H2ODeepLearningEstimator(distribution="huber",\n        ...                                         huber_alpha=0.9,\n        ...                                         seed=1234)\n        >>> insurance_dl.train(x=predictors,\n        ...                    y=response,\n        ...                    training_frame=train,\n        ...                    validation_frame=valid)\n        >>> insurance_dl.mse()\n        '
        return self._parms.get('huber_alpha')

    @huber_alpha.setter
    def huber_alpha(self, huber_alpha):
        if False:
            while True:
                i = 10
        assert_is_type(huber_alpha, None, numeric)
        self._parms['huber_alpha'] = huber_alpha

    @property
    def score_interval(self):
        if False:
            print('Hello World!')
        '\n        Shortest time interval (in seconds) between model scoring.\n\n        Type: ``float``, defaults to ``5.0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars_dl = H2ODeepLearningEstimator(score_interval=3,\n        ...                                    seed=1234) \n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('score_interval')

    @score_interval.setter
    def score_interval(self, score_interval):
        if False:
            print('Hello World!')
        assert_is_type(score_interval, None, numeric)
        self._parms['score_interval'] = score_interval

    @property
    def score_training_samples(self):
        if False:
            while True:
                i = 10
        '\n        Number of training set samples for scoring (0 for all).\n\n        Type: ``int``, defaults to ``10000``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars_dl = H2ODeepLearningEstimator(score_training_samples=10000,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('score_training_samples')

    @score_training_samples.setter
    def score_training_samples(self, score_training_samples):
        if False:
            while True:
                i = 10
        assert_is_type(score_training_samples, None, int)
        self._parms['score_training_samples'] = score_training_samples

    @property
    def score_validation_samples(self):
        if False:
            print('Hello World!')
        '\n        Number of validation set samples for scoring (0 for all).\n\n        Type: ``int``, defaults to ``0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(score_validation_samples=3,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('score_validation_samples')

    @score_validation_samples.setter
    def score_validation_samples(self, score_validation_samples):
        if False:
            while True:
                i = 10
        assert_is_type(score_validation_samples, None, int)
        self._parms['score_validation_samples'] = score_validation_samples

    @property
    def score_duty_cycle(self):
        if False:
            i = 10
            return i + 15
        '\n        Maximum duty cycle fraction for scoring (lower: more training, higher: more scoring).\n\n        Type: ``float``, defaults to ``0.1``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> cars_dl = H2ODeepLearningEstimator(score_duty_cycle=0.2,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('score_duty_cycle')

    @score_duty_cycle.setter
    def score_duty_cycle(self, score_duty_cycle):
        if False:
            return 10
        assert_is_type(score_duty_cycle, None, numeric)
        self._parms['score_duty_cycle'] = score_duty_cycle

    @property
    def classification_stop(self):
        if False:
            while True:
                i = 10
        '\n        Stopping criterion for classification error fraction on training data (-1 to disable).\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> predictors = covtype.columns[0:54]\n        >>> response = \'C55\'\n        >>> train, valid = covtype.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(classification_stop=1.5,\n        ...                                    seed=1234)\n        >>> cov_dl.train(x=predictors,\n        ...              y=response,\n        ...              training_frame=train,\n        ...              validation_frame=valid)\n        >>> cov_dl.mse()\n        '
        return self._parms.get('classification_stop')

    @classification_stop.setter
    def classification_stop(self, classification_stop):
        if False:
            i = 10
            return i + 15
        assert_is_type(classification_stop, None, numeric)
        self._parms['classification_stop'] = classification_stop

    @property
    def regression_stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stopping criterion for regression error (MSE) on training data (-1 to disable).\n\n        Type: ``float``, defaults to ``1e-06``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator(regression_stop=1e-6,\n        ...                                        seed=1234)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.auc()\n        '
        return self._parms.get('regression_stop')

    @regression_stop.setter
    def regression_stop(self, regression_stop):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(regression_stop, None, numeric)
        self._parms['regression_stop'] = regression_stop

    @property
    def stopping_rounds(self):
        if False:
            return 10
        '\n        Early stopping based on convergence of stopping_metric. Stop if simple moving average of length k of the\n        stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)\n\n        Type: ``int``, defaults to ``5``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator(stopping_metric="auc",\n        ...                                        stopping_rounds=3,\n        ...                                        stopping_tolerance=1e-2,\n        ...                                        seed=1234)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.auc()\n        '
        return self._parms.get('stopping_rounds')

    @stopping_rounds.setter
    def stopping_rounds(self, stopping_rounds):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(stopping_rounds, None, int)
        self._parms['stopping_rounds'] = stopping_rounds

    @property
    def stopping_metric(self):
        if False:
            return 10
        '\n        Metric to use for early stopping (AUTO: logloss for classification, deviance for regression and anomaly_score\n        for Isolation Forest). Note that custom and custom_increasing can only be used in GBM and DRF with the Python\n        client.\n\n        Type: ``Literal["auto", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "lift_top_group",\n        "misclassification", "mean_per_class_error", "custom", "custom_increasing"]``, defaults to ``"auto"``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator(stopping_metric="auc",\n        ...                                        stopping_rounds=3,\n        ...                                        stopping_tolerance=1e-2,\n        ...                                        seed=1234)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.auc()\n        '
        return self._parms.get('stopping_metric')

    @stopping_metric.setter
    def stopping_metric(self, stopping_metric):
        if False:
            while True:
                i = 10
        assert_is_type(stopping_metric, None, Enum('auto', 'deviance', 'logloss', 'mse', 'rmse', 'mae', 'rmsle', 'auc', 'aucpr', 'lift_top_group', 'misclassification', 'mean_per_class_error', 'custom', 'custom_increasing'))
        self._parms['stopping_metric'] = stopping_metric

    @property
    def stopping_tolerance(self):
        if False:
            i = 10
            return i + 15
        '\n        Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at least this much)\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator(stopping_metric="auc",\n        ...                                        stopping_rounds=3,\n        ...                                        stopping_tolerance=1e-2,\n        ...                                        seed=1234)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.auc()\n        '
        return self._parms.get('stopping_tolerance')

    @stopping_tolerance.setter
    def stopping_tolerance(self, stopping_tolerance):
        if False:
            while True:
                i = 10
        assert_is_type(stopping_tolerance, None, numeric)
        self._parms['stopping_tolerance'] = stopping_tolerance

    @property
    def max_runtime_secs(self):
        if False:
            while True:
                i = 10
        '\n        Maximum allowed runtime in seconds for model training. Use 0 to disable.\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(max_runtime_secs=10,\n        ...                                    seed=1234) \n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('max_runtime_secs')

    @max_runtime_secs.setter
    def max_runtime_secs(self, max_runtime_secs):
        if False:
            i = 10
            return i + 15
        assert_is_type(max_runtime_secs, None, numeric)
        self._parms['max_runtime_secs'] = max_runtime_secs

    @property
    def score_validation_sampling(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method used to sample validation dataset for scoring.\n\n        Type: ``Literal["uniform", "stratified"]``, defaults to ``"uniform"``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(score_validation_sampling="uniform",\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('score_validation_sampling')

    @score_validation_sampling.setter
    def score_validation_sampling(self, score_validation_sampling):
        if False:
            i = 10
            return i + 15
        assert_is_type(score_validation_sampling, None, Enum('uniform', 'stratified'))
        self._parms['score_validation_sampling'] = score_validation_sampling

    @property
    def diagnostics(self):
        if False:
            print('Hello World!')
        '\n        Enable diagnostics for hidden layers.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> predictors = covtype.columns[0:54]\n        >>> response = \'C55\'\n        >>> train, valid = covtype.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(diagnostics=True,\n        ...                                    seed=1234)  \n        >>> cov_dl.train(x=predictors,\n        ...              y=response,\n        ...              training_frame=train,\n        ...              validation_frame=valid)\n        >>> cov_dl.mse()\n        '
        return self._parms.get('diagnostics')

    @diagnostics.setter
    def diagnostics(self, diagnostics):
        if False:
            return 10
        assert_is_type(diagnostics, None, bool)
        self._parms['diagnostics'] = diagnostics

    @property
    def fast_mode(self):
        if False:
            return 10
        '\n        Enable fast mode (minor approximation in back-propagation).\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(fast_mode=False,\n        ...                                    seed=1234)          \n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('fast_mode')

    @fast_mode.setter
    def fast_mode(self, fast_mode):
        if False:
            while True:
                i = 10
        assert_is_type(fast_mode, None, bool)
        self._parms['fast_mode'] = fast_mode

    @property
    def force_load_balance(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Force extra load balancing to increase training speed for small datasets (to keep all cores busy).\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(force_load_balance=False,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('force_load_balance')

    @force_load_balance.setter
    def force_load_balance(self, force_load_balance):
        if False:
            return 10
        assert_is_type(force_load_balance, None, bool)
        self._parms['force_load_balance'] = force_load_balance

    @property
    def variable_importances(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute variable importances for input features (Gedeon method) - can be slow for large networks.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator(variable_importances=True,\n        ...                                        seed=1234)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.mse()\n        '
        return self._parms.get('variable_importances')

    @variable_importances.setter
    def variable_importances(self, variable_importances):
        if False:
            while True:
                i = 10
        assert_is_type(variable_importances, None, bool)
        self._parms['variable_importances'] = variable_importances

    @property
    def replicate_training_data(self):
        if False:
            print('Hello World!')
        '\n        Replicate the entire training dataset onto every node for faster training on small datasets.\n\n        Type: ``bool``, defaults to ``True``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> airlines_dl = H2ODeepLearningEstimator(replicate_training_data=False)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=airlines) \n        >>> airlines_dl.auc()\n        '
        return self._parms.get('replicate_training_data')

    @replicate_training_data.setter
    def replicate_training_data(self, replicate_training_data):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(replicate_training_data, None, bool)
        self._parms['replicate_training_data'] = replicate_training_data

    @property
    def single_node_mode(self):
        if False:
            return 10
        '\n        Run on a single node for fine-tuning of model parameters.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(single_node_mode=True,\n        ...                                    seed=1234) \n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('single_node_mode')

    @single_node_mode.setter
    def single_node_mode(self, single_node_mode):
        if False:
            return 10
        assert_is_type(single_node_mode, None, bool)
        self._parms['single_node_mode'] = single_node_mode

    @property
    def shuffle_training_data(self):
        if False:
            i = 10
            return i + 15
        '\n        Enable shuffling of training data (recommended if training data is replicated and train_samples_per_iteration is\n        close to #nodes x #rows, of if using balance_classes).\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(shuffle_training_data=True,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('shuffle_training_data')

    @shuffle_training_data.setter
    def shuffle_training_data(self, shuffle_training_data):
        if False:
            while True:
                i = 10
        assert_is_type(shuffle_training_data, None, bool)
        self._parms['shuffle_training_data'] = shuffle_training_data

    @property
    def missing_values_handling(self):
        if False:
            while True:
                i = 10
        '\n        Handling of missing values. Either MeanImputation or Skip.\n\n        Type: ``Literal["mean_imputation", "skip"]``, defaults to ``"mean_imputation"``.\n\n        :examples:\n\n        >>> boston = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/BostonHousing.csv")\n        >>> predictors = boston.columns[:-1]\n        >>> response = "medv"\n        >>> boston[\'chas\'] = boston[\'chas\'].asfactor()\n        >>> boston.insert_missing_values()\n        >>> train, valid = boston.split_frame(ratios=[.8])\n        >>> boston_dl = H2ODeepLearningEstimator(missing_values_handling="skip")\n        >>> boston_dl.train(x=predictors,\n        ...                 y=response,\n        ...                 training_frame=train,\n        ...                 validation_frame=valid)\n        >>> boston_dl.mse()\n        '
        return self._parms.get('missing_values_handling')

    @missing_values_handling.setter
    def missing_values_handling(self, missing_values_handling):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(missing_values_handling, None, Enum('mean_imputation', 'skip'))
        self._parms['missing_values_handling'] = missing_values_handling

    @property
    def quiet_mode(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enable quiet mode for less output to standard output.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")\n        >>> titanic[\'survived\'] = titanic[\'survived\'].asfactor()\n        >>> predictors = titanic.columns\n        >>> del predictors[1:3]\n        >>> response = \'survived\'\n        >>> train, valid = titanic.split_frame(ratios=[.8], seed=1234)\n        >>> titanic_dl = H2ODeepLearningEstimator(quiet_mode=True,\n        ...                                       seed=1234)\n        >>> titanic_dl.train(x=predictors,\n        ...                  y=response,\n        ...                  training_frame=train,\n        ...                  validation_frame=valid)\n        >>> titanic_dl.mse()\n        '
        return self._parms.get('quiet_mode')

    @quiet_mode.setter
    def quiet_mode(self, quiet_mode):
        if False:
            while True:
                i = 10
        assert_is_type(quiet_mode, None, bool)
        self._parms['quiet_mode'] = quiet_mode

    @property
    def autoencoder(self):
        if False:
            i = 10
            return i + 15
        '\n        Auto-Encoder.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> cars_dl = H2ODeepLearningEstimator(autoencoder=True)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('autoencoder')

    @autoencoder.setter
    def autoencoder(self, autoencoder):
        if False:
            print('Hello World!')
        assert_is_type(autoencoder, bool)
        self._parms['autoencoder'] = autoencoder
        self.supervised_learning = not autoencoder

    @property
    def sparse(self):
        if False:
            while True:
                i = 10
        '\n        Sparse data handling (more efficient for data with lots of 0 values).\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "economy_20mpg"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(sparse=True,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=cars)\n        >>> cars_dl.auc()\n        '
        return self._parms.get('sparse')

    @sparse.setter
    def sparse(self, sparse):
        if False:
            return 10
        assert_is_type(sparse, None, bool)
        self._parms['sparse'] = sparse

    @property
    def col_major(self):
        if False:
            return 10
        '\n        #DEPRECATED Use a column major weight matrix for input layer. Can speed up forward propagation, but might slow\n        down backpropagation.\n\n        Type: ``bool``, defaults to ``False``.\n        '
        return self._parms.get('col_major')

    @col_major.setter
    def col_major(self, col_major):
        if False:
            i = 10
            return i + 15
        assert_is_type(col_major, None, bool)
        self._parms['col_major'] = col_major

    @property
    def average_activation(self):
        if False:
            return 10
        '\n        Average activation for sparse auto-encoder. #Experimental\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> cars_dl = H2ODeepLearningEstimator(average_activation=1.5,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('average_activation')

    @average_activation.setter
    def average_activation(self, average_activation):
        if False:
            print('Hello World!')
        assert_is_type(average_activation, None, numeric)
        self._parms['average_activation'] = average_activation

    @property
    def sparsity_beta(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sparsity regularization. #Experimental\n\n        Type: ``float``, defaults to ``0.0``.\n\n        :examples:\n\n        >>> from h2o.estimators import H2OAutoEncoderEstimator\n        >>> resp = 784\n        >>> nfeatures = 20\n        >>> train = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/train.csv.gz")\n        >>> test = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/mnist/test.csv.gz")\n        >>> train[resp] = train[resp].asfactor()\n        >>> test[resp] = test[resp].asfactor()\n        >>> sid = train[0].runif(0)\n        >>> train_unsupervised = train[sid>=0.5]\n        >>> train_unsupervised.pop(resp)\n        >>> ae_model = H2OAutoEncoderEstimator(activation="Tanh",\n        ...                                    hidden=[nfeatures],\n        ...                                    epochs=1,\n        ...                                    ignore_const_cols=False,\n        ...                                    reproducible=True,\n        ...                                    sparsity_beta=0.5,\n        ...                                    seed=1234)\n        >>> ae_model.train(list(range(resp)),\n        ...                training_frame=train_unsupervised)\n        >>> ae_model.mse()\n        '
        return self._parms.get('sparsity_beta')

    @sparsity_beta.setter
    def sparsity_beta(self, sparsity_beta):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(sparsity_beta, None, numeric)
        self._parms['sparsity_beta'] = sparsity_beta

    @property
    def max_categorical_features(self):
        if False:
            print('Hello World!')
        '\n        Max. number of categorical features, enforced via hashing. #Experimental\n\n        Type: ``int``, defaults to ``2147483647``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> predictors = covtype.columns[0:54]\n        >>> response = \'C55\'\n        >>> train, valid = covtype.split_frame(ratios=[.8], seed=1234)\n        >>> cov_dl = H2ODeepLearningEstimator(balance_classes=True,\n        ...                                   max_categorical_features=2147483647,\n        ...                                   seed=1234)\n        >>> cov_dl.train(x=predictors,\n        ...              y=response,\n        ...              training_frame=train,\n        ...              validation_frame=valid)\n        >>> cov_dl.logloss()\n        '
        return self._parms.get('max_categorical_features')

    @max_categorical_features.setter
    def max_categorical_features(self, max_categorical_features):
        if False:
            i = 10
            return i + 15
        assert_is_type(max_categorical_features, None, int)
        self._parms['max_categorical_features'] = max_categorical_features

    @property
    def reproducible(self):
        if False:
            return 10
        '\n        Force reproducibility on small data (will be slow - only uses 1 thread).\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> airlines_dl = H2ODeepLearningEstimator(reproducible=True)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.auc()\n        '
        return self._parms.get('reproducible')

    @reproducible.setter
    def reproducible(self, reproducible):
        if False:
            while True:
                i = 10
        assert_is_type(reproducible, None, bool)
        self._parms['reproducible'] = reproducible

    @property
    def export_weights_and_biases(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether to export Neural Network weights and biases to H2O Frames.\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(export_weights_and_biases=True,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('export_weights_and_biases')

    @export_weights_and_biases.setter
    def export_weights_and_biases(self, export_weights_and_biases):
        if False:
            return 10
        assert_is_type(export_weights_and_biases, None, bool)
        self._parms['export_weights_and_biases'] = export_weights_and_biases

    @property
    def mini_batch_size(self):
        if False:
            i = 10
            return i + 15
        '\n        Mini-batch size (smaller leads to better fit, larger can speed up and generalize better).\n\n        Type: ``int``, defaults to ``1``.\n\n        :examples:\n\n        >>> covtype = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/covtype/covtype.20k.data")\n        >>> covtype[54] = covtype[54].asfactor()\n        >>> predictors = covtype.columns[0:54]\n        >>> response = \'C55\'\n        >>> train, valid = covtype.split_frame(ratios=[.8], seed=1234)\n        >>> cov_dl = H2ODeepLearningEstimator(activation="RectifierWithDropout",\n        ...                                   hidden=[10,10],\n        ...                                   epochs=10,\n        ...                                   input_dropout_ratio=0.2,\n        ...                                   l1=1e-5,\n        ...                                   max_w2=10.5,\n        ...                                   stopping_rounds=0)\n        ...                                   mini_batch_size=35\n        >>> cov_dl.train(x=predictors,\n        ...              y=response,\n        ...              training_frame=train,\n        ...              validation_frame=valid)\n        >>> cov_dl.mse()\n        '
        return self._parms.get('mini_batch_size')

    @mini_batch_size.setter
    def mini_batch_size(self, mini_batch_size):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(mini_batch_size, None, int)
        self._parms['mini_batch_size'] = mini_batch_size

    @property
    def categorical_encoding(self):
        if False:
            print('Hello World!')
        '\n        Encoding scheme for categorical features\n\n        Type: ``Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",\n        "sort_by_response", "enum_limited"]``, defaults to ``"auto"``.\n\n        :examples:\n\n        >>> airlines= h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")\n        >>> airlines["Year"]= airlines["Year"].asfactor()\n        >>> airlines["Month"]= airlines["Month"].asfactor()\n        >>> airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()\n        >>> airlines["Cancelled"] = airlines["Cancelled"].asfactor()\n        >>> airlines[\'FlightNum\'] = airlines[\'FlightNum\'].asfactor()\n        >>> predictors = ["Origin", "Dest", "Year", "UniqueCarrier",\n        ...               "DayOfWeek", "Month", "Distance", "FlightNum"]\n        >>> response = "IsDepDelayed"\n        >>> train, valid= airlines.split_frame(ratios=[.8], seed=1234)\n        >>> encoding = "one_hot_internal"\n        >>> airlines_dl = H2ODeepLearningEstimator(categorical_encoding=encoding,\n        ...                                        seed=1234)\n        >>> airlines_dl.train(x=predictors,\n        ...                   y=response,\n        ...                   training_frame=train,\n        ...                   validation_frame=valid)\n        >>> airlines_dl.mse()\n        '
        return self._parms.get('categorical_encoding')

    @categorical_encoding.setter
    def categorical_encoding(self, categorical_encoding):
        if False:
            return 10
        assert_is_type(categorical_encoding, None, Enum('auto', 'enum', 'one_hot_internal', 'one_hot_explicit', 'binary', 'eigen', 'label_encoder', 'sort_by_response', 'enum_limited'))
        self._parms['categorical_encoding'] = categorical_encoding

    @property
    def elastic_averaging(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Elastic averaging between compute nodes can improve distributed model convergence. #Experimental\n\n        Type: ``bool``, defaults to ``False``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(elastic_averaging=True,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('elastic_averaging')

    @elastic_averaging.setter
    def elastic_averaging(self, elastic_averaging):
        if False:
            while True:
                i = 10
        assert_is_type(elastic_averaging, None, bool)
        self._parms['elastic_averaging'] = elastic_averaging

    @property
    def elastic_averaging_moving_rate(self):
        if False:
            return 10
        '\n        Elastic averaging moving rate (only if elastic averaging is enabled).\n\n        Type: ``float``, defaults to ``0.9``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(elastic_averaging_moving_rate=.8,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('elastic_averaging_moving_rate')

    @elastic_averaging_moving_rate.setter
    def elastic_averaging_moving_rate(self, elastic_averaging_moving_rate):
        if False:
            return 10
        assert_is_type(elastic_averaging_moving_rate, None, numeric)
        self._parms['elastic_averaging_moving_rate'] = elastic_averaging_moving_rate

    @property
    def elastic_averaging_regularization(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Elastic averaging regularization strength (only if elastic averaging is enabled).\n\n        Type: ``float``, defaults to ``0.001``.\n\n        :examples:\n\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> cars_dl = H2ODeepLearningEstimator(elastic_averaging_regularization=.008,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> cars_dl.mse()\n        '
        return self._parms.get('elastic_averaging_regularization')

    @elastic_averaging_regularization.setter
    def elastic_averaging_regularization(self, elastic_averaging_regularization):
        if False:
            i = 10
            return i + 15
        assert_is_type(elastic_averaging_regularization, None, numeric)
        self._parms['elastic_averaging_regularization'] = elastic_averaging_regularization

    @property
    def export_checkpoints_dir(self):
        if False:
            i = 10
            return i + 15
        '\n        Automatically export generated models to this directory.\n\n        Type: ``str``.\n\n        :examples:\n\n        >>> import tempfile\n        >>> from os import listdir\n        >>> cars = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")\n        >>> cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()\n        >>> predictors = ["displacement","power","weight","acceleration","year"]\n        >>> response = "cylinders"\n        >>> train, valid = cars.split_frame(ratios=[.8], seed=1234)\n        >>> checkpoints_dir = tempfile.mkdtemp()\n        >>> cars_dl = H2ODeepLearningEstimator(export_checkpoints_dir=checkpoints_dir,\n        ...                                    seed=1234)\n        >>> cars_dl.train(x=predictors,\n        ...               y=response,\n        ...               training_frame=train,\n        ...               validation_frame=valid)\n        >>> len(listdir(checkpoints_dir))\n        '
        return self._parms.get('export_checkpoints_dir')

    @export_checkpoints_dir.setter
    def export_checkpoints_dir(self, export_checkpoints_dir):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(export_checkpoints_dir, None, str)
        self._parms['export_checkpoints_dir'] = export_checkpoints_dir

    @property
    def auc_type(self):
        if False:
            i = 10
            return i + 15
        '\n        Set default multinomial AUC type.\n\n        Type: ``Literal["auto", "none", "macro_ovr", "weighted_ovr", "macro_ovo", "weighted_ovo"]``, defaults to\n        ``"auto"``.\n        '
        return self._parms.get('auc_type')

    @auc_type.setter
    def auc_type(self, auc_type):
        if False:
            i = 10
            return i + 15
        assert_is_type(auc_type, None, Enum('auto', 'none', 'macro_ovr', 'weighted_ovr', 'macro_ovo', 'weighted_ovo'))
        self._parms['auc_type'] = auc_type

    @property
    def custom_metric_func(self):
        if False:
            i = 10
            return i + 15
        '\n        Reference to custom evaluation function, format: `language:keyName=funcName`\n\n        Type: ``str``.\n        '
        return self._parms.get('custom_metric_func')

    @custom_metric_func.setter
    def custom_metric_func(self, custom_metric_func):
        if False:
            for i in range(10):
                print('nop')
        assert_is_type(custom_metric_func, None, str)
        self._parms['custom_metric_func'] = custom_metric_func

class H2OAutoEncoderEstimator(H2ODeepLearningEstimator):
    """
    :examples:

    >>> import h2o as ml
    >>> from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
    >>> ml.init()
    >>> rows = [[1,2,3,4,0]*50, [2,1,2,4,1]*50, [2,1,4,2,1]*50, [0,1,2,34,1]*50, [2,3,4,1,0]*50]
    >>> fr = ml.H2OFrame(rows)
    >>> fr[4] = fr[4].asfactor()
    >>> model = H2OAutoEncoderEstimator()
    >>> model.train(x=list(range(4)), training_frame=fr)
    """
    supervised_learning = False

    def __init__(self, **kwargs):
        if False:
            return 10
        super(H2OAutoEncoderEstimator, self).__init__(**kwargs)
        self.autoencoder = True