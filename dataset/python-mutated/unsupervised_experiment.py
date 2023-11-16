import datetime
import gc
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from joblib.memory import Memory
from sklearn.base import clone
from pycaret.containers.metrics.clustering import get_all_metric_containers
from pycaret.containers.models.clustering import ALL_ALLOWED_ENGINES, get_container_default_engines
from pycaret.internal.display import CommonDisplay
from pycaret.internal.logging import get_logger, redirect_output
from pycaret.internal.pipeline import Pipeline as InternalPipeline
from pycaret.internal.pipeline import estimator_pipeline, get_pipeline_fit_kwargs
from pycaret.internal.preprocess.preprocessor import Preprocessor
from pycaret.internal.pycaret_experiment.tabular_experiment import _TabularExperiment
from pycaret.internal.validation import is_sklearn_pipeline
from pycaret.loggers.base_logger import BaseLogger
from pycaret.utils.constants import DATAFRAME_LIKE, SEQUENCE_LIKE
from pycaret.utils.generic import MLUsecase, highlight_setup
LOGGER = get_logger()

class _UnsupervisedExperiment(_TabularExperiment, Preprocessor):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._variable_keys = self._variable_keys.union({'X'})
        return

    def _calculate_metrics(self, X, labels, ground_truth=None, ml_usecase=None) -> dict:
        if False:
            print('Hello World!')
        '\n        Calculate all metrics in _all_metrics.\n        '
        from pycaret.utils.generic import calculate_unsupervised_metrics
        if ml_usecase is None:
            ml_usecase = self._ml_usecase
        try:
            return calculate_unsupervised_metrics(metrics=self._all_metrics, X=X, labels=labels, ground_truth=ground_truth)
        except Exception:
            if ml_usecase == MLUsecase.CLUSTERING:
                metrics = get_all_metric_containers(self.variables, True)
            return calculate_unsupervised_metrics(metrics=metrics, X=X, labels=labels, ground_truth=ground_truth)

    def _is_unsupervised(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    def _set_up_logging(self, runtime, log_data, log_profile, experiment_custom_tags=None):
        if False:
            print('Hello World!')
        if self.logging_param:
            self.logging_param.log_experiment(self, log_profile, log_data, experiment_custom_tags, runtime)

    def setup(self, data: Optional[DATAFRAME_LIKE]=None, data_func: Optional[Callable[[], Union[pd.Series, pd.DataFrame]]]=None, index: Union[bool, int, str, SEQUENCE_LIKE]=True, ordinal_features: Optional[Dict[str, list]]=None, numeric_features: Optional[List[str]]=None, categorical_features: Optional[List[str]]=None, date_features: Optional[List[str]]=None, text_features: Optional[List[str]]=None, ignore_features: Optional[List[str]]=None, keep_features: Optional[List[str]]=None, preprocess: bool=True, create_date_columns: List[str]=['day', 'month', 'year'], imputation_type: Optional[str]='simple', numeric_imputation: str='mean', categorical_imputation: str='mode', text_features_method: str='tf-idf', max_encoding_ohe: int=-1, encoding_method: Optional[Any]=None, rare_to_value: Optional[float]=None, rare_value: str='rare', polynomial_features: bool=False, polynomial_degree: int=2, low_variance_threshold: Optional[float]=None, group_features: Optional[dict]=None, drop_groups: bool=False, remove_multicollinearity: bool=False, multicollinearity_threshold: float=0.9, bin_numeric_features: Optional[List[str]]=None, remove_outliers: bool=False, outliers_method: str='iforest', outliers_threshold: float=0.05, transformation: bool=False, transformation_method: str='yeo-johnson', normalize: bool=False, normalize_method: str='zscore', pca: bool=False, pca_method: str='linear', pca_components: Optional[Union[int, float, str]]=None, custom_pipeline: Optional[Any]=None, custom_pipeline_position: int=-1, n_jobs: Optional[int]=-1, use_gpu: bool=False, html: bool=True, session_id: Optional[int]=None, system_log: Union[bool, str, logging.Logger]=True, log_experiment: Union[bool, str, BaseLogger, List[Union[str, BaseLogger]]]=False, experiment_name: Optional[str]=None, experiment_custom_tags: Optional[Dict[str, Any]]=None, log_plots: Union[bool, list]=False, log_profile: bool=False, log_data: bool=False, verbose: bool=True, memory: Union[bool, str, Memory]=True, profile: bool=False, profile_kwargs: Optional[Dict[str, Any]]=None, engines: Optional[Dict[str, str]]=None):
        if False:
            return 10
        '\n\n        This function initializes the training environment and creates the transformation\n        pipeline. Setup function must be called before executing any other function. It\n        takes one mandatory parameter: ``data``. All the other parameters are optional.\n\n\n        Example\n        -------\n        >>> from pycaret.datasets import get_data\n        >>> jewellery = get_data(\'jewellery\')\n        >>> from pycaret.clustering import *\n        >>> exp_name = setup(data = jewellery)\n\n\n        data: dataframe-like\n            Data set with shape (n_samples, n_features), where n_samples is the\n            number of samples and n_features is the number of features. If data\n            is not a pandas dataframe, it\'s converted to one using default column\n            names.\n\n\n        data_func: Callable[[], DATAFRAME_LIKE] = None\n            The function that generate ``data`` (the dataframe-like input). This\n            is useful when the dataset is large, and you need parallel operations\n            such as ``compare_models``. It can avoid broadcasting large dataset\n            from driver to workers. Notice one and only one of ``data`` and\n            ``data_func`` must be set.\n\n        index: bool, int, str or sequence, default = True\n            Handle indices in the `data` dataframe.\n                - If False: Reset to RangeIndex.\n                - If True: Keep the provided index.\n                - If int: Position of the column to use as index.\n                - If str: Name of the column to use as index.\n                - If sequence: Array with shape=(n_samples,) to use as index.\n\n\n        ordinal_features: dict, default = None\n            Categorical features to be encoded ordinally. For example, a categorical\n            feature with \'low\', \'medium\', \'high\' values where low < medium < high can\n            be passed as ordinal_features = {\'column_name\' : [\'low\', \'medium\', \'high\']}.\n\n\n        numeric_features: list of str, default = None\n            If the inferred data types are not correct, the numeric_features param can\n            be used to define the data types. It takes a list of strings with column\n            names that are numeric.\n\n\n        categorical_features: list of str, default = None\n            If the inferred data types are not correct, the categorical_features param\n            can be used to define the data types. It takes a list of strings with column\n            names that are categorical.\n\n\n        date_features: list of str, default = None\n            If the inferred data types are not correct, the date_features param can be\n            used to overwrite the data types. It takes a list of strings with column\n            names that are DateTime.\n\n\n        text_features: list of str, default = None\n            Column names that contain a text corpus. If None, no text features are\n            selected.\n\n\n        ignore_features: list of str, default = None\n            ignore_features param can be used to ignore features during preprocessing\n            and model training. It takes a list of strings with column names that are\n            to be ignored.\n\n\n        keep_features: list of str, default = None\n            keep_features param can be used to always keep specific features during\n            preprocessing, i.e. these features are never dropped by any kind of\n            feature selection. It takes a list of strings with column names that are\n            to be kept.\n\n\n        preprocess: bool, default = True\n            When set to False, no transformations are applied except for train_test_split\n            and custom transformations passed in ``custom_pipeline`` param. Data must be\n            ready for modeling (no missing values, no dates, categorical data encoding),\n            when preprocess is set to False.\n\n\n        create_date_columns: list of str, default = ["day", "month", "year"]\n            Columns to create from the date features. Note that created features\n            with zero variance (e.g. the feature hour in a column that only contains\n            dates) are ignored. Allowed values are datetime attributes from\n            `pandas.Series.dt`. The datetime format of the feature is inferred\n            automatically from the first non NaN value.\n\n\n        imputation_type: str or None, default = \'simple\'\n            The type of imputation to use. Unsupervised learning only supports\n            \'imputation_type=simple\'. If None, no imputation of missing values\n            is performed.\n\n\n        numeric_imputation: str, default = \'mean\'\n            Missing values in numeric features are imputed with \'mean\' value of the feature\n            in the training dataset. The other available option is \'median\' or \'zero\'.\n\n\n        categorical_imputation: str, default = \'constant\'\n            Missing values in categorical features are imputed with a constant \'not_available\'\n            value. The other available option is \'mode\'.\n\n\n        text_features_method: str, default = "tf-idf"\n            Method with which to embed the text features in the dataset. Choose\n            between "bow" (Bag of Words - CountVectorizer) or "tf-idf" (TfidfVectorizer).\n            Be aware that the sparse matrix output of the transformer is converted\n            internally to its full array. This can cause memory issues for large\n            text embeddings.\n\n\n        max_encoding_ohe: int, default = -1\n            Categorical columns with `max_encoding_ohe` or less unique values are\n            encoded using OneHotEncoding. If more, the `encoding_method` estimator\n            is used. Note that columns with exactly two classes are always encoded\n            ordinally. Set to below 0 to always use OneHotEncoding.\n\n\n        encoding_method: category-encoders estimator, default = None\n            A `category-encoders` estimator to encode the categorical columns\n            with more than `max_encoding_ohe` unique values. If None,\n            `category_encoders.basen.BaseN` is used.\n\n\n        rare_to_value: float or None, default=None\n            Minimum fraction of category occurrences in a categorical column.\n            If a category is less frequent than `rare_to_value * len(X)`, it is\n            replaced with the string in `rare_value`. Use this parameter to group\n            rare categories before encoding the column. If None, ignores this step.\n\n\n        rare_value: str, default="rare"\n            Value with which to replace rare categories. Ignored when\n            ``rare_to_value`` is None.\n\n\n        polynomial_features: bool, default = False\n            When set to True, new features are derived using existing numeric features.\n\n\n        polynomial_degree: int, default = 2\n            Degree of polynomial features. For example, if an input sample is two dimensional\n            and of the form [a, b], the polynomial features with degree = 2 are:\n            [1, a, b, a^2, ab, b^2]. Ignored when ``polynomial_features`` is not True.\n\n\n        low_variance_threshold: float or None, default = None\n            Remove features with a training-set variance lower than the provided\n            threshold. If 0, keep all features with non-zero variance, i.e. remove\n            the features that have the same value in all samples. If None, skip\n            this transformation step.\n\n\n        group_features: dict or None, default = None\n            When the dataset contains features with related characteristics,\n            add new fetaures with the following statistical properties of that\n            group: min, max, mean, std, median and mode. The parameter takes a\n            dict with the group name as key and a list of feature names\n            belonging to that group as value.\n\n\n        drop_groups: bool, default=False\n            Whether to drop the original features in the group. Ignored when\n            ``group_features`` is None.\n\n\n        remove_multicollinearity: bool, default = False\n            When set to True, features with the inter-correlations higher than\n            the defined threshold are removed. For each group, it removes all\n            except the first feature.\n\n\n        multicollinearity_threshold: float, default = 0.9\n            Minimum absolute Pearson correlation to identify correlated\n            features. The default value removes equal columns. Ignored when\n            ``remove_multicollinearity`` is not True.\n\n\n        bin_numeric_features: list of str, default = None\n            To convert numeric features into categorical, bin_numeric_features parameter can\n            be used. It takes a list of strings with column names to be discretized. It does\n            so by using \'sturges\' rule to determine the number of clusters and then apply\n            KMeans algorithm. Original values of the feature are then replaced by the\n            cluster label.\n\n\n        remove_outliers: bool, default = False\n            When set to True, outliers from the training data are removed using an\n            Isolation Forest.\n\n\n        outliers_method: str, default = "iforest"\n            Method with which to remove outliers. Possible values are:\n                - \'iforest\': Uses sklearn\'s IsolationForest.\n                - \'ee\': Uses sklearn\'s EllipticEnvelope.\n                - \'lof\': Uses sklearn\'s LocalOutlierFactor.\n\n\n        outliers_threshold: float, default = 0.05\n            The percentage outliers to be removed from the dataset. Ignored\n            when ``remove_outliers=False``.\n\n\n        transformation: bool, default = False\n            When set to True, it applies the power transform to make data more Gaussian-like.\n            Type of transformation is defined by the ``transformation_method`` parameter.\n\n\n        transformation_method: str, default = \'yeo-johnson\'\n            Defines the method for transformation. By default, the transformation method is\n            set to \'yeo-johnson\'. The other available option for transformation is \'quantile\'.\n            Ignored when ``transformation`` is not True.\n\n\n        normalize: bool, default = False\n            When set to True, it transforms the features by scaling them to a given\n            range. Type of scaling is defined by the ``normalize_method`` parameter.\n\n\n        normalize_method: str, default = \'zscore\'\n            Defines the method for scaling. By default, normalize method is set to \'zscore\'\n            The standard zscore is calculated as z = (x - u) / s. Ignored when ``normalize``\n            is not True. The other options are:\n\n            - minmax: scales and translates each feature individually such that it is in\n            the range of 0 - 1.\n            - maxabs: scales and translates each feature individually such that the\n            maximal absolute value of each feature will be 1.0. It does not\n            shift/center the data, and thus does not destroy any sparsity.\n            - robust: scales and translates each feature according to the Interquartile\n            range. When the dataset contains outliers, robust scaler often gives\n            better results.\n\n\n        pca: bool, default = False\n            When set to True, dimensionality reduction is applied to project the data into\n            a lower dimensional space using the method defined in ``pca_method`` parameter.\n\n\n        pca_method: str, default = \'linear\'\n            Method with which to apply PCA. Possible values are:\n                - \'linear\': Uses Singular Value  Decomposition.\n                - \'kernel\': Dimensionality reduction through the use of RBF kernel.\n                - \'incremental\': Similar to \'linear\', but more efficient for large datasets.\n\n\n        pca_components: int, float, str or None, default = None\n            Number of components to keep. This parameter is ignored when `pca=False`.\n                - If None: All components are kept.\n                - If int: Absolute number of components.\n                - If float: Such an amount that the variance that needs to be explained\n                            is greater than the percentage specified by `n_components`.\n                            Value should lie between 0 and 1 (ony for pca_method=\'linear\').\n                - If "mle": Minka’s MLE is used to guess the dimension (ony for pca_method=\'linear\').\n\n\n        custom_pipeline: list of (str, transformer), dict or Pipeline, default = None\n            Addidiotnal custom transformers. If passed, they are applied to the\n            pipeline last, after all the build-in transformers.\n\n\n        custom_pipeline_position: int, default = -1\n            Position of the custom pipeline in the overal preprocessing pipeline.\n            The default value adds the custom pipeline last.\n\n\n        n_jobs: int, default = -1\n            The number of jobs to run in parallel (for functions that supports parallel\n            processing) -1 means using all processors. To run all functions on single\n            processor set n_jobs to None.\n\n\n        use_gpu: bool or str, default = False\n            When set to True, it will use GPU for training with algorithms that support it,\n            and fall back to CPU if they are unavailable. When set to \'force\', it will only\n            use GPU-enabled algorithms and raise exceptions when they are unavailable. When\n            False, all algorithms are trained using CPU only.\n\n            GPU enabled algorithms:\n\n            - None at this moment.\n\n\n        html: bool, default = True\n            When set to False, prevents runtime display of monitor. This must be set to False\n            when the environment does not support IPython. For example, command line terminal,\n            Databricks Notebook, Spyder and other similar IDEs.\n\n\n        session_id: int, default = None\n            Controls the randomness of experiment. It is equivalent to \'random_state\' in\n            scikit-learn. When None, a pseudo random number is generated. This can be used\n            for later reproducibility of the entire experiment.\n\n\n        system_log: bool or str or logging.Logger, default = True\n            Whether to save the system logging file (as logs.log). If the input\n            is a string, use that as the path to the logging file. If the input\n            already is a logger object, use that one instead.\n\n\n        log_experiment: bool, default = False\n            A (list of) PyCaret ``BaseLogger`` or str (one of \'mlflow\', \'wandb\', \'comet_ml\')\n            corresponding to a logger to determine which experiment loggers to use.\n            Setting to True will use just MLFlow.\n            If ``wandb`` (Weights & Biases) or ``comet_ml``is installed, will also log there.\n\n\n        experiment_name: str, default = None\n            Name of the experiment for logging. Ignored when ``log_experiment`` is False.\n\n\n        experiment_custom_tags: dict, default = None\n            Dictionary of tag_name: String -> value: (String, but will be string-ified\n            if not) passed to the mlflow.set_tags to add new custom tags for the experiment.\n\n\n        log_plots: bool or list, default = False\n            When set to True, certain plots are logged automatically in the ``MLFlow`` server.\n            To change the type of plots to be logged, pass a list containing plot IDs. Refer\n            to documentation of ``plot_model``. Ignored when ``log_experiment`` is False.\n\n\n        log_profile: bool, default = False\n            When set to True, data profile is logged on the ``MLflow`` server as a html file.\n            Ignored when ``log_experiment`` is False.\n\n\n        log_data: bool, default = False\n            When set to True, dataset is logged on the ``MLflow`` server as a csv file.\n            Ignored when ``log_experiment`` is False.\n\n\n        verbose: bool, default = True\n            When set to False, Information grid is not printed.\n\n\n        memory: str, bool or Memory, default=True\n            Used to cache the fitted transformers of the pipeline.\n                If False: No caching is performed.\n                If True: A default temp directory is used.\n                If str: Path to the caching directory.\n\n\n        profile: bool, default = False\n            When set to True, an interactive EDA report is displayed.\n\n\n        profile_kwargs: dict, default = {} (empty dict)\n            Dictionary of arguments passed to the ProfileReport method used\n            to create the EDA report. Ignored if ``profile`` is False.\n\n\n        Returns:\n            Global variables that can be changed using the ``set_config`` function.\n\n        '
        self._register_setup_params(dict(locals()))
        if data is None and data_func is None or (data is not None and data_func is not None):
            raise ValueError('One and only one of data and data_func must be set')
        runtime_start = time.time()
        self.all_allowed_engines = ALL_ALLOWED_ENGINES
        self._initialize_setup(n_jobs=n_jobs, use_gpu=use_gpu, html=html, session_id=session_id, system_log=system_log, log_experiment=log_experiment, experiment_name=experiment_name, memory=memory, verbose=verbose)
        self.log_plots_param = log_plots
        if self.log_plots_param is True:
            self.log_plots_param = self._get_default_plots_to_log()
        elif isinstance(self.log_plots_param, list):
            for i in self.log_plots_param:
                if i not in self._available_plots:
                    raise ValueError(f"Invalid value for log_plots '{i}'. Possible values are: {', '.join(self._available_plots.keys())}.")
        if data_func is not None:
            data = data_func()
        self.index = index
        self.data = self._set_index(self._prepare_dataset(data))
        self.idx = [self.data.index, None]
        self._prepare_column_types(ordinal_features=ordinal_features, numeric_features=numeric_features, categorical_features=categorical_features, date_features=date_features, text_features=text_features, ignore_features=ignore_features, keep_features=keep_features)
        self._set_exp_model_engines(container_default_engines=get_container_default_engines(), engine=engines)
        self.pipeline = InternalPipeline(steps=[('placeholder', None)], memory=self.memory)
        if preprocess:
            self.logger.info('Preparing preprocessing pipeline...')
            if self._fxs['Date']:
                self._date_feature_engineering(create_date_columns)
            if imputation_type == 'simple':
                self._simple_imputation(numeric_imputation, categorical_imputation)
            elif imputation_type is not None:
                raise ValueError(f'Invalid value for the imputation_type parameter, got {imputation_type}. Possible values are: simple.')
            if self._fxs['Text']:
                self._text_embedding(text_features_method)
            if self._fxs['Ordinal'] or self._fxs['Categorical']:
                self._encoding(max_encoding_ohe=max_encoding_ohe, encoding_method=encoding_method, rare_to_value=rare_to_value, rare_value=rare_value)
            if polynomial_features:
                self._polynomial_features(polynomial_degree)
            if low_variance_threshold is not None:
                self._low_variance(low_variance_threshold)
            if group_features:
                self._group_features(group_features, drop_groups)
            if remove_multicollinearity:
                self._remove_multicollinearity(multicollinearity_threshold)
            if bin_numeric_features:
                self._bin_numerical_features(bin_numeric_features)
            if remove_outliers:
                self._remove_outliers(outliers_method, outliers_threshold)
            if transformation:
                self._transformation(transformation_method)
            if normalize:
                self._normalization(normalize_method)
            if pca:
                self._pca(pca_method, pca_components)
        if custom_pipeline:
            self._add_custom_pipeline(custom_pipeline, custom_pipeline_position)
        if ('placeholder', None) in self.pipeline.steps and len(self.pipeline) > 1:
            self.pipeline.steps.remove(('placeholder', None))
        self.pipeline.fit(self.X)
        self.logger.info('Finished creating preprocessing pipeline.')
        self.logger.info(f'Pipeline: {self.pipeline}')
        self.logger.info('Creating final display dataframe.')
        container = []
        container.append(['Session id', self.seed])
        container.append(['Original data shape', self.data.shape])
        container.append(['Transformed data shape', self.dataset_transformed.shape])
        for (fx, cols) in self._fxs.items():
            if len(cols) > 0:
                container.append([f'{fx} features', len(cols)])
        if self.data.isna().sum().sum():
            n_nans = 100 * self.data.isna().any(axis=1).sum() / len(self.data)
            container.append(['Rows with missing values', f'{round(n_nans, 1)}%'])
        if preprocess:
            container.append(['Preprocess', preprocess])
            container.append(['Imputation type', imputation_type])
            if imputation_type == 'simple':
                container.append(['Numeric imputation', numeric_imputation])
                container.append(['Categorical imputation', categorical_imputation])
            if self._fxs['Text']:
                container.append(['Text features embedding method', text_features_method])
            if self._fxs['Categorical']:
                container.append(['Maximum one-hot encoding', max_encoding_ohe])
                container.append(['Encoding method', encoding_method])
            if polynomial_features:
                container.append(['Polynomial features', polynomial_features])
                container.append(['Polynomial degree', polynomial_degree])
            if low_variance_threshold is not None:
                container.append(['Low variance threshold', low_variance_threshold])
            if remove_multicollinearity:
                container.append(['Remove multicollinearity', remove_multicollinearity])
                container.append(['Multicollinearity threshold', multicollinearity_threshold])
            if remove_outliers:
                container.append(['Remove outliers', remove_outliers])
                container.append(['Outliers threshold', outliers_threshold])
            if transformation:
                container.append(['Transformation', transformation])
                container.append(['Transformation method', transformation_method])
            if normalize:
                container.append(['Normalize', normalize])
                container.append(['Normalize method', normalize_method])
            if pca:
                container.append(['PCA', pca])
                container.append(['PCA method', pca_method])
                container.append(['PCA components', pca_components])
            if custom_pipeline:
                container.append(['Custom pipeline', 'Yes'])
            container.append(['CPU Jobs', self.n_jobs_param])
            container.append(['Use GPU', self.gpu_param])
            container.append(['Log Experiment', self.logging_param])
            container.append(['Experiment Name', self.exp_name_log])
            container.append(['USI', self.USI])
        self._display_container = [pd.DataFrame(container, columns=['Description', 'Value'])]
        self.logger.info(f'Setup _display_container: {self._display_container[0]}')
        display = CommonDisplay(verbose=self.verbose, html_param=self.html_param)
        if self.verbose:
            pd.set_option('display.max_rows', 100)
            display.display(self._display_container[0].style.apply(highlight_setup))
            pd.reset_option('display.max_rows')
        self._profile(profile, profile_kwargs)
        (self._all_models, self._all_models_internal) = self._get_models()
        self._all_metrics = self._get_metrics()
        runtime = np.array(time.time() - runtime_start).round(2)
        self._set_up_logging(runtime, log_data, log_profile, experiment_custom_tags=experiment_custom_tags)
        self._setup_ran = True
        self.logger.info(f'setup() successfully completed in {runtime}s...............')
        return self

    def assign_model(self, model, transformation: bool=False, score: bool=True, verbose: bool=True) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        "\n        This function assigns cluster labels to the dataset for a given model.\n\n\n        Example\n        -------\n        >>> from pycaret.datasets import get_data\n        >>> jewellery = get_data('jewellery')\n        >>> from pycaret.clustering import *\n        >>> exp_name = setup(data = jewellery)\n        >>> kmeans = create_model('kmeans')\n        >>> kmeans_df = assign_model(kmeans)\n\n\n\n        model: scikit-learn compatible object\n            Trained model object\n\n\n        transformation: bool, default = False\n            Whether to apply cluster labels on the transformed dataset.\n\n\n        verbose: bool, default = True\n            Status update is not printed when verbose is set to False.\n\n\n        Returns:\n            pandas.DataFrame\n\n        "
        function_params_str = ', '.join([f'{k}={v}' for (k, v) in locals().items()])
        self.logger.info('Initializing assign_model()')
        self.logger.info(f'assign_model({function_params_str})')
        self.logger.info('Checking exceptions')
        if type(transformation) is not bool:
            raise TypeError('Transformation parameter can only take argument as True or False.')
        if type(verbose) is not bool:
            raise TypeError('Verbose parameter can only take argument as True or False.')
        '\n        error handling ends here\n        '
        if is_sklearn_pipeline(model):
            model = model.steps[-1][1]
        self.logger.info('Determining Trained Model')
        name = self._get_model_name(model)
        self.logger.info(f'Trained Model : {name}')
        self.logger.info('Copying data')
        if transformation:
            data = self.X_transformed.copy()
            self.logger.info('Transformation parameter set to True. Assigned clusters are attached on transformed dataset.')
        else:
            data = self.X.copy()
        if self._ml_usecase == MLUsecase.CLUSTERING:
            labels = [f'Cluster {i}' for i in model.labels_]
            data['Cluster'] = labels
        else:
            data['Anomaly'] = model.labels_
            if score:
                data['Anomaly_Score'] = model.decision_scores_
        self.logger.info(data.shape)
        self.logger.info('assign_model() successfully completed......................................')
        return data

    def predict_model(self, estimator, data: pd.DataFrame, ml_usecase: Optional[MLUsecase]=None) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        function_params_str = ', '.join([f'{k}={v}' for (k, v) in locals().items() if k != 'data'])
        self.logger.info('Initializing predict_model()')
        self.logger.info(f'predict_model({function_params_str})')
        if ml_usecase is None:
            ml_usecase = self._ml_usecase
        if data is None:
            if estimator.__class__.__name__ == 'Pipeline':
                data = self.X
            else:
                data = self.X_transformed
        elif estimator.__class__.__name__ == 'Pipeline':
            data = self._prepare_dataset(data)
        else:
            data = self.pipeline.transform(data)
        if hasattr(estimator, 'feature_names_in_'):
            data = data[list(estimator.feature_names_in_)]
        if hasattr(estimator, 'predict'):
            pass
        else:
            raise TypeError("Model doesn't have the predict method.")
        output = data.copy()
        pred = estimator.predict(data)
        if ml_usecase == MLUsecase.CLUSTERING:
            output['Cluster'] = [f'Cluster {i}' for i in pred]
        else:
            output['Anomaly'] = pred
            output['Anomaly_Score'] = estimator.decision_function(data)
        return output

    def _create_model(self, estimator, num_clusters: int=4, fraction: float=0.05, ground_truth: Optional[str]=None, round: int=4, fit_kwargs: Optional[dict]=None, experiment_custom_tags: Optional[Dict[str, Any]]=None, verbose: bool=True, system: bool=True, add_to_model_list: bool=True, raise_num_clusters: bool=False, X_data: Optional[pd.DataFrame]=None, display: Optional[CommonDisplay]=None, **kwargs) -> Any:
        if False:
            while True:
                i = 10
        '\n        Internal version of ``create_model`` with private arguments.\n        '
        function_params_str = ', '.join([f'{k}={v}' for (k, v) in locals().items() if k not in 'X_data'])
        self.logger.info('Initializing create_model()')
        self.logger.info(f'create_model({function_params_str})')
        self.logger.info('Checking exceptions')
        runtime_start = time.time()
        available_estimators = set(self._all_models_internal.keys())
        if not fit_kwargs:
            fit_kwargs = {}
        if isinstance(estimator, str):
            if estimator not in available_estimators:
                raise ValueError(f'Estimator {estimator} not available. Please see docstring for list of available estimators.')
        elif not hasattr(estimator, 'fit'):
            raise ValueError(f'Estimator {estimator} does not have the required fit() method.')
        if type(round) is not int:
            raise TypeError('Round parameter only accepts integer value.')
        if type(verbose) is not bool:
            raise TypeError('Verbose parameter can only take argument as True or False.')
        if type(system) is not bool:
            raise TypeError('System parameter can only take argument as True or False.')
        if fraction <= 0 or fraction >= 1:
            raise TypeError('Fraction parameter can only take value as float between 0 to 1.')
        if num_clusters <= 1:
            raise TypeError('num_clusters parameter can only take value integer value greater than 1.')
        if ground_truth is not None:
            if ground_truth not in self.dataset.columns:
                raise ValueError(f"ground_truth {ground_truth} doesn't exist in the dataset.")
        '\n\n        ERROR HANDLING ENDS HERE\n\n        '
        if not display:
            progress_args = {'max': 3}
            timestampStr = datetime.datetime.now().strftime('%H:%M:%S')
            monitor_rows = [['Initiated', '. . . . . . . . . . . . . . . . . .', timestampStr], ['Status', '. . . . . . . . . . . . . . . . . .', 'Loading Dependencies'], ['Estimator', '. . . . . . . . . . . . . . . . . .', 'Compiling Library']]
            display = CommonDisplay(verbose=verbose, html_param=self.html_param, progress_args=progress_args, monitor_rows=monitor_rows)
        np.random.seed(self.seed)
        data_X = self.X if X_data is None else X_data
        transformed_data = self.X_transformed if X_data is None else self.pipeline.transform(X_data)
        '\n        MONITOR UPDATE STARTS\n        '
        display.update_monitor(1, 'Selecting Estimator')
        '\n        MONITOR UPDATE ENDS\n        '
        self.logger.info('Importing untrained model')
        is_cblof = False
        if isinstance(estimator, str) and estimator in available_estimators:
            is_cblof = estimator == 'cluster'
            model_definition = self._all_models_internal[estimator]
            model_args = model_definition.args
            model_args = {**model_args, **kwargs}
            model = model_definition.class_def(**model_args)
            full_name = model_definition.name
        else:
            self.logger.info('Declaring custom model')
            model = clone(estimator)
            model.set_params(**kwargs)
            full_name = self._get_model_name(model)
        display.update_monitor(2, full_name)
        if self._ml_usecase == MLUsecase.CLUSTERING:
            if raise_num_clusters:
                model.set_params(n_clusters=num_clusters)
            else:
                try:
                    model.set_params(n_clusters=num_clusters)
                except Exception:
                    pass
        else:
            model.set_params(contamination=fraction)
        try:
            model = clone(model)
        except Exception:
            self.logger.warning(f'create_model() for {model} raised an exception when cloning:')
            self.logger.warning(traceback.format_exc())
        self.logger.info(f'{full_name} Imported successfully')
        display.move_progress()
        '\n        MONITOR UPDATE STARTS\n        '
        if self._ml_usecase == MLUsecase.CLUSTERING:
            display.update_monitor(1, f'Fitting {num_clusters} Clusters')
        else:
            display.update_monitor(1, f'Fitting {fraction} Fraction')
        '\n        MONITOR UPDATE ENDS\n        '
        with estimator_pipeline(self.pipeline, model) as pipeline_with_model:
            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            self.logger.info('Fitting Model')
            model_fit_start = time.time()
            with redirect_output(self.logger):
                if is_cblof and 'n_clusters' not in kwargs:
                    try:
                        pipeline_with_model.fit(data_X, **fit_kwargs)
                    except Exception:
                        try:
                            pipeline_with_model.set_params(actual_estimator__n_clusters=12)
                            model_fit_start = time.time()
                            pipeline_with_model.fit(data_X, **fit_kwargs)
                        except Exception as e:
                            raise RuntimeError('Could not form valid cluster separation. Try a different dataset or model.') from e
                else:
                    pipeline_with_model.fit(data_X, **fit_kwargs)
            model_fit_end = time.time()
            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
        display.move_progress()
        if ground_truth is not None:
            self.logger.info(f'ground_truth parameter set to {ground_truth}')
            gt = np.array(self.dataset[ground_truth])
        else:
            gt = None
        if self._ml_usecase == MLUsecase.CLUSTERING:
            with redirect_output(self.logger):
                metrics = self._calculate_metrics(transformed_data, model.labels_, ground_truth=gt)
        else:
            metrics = {}
        self.logger.info(str(model))
        self.logger.info('create_models() successfully completed......................................')
        runtime = time.time() - runtime_start
        if self.logging_param and system:
            metrics_log = {k: v for (k, v) in metrics.items()}
            self._log_model(model=model, model_results=None, score_dict=metrics_log, source='create_model', runtime=runtime, model_fit_time=model_fit_time, pipeline=self.pipeline, log_plots=self.log_plots_param, experiment_custom_tags=experiment_custom_tags, display=display)
        display.move_progress()
        self.logger.info('Uploading results into container')
        if metrics:
            model_results = pd.DataFrame(metrics, index=[0])
            model_results = model_results.round(round)
            self._display_container.append(model_results)
        else:
            model_results = None
        if add_to_model_list:
            self.logger.info('Uploading model into container now')
            self._master_model_container.append({'model': model, 'scores': model_results, 'cv': None})
        if model_results is not None and system:
            display.display(model_results.style.format(precision=round))
        else:
            display.close()
        self.logger.info(f'_master_model_container: {len(self._master_model_container)}')
        self.logger.info(f'_display_container: {len(self._display_container)}')
        self.logger.info(str(model))
        self.logger.info('create_model() successfully completed......................................')
        gc.collect()
        if not system:
            return (model, model_fit_time)
        return model

    def create_model(self, estimator, num_clusters: int=4, fraction: float=0.05, ground_truth: Optional[str]=None, round: int=4, fit_kwargs: Optional[dict]=None, experiment_custom_tags: Optional[Dict[str, Any]]=None, engine: Optional[str]=None, verbose: bool=True, **kwargs) -> Any:
        if False:
            return 10
        '\n        This function trains and evaluates the performance of a given model.\n        Metrics evaluated can be accessed using the ``get_metrics`` function.\n        Custom metrics can be added or removed using the ``add_metric`` and\n        ``remove_metric`` function. All the available models can be accessed\n        using the ``models`` function.\n\n\n        Example\n        -------\n        >>> from pycaret.datasets import get_data\n        >>> jewellery = get_data(\'jewellery\')\n        >>> from pycaret.clustering import *\n        >>> exp_name = setup(data = jewellery)\n        >>> kmeans = create_model(\'kmeans\')\n\n\n        model: str or scikit-learn compatible object\n            ID of an model available in the model library or pass an untrained\n            model object consistent with scikit-learn API. Models available\n            in the model library (ID - Name):\n\n            * \'kmeans\' - K-Means Clustering\n            * \'ap\' - Affinity Propagation\n            * \'meanshift\' - Mean shift Clustering\n            * \'sc\' - Spectral Clustering\n            * \'hclust\' - Agglomerative Clustering\n            * \'dbscan\' - Density-Based Spatial Clustering\n            * \'optics\' - OPTICS Clustering\n            * \'birch\' - Birch Clustering\n            * \'kmodes\' - K-Modes Clustering\n\n\n        num_clusters: int, default = 4\n            The number of clusters to form.\n\n\n        ground_truth: str, default = None\n            ground_truth to be provided to evaluate metrics that require true labels.\n            When None, such metrics are returned as 0.0. All metrics evaluated can\n            be accessed using ``get_metrics`` function.\n\n\n        round: int, default = 4\n            Number of decimal places the metrics in the score grid will be rounded to.\n\n\n        fit_kwargs: dict, default = {} (empty dict)\n            Dictionary of arguments passed to the fit method of the model.\n\n\n        verbose: bool, default = True\n            Status update is not printed when verbose is set to False.\n\n\n        engine: Optional[str] = None\n            The execution engine to use for the model, e.g. for K-Means Clustering ("kmeans"), users can\n            switch between "sklearn" and "sklearnex" by specifying\n            `engine="sklearnex"`.\n\n\n        experiment_custom_tags: dict, default = None\n            Dictionary of tag_name: String -> value: (String, but will be string-ified\n            if not) passed to the mlflow.set_tags to add new custom tags for the experiment.\n\n\n        **kwargs:\n            Additional keyword arguments to pass to the estimator.\n\n\n        Returns:\n            Trained Model\n\n\n        Warnings\n        --------\n        - ``num_clusters`` param not required for Affinity Propagation (\'ap\'),\n        Mean shift (\'meanshift\'), Density-Based Spatial Clustering (\'dbscan\')\n        and OPTICS Clustering (\'optics\').\n\n        - When fit doesn\'t converge in Affinity Propagation (\'ap\') model, all\n        datapoints are labelled as -1.\n\n        - Noisy samples are given the label -1, when using Density-Based Spatial\n        (\'dbscan\') or OPTICS Clustering (\'optics\').\n\n        - OPTICS (\'optics\') clustering may take longer training times on large\n        datasets.\n\n\n        '
        assert not any((x in ('system', 'add_to_model_list', 'raise_num_clusters', 'X_data', 'metrics') for x in kwargs))
        if engine is not None:
            initial_default_model_engines = self.exp_model_engines.copy()
            self._set_engine(estimator=estimator, engine=engine, severity='error')
        try:
            return_values = self._create_model(estimator=estimator, num_clusters=num_clusters, fraction=fraction, ground_truth=ground_truth, round=round, fit_kwargs=fit_kwargs, experiment_custom_tags=experiment_custom_tags, verbose=verbose, **kwargs)
        finally:
            if engine is not None:
                self._set_exp_model_engines(container_default_engines=get_container_default_engines(), engine=initial_default_model_engines)
        return return_values

    def evaluate_model(self, estimator, fold: Optional[Union[int, Any]]=None, fit_kwargs: Optional[dict]=None, plot_kwargs: Optional[dict]=None, feature_name: Optional[str]=None, groups: Optional[Union[str, Any]]=None):
        if False:
            return 10
        "\n        This function displays a user interface for analyzing performance of a trained\n        model. It calls the ``plot_model`` function internally.\n\n        Example\n        --------\n        >>> from pycaret.datasets import get_data\n        >>> jewellery = get_data('jewellery')\n        >>> from pycaret.clustering import *\n        >>> exp_name = setup(data = jewellery)\n        >>> kmeans = create_model('kmeans')\n        >>> evaluate_model(kmeans)\n\n\n        model: scikit-learn compatible object\n            Trained model object\n\n\n        feature: str, default = None\n            Feature to be evaluated when plot = 'distribution'. When ``plot`` type is\n            'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or\n            label when the ``label`` param is set to True. When the ``plot`` type is\n            'cluster' or 'tsne' and feature is None, first column of the dataset is\n            used.\n\n\n        fit_kwargs: dict, default = {} (empty dict)\n            Dictionary of arguments passed to the fit method of the model.\n\n\n        Returns:\n            None\n\n\n        Warnings\n        --------\n        -   This function only works in IPython enabled Notebook.\n\n        "
        return super().evaluate_model(estimator, fold, fit_kwargs, plot_kwargs, feature_name, groups)