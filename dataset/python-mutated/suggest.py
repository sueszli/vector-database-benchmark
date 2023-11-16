import numpy as np
import logging
import pathlib
import json
from flaml.automl.data import DataTransformer
from flaml.automl.task.task import CLASSIFICATION, get_classification_objective
from flaml.automl.task.generic_task import len_labels
from flaml.automl.task.factory import task_factory
from flaml.version import __version__
try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    pass
LOCATION = pathlib.Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)
CONFIG_PREDICTORS = {}

def meta_feature(task, X_train, y_train, meta_feature_names):
    if False:
        print('Hello World!')
    this_feature = []
    n_row = X_train.shape[0]
    n_feat = X_train.shape[1]
    is_classification = task in CLASSIFICATION
    for each_feature_name in meta_feature_names:
        if each_feature_name == 'NumberOfInstances':
            this_feature.append(n_row)
        elif each_feature_name == 'NumberOfFeatures':
            this_feature.append(n_feat)
        elif each_feature_name == 'NumberOfClasses':
            this_feature.append(len_labels(y_train) if is_classification else 0)
        elif each_feature_name == 'PercentageOfNumericFeatures':
            try:
                this_feature.append(X_train.select_dtypes(include=[np.number, 'float', 'int', 'long']).shape[1] / n_feat)
            except AttributeError:
                this_feature.append(1)
        else:
            raise ValueError('Feature {} not implemented. '.format(each_feature_name))
    return this_feature

def load_config_predictor(estimator_name, task, location=None):
    if False:
        while True:
            i = 10
    task = str(task)
    key = f'{location}/{estimator_name}/{task}'
    predictor = CONFIG_PREDICTORS.get(key)
    if predictor:
        return predictor
    task = 'multiclass' if task == 'multi' else task
    try:
        location = location or LOCATION
        with open(f'{location}/{estimator_name}/{task}.json', 'r') as f:
            CONFIG_PREDICTORS[key] = predictor = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'Portfolio has not been built for {estimator_name} on {task} task.')
    return predictor

def suggest_config(task, X, y, estimator_or_predictor, location=None, k=None, meta_feature_fn=meta_feature):
    if False:
        while True:
            i = 10
    'Suggest a list of configs for the given task and training data.\n\n    The returned configs can be used as starting points for AutoML.fit().\n    `FLAML_sample_size` is removed from the configs.\n    '
    from packaging.version import parse as version_parse
    task = get_classification_objective(len_labels(y)) if task == 'classification' and y is not None else task
    predictor = load_config_predictor(estimator_or_predictor, task, location) if isinstance(estimator_or_predictor, str) else estimator_or_predictor
    older_version = '1.0.2'
    assert version_parse(__version__) >= version_parse(predictor['version']) >= version_parse(older_version)
    prep = predictor['preprocessing']
    feature = meta_feature_fn(task, X_train=X, y_train=y, meta_feature_names=predictor['meta_feature_names'])
    feature = (np.array(feature) - np.array(prep['center'])) / np.array(prep['scale'])
    neighbors = predictor['neighbors']
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit([x['features'] for x in neighbors])
    (dist, ind) = nn.kneighbors(feature.reshape(1, -1), return_distance=True)
    logger.info(f'metafeature distance: {dist.item()}')
    ind = int(ind.item())
    choice = neighbors[ind]['choice'] if k is None else neighbors[ind]['choice'][:k]
    configs = [predictor['portfolio'][x] for x in choice]
    for config in configs:
        if 'hyperparameters' in config:
            hyperparams = config['hyperparameters']
            if hyperparams and 'FLAML_sample_size' in hyperparams:
                hyperparams.pop('FLAML_sample_size')
    return configs

def suggest_learner(task, X, y, estimator_or_predictor='all', estimator_list=None, location=None):
    if False:
        return 10
    'Suggest best learner within estimator_list.'
    configs = suggest_config(task, X, y, estimator_or_predictor, location)
    if not estimator_list:
        return configs[0]['class']
    for c in configs:
        if c['class'] in estimator_list:
            return c['class']
    return estimator_list[0]

def suggest_hyperparams(task, X, y, estimator_or_predictor, location=None):
    if False:
        for i in range(10):
            print('nop')
    'Suggest hyperparameter configurations and an estimator class.\n\n    The configurations can be used to initialize the estimator class like lightgbm.LGBMRegressor.\n\n    Example:\n\n    ```python\n    hyperparams, estimator_class = suggest_hyperparams("regression", X_train, y_train, "lgbm")\n    model = estimator_class(**hyperparams)  # estimator_class is LGBMRegressor\n    model.fit(X_train, y_train)\n    ```\n\n    Args:\n        task: A string of the task type, e.g.,\n            \'classification\', \'regression\', \'ts_forecast\', \'rank\',\n            \'seq-classification\', \'seq-regression\'.\n        X: A dataframe of training data in shape n*m.\n            For \'ts_forecast\' task, the first column of X_train\n            must be the timestamp column (datetime type). Other\n            columns in the dataframe are assumed to be exogenous\n            variables (categorical or numeric).\n        y: A series of labels in shape n*1.\n        estimator_or_predictor: A str of the learner name or a dict of the learned config predictor.\n            If a dict, it contains:\n            - "version": a str of the version number.\n            - "preprocessing": a dictionary containing:\n                * "center": a list of meta feature value offsets for normalization.\n                * "scale": a list of meta feature scales to normalize each dimension.\n            - "neighbors": a list of dictionaries. Each dictionary contains:\n                * "features": a list of the normalized meta features for a neighbor.\n                * "choice": an integer of the configuration id in the portfolio.\n            - "portfolio": a list of dictionaries, each corresponding to a configuration:\n                * "class": a str of the learner name.\n                * "hyperparameters": a dict of the config. The key "FLAML_sample_size" will be ignored.\n        location: (Optional) A str of the location containing mined portfolio file.\n            Only valid when the portfolio is a str, by default the location is flaml/default.\n\n    Returns:\n        hyperparams: A dict of the hyperparameter configurations.\n        estiamtor_class: A class of the underlying estimator, e.g., lightgbm.LGBMClassifier.\n    '
    config = suggest_config(task, X, y, estimator_or_predictor, location=location, k=1)[0]
    estimator = config['class']
    task = task_factory(task)
    model_class = task.estimator_class_from_str(estimator)
    hyperparams = config['hyperparameters']
    model = model_class(task=task.name, **hyperparams)
    estimator_class = model.estimator_class
    hyperparams = hyperparams and model.params
    return (hyperparams, estimator_class)

class AutoMLTransformer:

    def __init__(self, model, data_transformer):
        if False:
            for i in range(10):
                print('nop')
        self._model = model
        self._dt = data_transformer

    def transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        return self._model._preprocess(self._dt.transform(X))

def preprocess_and_suggest_hyperparams(task, X, y, estimator_or_predictor, location=None):
    if False:
        while True:
            i = 10
    'Preprocess the data and suggest hyperparameters.\n\n    Example:\n\n    ```python\n    hyperparams, estimator_class, X, y, feature_transformer, label_transformer =         preprocess_and_suggest_hyperparams("classification", X_train, y_train, "xgb_limitdepth")\n    model = estimator_class(**hyperparams)  # estimator_class is XGBClassifier\n    model.fit(X, y)\n    X_test = feature_transformer.transform(X_test)\n    y_pred = label_transformer.inverse_transform(pd.Series(model.predict(X_test).astype(int)))\n    ```\n\n    Args:\n        task: A string of the task type, e.g.,\n            \'classification\', \'regression\', \'ts_forecast\', \'rank\',\n            \'seq-classification\', \'seq-regression\'.\n        X: A dataframe of training data in shape n*m.\n            For \'ts_forecast\' task, the first column of X_train\n            must be the timestamp column (datetime type). Other\n            columns in the dataframe are assumed to be exogenous\n            variables (categorical or numeric).\n        y: A series of labels in shape n*1.\n        estimator_or_predictor: A str of the learner name or a dict of the learned config predictor.\n            "choose_xgb" means choosing between xgb_limitdepth and xgboost.\n            If a dict, it contains:\n            - "version": a str of the version number.\n            - "preprocessing": a dictionary containing:\n                * "center": a list of meta feature value offsets for normalization.\n                * "scale": a list of meta feature scales to normalize each dimension.\n            - "neighbors": a list of dictionaries. Each dictionary contains:\n                * "features": a list of the normalized meta features for a neighbor.\n                * "choice": a integer of the configuration id in the portfolio.\n            - "portfolio": a list of dictionaries, each corresponding to a configuration:\n                * "class": a str of the learner name.\n                * "hyperparameters": a dict of the config. They key "FLAML_sample_size" will be ignored.\n        location: (Optional) A str of the location containing mined portfolio file.\n            Only valid when the portfolio is a str, by default the location is flaml/default.\n\n    Returns:\n        hyperparams: A dict of the hyperparameter configurations.\n        estiamtor_class: A class of the underlying estimator, e.g., lightgbm.LGBMClassifier.\n        X: the preprocessed X.\n        y: the preprocessed y.\n        feature_transformer: a data transformer that can be applied to X_test.\n        label_transformer: a label transformer that can be applied to y_test.\n    '
    dt = DataTransformer()
    (X, y) = dt.fit_transform(X, y, task)
    if 'choose_xgb' == estimator_or_predictor:
        estimator_or_predictor = suggest_learner(task, X, y, estimator_list=['xgb_limitdepth', 'xgboost'], location=location)
    config = suggest_config(task, X, y, estimator_or_predictor, location=location, k=1)[0]
    estimator = config['class']
    model_class = task_factory(task).estimator_class_from_str(estimator)
    hyperparams = config['hyperparameters']
    model = model_class(task=task, **hyperparams)
    if model.estimator_class is None:
        return (hyperparams, model_class, X, y, None, None)
    else:
        estimator_class = model.estimator_class
        X = model._preprocess(X)
        hyperparams = hyperparams and model.params
        transformer = AutoMLTransformer(model, dt)
        return (hyperparams, estimator_class, X, y, transformer, dt.label_transformer)