"""@package turicreate.toolkits
This module defines the (internal) functions used by the supervised_learning_models.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _turicreate
from turicreate.toolkits._model import Model
from turicreate.toolkits._internal_utils import _raise_error_if_not_sframe
from turicreate.toolkits._internal_utils import _validate_data
from turicreate.toolkits._main import ToolkitError
from turicreate._cython.cy_server import QuietProgress

class SupervisedLearningModel(Model):
    """
    Supervised learning module to predict a target variable as a function of
    several feature variables.
    """

    def __init__(self, model_proxy=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        self.__proxy__ = model_proxy
        self.__name__ = name

    @classmethod
    def _native_name(cls):
        if False:
            for i in range(10):
                print('nop')
        return None

    def __str__(self):
        if False:
            while True:
                i = 10
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the model.\n        '
        return self.__class__.__name__

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a string description of the model, including (where relevant)\n        the schema of the training data, description of the training data,\n        training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        out : string\n            A description of the model.\n        '
        return self.__class__.__name__

    def predict(self, dataset, missing_value_action='auto', output_type='', options={}, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return predictions for ``dataset``, using the trained supervised_learning\n        model. Predictions are generated as class labels (0 or\n        1).\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        missing_value_action: str, optional\n            Action to perform when missing values are encountered. This can be\n            one of:\n\n            - 'auto': Choose a model dependent missing value policy.\n            - 'impute': Proceed with evaluation by filling in the missing\n                        values with the mean of the training data. Missing\n                        values are also imputed if an entire column of data is\n                        missing during evaluation.\n            - 'none': Treat missing value as is. Model must be able to handle missing value.\n            - 'error' : Do not proceed with prediction and terminate with\n                        an error message.\n\n        output_type : str, optional\n            output type that maybe needed by some of the toolkits\n\n        options : dict\n            additional options to be passed in to prediction\n\n        kwargs : dict\n            additional options to be passed into prediction\n\n        Returns\n        -------\n        out : SArray\n            An SArray with model predictions.\n        "
        if missing_value_action == 'auto':
            missing_value_action = select_default_missing_value_policy(self, 'predict')
        if isinstance(dataset, list):
            return self.__proxy__.fast_predict(dataset, missing_value_action, output_type)
        if isinstance(dataset, dict):
            return self.__proxy__.fast_predict([dataset], missing_value_action, output_type)
        else:
            _raise_error_if_not_sframe(dataset, 'dataset')
            return self.__proxy__.predict(dataset, missing_value_action, output_type)

    def evaluate(self, dataset, metric='auto', missing_value_action='auto', with_predictions=False, options={}, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Evaluate the model by making predictions of target values and comparing\n        these to actual values.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset in the same format used for training. The columns names and\n            types of the dataset must be the same as that used in training.\n\n        metric : str, list[str]\n            Evaluation metric(s) to be computed.\n\n        missing_value_action: str, optional\n            Action to perform when missing values are encountered. This can be\n            one of:\n\n            - 'auto': Choose a model dependent missing value policy.\n            - 'impute': Proceed with evaluation by filling in the missing\n                        values with the mean of the training data. Missing\n                        values are also imputed if an entire column of data is\n                        missing during evaluation.\n            - 'none': Treat missing value as is. Model must be able to handle missing value.\n            - 'error' : Do not proceed with prediction and terminate with\n                        an error message.\n\n        options : dict\n            additional options to be passed in to prediction\n\n        kwargs : dict\n            additional options to be passed into prediction\n        "
        if missing_value_action == 'auto':
            missing_value_action = select_default_missing_value_policy(self, 'evaluate')
        _raise_error_if_not_sframe(dataset, 'dataset')
        results = self.__proxy__.evaluate(dataset, missing_value_action, metric, with_predictions=with_predictions)
        return results

    def _training_stats(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a dictionary containing statistics collected during model\n        training. These statistics are also available with the ``get`` method,\n        and are described in more detail in the documentation for that method.\n\n        Notes\n        -----\n        '
        return self.__proxy__.get_train_stats()

    def _get(self, field):
        if False:
            while True:
                i = 10
        '\n        Get the value of a given field.\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out : [various]\n            The current value of the requested field.\n        '
        return self.__proxy__.get_value(field)

class Classifier(SupervisedLearningModel):
    """
    Classifier module to predict a discrete target variable as a function of
    several feature variables.
    """

    @classmethod
    def _native_name(cls):
        if False:
            return 10
        return None

    def classify(self, dataset, missing_value_action='auto'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return predictions for ``dataset``, using the trained supervised_learning\n        model. Predictions are generated as class labels (0 or\n        1).\n\n        Parameters\n        ----------\n        dataset: SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        missing_value_action: str, optional\n            Action to perform when missing values are encountered. This can be\n            one of:\n\n            - 'auto': Choose model dependent missing value action\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with prediction and terminate with\n              an error message.\n        Returns\n        -------\n        out : SFrame\n            An SFrame with model predictions.\n        "
        if missing_value_action == 'auto':
            missing_value_action = select_default_missing_value_policy(self, 'classify')
        if isinstance(dataset, list):
            return self.__proxy__.fast_classify(dataset, missing_value_action)
        if isinstance(dataset, dict):
            return self.__proxy__.fast_classify([dataset], missing_value_action)
        _raise_error_if_not_sframe(dataset, 'dataset')
        return self.__proxy__.classify(dataset, missing_value_action)

def print_validation_track_notification():
    if False:
        return 10
    print('PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.\n          You can set ``validation_set=None`` to disable validation tracking.\n')

def create(dataset, target, model_name, features=None, validation_set='auto', distributed='auto', verbose=True, seed=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Create a :class:`~turicreate.toolkits.SupervisedLearningModel`,\n\n    This is generic function that allows you to create any model that\n    implements SupervisedLearningModel This function is normally not called, call\n    specific model's create function instead\n\n    Parameters\n    ----------\n    dataset : SFrame\n        Dataset for training the model.\n\n    target : string\n        Name of the column containing the target variable. The values in this\n        column must be 0 or 1, of integer type.\n\n    model_name : string\n        Name of the model\n\n    features : list[string], optional\n        List of feature names used by feature column\n\n    validation_set : SFrame, optional\n        A dataset for monitoring the model's generalization performance.\n        For each row of the progress table, the chosen metrics are computed\n        for both the provided training dataset and the validation_set. The\n        format of this SFrame must be the same as the training set.\n        By default this argument is set to 'auto' and a validation set is\n        automatically sampled and used for progress printing. If\n        validation_set is set to None, then no additional metrics\n        are computed. The default value is 'auto'.\n\n    distributed: env\n        The distributed environment\n\n    verbose : boolean\n        whether print out messages during training\n\n    seed : int, optional\n        Seed for random number generation. Set this value to ensure that the\n        same model is created every time.\n\n    kwargs : dict\n        Additional parameter options that can be passed\n    "
    (dataset, validation_set) = _validate_data(dataset, target, features, validation_set)
    if isinstance(validation_set, str):
        assert validation_set == 'auto'
        if dataset.num_rows() >= 100:
            if verbose:
                print_validation_track_notification()
            (dataset, validation_set) = dataset.random_split(0.95, seed=seed, exact=True)
        else:
            validation_set = _turicreate.SFrame()
    elif validation_set is None:
        validation_set = _turicreate.SFrame()
    options = {k.lower(): kwargs[k] for k in kwargs}
    model = _turicreate.extensions.__dict__[model_name]()
    with QuietProgress(verbose):
        model.train(dataset, target, validation_set, options)
    return SupervisedLearningModel(model, model_name)

def create_classification_with_model_selector(dataset, target, model_selector, features=None, validation_set='auto', verbose=True):
    if False:
        print('Hello World!')
    "\n    Create a :class:`~turicreate.toolkits.SupervisedLearningModel`,\n\n    This is generic function that allows you to create any model that\n    implements SupervisedLearningModel. This function is normally not called, call\n    specific model's create function instead.\n\n    Parameters\n    ----------\n    dataset : SFrame\n        Dataset for training the model.\n\n    target : string\n        Name of the column containing the target variable. The values in this\n        column must be 0 or 1, of integer type.\n\n    model_name : string\n        Name of the model\n\n    model_selector: function\n        Provide a model selector.\n\n    features : list[string], optional\n        List of feature names used by feature column\n\n    verbose : boolean\n        whether print out messages during training\n\n    "
    (dataset, validation_set) = _validate_data(dataset, target, features, validation_set)
    features_sframe = dataset
    if features_sframe.num_rows() > 100000.0:
        fraction = 1.0 * 100000.0 / features_sframe.num_rows()
        features_sframe = features_sframe.sample(fraction, seed=0)
    num_classes = len(dataset[target].unique())
    selected_model_names = model_selector(num_classes, features_sframe)
    if isinstance(validation_set, str):
        if validation_set == 'auto':
            if dataset.num_rows() >= 100:
                if verbose:
                    print_validation_track_notification()
                (dataset, validation_set) = dataset.random_split(0.95, exact=True)
            else:
                validation_set = None
        else:
            raise TypeError('Unrecognized value for validation_set.')
    python_names = {'boosted_trees_classifier': 'BoostedTreesClassifier', 'random_forest_classifier': 'RandomForestClassifier', 'decision_tree_classifier': 'DecisionTreeClassifier', 'classifier_logistic_regression': 'LogisticClassifier', 'classifier_svm': 'SVMClassifier'}
    if verbose:
        print('PROGRESS: The following methods are available for this type of problem.')
        print('PROGRESS: ' + ', '.join([python_names[x] for x in selected_model_names]))
        if len(selected_model_names) > 1:
            print('PROGRESS: The returned model will be chosen according to validation accuracy.')
    models = {}
    metrics = {}
    for model_name in selected_model_names:
        m = create_selected(model_name, dataset, target, features, validation_set, verbose)
        models[model_name] = m
        if 'validation_accuracy' in m._list_fields():
            metrics[model_name] = m.validation_accuracy
        elif 'training_accuracy' in m._list_fields():
            metrics[model_name] = m.training_accuracy
        elif 'progress' in m._list_fields():
            prog = m.progress
            validation_column = 'Validation Accuracy'
            accuracy_column = 'Training Accuracy'
            if validation_column in prog.column_names():
                metrics[model_name] = float(prog[validation_column].tail(1)[0])
            else:
                metrics[model_name] = float(prog[accuracy_column].tail(1)[0])
        else:
            raise ValueError('Model does not have metrics that can be used for model selection.')
    best_model = None
    best_acc = None
    for model_name in selected_model_names:
        if best_acc is None:
            best_model = model_name
            best_acc = metrics[model_name]
        if best_acc is not None and best_acc < metrics[model_name]:
            best_model = model_name
            best_acc = metrics[model_name]
    ret = []
    width = 32
    if len(selected_model_names) > 1:
        ret.append('PROGRESS: Model selection based on validation accuracy:')
        ret.append('---------------------------------------------')
        key_str = '{:<{}}: {}'
        for model_name in selected_model_names:
            name = python_names[model_name]
            row = key_str.format(name, width, str(metrics[model_name]))
            ret.append(row)
        ret.append('---------------------------------------------')
        ret.append('Selecting ' + python_names[best_model] + ' based on validation set performance.')
    if verbose:
        print('\nPROGRESS: '.join(ret))
    return models[best_model]

def create_selected(selected_model_name, dataset, target, features, validation_set='auto', verbose=True):
    if False:
        for i in range(10):
            print('nop')
    model = create(dataset, target, selected_model_name, features=features, validation_set=validation_set, verbose=verbose)
    return wrap_model_proxy(model.__proxy__)

def wrap_model_proxy(model_proxy):
    if False:
        while True:
            i = 10
    selected_model_name = model_proxy.__class__.__name__
    if selected_model_name == 'boosted_trees_regression':
        return _turicreate.boosted_trees_regression.BoostedTreesRegression(model_proxy)
    elif selected_model_name == 'random_forest_regression':
        return _turicreate.random_forest_regression.RandomForestRegression(model_proxy)
    elif selected_model_name == 'decision_tree_regression':
        return _turicreate.decision_tree_classifier.DecisionTreeRegression(model_proxy)
    elif selected_model_name == 'regression_linear_regression':
        return _turicreate.linear_regression.LinearRegression(model_proxy)
    elif selected_model_name == 'boosted_trees_classifier':
        return _turicreate.boosted_trees_classifier.BoostedTreesClassifier(model_proxy)
    elif selected_model_name == 'random_forest_classifier':
        return _turicreate.random_forest_classifier.RandomForestClassifier(model_proxy)
    elif selected_model_name == 'decision_tree_classifier':
        return _turicreate.decision_tree_classifier.DecisionTreeClassifier(model_proxy)
    elif selected_model_name == 'classifier_logistic_regression':
        return _turicreate.logistic_classifier.LogisticClassifier(model_proxy)
    elif selected_model_name == 'classifier_svm':
        return _turicreate.svm_classifier.SVMClassifier(model_proxy)
    else:
        raise ToolkitError('Internal error: Incorrect model returned.')

def select_default_missing_value_policy(model, action):
    if False:
        for i in range(10):
            print('nop')
    from .classifier.boosted_trees_classifier import BoostedTreesClassifier
    from .classifier.random_forest_classifier import RandomForestClassifier
    from .classifier.decision_tree_classifier import DecisionTreeClassifier
    from .regression.boosted_trees_regression import BoostedTreesRegression
    from .regression.random_forest_regression import RandomForestRegression
    from .regression.decision_tree_regression import DecisionTreeRegression
    tree_models = [BoostedTreesClassifier, BoostedTreesRegression, RandomForestClassifier, RandomForestRegression, DecisionTreeClassifier, DecisionTreeRegression]
    if any((isinstance(model, tree_model) for tree_model in tree_models)):
        return 'none'
    else:
        return 'impute'