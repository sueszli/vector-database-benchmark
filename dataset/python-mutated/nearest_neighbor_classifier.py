"""
Methods for creating and using a nearest neighbor classifier model.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import time as _time
import copy as _copy
import array as _array
import logging as _logging
import turicreate as _tc
from turicreate.toolkits._model import CustomModel as _CustomModel
from turicreate.toolkits._internal_utils import _toolkit_repr_print
from turicreate.toolkits._internal_utils import _raise_error_if_sframe_empty
from turicreate.toolkits._internal_utils import _raise_error_if_not_sframe
from turicreate.toolkits._internal_utils import _raise_error_if_column_exists
from turicreate.toolkits._internal_utils import _raise_error_evaluation_metric_is_valid
from turicreate.toolkits._main import ToolkitError as _ToolkitError
from turicreate.toolkits import evaluation as _evaluation
from turicreate.toolkits._model import PythonProxy as _PythonProxy

def _sort_topk_votes(x, k):
    if False:
        print('Hello World!')
    "\n    Sort a dictionary of classes and corresponding vote totals according to the\n    votes, then truncate to the highest 'k' classes.\n    "
    y = sorted(x.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{'class': i[0], 'votes': i[1]} for i in y]

def _construct_auto_distance(features, column_types):
    if False:
        print('Hello World!')
    '\n    Construct a composite distance function for a set of features, based on the\n    types of those features.\n\n    NOTE: This function is very similar to\n    `:func:_nearest_neighbors.choose_auto_distance`. The function is separate\n    because the auto-distance logic different than for each nearest\n    neighbors-based toolkit.\n\n    Parameters\n    ----------\n    features : list[str]\n        Names of for which to construct a distance function.\n\n    column_types : dict(string, type)\n        Names and types of all columns.\n\n    Returns\n    -------\n    dist : list[list]\n        A composite distance function. Each element of the inner list has three\n        elements: a list of feature names (strings), a distance function name\n        (string), and a weight (float).\n    '
    numeric_ftrs = []
    string_ftrs = []
    dict_ftrs = []
    for ftr in features:
        try:
            ftr_type = column_types[ftr]
        except:
            raise ValueError('The specified feature does not exist in the ' + 'input data.')
        if ftr_type == str:
            string_ftrs.append(ftr)
        elif ftr_type == dict:
            dict_ftrs.append(ftr)
        elif ftr_type in [int, float, _array.array]:
            numeric_ftrs.append(ftr)
        else:
            raise TypeError('Unable to automatically construct a distance ' + "function for feature '{}'. ".format(ftr) + 'For the nearest neighbor classifier, features ' + 'must be of type integer, float, string, dictionary, ' + 'or array.array.')
    dist = []
    for ftr in string_ftrs:
        dist.append([[ftr], 'levenshtein', 1])
    if len(dict_ftrs) > 0:
        dist.append([dict_ftrs, 'weighted_jaccard', len(dict_ftrs)])
    if len(numeric_ftrs) > 0:
        dist.append([numeric_ftrs, 'euclidean', len(numeric_ftrs)])
    return dist

def create(dataset, target, features=None, distance=None, verbose=True):
    if False:
        i = 10
        return i + 15
    "\n    Create a\n    :class:`~turicreate.nearest_neighbor_classifier.NearestNeighborClassifier`\n    model. This model predicts the class of a query instance by finding the most\n    common class among the query's nearest neighbors.\n\n    Parameters\n    ----------\n    dataset : SFrame\n        Dataset for training the model.\n\n    target : str\n        Name of the column containing the target variable. The values in this\n        column must be of string or integer type.\n\n    features : list[str], optional\n        Name of the columns with features to use in comparing records. 'None'\n        (the default) indicates that all columns except the target variable\n        should be used. Please note: if `distance` is specified as a composite\n        distance, then that parameter controls which features are used in the\n        model. Each column can be one of the following types:\n\n        - *Numeric*: values of numeric type integer or float.\n\n        - *Array*: array of numeric (integer or float) values. Each array\n          element is treated as a separate variable in the model.\n\n        - *Dictionary*: key-value pairs with numeric (integer or float) values.\n          Each key indicates a separate variable in the model.\n\n        - *String*: string values.\n\n        Please note: if `distance` is specified as a composite distance, then\n        that parameter controls which features are used in the model.\n\n    distance : str, function, or list[list], optional\n        Function to measure the distance between any two input data rows. This\n        may be one of three types:\n\n        - *String*: the name of a standard distance function. One of\n          'euclidean', 'squared_euclidean', 'manhattan', 'levenshtein',\n          'jaccard', 'weighted_jaccard', 'cosine' or 'transformed_dot_product'.\n\n        - *Function*: a function handle from the\n          :mod:`~turicreate.toolkits.distances` module.\n\n        - *Composite distance*: the weighted sum of several standard distance\n          functions applied to various features. This is specified as a list of\n          distance components, each of which is itself a list containing three\n          items:\n\n          1. list or tuple of feature names (str)\n\n          2. standard distance name (str)\n\n          3. scaling factor (int or float)\n\n        For more information about Turi Create distance functions, please\n        see the :py:mod:`~turicreate.toolkits.distances` module.\n\n        For sparse vectors, missing keys are assumed to have value 0.0.\n\n        If 'distance' is left unspecified or set to 'auto', a composite distance\n        is constructed automatically based on feature types.\n\n    verbose : bool, optional\n        If True, print progress updates and model details.\n\n    Returns\n    -------\n    out : NearestNeighborClassifier\n        A trained model of type\n        :class:`~turicreate.nearest_neighbor_classifier.NearestNeighborClassifier`.\n\n    See Also\n    --------\n    NearestNeighborClassifier\n    turicreate.toolkits.nearest_neighbors\n    turicreate.toolkits.distances\n\n    References\n    ----------\n    - `Wikipedia - nearest neighbors classifier\n      <http://en.wikipedia.org/wiki/Nearest_neighbour_classifiers>`_\n\n    - Hastie, T., Tibshirani, R., Friedman, J. (2009). `The Elements of\n      Statistical Learning <https://web.stanford.edu/~hastie/ElemStatLearn/>`_.\n      Vol. 2. New York. Springer. pp. 463-481.\n\n    Examples\n    --------\n    >>> sf = turicreate.SFrame({'species': ['cat', 'dog', 'fossa', 'dog'],\n    ...                       'height': [9, 25, 20, 23],\n    ...                       'weight': [13, 28, 33, 22]})\n    ...\n    >>> model = turicreate.nearest_neighbor_classifier.create(sf, target='species')\n\n    As with the nearest neighbors toolkit, the nearest neighbor classifier\n    accepts composite distance functions.\n\n    >>> my_dist = [[('height', 'weight'), 'euclidean', 2.7],\n    ...            [('height', 'weight'), 'manhattan', 1.6]]\n    ...\n    >>> model = turicreate.nearest_neighbor_classifier.create(sf, target='species',\n    ...                                                     distance=my_dist)\n    "
    start_time = _time.time()
    _raise_error_if_not_sframe(dataset, 'dataset')
    _raise_error_if_sframe_empty(dataset, 'dataset')
    if not isinstance(target, str) or target not in dataset.column_names():
        raise _ToolkitError("The 'target' parameter must be the name of a column in the input dataset.")
    if not dataset[target].dtype == str and (not dataset[target].dtype == int):
        raise TypeError('The target column must contain integers or strings.')
    if dataset[target].countna() > 0:
        _logging.warning('Missing values detected in the target column. This ' + "may lead to ambiguous 'None' predictions, if the " + "'radius' parameter is set too small in the prediction, " + 'classification, or evaluation methods.')
    if features is None:
        _features = [x for x in dataset.column_names() if x != target]
    else:
        _features = [x for x in features if x != target]
    if isinstance(distance, list):
        distance = _copy.deepcopy(distance)
    elif hasattr(distance, '__call__') or (isinstance(distance, str) and (not distance == 'auto')):
        distance = [[_features, distance, 1]]
    elif distance is None or distance == 'auto':
        col_types = {k: v for (k, v) in zip(dataset.column_names(), dataset.column_types())}
        distance = _construct_auto_distance(_features, col_types)
    else:
        raise TypeError("Input 'distance' not understood. The 'distance' " + 'parameter must be a string or a composite distance, ' + ' or left unspecified.')
    knn_model = _tc.nearest_neighbors.create(dataset, label=target, distance=distance, verbose=verbose)
    state = {'verbose': verbose, 'distance': knn_model.distance, 'num_distance_components': knn_model.num_distance_components, 'num_examples': dataset.num_rows(), 'features': knn_model.features, 'target': target, 'num_classes': len(dataset[target].unique()), 'num_features': knn_model.num_features, 'num_unpacked_features': knn_model.num_unpacked_features, 'training_time': _time.time() - start_time, '_target_type': dataset[target].dtype}
    model = NearestNeighborClassifier(knn_model, state)
    return model

class NearestNeighborClassifier(_CustomModel):
    """
    Nearest neighbor classifier model. Nearest neighbor classifiers predict the
    class of any observation to be the most common class among the observation's
    closest neighbors.

    This model should not be constructed directly. Instead, use
    :func:`turicreate.nearest_neighbor_classifier.create` to create an instance of
    this model.
    """
    _PYTHON_NN_CLASSIFIER_MODEL_VERSION = 2

    def __init__(self, knn_model, state=None):
        if False:
            for i in range(10):
                print('nop')
        self.__proxy__ = _PythonProxy(state)
        assert isinstance(knn_model, _tc.nearest_neighbors.NearestNeighborsModel)
        self._knn_model = knn_model

    @classmethod
    def _native_name(cls):
        if False:
            while True:
                i = 10
        return 'nearest_neighbor_classifier'

    def _get_version(self):
        if False:
            i = 10
            return i + 15
        return self._PYTHON_NN_CLASSIFIER_MODEL_VERSION

    def _get_native_state(self):
        if False:
            for i in range(10):
                print('nop')
        retstate = self.__proxy__.get_state()
        retstate['knn_model'] = self._knn_model.__proxy__
        retstate['_target_type'] = self._target_type.__name__
        return retstate

    @classmethod
    def _load_version(cls, state, version):
        if False:
            while True:
                i = 10
        '\n        A function to load a previously saved NearestNeighborClassifier model.\n\n        Parameters\n        ----------\n        unpickler : GLUnpickler\n            A GLUnpickler file handler.\n\n        version : int\n            Version number maintained by the class writer.\n        '
        assert version == cls._PYTHON_NN_CLASSIFIER_MODEL_VERSION
        knn_model = _tc.nearest_neighbors.NearestNeighborsModel(state['knn_model'])
        del state['knn_model']
        state['_target_type'] = eval(state['_target_type'])
        return cls(knn_model, state)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Print a string description of the model when the model name is entered\n        in the terminal.\n        '
        (sections, section_titles) = self._get_summary_struct()
        out = _toolkit_repr_print(self, sections, section_titles, width=36)
        return out

    def __str__(self):
        if False:
            while True:
                i = 10
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : str\n            A description of the NearestNeighborClassifier model.\n        '
        return self.__repr__()

    def _get_summary_struct(self):
        if False:
            print('Hello World!')
        "\n        Returns a structured description of the model, including (where relevant)\n        the schema of the training data, description of the training data,\n        training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        model_fields = [('Number of examples', 'num_examples'), ('Number of feature columns', 'num_features'), ('Number of unpacked features', 'num_unpacked_features'), ('Number of distance components', 'num_distance_components'), ('Number of classes', 'num_classes')]
        training_fields = [('Training time (seconds)', 'training_time')]
        section_titles = ['Schema', 'Training Summary']
        return ([model_fields, training_fields], section_titles)

    def classify(self, dataset, max_neighbors=10, radius=None, verbose=True):
        if False:
            i = 10
            return i + 15
        "\n        Return the predicted class for each observation in *dataset*. This\n        prediction is made based on the closest neighbors stored in the nearest\n        neighbors classifier model.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        verbose : bool, optional\n            If True, print progress updates.\n\n        max_neighbors : int, optional\n            Maximum number of neighbors to consider for each point.\n\n        radius : float, optional\n            Maximum distance from each point to a neighbor in the reference\n            dataset.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame with model predictions. The first column is the most\n            likely class according to the model, and the second column is the\n            predicted probability for that class.\n\n        See Also\n        --------\n        create, predict, predict_topk\n\n        Notes\n        -----\n        - If the 'radius' parameter is small, it is possible that a query point\n          has no qualified neighbors in the training dataset. In this case, the\n          resulting class and probability for that query are 'None' in the\n          SFrame output by this method. If the target column in the training\n          dataset has missing values, these predictions will be ambiguous.\n\n        - Ties between predicted classes are broken randomly.\n\n        Examples\n        --------\n        >>> sf_train = turicreate.SFrame({'species': ['cat', 'dog', 'fossa', 'dog'],\n        ...                             'height': [9, 25, 20, 23],\n        ...                             'weight': [13, 28, 33, 22]})\n        ...\n        >>> sf_new = turicreate.SFrame({'height': [26, 19],\n        ...                           'weight': [25, 35]})\n        ...\n        >>> m = turicreate.nearest_neighbor_classifier.create(sf, target='species')\n        >>> ystar = m.classify(sf_new, max_neighbors=2)\n        >>> print ystar\n        +-------+-------------+\n        | class | probability |\n        +-------+-------------+\n        |  dog  |     1.0     |\n        | fossa |     0.5     |\n        +-------+-------------+\n        "
        _raise_error_if_not_sframe(dataset, 'dataset')
        _raise_error_if_sframe_empty(dataset, 'dataset')
        n_query = dataset.num_rows()
        if max_neighbors is not None:
            if not isinstance(max_neighbors, int):
                raise ValueError("Input 'max_neighbors' must be an integer.")
            if max_neighbors <= 0:
                raise ValueError("Input 'max_neighbors' must be larger than 0.")
        knn = self._knn_model.query(dataset, k=max_neighbors, radius=radius, verbose=verbose)
        if knn.num_rows() == 0:
            ystar = _tc.SFrame({'class': _tc.SArray([None] * n_query, self._target_type), 'probability': _tc.SArray([None] * n_query, int)})
        else:
            grp = knn.groupby(['query_label', 'reference_label'], _tc.aggregate.COUNT)
            ystar = grp.groupby('query_label', {'class': _tc.aggregate.ARGMAX('Count', 'reference_label'), 'max_votes': _tc.aggregate.MAX('Count'), 'total_votes': _tc.aggregate.SUM('Count')})
            ystar['probability'] = ystar['max_votes'] / ystar['total_votes']
            row_ids = _tc.SFrame({'query_label': range(n_query)})
            ystar = ystar.join(row_ids, how='right')
            ystar = ystar.sort('query_label', ascending=True)
            ystar = ystar[['class', 'probability']]
        return ystar

    def predict(self, dataset, max_neighbors=10, radius=None, output_type='class', verbose=True):
        if False:
            print('Hello World!')
        "\n        Return predicted class labels for instances in *dataset*. This model\n        makes predictions based on the closest neighbors stored in the nearest\n        neighbors classifier model.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include the features used for\n            model training, but does not require a target column. Additional\n            columns are ignored.\n\n        max_neighbors : int, optional\n            Maximum number of neighbors to consider for each point.\n\n        radius : float, optional\n            Maximum distance from each point to a neighbor in the reference\n            dataset.\n\n        output_type : {'class', 'probability'}, optional\n            Type of prediction output:\n\n            - `class`: Predicted class label. The class with the maximum number\n              of votes among the nearest neighbors in the reference dataset.\n\n            - `probability`: Maximum number of votes for any class out of all\n              nearest neighbors in the reference dataset.\n\n        Returns\n        -------\n        out : SArray\n            An SArray with model predictions.\n\n        See Also\n        ----------\n        create, classify, predict_topk\n\n        Notes\n        -----\n        - If the 'radius' parameter is small, it is possible that a query point\n          has no qualified neighbors in the training dataset. In this case, the\n          result for that query is 'None' in the SArray output by this method.\n          If the target column in the training dataset has missing values, these\n          predictions will be ambiguous.\n\n        - Ties between predicted classes are broken randomly.\n\n        Examples\n        --------\n        >>> sf_train = turicreate.SFrame({'species': ['cat', 'dog', 'fossa', 'dog'],\n        ...                             'height': [9, 25, 20, 23],\n        ...                             'weight': [13, 28, 33, 22]})\n        ...\n        >>> sf_new = turicreate.SFrame({'height': [26, 19],\n        ...                           'weight': [25, 35]})\n        ...\n        >>> m = turicreate.nearest_neighbor_classifier.create(sf, target='species')\n        >>> ystar = m.predict(sf_new, max_neighbors=2, output_type='class')\n        >>> print ystar\n        ['dog', 'fossa']\n        "
        ystar = self.classify(dataset=dataset, max_neighbors=max_neighbors, radius=radius, verbose=verbose)
        if output_type == 'class':
            return ystar['class']
        elif output_type == 'probability':
            return ystar['probability']
        else:
            raise ValueError("Input 'output_type' not understood. 'output_type' must be either 'class' or 'probability'.")

    def predict_topk(self, dataset, max_neighbors=10, radius=None, k=3, verbose=False):
        if False:
            while True:
                i = 10
        "\n        Return top-k most likely predictions for each observation in\n        ``dataset``. Predictions are returned as an SFrame with three columns:\n        `row_id`, `class`, and `probability`.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include the features used for\n            model training, but does not require a target column. Additional\n            columns are ignored.\n\n        max_neighbors : int, optional\n            Maximum number of neighbors to consider for each point.\n\n        radius : float, optional\n            Maximum distance from each point to a neighbor in the reference\n            dataset.\n\n        k : int, optional\n            Number of classes to return for each input example.\n\n        Returns\n        -------\n        out : SFrame\n\n        See Also\n        ----------\n        create, classify, predict\n\n        Notes\n        -----\n        - If the 'radius' parameter is small, it is possible that a query point\n          has no neighbors in the training dataset. In this case, the query is\n          dropped from the SFrame output by this method. If all queries have no\n          neighbors, then the result is an empty SFrame. If the target column in\n          the training dataset has missing values, these predictions will be\n          ambiguous.\n\n        - Ties between predicted classes are broken randomly.\n\n        Examples\n        --------\n        >>> sf_train = turicreate.SFrame({'species': ['cat', 'dog', 'fossa', 'dog'],\n        ...                             'height': [9, 25, 20, 23],\n        ...                             'weight': [13, 28, 33, 22]})\n        ...\n        >>> sf_new = turicreate.SFrame({'height': [26, 19],\n        ...                           'weight': [25, 35]})\n        ...\n        >>> m = turicreate.nearest_neighbor_classifier.create(sf_train, target='species')\n        >>> ystar = m.predict_topk(sf_new, max_neighbors=2)\n        >>> print ystar\n        +--------+-------+-------------+\n        | row_id | class | probability |\n        +--------+-------+-------------+\n        |   0    |  dog  |     1.0     |\n        |   1    | fossa |     0.5     |\n        |   1    |  dog  |     0.5     |\n        +--------+-------+-------------+\n        "
        if not isinstance(k, int) or k < 1:
            raise TypeError('The number of results to return for each point, ' + "'k', must be an integer greater than 0.")
        _raise_error_if_not_sframe(dataset, 'dataset')
        _raise_error_if_sframe_empty(dataset, 'dataset')
        if max_neighbors is not None:
            if not isinstance(max_neighbors, int):
                raise ValueError("Input 'max_neighbors' must be an integer.")
            if max_neighbors <= 0:
                raise ValueError("Input 'max_neighbors' must be larger than 0.")
        knn = self._knn_model.query(dataset, k=max_neighbors, radius=radius, verbose=verbose)
        if knn.num_rows() == 0:
            ystar = _tc.SFrame({'row_id': [], 'class': [], 'probability': []})
            ystar['row_id'] = ystar['row_id'].astype(int)
            ystar['class'] = ystar['class'].astype(str)
        else:
            grp = knn.groupby(['query_label', 'reference_label'], _tc.aggregate.COUNT)
            ystar = grp.unstack(column_names=['reference_label', 'Count'], new_column_name='votes')
            ystar['topk'] = ystar['votes'].apply(lambda x: _sort_topk_votes(x, k))
            ystar['total_votes'] = ystar['votes'].apply(lambda x: sum(x.values()))
            ystar = ystar.stack('topk', new_column_name='topk')
            ystar = ystar.unpack('topk')
            ystar.rename({'topk.class': 'class', 'query_label': 'row_id'}, inplace=True)
            ystar['probability'] = ystar['topk.votes'] / ystar['total_votes']
            ystar = ystar[['row_id', 'class', 'probability']]
        return ystar

    def evaluate(self, dataset, metric='auto', max_neighbors=10, radius=None):
        if False:
            while True:
                i = 10
        "\n        Evaluate the model's predictive accuracy. This is done by predicting the\n        target class for instances in a new dataset and comparing to known\n        target values.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the target and features used for model training. Additional\n            columns are ignored.\n\n        metric : str, optional\n            Name of the evaluation metric.  Possible values are:\n\n            - 'auto': Returns all available metrics.\n\n            - 'accuracy': Classification accuracy.\n\n            - 'confusion_matrix': An SFrame with counts of possible\n              prediction/true label combinations.\n\n            - 'roc_curve': An SFrame containing information needed for an roc\n              curve (binary classification only).\n\n        max_neighbors : int, optional\n            Maximum number of neighbors to consider for each point.\n\n        radius : float, optional\n            Maximum distance from each point to a neighbor in the reference\n            dataset.\n\n        Returns\n        -------\n        out : dict\n            Evaluation results. The dictionary keys are *accuracy* and\n            *confusion_matrix* and *roc_curve* (if applicable).\n\n        See also\n        --------\n        create, predict, predict_topk, classify\n\n        Notes\n        -----\n        - Because the model randomly breaks ties between predicted classes, the\n          results of repeated calls to `evaluate` method may differ.\n\n        Examples\n        --------\n        >>> sf_train = turicreate.SFrame({'species': ['cat', 'dog', 'fossa', 'dog'],\n        ...                             'height': [9, 25, 20, 23],\n        ...                             'weight': [13, 28, 33, 22]})\n        >>> m = turicreate.nearest_neighbor_classifier.create(sf, target='species')\n        >>> ans = m.evaluate(sf_train, max_neighbors=2,\n        ...                  metric='confusion_matrix')\n        >>> print ans['confusion_matrix']\n        +--------------+-----------------+-------+\n        | target_label | predicted_label | count |\n        +--------------+-----------------+-------+\n        |     cat      |       dog       |   1   |\n        |     dog      |       dog       |   2   |\n        |    fossa     |       dog       |   1   |\n        +--------------+-----------------+-------+\n        "
        _raise_error_evaluation_metric_is_valid(metric, ['auto', 'accuracy', 'confusion_matrix', 'roc_curve'])
        target = self.target
        _raise_error_if_column_exists(dataset, target, 'dataset', target)
        if not dataset[target].dtype == str and (not dataset[target].dtype == int):
            raise TypeError('The target column of the evaluation dataset must contain integers or strings.')
        if self.num_classes != 2:
            if metric == 'roc_curve' or metric == ['roc_curve']:
                err_msg = 'Currently, ROC curve is not supported for '
                err_msg += 'multi-class classification in this model.'
                raise _ToolkitError(err_msg)
            else:
                warn_msg = 'WARNING: Ignoring `roc_curve`. '
                warn_msg += 'Not supported for multi-class classification.'
                print(warn_msg)
        ystar = self.predict(dataset, output_type='class', max_neighbors=max_neighbors, radius=radius)
        ystar_prob = self.predict(dataset, output_type='probability', max_neighbors=max_neighbors, radius=radius)
        results = {}
        if metric in ['accuracy', 'auto']:
            results['accuracy'] = _evaluation.accuracy(targets=dataset[target], predictions=ystar)
        if metric in ['confusion_matrix', 'auto']:
            results['confusion_matrix'] = _evaluation.confusion_matrix(targets=dataset[target], predictions=ystar)
        if self.num_classes == 2:
            if metric in ['roc_curve', 'auto']:
                results['roc_curve'] = _evaluation.roc_curve(targets=dataset[target], predictions=ystar_prob)
        return results