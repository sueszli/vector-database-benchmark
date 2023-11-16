"""
Class definition and utilities for the activity classification toolkit.
"""
from __future__ import absolute_import as _
from __future__ import print_function as _
from __future__ import division as _
import time as _time
import six as _six
from turicreate import SFrame as _SFrame
import turicreate.toolkits._internal_utils as _tkutl
from turicreate.toolkits import _coreml_utils
import turicreate.toolkits._feature_engineering._internal_utils as _fe_tkutl
from turicreate.toolkits._main import ToolkitError as _ToolkitError
from turicreate._deps.minimal_package import _minimal_package_import_check
from turicreate.toolkits._model import Model as _Model

def create(dataset, session_id, target, features=None, prediction_window=100, validation_set='auto', max_iterations=10, batch_size=32, verbose=True, random_seed=None):
    if False:
        return 10
    "\n    Create an :class:`ActivityClassifier` model.\n\n    Parameters\n    ----------\n    dataset : SFrame\n        Input data which consists of `sessions` of data where each session is\n        a sequence of data. The data must be in `stacked` format, grouped by\n        session. Within each session, the data is assumed to be sorted\n        temporally. Columns in `features` will be used to train a model that\n        will make a prediction using labels in the `target` column.\n\n    session_id : string\n        Name of the column that contains a unique ID for each session.\n\n    target : string\n        Name of the column containing the target variable. The values in this\n        column must be of string or integer type. Use `model.classes` to\n        retrieve the order in which the classes are mapped.\n\n    features : list[string], optional\n        Name of the columns containing the input features that will be used\n        for classification. If set to `None`, all columns except `session_id`\n        and `target` will be used.\n\n    prediction_window : int, optional\n        Number of time units between predictions. For example, if your input\n        data is sampled at 100Hz, and the `prediction_window` is set to 100,\n        then this model will make a prediction every 1 second.\n\n    validation_set : SFrame, optional\n        A dataset for monitoring the model's generalization performance to\n        prevent the model from overfitting to the training data.\n\n        For each row of the progress table, accuracy is measured over the\n        provided training dataset and the `validation_set`. The format of this\n        SFrame must be the same as the training set.\n\n        When set to 'auto', a validation set is automatically sampled from the\n        training data (if the training data has > 100 sessions). If\n        validation_set is set to None, then all the data will be used for\n        training.\n\n    max_iterations : int , optional\n        Maximum number of iterations/epochs made over the data during the\n        training phase.\n\n    batch_size : int, optional\n        Number of sequence chunks used per training step. Must be greater than\n        the number of GPUs in use.\n\n    verbose : bool, optional\n        If True, print progress updates and model details.\n\n    random_seed : int, optional\n        The results can be reproduced when given the same seed.\n\n    Returns\n    -------\n    out : ActivityClassifier\n        A trained :class:`ActivityClassifier` model.\n\n    Examples\n    --------\n    .. sourcecode:: python\n\n        >>> import turicreate as tc\n\n        # Training on dummy data\n        >>> data = tc.SFrame({\n        ...    'accelerometer_x': [0.1, 0.2, 0.3, 0.4, 0.5] * 10,\n        ...    'accelerometer_y': [0.5, 0.4, 0.3, 0.2, 0.1] * 10,\n        ...    'accelerometer_z': [0.01, 0.01, 0.02, 0.02, 0.01] * 10,\n        ...    'session_id': [0, 0, 0] * 10 + [1, 1] * 10,\n        ...    'activity': ['walk', 'run', 'run'] * 10 + ['swim', 'swim'] * 10\n        ... })\n\n        # Create an activity classifier\n        >>> model = tc.activity_classifier.create(data,\n        ...     session_id='session_id', target='activity',\n        ...     features=['accelerometer_x', 'accelerometer_y', 'accelerometer_z'])\n\n        # Make predictions (as probability vector, or class)\n        >>> predictions = model.predict(data)\n        >>> predictions = model.predict(data, output_type='probability_vector')\n\n        # Get both predictions and classes together\n        >>> predictions = model.classify(data)\n\n        # Get topk predictions (instead of only top-1) if your labels have more\n        # 2 classes\n        >>> predictions = model.predict_topk(data, k = 3)\n\n        # Evaluate the model\n        >>> results = model.evaluate(data)\n\n    See Also\n    --------\n    ActivityClassifier, util.random_split_by_session\n    "
    _tkutl._raise_error_if_not_sframe(dataset, 'dataset')
    if not isinstance(target, str):
        raise _ToolkitError('target must be of type str')
    if not isinstance(session_id, str):
        raise _ToolkitError('session_id must be of type str')
    if not isinstance(batch_size, int):
        raise _ToolkitError('batch_size must be of type int')
    _tkutl._raise_error_if_sframe_empty(dataset, 'dataset')
    _tkutl._numeric_param_check_range('prediction_window', prediction_window, 1, 400)
    _tkutl._numeric_param_check_range('max_iterations', max_iterations, 0, _six.MAXSIZE)
    if features is None:
        features = _fe_tkutl.get_column_names(dataset, interpret_as_excluded=True, column_names=[session_id, target])
    if not hasattr(features, '__iter__'):
        raise TypeError("Input 'features' must be a list.")
    if not all([isinstance(x, str) for x in features]):
        raise TypeError('Invalid member of "features": feature names must be of type str.')
    if len(features) == 0:
        raise TypeError("Input 'features' must contain at least one column name.")
    start_time = _time.time()
    dataset = _tkutl._toolkits_select_columns(dataset, features + [session_id, target])
    _tkutl._raise_error_if_sarray_not_expected_dtype(dataset[target], target, [str, int])
    _tkutl._raise_error_if_sarray_not_expected_dtype(dataset[session_id], session_id, [str, int])
    for feature in features:
        _tkutl._handle_missing_values(dataset, feature, 'training_dataset')
    if isinstance(validation_set, _SFrame):
        _tkutl._raise_error_if_sframe_empty(validation_set, 'validation_set')
        for feature in features:
            _tkutl._handle_missing_values(validation_set, feature, 'validation_set')
    name = 'activity_classifier'
    import turicreate as _turicreate
    _minimal_package_import_check('turicreate.toolkits.libtctensorflow')
    model = _turicreate.extensions.activity_classifier()
    options = {}
    options['prediction_window'] = prediction_window
    options['batch_size'] = batch_size
    options['max_iterations'] = max_iterations
    options['verbose'] = verbose
    options['_show_loss'] = False
    options['random_seed'] = random_seed
    model.train(dataset, target, session_id, validation_set, options)
    return ActivityClassifier(model_proxy=model, name=name)

def _encode_target(data, target, mapping=None):
    if False:
        return 10
    ' Encode targets to integers in [0, num_classes - 1] '
    if mapping is None:
        mapping = {t: i for (i, t) in enumerate(sorted(data[target].unique()))}
    data[target] = data[target].apply(lambda t: mapping[t])
    return (data, mapping)

class ActivityClassifier(_Model):
    """
    A trained model using C++ implementation that is ready to use for classification or export to
    CoreML.

    This model should not be constructed directly.
    """
    _CPP_ACTIVITY_CLASSIFIER_VERSION = 1

    def __init__(self, model_proxy=None, name=None):
        if False:
            while True:
                i = 10
        self.__proxy__ = model_proxy
        self.__name__ = name

    @classmethod
    def _native_name(cls):
        if False:
            i = 10
            return i + 15
        return 'activity_classifier'

    def __str__(self):
        if False:
            while True:
                i = 10
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the ActivityClassifier.\n        '
        return self.__repr__()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Print a string description of the model when the model name is entered\n        in the terminal.\n        '
        width = 40
        (sections, section_titles) = self._get_summary_struct()
        out = _tkutl._toolkit_repr_print(self, sections, section_titles, width=width)
        return out

    def _get_version(self):
        if False:
            print('Hello World!')
        return self._CPP_ACTIVITY_CLASSIFIER_VERSION

    def export_coreml(self, filename):
        if False:
            return 10
        '\n        Export the model in Core ML format.\n\n        Parameters\n        ----------\n        filename: str\n          A valid filename where the model can be saved.\n\n        Examples\n        --------\n        >>> model.export_coreml("MyModel.mlmodel")\n        '
        short_description = _coreml_utils._mlmodel_short_description('Activity classifier')
        additional_user_defined_metadata = _coreml_utils._get_tc_version_info()
        self.__proxy__.export_to_coreml(filename, short_description, additional_user_defined_metadata)

    def predict(self, dataset, output_type='class', output_frequency='per_row'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return predictions for ``dataset``, using the trained activity classifier.\n        Predictions can be generated as class labels, or as a probability\n        vector with probabilities for each class.\n\n        The activity classifier generates a single prediction for each\n        ``prediction_window`` rows in ``dataset``, per ``session_id``. The number\n        of these predictions is smaller than the length of ``dataset``. By default,\n        when ``output_frequency='per_row'``, each prediction is repeated ``prediction_window`` to return\n        a prediction for each row of ``dataset``. Use ``output_frequency=per_window`` to\n        get the unreplicated predictions.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        output_type : {'class', 'probability_vector'}, optional\n            Form of each prediction which is one of:\n\n            - 'probability_vector': Prediction probability associated with each\n              class as a vector. The probability of the first class (sorted\n              alphanumerically by name of the class in the training set) is in\n              position 0 of the vector, the second in position 1 and so on.\n            - 'class': Class prediction. This returns the class with maximum\n              probability.\n\n        output_frequency : {'per_row', 'per_window'}, optional\n            The frequency of the predictions which is one of:\n\n            - 'per_window': Return a single prediction for each\n              ``prediction_window`` rows in ``dataset`` per ``session_id``.\n            - 'per_row': Convenience option to make sure the number of\n              predictions match the number of rows in the dataset. Each\n              prediction from the model is repeated ``prediction_window``\n              times during that window.\n\n        Returns\n        -------\n        out : SArray | SFrame\n            If ``output_frequency`` is 'per_row' return an SArray with predictions\n            for each row in ``dataset``.\n            If ``output_frequency`` is 'per_window' return an SFrame with\n            predictions for ``prediction_window`` rows in ``dataset``.\n\n        See Also\n        ----------\n        create, evaluate, classify\n\n        Examples\n        --------\n\n        .. sourcecode:: python\n\n            # One prediction per row\n            >>> probability_predictions = model.predict(\n            ...     data, output_type='probability_vector', output_frequency='per_row')[:4]\n            >>> probability_predictions\n\n            dtype: array\n            Rows: 4\n            [array('d', [0.01857384294271469, 0.0348394550383091, 0.026018327102065086]),\n             array('d', [0.01857384294271469, 0.0348394550383091, 0.026018327102065086]),\n             array('d', [0.01857384294271469, 0.0348394550383091, 0.026018327102065086]),\n             array('d', [0.01857384294271469, 0.0348394550383091, 0.026018327102065086])]\n\n            # One prediction per window\n            >>> class_predictions = model.predict(\n            ...     data, output_type='class', output_frequency='per_window')\n            >>> class_predictions\n\n            +---------------+------------+-----+\n            | prediction_id | session_id |class|\n            +---------------+------------+-----+\n            |       0       |     3      |  5  |\n            |       1       |     3      |  5  |\n            |       2       |     3      |  5  |\n            |       3       |     3      |  5  |\n            |       4       |     3      |  5  |\n            |       5       |     3      |  5  |\n            |       6       |     3      |  5  |\n            |       7       |     3      |  4  |\n            |       8       |     3      |  4  |\n            |       9       |     3      |  4  |\n            |      ...      |    ...     | ... |\n            +---------------+------------+-----+\n        "
        _tkutl._check_categorical_option_type('output_frequency', output_frequency, ['per_window', 'per_row'])
        if output_frequency == 'per_row':
            return self.__proxy__.predict(dataset, output_type)
        elif output_frequency == 'per_window':
            return self.__proxy__.predict_per_window(dataset, output_type)

    def evaluate(self, dataset, metric='auto'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Evaluate the model by making predictions of target values and comparing\n        these to actual values.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the session_id, target and features used for model training.\n            Additional columns are ignored.\n\n        metric : str, optional\n            Name of the evaluation metric.  Possible values are:\n\n            - 'auto'             : Returns all available metrics.\n            - 'accuracy'         : Classification accuracy (micro average).\n            - 'auc'              : Area under the ROC curve (macro average)\n            - 'precision'        : Precision score (macro average)\n            - 'recall'           : Recall score (macro average)\n            - 'f1_score'         : F1 score (macro average)\n            - 'log_loss'         : Log loss\n            - 'confusion_matrix' : An SFrame with counts of possible\n                                   prediction/true label combinations.\n            - 'roc_curve'        : An SFrame containing information needed for an\n                                   ROC curve\n\n        Returns\n        -------\n        out : dict\n            Dictionary of evaluation results where the key is the name of the\n            evaluation metric (e.g. `accuracy`) and the value is the evaluation\n            score.\n\n        See Also\n        ----------\n        create, predict\n\n        Examples\n        ----------\n        .. sourcecode:: python\n\n          >>> results = model.evaluate(data)\n          >>> print results['accuracy']\n        "
        return self.__proxy__.evaluate(dataset, metric)

    def predict_topk(self, dataset, output_type='probability', k=3, output_frequency='per_row'):
        if False:
            i = 10
            return i + 15
        "\n        Return top-k predictions for the ``dataset``, using the trained model.\n        Predictions are returned as an SFrame with three columns: `prediction_id`,\n        `class`, and `probability`, or `rank`, depending on the ``output_type``\n        parameter.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the features and session id used for model training, but\n            does not require a target column. Additional columns are ignored.\n\n        output_type : {'probability', 'rank'}, optional\n            Choose the return type of the prediction:\n\n            - `probability`: Probability associated with each label in the prediction.\n            - `rank`       : Rank associated with each label in the prediction.\n\n        k : int, optional\n            Number of classes to return for each input example.\n\n        output_frequency : {'per_row', 'per_window'}, optional\n            The frequency of the predictions which is one of:\n\n            - 'per_row': Each prediction is returned ``prediction_window`` times.\n            - 'per_window': Return a single prediction for each\n              ``prediction_window`` rows in ``dataset`` per ``session_id``.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame with model predictions.\n\n        See Also\n        --------\n        predict, classify, evaluate\n\n        Examples\n        --------\n        >>> pred = m.predict_topk(validation_data, k=3)\n        >>> pred\n        +---------------+-------+-------------------+\n        |     row_id    | class |    probability    |\n        +---------------+-------+-------------------+\n        |       0       |   4   |   0.995623886585  |\n        |       0       |   9   |  0.0038311756216  |\n        |       0       |   7   | 0.000301006948575 |\n        |       1       |   1   |   0.928708016872  |\n        |       1       |   3   |  0.0440889261663  |\n        |       1       |   2   |  0.0176190119237  |\n        |       2       |   3   |   0.996967732906  |\n        |       2       |   2   |  0.00151345680933 |\n        |       2       |   7   | 0.000637513934635 |\n        |       3       |   1   |   0.998070061207  |\n        |      ...      |  ...  |        ...        |\n        +---------------+-------+-------------------+\n        "
        if not isinstance(k, int):
            raise TypeError('k must be of type int')
        _tkutl._numeric_param_check_range('k', k, 1, _six.MAXSIZE)
        return self.__proxy__.predict_topk(dataset, output_type, k, output_frequency)

    def classify(self, dataset, output_frequency='per_row'):
        if False:
            i = 10
            return i + 15
        "\n        Return a classification, for each ``prediction_window`` examples in the\n        ``dataset``, using the trained activity classification model. The output\n        SFrame contains predictions as both class labels as well as probabilities\n        that the predicted value is the associated label.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the features and session id used for model training, but\n            does not require a target column. Additional columns are ignored.\n\n        output_frequency : {'per_row', 'per_window'}, optional\n            The frequency of the predictions which is one of:\n\n            - 'per_row': Each prediction is returned ``prediction_window`` times.\n            - 'per_window': Return a single prediction for each\n              ``prediction_window`` rows in ``dataset`` per ``session_id``.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame with model predictions i.e class labels and probabilities.\n\n        See Also\n        ----------\n        create, evaluate, predict\n\n        Examples\n        ----------\n        >>> classes = model.classify(data)\n        "
        return self.__proxy__.classify(dataset, output_frequency)

    def _get_summary_struct(self):
        if False:
            return 10
        "\n        Returns a structured description of the model, including (where\n        relevant) the schema of the training data, description of the training\n        data, training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        model_fields = [('Number of examples', 'num_examples'), ('Number of sessions', 'num_sessions'), ('Number of classes', 'num_classes'), ('Number of feature columns', 'num_features'), ('Prediction window', 'prediction_window')]
        training_fields = [('Log-likelihood', 'training_log_loss'), ('Training time (sec)', 'training_time')]
        section_titles = ['Schema', 'Training summary']
        return ([model_fields, training_fields], section_titles)