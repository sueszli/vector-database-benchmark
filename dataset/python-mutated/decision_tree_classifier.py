"""
This package contains the decision tree model class and the create function.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from turicreate.toolkits._supervised_learning import Classifier as _Classifier
import turicreate.toolkits._supervised_learning as _sl
from turicreate.toolkits._internal_utils import _toolkit_repr_print
from turicreate.toolkits._internal_utils import _raise_error_evaluation_metric_is_valid
from turicreate.toolkits._internal_utils import _raise_error_if_not_sframe
from turicreate.toolkits._internal_utils import _check_categorical_option_type
from turicreate.toolkits._tree_model_mixin import TreeModelMixin as _TreeModelMixin
_DECISION_TREE_MODEL_PARAMS_KEYS = ['max_depth', 'min_child_weight', 'min_loss_reduction']
_DECISION_TREE_TRAINING_PARAMS_KEYS = ['objective', 'training_time', 'training_error', 'validation_error', 'evaluation_metric']
_DECISION_TREE_TRAINING_DATA_PARAMS_KEYS = ['target', 'features', 'num_features', 'num_examples', 'num_validation_examples']
__doc_string_context = "\n      >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'\n      >>> data = turicreate.SFrame.read_csv(url)\n\n      >>> train, test = data.random_split(0.8)\n      >>> model = turicreate.decision_tree_classifier.create(train, target='label')\n"

class DecisionTreeClassifier(_Classifier, _TreeModelMixin):
    """
    Special case of gradient boosted trees with the number of trees set to 1.

    The decision tree model can be used as a classifier for predictive tasks.
    Different from linear models like logistic regression or SVM, this
    algorithm can model non-linear interactions between the features and the
    target. This model is suitable for handling numerical features and
    categorical features with tens of categories but is less suitable for
    highly sparse features (text data), or with categorical variables that
    encode a large number of categories.

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.decision_tree_classifier.create` to create an instance of
    this model.  Additional details on parameter options and code samples are
    available in the documentation for the create function.

    See Also
    --------
    create

    """

    def __init__(self, proxy):
        if False:
            for i in range(10):
                print('nop')
        '__init__(self)'
        self.__proxy__ = proxy
        self.__name__ = self.__class__._native_name()

    @classmethod
    def _native_name(cls):
        if False:
            while True:
                i = 10
        return 'decision_tree_classifier'

    def __str__(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the model.\n        '
        return self.__repr__()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print a string description of the model, when the model name is entered\n        in the terminal.\n        '
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=30)

    def _get(self, field):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the value of a given field. The following table describes each\n        of the fields below.\n\n        +-------------------------+--------------------------------------------------------------------------------+\n        | Field                   | Description                                                                    |\n        +=========================+================================================================================+\n        | column_subsample        | Percentage of the columns for training each individual tree                    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | features                | Names of the feature columns                                                   |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | max_depth               | The maximum depth of the individual decision trees                             |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | min_child_weight        | Minimum weight assigned to leaf nodes                                          |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | min_loss_reduction      | Minimum loss reduction required for splitting a node                           |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_features            | Number of feature columns in the model                                         |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_unpacked_features   | Number of features in the model (including unpacked dict/list type columns)    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_examples            | Number of training examples                                                    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_validation_examples | Number of validation examples                                                  |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | target                  | Name of the target column                                                      |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | training_accuracy       | Classification accuracy measured on the training data                          |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | training_time           | Time spent on training the model in seconds                                    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | trees_json              | Tree encoded using JSON                                                        |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | validation_accuracy     | Classification accuracy measured on the validation set                         |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | unpacked_features       | Feature names (including expanded list/dict features)                          |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | metric                  | Performance metric(s) that are tracked during training                         |\n        +-------------------------+--------------------------------------------------------------------------------+\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out : [various]\n            The current value of the requested field.\n        '
        return super(_Classifier, self)._get(field)

    def evaluate(self, dataset, metric='auto', missing_value_action='auto'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Evaluate the model by making predictions of target values and comparing\n        these to actual values.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the target and features used for model training. Additional\n            columns are ignored.\n\n        metric : str, optional\n            Name of the evaluation metric.  Possible values are:\n\n            - 'auto'             : Returns all available metrics.\n            - 'accuracy'         : Classification accuracy (micro average).\n            - 'auc'              : Area under the ROC curve (macro average)\n            - 'precision'        : Precision score (macro average)\n            - 'recall'           : Recall score (macro average)\n            - 'f1_score'         : F1 score (macro average)\n            - 'log_loss'         : Log loss\n            - 'confusion_matrix' : An SFrame with counts of possible prediction/true label combinations.\n            - 'roc_curve'        : An SFrame containing information needed for an ROC curve\n\n            For more flexibility in calculating evaluation metrics, use the\n            :class:`~turicreate.evaluation` module.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. This can be\n            one of:\n\n            - 'auto': Default to 'impute'\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : dict\n            Dictionary of evaluation results where the key is the name of the\n            evaluation metric (e.g. `accuracy`) and the value is the evaluation\n            score.\n\n        See Also\n        ----------\n        create, predict, classify\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n          >>> results = model.evaluate(test_data)\n          >>> results = model.evaluate(test_data, metric='accuracy')\n          >>> results = model.evaluate(test_data, metric='confusion_matrix')\n\n        "
        _raise_error_evaluation_metric_is_valid(metric, ['auto', 'accuracy', 'confusion_matrix', 'roc_curve', 'auc', 'log_loss', 'precision', 'recall', 'f1_score'])
        return super(_Classifier, self).evaluate(dataset, missing_value_action=missing_value_action, metric=metric)

    def predict(self, dataset, output_type='class', missing_value_action='auto'):
        if False:
            return 10
        "\n        A flexible and advanced prediction API.\n\n        The target column is provided during\n        :func:`~turicreate.decision_tree.create`. If the target column is in the\n        `dataset` it will be ignored.\n\n        Parameters\n        ----------\n        dataset : SFrame\n          A dataset that has the same columns that were used during training.\n          If the target column exists in ``dataset`` it will be ignored\n          while making predictions.\n\n        output_type : {'probability', 'margin', 'class', 'probability_vector'}, optional.\n            Form of the predictions which are one of:\n\n            - 'probability': Prediction probability associated with the True\n               class (not applicable for multi-class classification)\n            - 'margin': Margin associated with the prediction (not applicable\n              for multi-class classification)\n            - 'probability_vector': Prediction probability associated with each\n              class as a vector. The probability of the first class (sorted\n              alphanumerically by name of the class in the training set) is in\n              position 0 of the vector, the second in position 1 and so on.\n            - 'class': Class prediction. For multi-class classification, this\n               returns the class with maximum probability.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. Can be\n            one of:\n\n            - 'auto': By default the model will treat missing value as is.\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n\n        Returns\n        -------\n        out : SArray\n           Predicted target value for each example (i.e. row) in the dataset.\n\n        See Also\n        ----------\n        create, evaluate, classify\n\n        Examples\n        --------\n        >>> m.predict(testdata)\n        >>> m.predict(testdata, output_type='probability')\n        >>> m.predict(testdata, output_type='margin')\n        "
        _check_categorical_option_type('output_type', output_type, ['class', 'margin', 'probability', 'probability_vector'])
        return super(_Classifier, self).predict(dataset, output_type=output_type, missing_value_action=missing_value_action)

    def predict_topk(self, dataset, output_type='probability', k=3, missing_value_action='auto'):
        if False:
            i = 10
            return i + 15
        "\n        Return top-k predictions for the ``dataset``, using the trained model.\n        Predictions are returned as an SFrame with three columns: `id`,\n        `class`, and `probability`, `margin`,  or `rank`, depending on the ``output_type``\n        parameter. Input dataset size must be the same as for training of the model.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            A dataset that has the same columns that were used during training.\n            If the target column exists in ``dataset`` it will be ignored\n            while making predictions.\n\n        output_type : {'probability', 'rank', 'margin'}, optional\n            Choose the return type of the prediction:\n\n            - `probability`: Probability associated with each label in the prediction.\n            - `rank`       : Rank associated with each label in the prediction.\n            - `margin`     : Margin associated with each label in the prediction.\n\n        k : int, optional\n            Number of classes to return for each input example.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. Can be\n            one of:\n\n            - 'auto': By default the model will treat missing value as is.\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame with model predictions.\n\n        See Also\n        --------\n        predict, classify, evaluate\n\n        Examples\n        --------\n        >>> pred = m.predict_topk(validation_data, k=3)\n        >>> pred\n        +--------+-------+-------------------+\n        | id     | class |   probability     |\n        +--------+-------+-------------------+\n        |   0    |   4   |   0.995623886585  |\n        |   0    |   9   |  0.0038311756216  |\n        |   0    |   7   | 0.000301006948575 |\n        |   1    |   1   |   0.928708016872  |\n        |   1    |   3   |  0.0440889261663  |\n        |   1    |   2   |  0.0176190119237  |\n        |   2    |   3   |   0.996967732906  |\n        |   2    |   2   |  0.00151345680933 |\n        |   2    |   7   | 0.000637513934635 |\n        |   3    |   1   |   0.998070061207  |\n        |  ...   |  ...  |        ...        |\n        +--------+-------+-------------------+\n        [35688 rows x 3 columns]\n        "
        _check_categorical_option_type('output_type', output_type, ['rank', 'margin', 'probability'])
        if missing_value_action == 'auto':
            missing_value_action = _sl.select_default_missing_value_policy(self, 'predict')
        if isinstance(dataset, list):
            return self.__proxy__.fast_predict_topk(dataset, missing_value_action, output_type, k)
        if isinstance(dataset, dict):
            return self.__proxy__.fast_predict_topk([dataset], missing_value_action, output_type, k)
        _raise_error_if_not_sframe(dataset, 'dataset')
        return self.__proxy__.predict_topk(dataset, missing_value_action, output_type, k)

    def classify(self, dataset, missing_value_action='auto'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a classification, for each example in the ``dataset``, using the\n        trained model. The output SFrame contains predictions as class labels\n        (0 or 1) and probabilities associated with the the example.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. Can be\n            one of:\n\n            - 'auto': By default the model will treat missing value as is.\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame with model predictions i.e class labels and probabilities\n            associated with each of the class labels.\n\n        See Also\n        ----------\n        create, evaluate, predict\n\n        Examples\n        ----------\n        >>> data =  turicreate.SFrame('https://static.turi.com/datasets/regression/houses.csv')\n\n        >>> data['is_expensive'] = data['price'] > 30000\n        >>> model = turicreate.decision_tree_classifier.create(data,\n        >>>                                                  target='is_expensive',\n        >>>                                                  features=['bath', 'bedroom', 'size'])\n\n        >>> classes = model.classify(data)\n\n        "
        return super(DecisionTreeClassifier, self).classify(dataset, missing_value_action=missing_value_action)

    def export_coreml(self, filename):
        if False:
            while True:
                i = 10
        '\n        Export the model in Core ML format.\n\n        Parameters\n        ----------\n        filename: str\n          A valid filename where the model can be saved.\n\n        Examples\n        --------\n        >>> model.export_coreml("MyModel.mlmodel")\n        '
        from turicreate.toolkits import _coreml_utils
        display_name = 'decision tree classifier'
        short_description = _coreml_utils._mlmodel_short_description(display_name)
        context = {'mode': 'classification', 'model_type': 'decision_tree', 'class': self.__class__.__name__, 'short_description': short_description}
        self._export_coreml_impl(filename, context)

def create(dataset, target, features=None, validation_set='auto', class_weights=None, max_depth=6, min_loss_reduction=0.0, min_child_weight=0.1, verbose=True, random_seed=None, metric='auto', **kwargs):
    if False:
        while True:
            i = 10
    "\n    Create a (binary or multi-class) classifier model of type\n    :class:`~turicreate.decision_tree_classifier.DecisionTreeClassifier`. This\n    algorithm is a special case of boosted trees classifier with the number\n    of trees set to 1.\n\n    Parameters\n    ----------\n    dataset : SFrame\n        A training dataset containing feature columns and a target column.\n\n    target : str\n        Name of the column containing the target variable. The values in this\n        column must be of string or integer type.  String target variables are\n        automatically mapped to integers in alphabetical order of the variable values.\n        For example, a target variable with 'cat', 'dog', and 'foosa' as possible\n        values is mapped to 0, 1, and, 2 respectively.\n\n    features : list[str], optional\n        A list of columns names of features used for training the model.\n        Defaults to None, which uses all columns in the SFrame ``dataset``\n        excepting the target column..\n\n    validation_set : SFrame, optional\n        A dataset for monitoring the model's generalization performance.\n        For each row of the progress table, the chosen metrics are computed\n        for both the provided training dataset and the validation_set. The\n        format of this SFrame must be the same as the training set.\n        By default this argument is set to 'auto' and a validation set is\n        automatically sampled and used for progress printing. If\n        validation_set is set to None, then no additional metrics\n        are computed. This is computed once per full iteration. Large\n        differences in model accuracy between the training data and validation\n        data is indicative of overfitting. The default value is 'auto'.\n\n    class_weights : {dict, `auto`}, optional\n\n        Weights the examples in the training data according to the given class\n        weights. If provided, the dictionary must contain a key for each class\n        label. The value can be any positive number greater than 1e-20. Weights\n        are interpreted as relative to each other. So setting the weights to be\n        2.0 for the positive class and 1.0 for the negative class has the same\n        effect as setting them to be 20.0 and 10.0, respectively. If set to\n        `None`, all classes are taken to have weight 1.0. The `auto` mode sets\n        the class weight to be inversely proportional to the number of examples\n        in the training data with the given class.\n\n    max_depth : float, optional\n        Maximum depth of a tree. Must be at least 1.\n\n    min_loss_reduction : float, optional (non-negative)\n        Minimum loss reduction required to make a further partition/split a\n        node during the tree learning phase. Larger (more positive) values\n        can help prevent overfitting by avoiding splits that do not\n        sufficiently reduce the loss function.\n\n    min_child_weight : float, optional (non-negative)\n        Controls the minimum weight of each leaf node. Larger values result in\n        more conservative tree learning and help prevent overfitting.\n        Formally, this is minimum sum of instance weights (hessians) in each\n        node. If the tree learning algorithm results in a leaf node with the\n        sum of instance weights less than `min_child_weight`, tree building\n        will terminate.\n\n    verbose : boolean, optional\n        Print progress information during training (if set to true).\n\n    random_seed : int, optional\n        Seeds random operations such as column and row subsampling, such that\n        results are reproducible.\n\n    metric : str or list[str], optional\n        Performance metric(s) that are tracked during training. When specified,\n        the progress table will display the tracked metric(s) on training and\n        validation set.\n        Supported metrics are: {'accuracy', 'auc', 'log_loss'}\n\n    Returns\n    -------\n      out : DecisionTreeClassifier\n          A trained decision tree model for classifications tasks.\n\n    References\n    ----------\n\n    - `Wikipedia - Gradient tree boosting\n      <http://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting>`_\n    - `Trevor Hastie's slides on Boosted Trees and Random Forest\n      <http://jessica2.msri.org/attachments/10778/10778-boost.pdf>`_\n\n    See Also\n    --------\n    turicreate.logistic_classifier.LogisticClassifier, turicreate.svm_classifier.SVMClassifier\n\n    Examples\n    --------\n\n    .. sourcecode:: python\n\n      >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'\n      >>> data = turicreate.SFrame.read_csv(url)\n\n      >>> train, test = data.random_split(0.8)\n      >>> model = turicreate.decision_tree_classifier.create(train, target='label')\n\n      >>> predictions = model.classify(test)\n      >>> results = model.evaluate(test)\n    "
    if random_seed is not None:
        kwargs['random_seed'] = random_seed
    model = _sl.create(dataset=dataset, target=target, features=features, model_name='decision_tree_classifier', validation_set=validation_set, class_weights=class_weights, max_depth=max_depth, min_loss_reduction=min_loss_reduction, min_child_weight=min_child_weight, verbose=verbose, metric=metric, **kwargs)
    return DecisionTreeClassifier(model.__proxy__)