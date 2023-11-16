"""
This package contains the Random Forest model class and the create function.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from turicreate.toolkits._supervised_learning import Classifier as _Classifier
import turicreate.toolkits._supervised_learning as _sl
from turicreate.toolkits._internal_utils import _toolkit_repr_print
from turicreate.toolkits._internal_utils import _raise_error_evaluation_metric_is_valid
from turicreate.toolkits._internal_utils import _check_categorical_option_type
from turicreate.toolkits._tree_model_mixin import TreeModelMixin as _TreeModelMixin
from turicreate.util import _make_internal_url
__doc_string_context = "\n      >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'\n      >>> data = turicreate.SFrame.read_csv(url)\n\n      >>> train, test = data.random_split(0.8)\n      >>> model = turicreate.random_forest_classifier.create(train, target='label')\n"

class RandomForestClassifier(_Classifier, _TreeModelMixin):
    """
    The random forest model can be used as a classifier for predictive
    tasks.

    The prediction is based on a collection of base learners i.e
    `decision tree classifiers <http://en.wikipedia.org/wiki/Decision_tree_learning>`_
    and combines them through a technique called `random forest
    <http://en.wikipedia.org/wiki/Random_forest>`_.

    Different from linear models like logistic regression or SVM, gradient
    boosted trees can model non-linear interactions between the features and the
    target. This model is suitable for handling numerical features and
    categorical features with tens of categories but is less suitable for highly
    sparse features (text data), or with categorical variables that encode a
    large number of categories.

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.random_forest_classifier.create` to create an instance of
    this model.
    Additional details on parameter options and code samples are available in
    the documentation for the create function.

    See Also
    --------
    create

    """

    def __init__(self, proxy):
        if False:
            while True:
                i = 10
        '__init__(self)'
        self.__proxy__ = proxy
        self.__name__ = self.__class__._native_name()

    @classmethod
    def _native_name(cls):
        if False:
            for i in range(10):
                print('nop')
        return 'random_forest_classifier'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the model.\n        '
        return self.__repr__()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Print a string description of the model, when the model name is entered\n        in the terminal.\n        '
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=30)

    def _get(self, field):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the value of a given field. The following table describes each\n        of the fields below.\n\n        +-------------------------+--------------------------------------------------------------------------------+\n        | Field                   | Description                                                                    |\n        +=========================+================================================================================+\n        | column_subsample        | Percentage of the columns for training each individual tree                    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | features                | Names of the feature columns                                                   |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | max_depth               | The maximum depth of the individual decision trees                             |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | max_iterations          | Maximum number of iterations for training (one tree is trained per iteration)  |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | min_child_weight        | Minimum weight assigned to leaf nodes                                          |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | min_loss_reduction      | Minimum loss reduction required for splitting a node                           |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_features            | Number of feature columns in the model                                         |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_unpacked_features   | Number of features in the model (including unpacked dict/list type columns)    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_examples            | Number of training examples                                                    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_trees               | Number of trees created during training.                                       |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_validation_examples | Number of validation examples                                                  |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | row_subsample           | Percentage of the rows sampled for training each individual tree               |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | target                  | Name of the target column                                                      |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | training_accuracy       | Classification accuracy measured on the training data                          |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | training_time           | Time spent on training the model in seconds                                    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | trees_json              | Tree encoded using JSON                                                        |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | validation_accuracy     | Classification accuracy measured on the validation set                         |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | unpacked_features       | Feature names (including expanded list/dict features)                          |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | random_seed             | Seed for row and column subselection                                           |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | metric                  | Performance metric(s) that are tracked during training                         |\n        +-------------------------+--------------------------------------------------------------------------------+\n\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out : [various]\n            The current value of the requested field.\n        '
        return super(_Classifier, self)._get(field)

    def evaluate(self, dataset, metric='auto', missing_value_action='auto'):
        if False:
            while True:
                i = 10
        "\n        Evaluate the model by making predictions of target values and comparing\n        these to actual values.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the target and features used for model training. Additional\n            columns are ignored.\n\n        metric : str, optional\n            Name of the evaluation metric.  Possible values are:\n\n            - 'auto'             : Returns all available metrics.\n            - 'accuracy'         : Classification accuracy (micro average).\n            - 'auc'              : Area under the ROC curve (macro average)\n            - 'precision'        : Precision score (macro average)\n            - 'recall'           : Recall score (macro average)\n            - 'f1_score'         : F1 score (macro average)\n            - 'log_loss'         : Log loss\n            - 'confusion_matrix' : An SFrame with counts of possible prediction/true label combinations.\n            - 'roc_curve'        : An SFrame containing information needed for an ROC curve\n\n            For more flexibility in calculating evaluation metrics, use the\n            :class:`~turicreate.evaluation` module.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. This can be\n            one of:\n\n            - 'auto': Default to 'impute'\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : dict\n            Dictionary of evaluation results where the key is the name of the\n            evaluation metric (e.g. `accuracy`) and the value is the evaluation\n            score.\n\n        See Also\n        ----------\n        create, predict, classify\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n          >>> results = model.evaluate(test_data)\n          >>> results = model.evaluate(test_data, metric='accuracy')\n          >>> results = model.evaluate(test_data, metric='confusion_matrix')\n\n        "
        _raise_error_evaluation_metric_is_valid(metric, ['auto', 'accuracy', 'confusion_matrix', 'roc_curve', 'auc', 'log_loss', 'precision', 'recall', 'f1_score'])
        return super(_Classifier, self).evaluate(dataset, missing_value_action=missing_value_action, metric=metric)

    def predict(self, dataset, output_type='class', missing_value_action='auto'):
        if False:
            while True:
                i = 10
        "\n        A flexible and advanced prediction API.\n\n        The target column is provided during\n        :func:`~turicreate.random_forest.create`. If the target column is in the\n        `dataset` it will be ignored.\n\n        Parameters\n        ----------\n        dataset : SFrame\n          A dataset that has the same columns that were used during training.\n          If the target column exists in ``dataset`` it will be ignored\n          while making predictions.\n\n        output_type : {'probability', 'margin', 'class', 'probability_vector'}, optional.\n            Form of the predictions which are one of:\n\n            - 'probability': Prediction probability associated with the True\n               class (not applicable for multi-class classification)\n            - 'margin': Margin associated with the prediction (not applicable\n              for multi-class classification)\n            - 'probability_vector': Prediction probability associated with each\n              class as a vector. The probability of the first class (sorted\n              alphanumerically by name of the class in the training set) is in\n              position 0 of the vector, the second in position 1 and so on.\n            - 'class': Class prediction. For multi-class classification, this\n               returns the class with maximum probability.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. Can be\n            one of:\n\n            - 'auto': By default the model will treat missing value as is.\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : SArray\n           Predicted target value for each example (i.e. row) in the dataset.\n\n        See Also\n        ----------\n        create, evaluate, classify\n\n        Examples\n        --------\n        >>> m.predict(testdata)\n        >>> m.predict(testdata, output_type='probability')\n        >>> m.predict(testdata, output_type='margin')\n        "
        _check_categorical_option_type('output_type', output_type, ['class', 'margin', 'probability', 'probability_vector'])
        return super(_Classifier, self).predict(dataset, output_type=output_type, missing_value_action=missing_value_action)

    def predict_topk(self, dataset, output_type='probability', k=3, missing_value_action='auto'):
        if False:
            while True:
                i = 10
        "\n        Return top-k predictions for the ``dataset``, using the trained model.\n        Predictions are returned as an SFrame with three columns: `id`,\n        `class`, and `probability`, `margin`,  or `rank`, depending on the ``output_type``\n        parameter. Input dataset size must be the same as for training of the model.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            A dataset that has the same columns that were used during training.\n            If the target column exists in ``dataset`` it will be ignored\n            while making predictions.\n\n        output_type : {'probability', 'rank', 'margin'}, optional\n            Choose the return type of the prediction:\n\n            - `probability`: Probability associated with each label in the prediction.\n            - `rank`       : Rank associated with each label in the prediction.\n            - `margin`     : Margin associated with each label in the prediction.\n\n        k : int, optional\n            Number of classes to return for each input example.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. Can be\n            one of:\n\n            - 'auto': By default the model will treat missing value as is.\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame with model predictions.\n\n        See Also\n        --------\n        predict, classify, evaluate\n\n        Examples\n        --------\n        >>> pred = m.predict_topk(validation_data, k=3)\n        >>> pred\n        +--------+-------+-------------------+\n        | id     | class |   probability     |\n        +--------+-------+-------------------+\n        |   0    |   4   |   0.995623886585  |\n        |   0    |   9   |  0.0038311756216  |\n        |   0    |   7   | 0.000301006948575 |\n        |   1    |   1   |   0.928708016872  |\n        |   1    |   3   |  0.0440889261663  |\n        |   1    |   2   |  0.0176190119237  |\n        |   2    |   3   |   0.996967732906  |\n        |   2    |   2   |  0.00151345680933 |\n        |   2    |   7   | 0.000637513934635 |\n        |   3    |   1   |   0.998070061207  |\n        |  ...   |  ...  |        ...        |\n        +--------+-------+-------------------+\n        [35688 rows x 3 columns]\n        "
        _check_categorical_option_type('output_type', output_type, ['rank', 'margin', 'probability'])
        if missing_value_action == 'auto':
            missing_value_action = _sl.select_default_missing_value_policy(self, 'predict')
        if isinstance(dataset, list):
            return self.__proxy__.fast_predict_topk(dataset, missing_value_action, output_type, k)
        if isinstance(dataset, dict):
            return self.__proxy__.fast_predict_topk([dataset], missing_value_action, output_type, k)
        return self.__proxy__.predict_topk(dataset, missing_value_action, output_type, k)

    def classify(self, dataset, missing_value_action='auto'):
        if False:
            print('Hello World!')
        "\n        Return a classification, for each example in the ``dataset``, using the\n        trained random forest model. The output SFrame contains predictions\n        as class labels (0 or 1) and probabilities associated with the the example.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. Can be\n            one of:\n\n            - 'auto': By default the model will treat missing value as is.\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame with model predictions i.e class labels and probabilities\n            associated with each of the class labels.\n\n        See Also\n        ----------\n        create, evaluate, predict\n\n        Examples\n        ----------\n        >>> data =  turicreate.SFrame('https://static.turi.com/datasets/regression/houses.csv')\n\n        >>> data['is_expensive'] = data['price'] > 30000\n        >>> model = turicreate.random_forest_classifier.create(data,\n        >>>                                                  target='is_expensive',\n        >>>                                                  features=['bath', 'bedroom', 'size'])\n        >>> classes = model.classify(data)\n        "
        return super(RandomForestClassifier, self).classify(dataset, missing_value_action=missing_value_action)

    def export_coreml(self, filename):
        if False:
            while True:
                i = 10
        '\n        Export the model in Core ML format.\n\n        Parameters\n        ----------\n        filename: str\n          A valid filename where the model can be saved.\n\n        Examples\n        --------\n        >>> model.export_coreml("MyModel.mlmodel")\n        '
        from turicreate.toolkits import _coreml_utils
        display_name = 'random forest classifier'
        short_description = _coreml_utils._mlmodel_short_description(display_name)
        context = {'mode': 'classification', 'model_type': 'random_forest', 'class': self.__class__.__name__, 'short_description': short_description}
        self._export_coreml_impl(filename, context)

def create(dataset, target, features=None, max_iterations=10, validation_set='auto', verbose=True, class_weights=None, random_seed=None, metric='auto', **kwargs):
    if False:
        while True:
            i = 10
    "\n    Create a (binary or multi-class) classifier model of type\n    :class:`~turicreate.random_forest_classifier.RandomForestClassifier` using\n    an ensemble of decision trees trained on subsets of the data.\n\n    Parameters\n    ----------\n    dataset : SFrame\n        A training dataset containing feature columns and a target column.\n\n    target : str\n        Name of the column containing the target variable. The values in this\n        column must be of string or integer type.  String target variables are\n        automatically mapped to integers in alphabetical order of the variable values.\n        For example, a target variable with 'cat', 'dog', and 'foosa' as possible\n        values is mapped to 0, 1, and, 2 respectively.\n\n    features : list[str], optional\n        A list of columns names of features used for training the model.\n        Defaults to None, which uses all columns in the SFrame ``dataset``\n        excepting the target column..\n\n    max_iterations : int, optional\n        The maximum number of iterations to perform. For multi-class\n        classification with K classes, each iteration will create K-1 trees.\n\n    max_depth : float, optional\n        Maximum depth of a tree.\n\n    class_weights : {dict, `auto`}, optional\n        Weights the examples in the training data according to the given class\n        weights. If set to `None`, all classes are supposed to have weight one. The\n        `auto` mode set the class weight to be inversely proportional to number of\n        examples in the training data with the given class.\n\n    min_loss_reduction : float, optional (non-negative)\n        Minimum loss reduction required to make a further partition on a\n        leaf node of the tree. The larger it is, the more conservative the\n        algorithm will be. Must be non-negative.\n\n    min_child_weight : float, optional (non-negative)\n        Controls the minimum weight of each leaf node. Larger values result in\n        more conservative tree learning and help prevent overfitting.\n        Formally, this is minimum sum of instance weights (hessians) in each\n        node. If the tree learning algorithm results in a leaf node with the\n        sum of instance weights less than `min_child_weight`, tree building\n        will terminate.\n\n    row_subsample : float, optional\n        Subsample the ratio of the training set in each iteration of tree\n        construction.  This is called the bagging trick and can usually help\n        prevent overfitting.  Setting this to a value of 0.5 results in the\n        model randomly sampling half of the examples (rows) to grow each tree.\n\n    column_subsample : float, optional\n        Subsample ratio of the columns in each iteration of tree\n        construction.  Like row_subsample, this can also help prevent\n        model overfitting.  Setting this to a value of 0.5 results in the\n        model randomly sampling half of the columns to grow each tree.\n\n    validation_set : SFrame, optional\n        A dataset for monitoring the model's generalization performance.\n        For each row of the progress table, the chosen metrics are computed\n        for both the provided training dataset and the validation_set. The\n        format of this SFrame must be the same as the training set.\n        By default this argument is set to 'auto' and a validation set is\n        automatically sampled and used for progress printing. If\n        validation_set is set to None, then no additional metrics\n        are computed. This is computed once per full iteration. Large\n        differences in model accuracy between the training data and validation\n        data is indicative of overfitting. The default value is 'auto'.\n\n    verbose : boolean, optional\n        Print progress information during training (if set to true).\n\n    random_seed : int, optional\n        Seeds random operations such as column and row subsampling, such that\n        results are reproducible.\n\n    metric : str or list[str], optional\n        Performance metric(s) that are tracked during training. When specified,\n        the progress table will display the tracked metric(s) on training and\n        validation set.\n        Supported metrics are: {'accuracy', 'auc', 'log_loss'}\n\n    kwargs : dict, optional\n        Additional arguments for training the model.\n\n        - ``model_checkpoint_path`` : str, default None\n            If specified, checkpoint the model training to the given path every n iterations,\n            where n is specified by ``model_checkpoint_interval``.\n            For instance, if `model_checkpoint_interval` is 5, and `model_checkpoint_path` is\n            set to ``/tmp/model_tmp``, the checkpoints will be saved into\n            ``/tmp/model_tmp/model_checkpoint_5``, ``/tmp/model_tmp/model_checkpoint_10``, ... etc.\n            Training can be resumed by setting ``resume_from_checkpoint`` to one of these checkpoints.\n\n        - ``model_checkpoint_interval`` : int, default 5\n            If model_check_point_path is specified,\n            save the model to the given path every n iterations.\n\n        - ``resume_from_checkpoint`` : str, default None\n            Continues training from a model checkpoint. The model must take\n            exact the same training data as the checkpointed model.\n\n\n    Returns\n    -------\n      out : RandomForestClassifier\n          A trained random forest model for classification tasks.\n\n    References\n    ----------\n    - `Trevor Hastie's slides on Boosted Trees and Random Forest\n      <http://jessica2.msri.org/attachments/10778/10778-boost.pdf>`_\n\n    See Also\n    --------\n    RandomForestClassifier, turicreate.logistic_classifier.LogisticClassifier, turicreate.svm_classifier.SVMClassifier\n\n\n    Examples\n    --------\n\n    .. sourcecode:: python\n\n      >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'\n      >>> data = turicreate.SFrame.read_csv(url)\n\n      >>> train, test = data.random_split(0.8)\n      >>> model = turicreate.random_forest_classifier.create(train, target='label')\n\n      >>> predictions = model.classify(test)\n      >>> results = model.evaluate(test)\n    "
    if random_seed is not None:
        kwargs['random_seed'] = random_seed
    if 'model_checkpoint_path' in kwargs:
        kwargs['model_checkpoint_path'] = _make_internal_url(kwargs['model_checkpoint_path'])
    if 'resume_from_checkpoint' in kwargs:
        kwargs['resume_from_checkpoint'] = _make_internal_url(kwargs['resume_from_checkpoint'])
    model = _sl.create(dataset=dataset, target=target, features=features, model_name='random_forest_classifier', max_iterations=max_iterations, validation_set=validation_set, class_weights=class_weights, verbose=verbose, metric=metric, **kwargs)
    return RandomForestClassifier(model.__proxy__)