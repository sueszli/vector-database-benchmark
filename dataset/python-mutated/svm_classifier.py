"""
Methods for creating and using an SVM model.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate.toolkits._supervised_learning as _sl
from turicreate.toolkits._supervised_learning import Classifier as _Classifier
from turicreate.toolkits._internal_utils import _toolkit_repr_print, _toolkit_get_topk_bottomk, _raise_error_evaluation_metric_is_valid, _check_categorical_option_type, _summarize_coefficients
_DEFAULT_SOLVER_OPTIONS = {'convergence_threshold': 0.01, 'max_iterations': 10, 'lbfgs_memory_level': 11}

def create(dataset, target, features=None, penalty=1.0, solver='auto', feature_rescaling=True, convergence_threshold=_DEFAULT_SOLVER_OPTIONS['convergence_threshold'], lbfgs_memory_level=_DEFAULT_SOLVER_OPTIONS['lbfgs_memory_level'], max_iterations=_DEFAULT_SOLVER_OPTIONS['max_iterations'], class_weights=None, validation_set='auto', verbose=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a :class:`~turicreate.svm_classifier.SVMClassifier` to predict the class of a binary\n    target variable based on a model of which side of a hyperplane the example\n    falls on. In addition to standard numeric and categorical types, features\n    can also be extracted automatically from list- or dictionary-type SFrame\n    columns.\n\n    This loss function for the SVM model is the sum of an L1 mis-classification\n    loss (multiplied by the 'penalty' term) and a l2-norm on the weight vectors.\n\n    Parameters\n    ----------\n    dataset : SFrame\n        Dataset for training the model.\n\n    target : string\n        Name of the column containing the target variable. The values in this\n        column must be of string or integer type. String target variables are\n        automatically mapped to integers in alphabetical order of the variable\n        values. For example, a target variable with 'cat' and 'dog' as possible\n        values is mapped to 0 and 1 respectively with 0 being the base class\n        and 1 being the reference class.\n\n    features : list[string], optional\n        Names of the columns containing features. 'None' (the default) indicates\n        that all columns except the target variable should be used as features.\n\n        The features are columns in the input SFrame that can be of the\n        following types:\n\n        - *Numeric*: values of numeric type integer or float.\n\n        - *Categorical*: values of type string.\n\n        - *Array*: list of numeric (integer or float) values. Each list element\n          is treated as a separate feature in the model.\n\n        - *Dictionary*: key-value pairs with numeric (integer or float) values\n          Each key of a dictionary is treated as a separate feature and the\n          value in the dictionary corresponds to the value of the feature.\n          Dictionaries are ideal for representing sparse data.\n\n        Columns of type *list* are not supported. Convert them to array in\n        case all entries in the list are of numeric types and separate them\n        out into different columns if they are of mixed type.\n\n    penalty : float, optional\n        Penalty term on the mis-classification loss of the model. The larger\n        this weight, the more the model coefficients shrink toward 0.  The\n        larger the penalty, the lower is the emphasis placed on misclassified\n        examples, and the classifier would spend more time maximizing the\n        margin for correctly classified examples. The default value is 1.0;\n        this parameter must be set to a value of at least 1e-10.\n\n\n    solver : string, optional\n        Name of the solver to be used to solve the problem. See the\n        references for more detail on each solver. Available solvers are:\n\n        - *auto (default)*: automatically chooses the best solver (from the ones\n          listed below) for the data and model parameters.\n        - *lbfgs*: lLimited memory BFGS (``lbfgs``) is a robust solver for wide\n          datasets(i.e datasets with many coefficients).\n\n        The solvers are all automatically tuned and the default options should\n        function well. See the solver options guide for setting additional\n        parameters for each of the solvers.\n\n    feature_rescaling : bool, default = true\n\n        Feature rescaling is an important pre-processing step that ensures\n        that all features are on the same scale. An l2-norm rescaling is\n        performed to make sure that all features are of the same norm. Categorical\n        features are also rescaled by rescaling the dummy variables that\n        are used to represent them. The coefficients are returned in original\n        scale of the problem.\n\n    convergence_threshold :\n\n        Convergence is tested using variation in the training objective. The\n        variation in the training objective is calculated using the difference\n        between the objective values between two steps. Consider reducing this\n        below the default value (0.01) for a more accurately trained model.\n        Beware of overfitting (i.e a model that works well only on the training\n        data) if this parameter is set to a very low value.\n\n    max_iterations : int, optional\n\n        The maximum number of allowed passes through the data. More passes over\n        the data can result in a more accurately trained model. Consider\n        increasing this (the default value is 10) if the training accuracy is\n        low and the *Grad-Norm* in the display is large.\n\n    lbfgs_memory_level : int, optional\n\n        The L-BFGS algorithm keeps track of gradient information from the\n        previous ``lbfgs_memory_level`` iterations. The storage requirement for\n        each of these gradients is the ``num_coefficients`` in the problem.\n        Increasing the ``lbfgs_memory_level`` can help improve the quality of\n        the model trained. Setting this to more than ``max_iterations`` has the\n        same effect as setting it to ``max_iterations``.\n\n    class_weights : {dict, `auto`}, optional\n\n        Weights the examples in the training data according to the given class\n        weights. If set to `None`, all classes are supposed to have weight one. The\n        `auto` mode set the class weight to be inversely proportional to number of\n        examples in the training data with the given class.\n\n    validation_set : SFrame, optional\n\n        A dataset for monitoring the model's generalization performance.\n        For each row of the progress table, the chosen metrics are computed\n        for both the provided training dataset and the validation_set. The\n        format of this SFrame must be the same as the training set.\n        By default this argument is set to 'auto' and a validation set is\n        automatically sampled and used for progress printing. If\n        validation_set is set to None, then no additional metrics\n        are computed. The default value is 'auto'.\n\n    verbose : bool, optional\n        If True, print progress updates.\n\n    Returns\n    -------\n    out : SVMClassifier\n        A trained model of type\n        :class:`~turicreate.svm_classifier.SVMClassifier`.\n\n    See Also\n    --------\n    SVMClassifier\n\n    Notes\n    -----\n    - Categorical variables are encoded by creating dummy variables. For\n      a variable with :math:`K` categories, the encoding creates :math:`K-1`\n      dummy variables, while the first category encountered in the data is used\n      as the baseline.\n\n    - For prediction and evaluation of SVM models with sparse dictionary\n      inputs, new keys/columns that were not seen during training are silently\n      ignored.\n\n    - The penalty parameter is analogous to the 'C' term in the C-SVM. See the\n      reference on training SVMs for more details.\n\n    - Any 'None' values in the data will result in an error being thrown.\n\n    - A constant term of '1' is automatically added for the model intercept to\n      model the bias term.\n\n    - Note that the hinge loss is approximated by the scaled logistic loss\n      function. (See user guide for details)\n\n    References\n    ----------\n    - `Wikipedia - Support Vector Machines\n      <http://en.wikipedia.org/wiki/svm>`_\n\n    - Zhang et al. - Modified Logistic Regression: An Approximation to\n      SVM and its Applications in Large-Scale Text Categorization (ICML 2003)\n\n\n    Examples\n    --------\n\n    Given an :class:`~turicreate.SFrame` ``sf``, a list of feature columns\n    [``feature_1`` ... ``feature_K``], and a target column ``target`` with 0 and\n    1 values, create a\n    :class:`~turicreate.svm.SVMClassifier` as follows:\n\n    >>> data =  turicreate.SFrame('https://static.turi.com/datasets/regression/houses.csv')\n    >>> data['is_expensive'] = data['price'] > 30000\n    >>> model = turicreate.svm_classifier.create(data, 'is_expensive')\n    "
    model_name = 'classifier_svm'
    solver = solver.lower()
    model = _sl.create(dataset, target, model_name, features=features, validation_set=validation_set, verbose=verbose, penalty=penalty, feature_rescaling=feature_rescaling, convergence_threshold=convergence_threshold, lbfgs_memory_level=lbfgs_memory_level, max_iterations=max_iterations, class_weights=class_weights)
    return SVMClassifier(model.__proxy__)

class SVMClassifier(_Classifier):
    """
    Support Vector Machines can be used to predict binary target variable using
    several feature variables.

    The :py:class:`~turicreate.svm.SVMClassifier` model predicts a binary target
    variable given one or more feature variables. In an SVM model, the examples
    are represented as points in space, mapped so that the examples from the
    two classes being classified are divided by linear separator.

    Given a set of features :math:`x_i`, and a label :math:`y_i \\in \\{0,1\\}`,
    SVM minimizes the loss function:

        .. math::
          f_i(\\theta) =  \\max(1 - \\theta^T x, 0)

    An intercept term is added by appending a column of 1's to the features.
    Regularization is often required to prevent over fitting by penalizing
    models with extreme parameter values. The composite objective being
    optimized for is the following:

        .. math::
           \\min_{\\theta} \\sum_{i = 1}^{n} f_i(\\theta) + \\lambda ||\\theta||^{2}_{2}

    where :math:`\\lambda` is the ``penalty`` parameter.

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.svm_classifier.create` to create an instance of this model.
    Additional details on parameter options and code samples are available in
    the documentation for the create function.

    Examples
    --------

    .. sourcecode:: python

        # Load the data (From an S3 bucket)
        >>> import turicreate as tc
        >>> data =  tc.SFrame('https://static.turi.com/datasets/regression/houses.csv')

        # Make sure the target is binary 0/1
        >>> data['is_expensive'] = data['price'] > 30000

        # Make a logistic regression model
        >>> model = tc.svm_classifier.create(data, target='is_expensive'
                                        , features=['bath', 'bedroom', 'size'])

        # Extract the coefficients
        >>> coefficients = model.coefficients # an SFrame

        # Make predictions (as margins, or class)
        >>> predictions = model.progressredict(data)    # Predicts 0/1
        >>> predictions = model.progressredict(data, output_type='margin')

        # Evaluate the model
        >>> results = model.progressvaluate(data)               # a dictionary

    See Also
    --------
    create

    """

    def __init__(self, model_proxy):
        if False:
            while True:
                i = 10
        '__init__(self)'
        self.__proxy__ = model_proxy
        self.__name__ = self.__class__._native_name()

    @classmethod
    def _native_name(cls):
        if False:
            while True:
                i = 10
        return 'classifier_svm'

    def __str__(self):
        if False:
            return 10
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the model.\n        '
        return self.__repr__()

    def _get_summary_struct(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns a structured description of the model, including (where relevant)\n        the schema of the training data, description of the training data,\n        training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        model_fields = [('Number of coefficients', 'num_coefficients'), ('Number of examples', 'num_examples'), ('Number of classes', 'num_classes'), ('Number of feature columns', 'num_features'), ('Number of unpacked features', 'num_unpacked_features')]
        hyperparam_fields = [('Mis-classification penalty', 'penalty')]
        solver_fields = [('Solver', 'solver'), ('Solver iterations', 'training_iterations'), ('Solver status', 'training_solver_status'), ('Training time (sec)', 'training_time')]
        training_fields = [('Train Loss', 'training_loss')]
        coefs = self.coefficients
        (top_coefs, bottom_coefs) = _toolkit_get_topk_bottomk(coefs, k=5)
        (coefs_list, titles_list) = _summarize_coefficients(top_coefs, bottom_coefs)
        return ([model_fields, hyperparam_fields, solver_fields, training_fields] + coefs_list, ['Schema', 'Hyperparameters', 'Training Summary', 'Settings'] + titles_list)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Print a string description of the model, when the model name is entered\n        in the terminal.\n        '
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=30)

    def export_coreml(self, filename):
        if False:
            for i in range(10):
                print('nop')
        '\n        Export the model in Core ML format.\n\n        Parameters\n        ----------\n        filename: str\n          A valid filename where the model can be saved.\n\n        Examples\n        --------\n        >>> model.export_coreml("MyModel.mlmodel")\n        '
        from turicreate.extensions import _linear_svm_export_as_model_asset
        from turicreate.toolkits import _coreml_utils
        display_name = 'svm classifier'
        short_description = _coreml_utils._mlmodel_short_description(display_name)
        context = {'class': self.__class__.__name__, 'short_description': short_description}
        context['user_defined'] = _coreml_utils._get_model_metadata(self.__class__.__name__, None)
        _linear_svm_export_as_model_asset(self.__proxy__, filename, context)

    def _get(self, field):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the value of a given field. The list of all queryable fields is\n        detailed below, and can be obtained programmatically with the\n        :func:`~turicreate.svm.SVMClassifier._list_fields` method.\n\n\n        +------------------------+-------------------------------------------------------------+\n        |      Field             | Description                                                 |\n        +========================+=============================================================+\n        | coefficients           | Classifier coefficients                                     |\n        +------------------------+-------------------------------------------------------------+\n        | convergence_threshold  | Desired solver accuracy                                     |\n        +------------------------+-------------------------------------------------------------+\n        | feature_rescaling      | Bool indicating l2-rescaling of features                    |\n        +------------------------+---------+---------------------------------------------------+\n        | features               | Feature column names                                        |\n        +------------------------+-------------------------------------------------------------+\n        | lbfgs_memory_level     | Number of updates to store (lbfgs only)                     |\n        +------------------------+-------------------------------------------------------------+\n        | max_iterations         | Maximum number of solver iterations                         |\n        +------------------------+-------------------------------------------------------------+\n        | num_coefficients       | Number of coefficients in the model                         |\n        +------------------------+-------------------------------------------------------------+\n        | num_examples           | Number of examples used for training                        |\n        +------------------------+-------------------------------------------------------------+\n        | num_features           | Number of dataset columns used for training                 |\n        +------------------------+-------------------------------------------------------------+\n        | num_unpacked_features  | Number of features (including expanded list/dict features)  |\n        +------------------------+-------------------------------------------------------------+\n        | penalty                | Misclassification penalty term                              |\n        +------------------------+-------------------------------------------------------------+\n        | solver                 | Type of solver                                              |\n        +------------------------+-------------------------------------------------------------+\n        | target                 | Target column name                                          |\n        +------------------------+-------------------------------------------------------------+\n        | training_iterations    | Number of solver iterations                                 |\n        +------------------------+-------------------------------------------------------------+\n        | training_loss          | Maximized Log-likelihood                                    |\n        +------------------------+-------------------------------------------------------------+\n        | training_solver_status | Solver status after training                                |\n        +------------------------+-------------------------------------------------------------+\n        | training_time          | Training time (excludes preprocessing)                      |\n        +------------------------+-------------------------------------------------------------+\n        | unpacked_features      | Feature names (including expanded list/dict features)       |\n        +------------------------+-------------------------------------------------------------+\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out\n            Value of the requested fields.\n        '
        return super(_Classifier, self)._get(field)

    def predict(self, dataset, output_type='class', missing_value_action='auto'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return predictions for ``dataset``, using the trained logistic\n        regression model. Predictions can be generated as class labels (0 or\n        1), or margins (i.e. the distance of the observations from the hyperplane\n        separating the classes). By default, the predict method returns class\n        labels.\n\n        For each new example in ``dataset``, the margin---also known as the\n        linear predictor---is the inner product of the example and the model\n        coefficients plus the intercept term. Predicted classes are obtained by\n        thresholding the margins at 0.\n\n        Parameters\n        ----------\n        dataset : SFrame | dict\n            Dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        output_type : {'margin', 'class'}, optional\n            Form of the predictions which are one of:\n\n            - 'margin': Distance of the observations from the hyperplane\n              separating the classes.\n            - 'class': Class prediction.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. This can be\n            one of:\n\n            - 'auto': Default to 'impute'\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error' : Do not proceed with prediction and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : SArray\n            An SArray with model predictions.\n\n        See Also\n        ----------\n        create, evaluate, classify\n\n        Examples\n        ----------\n        >>> data =  turicreate.SFrame('https://static.turi.com/datasets/regression/houses.csv')\n\n        >>> data['is_expensive'] = data['price'] > 30000\n        >>> model = turicreate.svm_classifier.create(data,\n                                  target='is_expensive',\n                                  features=['bath', 'bedroom', 'size'])\n\n        >>> class_predictions = model.progressredict(data)\n        >>> margin_predictions = model.progressredict(data, output_type='margin')\n\n        "
        _check_categorical_option_type('output_type', output_type, ['class', 'margin'])
        return super(_Classifier, self).predict(dataset, output_type=output_type, missing_value_action=missing_value_action)

    def classify(self, dataset, missing_value_action='auto'):
        if False:
            while True:
                i = 10
        "\n        Return a classification, for each example in the ``dataset``, using the\n        trained SVM model. The output SFrame contains predictions\n        as class labels (0 or 1) associated with the the example.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. This can be\n            one of:\n\n            - 'auto': Default to 'impute'\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error' : Do not proceed with prediction and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame with model predictions i.e class labels.\n\n        See Also\n        ----------\n        create, evaluate, predict\n\n        Examples\n        ----------\n        >>> data =  turicreate.SFrame('https://static.turi.com/datasets/regression/houses.csv')\n\n        >>> data['is_expensive'] = data['price'] > 30000\n        >>> model = turicreate.svm_classifier.create(data, target='is_expensive',\n                                      features=['bath', 'bedroom', 'size'])\n\n        >>> classes = model.classify(data)\n\n        "
        return super(SVMClassifier, self).classify(dataset, missing_value_action=missing_value_action)

    def evaluate(self, dataset, metric='auto', missing_value_action='auto', with_predictions=False):
        if False:
            i = 10
            return i + 15
        "\n        Evaluate the model by making predictions of target values and comparing\n        these to actual values.\n\n        Two metrics are used to evaluate SVM. The confusion table contains the\n        cross-tabulation of actual and predicted classes for the target\n        variable. classifier accuracy is the fraction of examples whose\n        predicted and actual classes match.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the target and features used for model training. Additional\n            columns are ignored.\n\n        metric : str, optional\n            Name of the evaluation metric.  Possible values are:\n\n            - 'auto'             : Returns all available metrics.\n            - 'accuracy '        : Classification accuracy (micro average).\n            - 'precision'        : Precision score (micro average)\n            - 'recall'           : Recall score (micro average)\n            - 'f1_score'         : F1 score (micro average)\n            - 'confusion_matrix' : An SFrame with counts of possible prediction/true\n                                   label combinations.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. This can be\n            one of:\n\n            - 'auto': Default to 'impute'\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : dict\n            Dictionary of evaluation results where the key is the name of the\n            evaluation metric (e.g. `accuracy`) and the value is the evaluation\n            score.\n\n        See Also\n        ----------\n        create, predict, classify\n\n        Examples\n        ----------\n        .. sourcecode:: python\n\n          >>> data =  turicreate.SFrame('https://static.turi.com/datasets/regression/houses.csv')\n\n          >>> data['is_expensive'] = data['price'] > 30000\n          >>> model = turicreate.svm_classifier.create(data,           ...                                        target='is_expensive',           ...                                        features=['bath', 'bedroom', 'size'])\n          >>> results = model.progressvaluate(data)\n          >>> print results['accuracy']\n        "
        _raise_error_evaluation_metric_is_valid(metric, ['auto', 'accuracy', 'confusion_matrix', 'precision', 'recall', 'f1_score'])
        return super(_Classifier, self).evaluate(dataset, missing_value_action=missing_value_action, metric=metric, with_predictions=with_predictions)