"""
This package contains the Random Forest model class and the create function.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from turicreate.toolkits._supervised_learning import SupervisedLearningModel as _SupervisedLearningModel
import turicreate.toolkits._supervised_learning as _sl
from turicreate.toolkits._internal_utils import _toolkit_repr_print
from turicreate.toolkits._internal_utils import _raise_error_evaluation_metric_is_valid
from turicreate.toolkits._tree_model_mixin import TreeModelMixin as _TreeModelMixin
import logging as _logging
from turicreate.util import _make_internal_url
_RANDOM_FOREST_MODEL_PARAMS_KEYS = ['max_depth', 'min_child_weight', 'min_loss_reduction', 'row_subsample']
_RANDOM_FOREST_TRAINING_PARAMS_KEYS = ['objective', 'training_time', 'training_error', 'validation_error', 'evaluation_metric']
_RANDOM_FOREST_TRAINING_DATA_PARAMS_KEYS = ['target', 'features', 'num_features', 'num_examples', 'num_validation_examples']

class RandomForestRegression(_SupervisedLearningModel, _TreeModelMixin):
    """
    Encapsulates random forest models for regression tasks.

    The prediction is based on a collection of base learners, `regression trees
    <http://en.wikipedia.org/wiki/Decision_tree_learning>`_ and combines them
    through a technique called `random forest <http://en.wikipedia.org/wiki/Random_forest>`_.

    Different from linear models, e.g. linear regression,
    the random forests are able to model non-linear interactions
    between the features and the target using decision trees as the subroutine.
    It is good for handling numerical features and categorical features with
    tens of categories but is less suitable for highly sparse features such as
    text data.

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.random_forest_regression.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

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
            while True:
                i = 10
        return 'random_forest_regression'

    def __str__(self):
        if False:
            print('Hello World!')
        '\n        Return a string description of the model to the ``print`` method.\n\n        Returns\n        -------\n        out : string\n            A description of the model.\n        '
        return self.__repr__()

    def _get_summary_struct(self):
        if False:
            return 10
        "\n        Returns a structured description of the model, including (where relevant)\n        the schema of the training data, description of the training data,\n        training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        data_fields = [('Number of examples', 'num_examples'), ('Number of feature columns', 'num_features'), ('Number of unpacked features', 'num_unpacked_features')]
        training_fields = [('Maximum number of iterations', 'max_iterations'), ('Number of trees', 'num_trees'), ('Max tree depth', 'max_depth'), ('Train RMSE', 'training_rmse'), ('Validation RMSE', 'validation_rmse'), ('Training time (sec)', 'training_time')]
        return ([data_fields, training_fields], ['Schema', 'Settings'])

    def __repr__(self):
        if False:
            print('Hello World!')
        '\n        Print a string description of the model, when the model name is entered\n        in the terminal.\n        '
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=30)

    def _get(self, field):
        if False:
            i = 10
            return i + 15
        '\n        Get the value of a given field. The list of all queryable fields is\n        detailed below, and can be obtained programmatically using the\n        :func:`~turicreate.random_forest_regression._list_fields` method.\n\n        +-------------------------+--------------------------------------------------------------------------------+\n        | Field                   | Description                                                                    |\n        +=========================+================================================================================+\n        | column_subsample        | Percentage of the columns for training each individual tree                    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | features                | Names of the feature columns                                                   |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | history                 | A list of string for the training history                                      |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | max_depth               | The maximum depth of individual trees                                          |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | max_iterations          | Number of iterations, equals to the number of trees                            |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | min_child_weight        | Minimum weight required on the leave nodes                                     |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | min_loss_reduction      | Minimum loss reduction required for splitting a node                           |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_features            | Number of features in the model                                                |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_unpacked_features   | Number of features in the model (including unpacked dict/list type columns)    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_examples            | Number of training examples                                                    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | num_validation_examples | Number of validation examples                                                  |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | row_subsample           | Percentage of the rows for training each individual tree                       |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | target                  | Name of the target column                                                      |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | training_error          | Error on training data                                                         |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | training_time           | Time spent on training the model in seconds                                    |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | trees_json              | Tree encoded using JSON                                                        |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | validation_error        | Error on validation data                                                       |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | unpacked_features       | Feature names (including expanded list/dict features)                          |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | random_seed             | Seed for row and column subselection                                           |\n        +-------------------------+--------------------------------------------------------------------------------+\n        | metric                  | Performance metric(s) that are tracked during training                         |\n        +-------------------------+--------------------------------------------------------------------------------+\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out : [various]\n            The current value of the requested field.\n        '
        return super(RandomForestRegression, self)._get(field)

    def evaluate(self, dataset, metric='auto', missing_value_action='auto'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Evaluate the model on the given dataset.\n\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset in the same format used for training. The columns names and\n            types of the dataset must be the same as that used in training.\n\n        metric : str, optional\n            Name of the evaluation metric.  Possible values are:\n            'auto'      : Compute all metrics.\n            'rmse'      : Rooted mean squared error.\n            'max_error' : Maximum error.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. Can be\n            one of:\n\n            - 'auto': By default the model will treat missing value as is.\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : dict\n            A dictionary containing the evaluation result.\n\n        See Also\n        --------\n        create, predict\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n          >>> results = model.evaluate(test_data, 'rmse')\n\n        "
        _raise_error_evaluation_metric_is_valid(metric, ['auto', 'rmse', 'max_error'])
        return super(RandomForestRegression, self).evaluate(dataset, missing_value_action=missing_value_action, metric=metric)

    def predict(self, dataset, missing_value_action='auto'):
        if False:
            while True:
                i = 10
        "\n        Predict the target column of the given dataset.\n\n        The target column is provided during\n        :func:`~turicreate.random_forest_regression.create`. If the target column is in the\n        `dataset` it will be ignored.\n\n        Parameters\n        ----------\n        dataset : SFrame\n          A dataset that has the same columns that were used during training.\n          If the target column exists in ``dataset`` it will be ignored\n          while making predictions.\n\n        missing_value_action : str, optional\n            Action to perform when missing values are encountered. Can be\n            one of:\n\n            - 'auto': By default the model will treat missing value as is.\n            - 'impute': Proceed with evaluation by filling in the missing\n              values with the mean of the training data. Missing\n              values are also imputed if an entire column of data is\n              missing during evaluation.\n            - 'error': Do not proceed with evaluation and terminate with\n              an error message.\n\n        Returns\n        -------\n        out : SArray\n           Predicted target value for each example (i.e. row) in the dataset.\n\n        See Also\n        ----------\n        create, predict\n\n        Examples\n        --------\n        >>> m.predict(testdata)\n        "
        return super(RandomForestRegression, self).predict(dataset, output_type='margin', missing_value_action=missing_value_action)

    def export_coreml(self, filename):
        if False:
            while True:
                i = 10
        '\n        Export the model in Core ML format.\n\n        Parameters\n        ----------\n        filename: str\n          A valid filename where the model can be saved.\n\n        Examples\n        --------\n        >>> model.export_coreml("MyModel.mlmodel")\n        '
        from turicreate.toolkits import _coreml_utils
        display_name = 'random forest regression'
        short_description = _coreml_utils._mlmodel_short_description(display_name)
        context = {'mode': 'regression', 'model_type': 'random_forest', 'class': self.__class__.__name__, 'short_description': short_description}
        self._export_coreml_impl(filename, context)

def create(dataset, target, features=None, max_iterations=10, validation_set='auto', verbose=True, random_seed=None, metric='auto', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a :class:`~turicreate.random_forest_regression.RandomForestRegression` to predict\n    a scalar target variable using one or more features. In addition to standard\n    numeric and categorical types, features can also be extracted automatically\n    from list- or dictionary-type SFrame columns.\n\n\n    Parameters\n    ----------\n    dataset : SFrame\n        A training dataset containing feature columns and a target column.\n        Only numerical typed (int, float) target column is allowed.\n\n    target : str\n        The name of the column in ``dataset`` that is the prediction target.\n        This column must have a numeric type.\n\n    features : list[str], optional\n        A list of columns names of features used for training the model.\n        Defaults to None, using all columns.\n\n    max_iterations : int, optional\n        The number of iterations to perform.\n\n    max_depth : float, optional\n        Maximum depth of a tree.\n\n    min_loss_reduction : float, optional (non-negative)\n        Minimum loss reduction required to make a further partition/split a\n        node during the tree learning phase. Larger (more positive) values\n        can help prevent overfitting by avoiding splits that do not\n        sufficiently reduce the loss function.\n\n    min_child_weight : float, optional (non-negative)\n        Controls the minimum weight of each leaf node. Larger values result in\n        more conservative tree learning and help prevent overfitting.\n        Formally, this is minimum sum of instance weights (hessians) in each\n        node. If the tree learning algorithm results in a leaf node with the\n        sum of instance weights less than `min_child_weight`, tree building\n        will terminate.\n\n    row_subsample : float, optional\n        Subsample the ratio of the training set in each iteration of tree\n        construction.  This is called the bagging trick and usually can help\n        prevent overfitting.  Setting it to 0.5 means that model randomly\n        collected half of the examples (rows) to grow each tree.\n\n    column_subsample : float, optional\n        Subsample ratio of the columns in each iteration of tree\n        construction.  Like row_subsample, this also usually can help\n        prevent overfitting.  Setting it to 0.5 means that model randomly\n        collected half of the columns to grow each tree.\n\n    validation_set : SFrame, optional\n        A dataset for monitoring the model's generalization performance.\n        For each row of the progress table, the chosen metrics are computed\n        for both the provided training dataset and the validation_set. The\n        format of this SFrame must be the same as the training set.\n        By default this argument is set to 'auto' and a validation set is\n        automatically sampled and used for progress printing. If\n        validation_set is set to None, then no additional metrics\n        are computed. This is computed once per full iteration. Large\n        differences in model accuracy between the training data and validation\n        data is indicative of overfitting. The default value is 'auto'.\n\n\n    verbose : boolean, optional\n        If True, print progress information during training.\n\n    random_seed : int, optional\n        Seeds random operations such as column and row subsampling, such that\n        results are reproducible.\n\n    metric : str or list[str], optional\n        Performance metric(s) that are tracked during training. When specified,\n        the progress table will display the tracked metric(s) on training and\n        validation set.\n        Supported metrics are: {'rmse', 'max_error'}\n\n    kwargs : dict, optional\n        Additional arguments for training the model.\n\n        - ``model_checkpoint_path`` : str, default None\n            If specified, checkpoint the model training to the given path every n iterations,\n            where n is specified by ``model_checkpoint_interval``.\n            For instance, if `model_checkpoint_interval` is 5, and `model_checkpoint_path` is\n            set to ``/tmp/model_tmp``, the checkpoints will be saved into\n            ``/tmp/model_tmp/model_checkpoint_5``, ``/tmp/model_tmp/model_checkpoint_10``, ... etc.\n            Training can be resumed by setting ``resume_from_checkpoint`` to one of these checkpoints.\n\n        - ``model_checkpoint_interval`` : int, default 5\n            If model_check_point_path is specified,\n            save the model to the given path every n iterations.\n\n        - ``resume_from_checkpoint`` : str, default None\n            Continues training from a model checkpoint. The model must take\n            exact the same training data as the checkpointed model.\n\n    Returns\n    -------\n      out : RandomForestRegression\n          A trained random forest model for regression tasks.\n\n    References\n    ----------\n    - `Trevor Hastie's slides on Boosted Trees and Random Forest\n      <http://jessica2.msri.org/attachments/10778/10778-boost.pdf>`_\n\n    See Also\n    --------\n    RandomForestRegression, turicreate.linear_regression.LinearRegression, turicreate.regression.create\n\n    Examples\n    --------\n\n    Setup the data:\n\n    >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'\n    >>> data = turicreate.SFrame.read_csv(url)\n    >>> data['label'] = data['label'] == 'p'\n\n    Split the data into training and test data:\n\n    >>> train, test = data.random_split(0.8)\n\n    Create the model:\n\n    >>> model = turicreate.random_forest_regression.create(train, target='label')\n\n    Make predictions and evaluate the model:\n\n    >>> predictions = model.predict(test)\n    >>> results = model.evaluate(test)\n\n    "
    if random_seed is not None:
        kwargs['random_seed'] = random_seed
    if 'model_checkpoint_path' in kwargs:
        kwargs['model_checkpoint_path'] = _make_internal_url(kwargs['model_checkpoint_path'])
    if 'resume_from_checkpoint' in kwargs:
        kwargs['resume_from_checkpoint'] = _make_internal_url(kwargs['resume_from_checkpoint'])
    if 'num_trees' in kwargs:
        logger = _logging.getLogger(__name__)
        logger.warning('The `num_trees` keyword argument is deprecated. Please use the `max_iterations` argument instead. Any value provided for `num_trees` will be used in place of `max_iterations`.')
        max_iterations = kwargs['num_trees']
        del kwargs['num_trees']
    model = _sl.create(dataset=dataset, target=target, features=features, model_name='random_forest_regression', max_iterations=max_iterations, validation_set=validation_set, verbose=verbose, metric=metric, **kwargs)
    return RandomForestRegression(model.__proxy__)