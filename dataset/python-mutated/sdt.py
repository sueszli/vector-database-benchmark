from __future__ import absolute_import, division, print_function, unicode_literals
from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric

class H2OSdtEstimator(H2OEstimator):
    """
    Single Decision Tree

    """
    algo = 'sdt'
    supervised_learning = True

    def __init__(self, model_id=None, training_frame=None, ignored_columns=None, ignore_const_cols=True, categorical_encoding='auto', response_column=None, max_depth=0):
        if False:
            print('Hello World!')
        '\n        :param model_id: Destination id for this model; auto-generated if not specified.\n               Defaults to ``None``.\n        :type model_id: Union[None, str, H2OEstimator], optional\n        :param training_frame: Id of the training data frame.\n               Defaults to ``None``.\n        :type training_frame: Union[None, str, H2OFrame], optional\n        :param ignored_columns: Names of columns to ignore for training.\n               Defaults to ``None``.\n        :type ignored_columns: List[str], optional\n        :param ignore_const_cols: Ignore constant columns.\n               Defaults to ``True``.\n        :type ignore_const_cols: bool\n        :param categorical_encoding: Encoding scheme for categorical features\n               Defaults to ``"auto"``.\n        :type categorical_encoding: Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",\n               "sort_by_response", "enum_limited"]\n        :param response_column: Response variable column.\n               Defaults to ``None``.\n        :type response_column: str, optional\n        :param max_depth: Max depth of tree.\n               Defaults to ``0``.\n        :type max_depth: int\n        '
        super(H2OSdtEstimator, self).__init__()
        self._parms = {}
        self._id = self._parms['model_id'] = model_id
        self.training_frame = training_frame
        self.ignored_columns = ignored_columns
        self.ignore_const_cols = ignore_const_cols
        self.categorical_encoding = categorical_encoding
        self.response_column = response_column
        self.max_depth = max_depth

    @property
    def training_frame(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Id of the training data frame.\n\n        Type: ``Union[None, str, H2OFrame]``.\n        '
        return self._parms.get('training_frame')

    @training_frame.setter
    def training_frame(self, training_frame):
        if False:
            i = 10
            return i + 15
        self._parms['training_frame'] = H2OFrame._validate(training_frame, 'training_frame')

    @property
    def ignored_columns(self):
        if False:
            while True:
                i = 10
        '\n        Names of columns to ignore for training.\n\n        Type: ``List[str]``.\n        '
        return self._parms.get('ignored_columns')

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        if False:
            while True:
                i = 10
        assert_is_type(ignored_columns, None, [str])
        self._parms['ignored_columns'] = ignored_columns

    @property
    def ignore_const_cols(self):
        if False:
            print('Hello World!')
        '\n        Ignore constant columns.\n\n        Type: ``bool``, defaults to ``True``.\n        '
        return self._parms.get('ignore_const_cols')

    @ignore_const_cols.setter
    def ignore_const_cols(self, ignore_const_cols):
        if False:
            return 10
        assert_is_type(ignore_const_cols, None, bool)
        self._parms['ignore_const_cols'] = ignore_const_cols

    @property
    def categorical_encoding(self):
        if False:
            return 10
        '\n        Encoding scheme for categorical features\n\n        Type: ``Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",\n        "sort_by_response", "enum_limited"]``, defaults to ``"auto"``.\n        '
        return self._parms.get('categorical_encoding')

    @categorical_encoding.setter
    def categorical_encoding(self, categorical_encoding):
        if False:
            i = 10
            return i + 15
        assert_is_type(categorical_encoding, None, Enum('auto', 'enum', 'one_hot_internal', 'one_hot_explicit', 'binary', 'eigen', 'label_encoder', 'sort_by_response', 'enum_limited'))
        self._parms['categorical_encoding'] = categorical_encoding

    @property
    def response_column(self):
        if False:
            i = 10
            return i + 15
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
    def max_depth(self):
        if False:
            print('Hello World!')
        '\n        Max depth of tree.\n\n        Type: ``int``, defaults to ``0``.\n        '
        return self._parms.get('max_depth')

    @max_depth.setter
    def max_depth(self, max_depth):
        if False:
            i = 10
            return i + 15
        assert_is_type(max_depth, None, int)
        self._parms['max_depth'] = max_depth