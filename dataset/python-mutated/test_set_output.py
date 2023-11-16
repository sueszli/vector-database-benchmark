from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.utils._set_output import _get_output_config, _safe_set_output, _SetOutputMixin, _wrap_in_pandas_container
from sklearn.utils.fixes import CSR_CONTAINERS

def test__wrap_in_pandas_container_dense():
    if False:
        return 10
    'Check _wrap_in_pandas_container for dense data.'
    pd = pytest.importorskip('pandas')
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    columns = np.asarray(['f0', 'f1', 'f2'], dtype=object)
    index = np.asarray([0, 1])
    dense_named = _wrap_in_pandas_container(X, columns=lambda : columns, index=index)
    assert isinstance(dense_named, pd.DataFrame)
    assert_array_equal(dense_named.columns, columns)
    assert_array_equal(dense_named.index, index)

def test__wrap_in_pandas_container_dense_update_columns_and_index():
    if False:
        for i in range(10):
            print('nop')
    'Check that _wrap_in_pandas_container overrides columns and index.'
    pd = pytest.importorskip('pandas')
    X_df = pd.DataFrame([[1, 0, 3], [0, 0, 1]], columns=['a', 'b', 'c'])
    new_columns = np.asarray(['f0', 'f1', 'f2'], dtype=object)
    new_index = [10, 12]
    new_df = _wrap_in_pandas_container(X_df, columns=new_columns, index=new_index)
    assert_array_equal(new_df.columns, new_columns)
    assert_array_equal(new_df.index, X_df.index)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test__wrap_in_pandas_container_error_validation(csr_container):
    if False:
        while True:
            i = 10
    'Check errors in _wrap_in_pandas_container.'
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    X_csr = csr_container(X)
    match = 'The transformer outputs a scipy sparse matrix.'
    with pytest.raises(ValueError, match=match):
        _wrap_in_pandas_container(X_csr, columns=['a', 'b', 'c'])

class EstimatorWithoutSetOutputAndWithoutTransform:
    pass

class EstimatorNoSetOutputWithTransform:

    def transform(self, X, y=None):
        if False:
            i = 10
            return i + 15
        return X

class EstimatorWithSetOutput(_SetOutputMixin):

    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        if False:
            return 10
        return X

    def get_feature_names_out(self, input_features=None):
        if False:
            i = 10
            return i + 15
        return np.asarray([f'X{i}' for i in range(self.n_features_in_)], dtype=object)

def test__safe_set_output():
    if False:
        i = 10
        return i + 15
    'Check _safe_set_output works as expected.'
    est = EstimatorWithoutSetOutputAndWithoutTransform()
    _safe_set_output(est, transform='pandas')
    est = EstimatorNoSetOutputWithTransform()
    with pytest.raises(ValueError, match='Unable to configure output'):
        _safe_set_output(est, transform='pandas')
    est = EstimatorWithSetOutput().fit(np.asarray([[1, 2, 3]]))
    _safe_set_output(est, transform='pandas')
    config = _get_output_config('transform', est)
    assert config['dense'] == 'pandas'
    _safe_set_output(est, transform='default')
    config = _get_output_config('transform', est)
    assert config['dense'] == 'default'
    _safe_set_output(est, transform=None)
    config = _get_output_config('transform', est)
    assert config['dense'] == 'default'

class EstimatorNoSetOutputWithTransformNoFeatureNamesOut(_SetOutputMixin):

    def transform(self, X, y=None):
        if False:
            print('Hello World!')
        return X

def test_set_output_mixin():
    if False:
        i = 10
        return i + 15
    'Estimator without get_feature_names_out does not define `set_output`.'
    est = EstimatorNoSetOutputWithTransformNoFeatureNamesOut()
    assert not hasattr(est, 'set_output')

def test__safe_set_output_error():
    if False:
        for i in range(10):
            print('nop')
    'Check transform with invalid config.'
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput()
    _safe_set_output(est, transform='bad')
    msg = "output config must be 'default'"
    with pytest.raises(ValueError, match=msg):
        est.transform(X)

def test_set_output_method():
    if False:
        while True:
            i = 10
    'Check that the output is pandas.'
    pd = pytest.importorskip('pandas')
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput().fit(X)
    est2 = est.set_output(transform=None)
    assert est2 is est
    X_trans_np = est2.transform(X)
    assert isinstance(X_trans_np, np.ndarray)
    est.set_output(transform='pandas')
    X_trans_pd = est.transform(X)
    assert isinstance(X_trans_pd, pd.DataFrame)

def test_set_output_method_error():
    if False:
        return 10
    'Check transform fails with invalid transform.'
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput().fit(X)
    est.set_output(transform='bad')
    msg = "output config must be 'default'"
    with pytest.raises(ValueError, match=msg):
        est.transform(X)

def test__get_output_config():
    if False:
        i = 10
        return i + 15
    'Check _get_output_config works as expected.'
    global_config = get_config()['transform_output']
    config = _get_output_config('transform')
    assert config['dense'] == global_config
    with config_context(transform_output='pandas'):
        config = _get_output_config('transform')
        assert config['dense'] == 'pandas'
        est = EstimatorNoSetOutputWithTransform()
        config = _get_output_config('transform', est)
        assert config['dense'] == 'pandas'
        est = EstimatorWithSetOutput()
        config = _get_output_config('transform', est)
        assert config['dense'] == 'pandas'
        est.set_output(transform='default')
        config = _get_output_config('transform', est)
        assert config['dense'] == 'default'
    est.set_output(transform='pandas')
    config = _get_output_config('transform', est)
    assert config['dense'] == 'pandas'

class EstimatorWithSetOutputNoAutoWrap(_SetOutputMixin, auto_wrap_output_keys=None):

    def transform(self, X, y=None):
        if False:
            while True:
                i = 10
        return X

def test_get_output_auto_wrap_false():
    if False:
        while True:
            i = 10
    'Check that auto_wrap_output_keys=None does not wrap.'
    est = EstimatorWithSetOutputNoAutoWrap()
    assert not hasattr(est, 'set_output')
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    assert X is est.transform(X)

def test_auto_wrap_output_keys_errors_with_incorrect_input():
    if False:
        i = 10
        return i + 15
    msg = 'auto_wrap_output_keys must be None or a tuple of keys.'
    with pytest.raises(ValueError, match=msg):

        class BadEstimator(_SetOutputMixin, auto_wrap_output_keys='bad_parameter'):
            pass

class AnotherMixin:

    def __init_subclass__(cls, custom_parameter, **kwargs):
        if False:
            return 10
        super().__init_subclass__(**kwargs)
        cls.custom_parameter = custom_parameter

def test_set_output_mixin_custom_mixin():
    if False:
        while True:
            i = 10
    'Check that multiple init_subclasses passes parameters up.'

    class BothMixinEstimator(_SetOutputMixin, AnotherMixin, custom_parameter=123):

        def transform(self, X, y=None):
            if False:
                while True:
                    i = 10
            return X

        def get_feature_names_out(self, input_features=None):
            if False:
                i = 10
                return i + 15
            return input_features
    est = BothMixinEstimator()
    assert est.custom_parameter == 123
    assert hasattr(est, 'set_output')

def test__wrap_in_pandas_container_column_errors():
    if False:
        while True:
            i = 10
    'If a callable `columns` errors, it has the same semantics as columns=None.'
    pd = pytest.importorskip('pandas')

    def get_columns():
        if False:
            while True:
                i = 10
        raise ValueError('No feature names defined')
    X_df = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [3, 4, 5]})
    X_wrapped = _wrap_in_pandas_container(X_df, columns=get_columns)
    assert_array_equal(X_wrapped.columns, X_df.columns)
    X_np = np.asarray([[1, 3], [2, 4], [3, 5]])
    X_wrapped = _wrap_in_pandas_container(X_np, columns=get_columns)
    assert_array_equal(X_wrapped.columns, range(X_np.shape[1]))

def test_set_output_mro():
    if False:
        print('Hello World!')
    'Check that multi-inheritance resolves to the correct class method.\n\n    Non-regression test gh-25293.\n    '

    class Base(_SetOutputMixin):

        def transform(self, X):
            if False:
                while True:
                    i = 10
            return 'Base'

    class A(Base):
        pass

    class B(Base):

        def transform(self, X):
            if False:
                for i in range(10):
                    print('nop')
            return 'B'

    class C(A, B):
        pass
    assert C().transform(None) == 'B'

class EstimatorWithSetOutputIndex(_SetOutputMixin):

    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        if False:
            return 10
        import pandas as pd
        return pd.DataFrame(X.to_numpy(), index=[f's{i}' for i in range(X.shape[0])])

    def get_feature_names_out(self, input_features=None):
        if False:
            i = 10
            return i + 15
        return np.asarray([f'X{i}' for i in range(self.n_features_in_)], dtype=object)

def test_set_output_pandas_keep_index():
    if False:
        while True:
            i = 10
    'Check that set_output does not override index.\n\n    Non-regression test for gh-25730.\n    '
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[0, 1])
    est = EstimatorWithSetOutputIndex().set_output(transform='pandas')
    est.fit(X)
    X_trans = est.transform(X)
    assert_array_equal(X_trans.index, ['s0', 's1'])

class EstimatorReturnTuple(_SetOutputMixin):

    def __init__(self, OutputTuple):
        if False:
            print('Hello World!')
        self.OutputTuple = OutputTuple

    def transform(self, X, y=None):
        if False:
            while True:
                i = 10
        return self.OutputTuple(X, 2 * X)

def test_set_output_named_tuple_out():
    if False:
        while True:
            i = 10
    'Check that namedtuples are kept by default.'
    Output = namedtuple('Output', 'X, Y')
    X = np.asarray([[1, 2, 3]])
    est = EstimatorReturnTuple(OutputTuple=Output)
    X_trans = est.transform(X)
    assert isinstance(X_trans, Output)
    assert_array_equal(X_trans.X, X)
    assert_array_equal(X_trans.Y, 2 * X)

class EstimatorWithListInput(_SetOutputMixin):

    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        assert isinstance(X, list)
        self.n_features_in_ = len(X[0])
        return self

    def transform(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        return X

    def get_feature_names_out(self, input_features=None):
        if False:
            return 10
        return np.asarray([f'X{i}' for i in range(self.n_features_in_)], dtype=object)

def test_set_output_list_input():
    if False:
        for i in range(10):
            print('nop')
    'Check set_output for list input.\n\n    Non-regression test for #27037.\n    '
    pd = pytest.importorskip('pandas')
    X = [[0, 1, 2, 3], [4, 5, 6, 7]]
    est = EstimatorWithListInput()
    est.set_output(transform='pandas')
    X_out = est.fit(X).transform(X)
    assert isinstance(X_out, pd.DataFrame)
    assert_array_equal(X_out.columns, ['X0', 'X1', 'X2', 'X3'])