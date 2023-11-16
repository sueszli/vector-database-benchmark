from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import HasMethods, Hidden, Interval, InvalidParameterError, MissingValues, Options, RealNotInt, StrOptions, _ArrayLikes, _Booleans, _Callables, _CVObjects, _InstancesOf, _IterablesNotString, _NanConstraint, _NoneConstraint, _PandasNAConstraint, _RandomStates, _SparseMatrices, _VerboseHelper, generate_invalid_param_val, generate_valid_param, make_constraint, validate_params

@validate_params({'a': [Real], 'b': [Real], 'c': [Real], 'd': [Real]}, prefer_skip_nested_validation=True)
def _func(a, b=0, *args, c, d=0, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'A function to test the validation of functions.'

class _Class:
    """A class to test the _InstancesOf constraint and the validation of methods."""

    @validate_params({'a': [Real]}, prefer_skip_nested_validation=True)
    def _method(self, a):
        if False:
            print('Hello World!')
        'A validated method'

    @deprecated()
    @validate_params({'a': [Real]}, prefer_skip_nested_validation=True)
    def _deprecated_method(self, a):
        if False:
            print('Hello World!')
        'A deprecated validated method'

class _Estimator(BaseEstimator):
    """An estimator to test the validation of estimator parameters."""
    _parameter_constraints: dict = {'a': [Real]}

    def __init__(self, a):
        if False:
            i = 10
            return i + 15
        self.a = a

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X=None, y=None):
        if False:
            for i in range(10):
                print('nop')
        pass

@pytest.mark.parametrize('interval_type', [Integral, Real])
def test_interval_range(interval_type):
    if False:
        i = 10
        return i + 15
    'Check the range of values depending on closed.'
    interval = Interval(interval_type, -2, 2, closed='left')
    assert -2 in interval
    assert 2 not in interval
    interval = Interval(interval_type, -2, 2, closed='right')
    assert -2 not in interval
    assert 2 in interval
    interval = Interval(interval_type, -2, 2, closed='both')
    assert -2 in interval
    assert 2 in interval
    interval = Interval(interval_type, -2, 2, closed='neither')
    assert -2 not in interval
    assert 2 not in interval

@pytest.mark.parametrize('interval_type', [Integral, Real])
def test_interval_large_integers(interval_type):
    if False:
        i = 10
        return i + 15
    'Check that Interval constraint work with large integers.\n\n    non-regression test for #26648.\n    '
    interval = Interval(interval_type, 0, 2, closed='neither')
    assert 2 ** 65 not in interval
    assert 2 ** 128 not in interval
    assert float(2 ** 65) not in interval
    assert float(2 ** 128) not in interval
    interval = Interval(interval_type, 0, 2 ** 128, closed='neither')
    assert 2 ** 65 in interval
    assert 2 ** 128 not in interval
    assert float(2 ** 65) in interval
    assert float(2 ** 128) not in interval
    assert 2 ** 1024 not in interval

def test_interval_inf_in_bounds():
    if False:
        print('Hello World!')
    'Check that inf is included iff a bound is closed and set to None.\n\n    Only valid for real intervals.\n    '
    interval = Interval(Real, 0, None, closed='right')
    assert np.inf in interval
    interval = Interval(Real, None, 0, closed='left')
    assert -np.inf in interval
    interval = Interval(Real, None, None, closed='neither')
    assert np.inf not in interval
    assert -np.inf not in interval

@pytest.mark.parametrize('interval', [Interval(Real, 0, 1, closed='left'), Interval(Real, None, None, closed='both')])
def test_nan_not_in_interval(interval):
    if False:
        print('Hello World!')
    'Check that np.nan is not in any interval.'
    assert np.nan not in interval

@pytest.mark.parametrize('params, error, match', [({'type': Integral, 'left': 1.0, 'right': 2, 'closed': 'both'}, TypeError, 'Expecting left to be an int for an interval over the integers'), ({'type': Integral, 'left': 1, 'right': 2.0, 'closed': 'neither'}, TypeError, 'Expecting right to be an int for an interval over the integers'), ({'type': Integral, 'left': None, 'right': 0, 'closed': 'left'}, ValueError, "left can't be None when closed == left"), ({'type': Integral, 'left': 0, 'right': None, 'closed': 'right'}, ValueError, "right can't be None when closed == right"), ({'type': Integral, 'left': 1, 'right': -1, 'closed': 'both'}, ValueError, "right can't be less than left")])
def test_interval_errors(params, error, match):
    if False:
        return 10
    'Check that informative errors are raised for invalid combination of parameters'
    with pytest.raises(error, match=match):
        Interval(**params)

def test_stroptions():
    if False:
        print('Hello World!')
    'Sanity check for the StrOptions constraint'
    options = StrOptions({'a', 'b', 'c'}, deprecated={'c'})
    assert options.is_satisfied_by('a')
    assert options.is_satisfied_by('c')
    assert not options.is_satisfied_by('d')
    assert "'c' (deprecated)" in str(options)

def test_options():
    if False:
        i = 10
        return i + 15
    'Sanity check for the Options constraint'
    options = Options(Real, {-0.5, 0.5, np.inf}, deprecated={-0.5})
    assert options.is_satisfied_by(-0.5)
    assert options.is_satisfied_by(np.inf)
    assert not options.is_satisfied_by(1.23)
    assert '-0.5 (deprecated)' in str(options)

@pytest.mark.parametrize('type, expected_type_name', [(int, 'int'), (Integral, 'int'), (Real, 'float'), (np.ndarray, 'numpy.ndarray')])
def test_instances_of_type_human_readable(type, expected_type_name):
    if False:
        while True:
            i = 10
    'Check the string representation of the _InstancesOf constraint.'
    constraint = _InstancesOf(type)
    assert str(constraint) == f"an instance of '{expected_type_name}'"

def test_hasmethods():
    if False:
        return 10
    'Check the HasMethods constraint.'
    constraint = HasMethods(['a', 'b'])

    class _Good:

        def a(self):
            if False:
                print('Hello World!')
            pass

        def b(self):
            if False:
                print('Hello World!')
            pass

    class _Bad:

        def a(self):
            if False:
                print('Hello World!')
            pass
    assert constraint.is_satisfied_by(_Good())
    assert not constraint.is_satisfied_by(_Bad())
    assert str(constraint) == "an object implementing 'a' and 'b'"

@pytest.mark.parametrize('constraint', [Interval(Real, None, 0, closed='left'), Interval(Real, 0, None, closed='left'), Interval(Real, None, None, closed='neither'), StrOptions({'a', 'b', 'c'}), MissingValues(), MissingValues(numeric_only=True), _VerboseHelper(), HasMethods('fit'), _IterablesNotString(), _CVObjects()])
def test_generate_invalid_param_val(constraint):
    if False:
        print('Hello World!')
    'Check that the value generated does not satisfy the constraint'
    bad_value = generate_invalid_param_val(constraint)
    assert not constraint.is_satisfied_by(bad_value)

@pytest.mark.parametrize('integer_interval, real_interval', [(Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, -5, 5, closed='both')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, -5, 5, closed='neither')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, 4, 5, closed='both')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, 5, None, closed='left')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, 4, None, closed='neither')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, -5, 5, closed='both')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, -5, 5, closed='neither')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, 1, 2, closed='both')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, None, -5, closed='left')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, None, -4, closed='neither')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, None, 1, closed='right')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, 1, None, closed='left')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, -10, -4, closed='neither')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, -10, -4, closed='right')), (Interval(Integral, -5, 5, closed='neither'), Interval(RealNotInt, 6, 10, closed='neither')), (Interval(Integral, -5, 5, closed='neither'), Interval(RealNotInt, 6, 10, closed='left')), (Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both')), (Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both'))])
def test_generate_invalid_param_val_2_intervals(integer_interval, real_interval):
    if False:
        i = 10
        return i + 15
    'Check that the value generated for an interval constraint does not satisfy any of\n    the interval constraints.\n    '
    bad_value = generate_invalid_param_val(constraint=real_interval)
    assert not real_interval.is_satisfied_by(bad_value)
    assert not integer_interval.is_satisfied_by(bad_value)
    bad_value = generate_invalid_param_val(constraint=integer_interval)
    assert not real_interval.is_satisfied_by(bad_value)
    assert not integer_interval.is_satisfied_by(bad_value)

@pytest.mark.parametrize('constraint', [_ArrayLikes(), _InstancesOf(list), _Callables(), _NoneConstraint(), _RandomStates(), _SparseMatrices(), _Booleans(), Interval(Integral, None, None, closed='neither')])
def test_generate_invalid_param_val_all_valid(constraint):
    if False:
        print('Hello World!')
    "Check that the function raises NotImplementedError when there's no invalid value\n    for the constraint.\n    "
    with pytest.raises(NotImplementedError):
        generate_invalid_param_val(constraint)

@pytest.mark.parametrize('constraint', [_ArrayLikes(), _Callables(), _InstancesOf(list), _NoneConstraint(), _RandomStates(), _SparseMatrices(), _Booleans(), _VerboseHelper(), MissingValues(), MissingValues(numeric_only=True), StrOptions({'a', 'b', 'c'}), Options(Integral, {1, 2, 3}), Interval(Integral, None, None, closed='neither'), Interval(Integral, 0, 10, closed='neither'), Interval(Integral, 0, None, closed='neither'), Interval(Integral, None, 0, closed='neither'), Interval(Real, 0, 1, closed='neither'), Interval(Real, 0, None, closed='both'), Interval(Real, None, 0, closed='right'), HasMethods('fit'), _IterablesNotString(), _CVObjects()])
def test_generate_valid_param(constraint):
    if False:
        while True:
            i = 10
    'Check that the value generated does satisfy the constraint.'
    value = generate_valid_param(constraint)
    assert constraint.is_satisfied_by(value)

@pytest.mark.parametrize('constraint_declaration, value', [(Interval(Real, 0, 1, closed='both'), 0.42), (Interval(Integral, 0, None, closed='neither'), 42), (StrOptions({'a', 'b', 'c'}), 'b'), (Options(type, {np.float32, np.float64}), np.float64), (callable, lambda x: x + 1), (None, None), ('array-like', [[1, 2], [3, 4]]), ('array-like', np.array([[1, 2], [3, 4]])), ('sparse matrix', csr_matrix([[1, 2], [3, 4]])), ('random_state', 0), ('random_state', np.random.RandomState(0)), ('random_state', None), (_Class, _Class()), (int, 1), (Real, 0.5), ('boolean', False), ('verbose', 1), ('nan', np.nan), (MissingValues(), -1), (MissingValues(), -1.0), (MissingValues(), 2 ** 1028), (MissingValues(), None), (MissingValues(), float('nan')), (MissingValues(), np.nan), (MissingValues(), 'missing'), (HasMethods('fit'), _Estimator(a=0)), ('cv_object', 5)])
def test_is_satisfied_by(constraint_declaration, value):
    if False:
        return 10
    'Sanity check for the is_satisfied_by method'
    constraint = make_constraint(constraint_declaration)
    assert constraint.is_satisfied_by(value)

@pytest.mark.parametrize('constraint_declaration, expected_constraint_class', [(Interval(Real, 0, 1, closed='both'), Interval), (StrOptions({'option1', 'option2'}), StrOptions), (Options(Real, {0.42, 1.23}), Options), ('array-like', _ArrayLikes), ('sparse matrix', _SparseMatrices), ('random_state', _RandomStates), (None, _NoneConstraint), (callable, _Callables), (int, _InstancesOf), ('boolean', _Booleans), ('verbose', _VerboseHelper), (MissingValues(numeric_only=True), MissingValues), (HasMethods('fit'), HasMethods), ('cv_object', _CVObjects), ('nan', _NanConstraint)])
def test_make_constraint(constraint_declaration, expected_constraint_class):
    if False:
        for i in range(10):
            print('nop')
    'Check that make_constraint dispatches to the appropriate constraint class'
    constraint = make_constraint(constraint_declaration)
    assert constraint.__class__ is expected_constraint_class

def test_make_constraint_unknown():
    if False:
        return 10
    'Check that an informative error is raised when an unknown constraint is passed'
    with pytest.raises(ValueError, match='Unknown constraint'):
        make_constraint('not a valid constraint')

def test_validate_params():
    if False:
        return 10
    'Check that validate_params works no matter how the arguments are passed'
    with pytest.raises(InvalidParameterError, match="The 'a' parameter of _func must be"):
        _func('wrong', c=1)
    with pytest.raises(InvalidParameterError, match="The 'b' parameter of _func must be"):
        _func(*[1, 'wrong'], c=1)
    with pytest.raises(InvalidParameterError, match="The 'c' parameter of _func must be"):
        _func(1, **{'c': 'wrong'})
    with pytest.raises(InvalidParameterError, match="The 'd' parameter of _func must be"):
        _func(1, c=1, d='wrong')
    with pytest.raises(InvalidParameterError, match="The 'b' parameter of _func must be"):
        _func(0, *['wrong', 2, 3], c=4, **{'e': 5})
    with pytest.raises(InvalidParameterError, match="The 'c' parameter of _func must be"):
        _func(0, *[1, 2, 3], c='four', **{'e': 5})

def test_validate_params_missing_params():
    if False:
        print('Hello World!')
    'Check that no error is raised when there are parameters without\n    constraints\n    '

    @validate_params({'a': [int]}, prefer_skip_nested_validation=True)
    def func(a, b):
        if False:
            while True:
                i = 10
        pass
    func(1, 2)

def test_decorate_validated_function():
    if False:
        return 10
    'Check that validate_params functions can be decorated'
    decorated_function = deprecated()(_func)
    with pytest.warns(FutureWarning, match='Function _func is deprecated'):
        decorated_function(1, 2, c=3)
    with pytest.warns(FutureWarning, match='Function _func is deprecated'):
        with pytest.raises(InvalidParameterError, match="The 'c' parameter of _func must be"):
            decorated_function(1, 2, c='wrong')

def test_validate_params_method():
    if False:
        i = 10
        return i + 15
    'Check that validate_params works with methods'
    with pytest.raises(InvalidParameterError, match="The 'a' parameter of _Class._method must be"):
        _Class()._method('wrong')
    with pytest.warns(FutureWarning, match='Function _deprecated_method is deprecated'):
        with pytest.raises(InvalidParameterError, match="The 'a' parameter of _Class._deprecated_method must be"):
            _Class()._deprecated_method('wrong')

def test_validate_params_estimator():
    if False:
        print('Hello World!')
    'Check that validate_params works with Estimator instances'
    est = _Estimator('wrong')
    with pytest.raises(InvalidParameterError, match="The 'a' parameter of _Estimator must be"):
        est.fit()

def test_stroptions_deprecated_subset():
    if False:
        for i in range(10):
            print('nop')
    'Check that the deprecated parameter must be a subset of options.'
    with pytest.raises(ValueError, match='deprecated options must be a subset'):
        StrOptions({'a', 'b', 'c'}, deprecated={'a', 'd'})

def test_hidden_constraint():
    if False:
        i = 10
        return i + 15
    'Check that internal constraints are not exposed in the error message.'

    @validate_params({'param': [Hidden(list), dict]}, prefer_skip_nested_validation=True)
    def f(param):
        if False:
            for i in range(10):
                print('nop')
        pass
    f({'a': 1, 'b': 2, 'c': 3})
    f([1, 2, 3])
    with pytest.raises(InvalidParameterError, match="The 'param' parameter") as exc_info:
        f(param='bad')
    err_msg = str(exc_info.value)
    assert "an instance of 'dict'" in err_msg
    assert "an instance of 'list'" not in err_msg

def test_hidden_stroptions():
    if False:
        for i in range(10):
            print('nop')
    'Check that we can have 2 StrOptions constraints, one being hidden.'

    @validate_params({'param': [StrOptions({'auto'}), Hidden(StrOptions({'warn'}))]}, prefer_skip_nested_validation=True)
    def f(param):
        if False:
            return 10
        pass
    f('auto')
    f('warn')
    with pytest.raises(InvalidParameterError, match="The 'param' parameter") as exc_info:
        f(param='bad')
    err_msg = str(exc_info.value)
    assert 'auto' in err_msg
    assert 'warn' not in err_msg

def test_validate_params_set_param_constraints_attribute():
    if False:
        return 10
    'Check that the validate_params decorator properly sets the parameter constraints\n    as attribute of the decorated function/method.\n    '
    assert hasattr(_func, '_skl_parameter_constraints')
    assert hasattr(_Class()._method, '_skl_parameter_constraints')

def test_boolean_constraint_deprecated_int():
    if False:
        for i in range(10):
            print('nop')
    'Check that validate_params raise a deprecation message but still passes\n    validation when using an int for a parameter accepting a boolean.\n    '

    @validate_params({'param': ['boolean']}, prefer_skip_nested_validation=True)
    def f(param):
        if False:
            print('Hello World!')
        pass
    f(True)
    f(np.bool_(False))
    with pytest.warns(FutureWarning, match='Passing an int for a boolean parameter is deprecated'):
        f(1)

def test_no_validation():
    if False:
        print('Hello World!')
    'Check that validation can be skipped for a parameter.'

    @validate_params({'param1': [int, None], 'param2': 'no_validation'}, prefer_skip_nested_validation=True)
    def f(param1=None, param2=None):
        if False:
            print('Hello World!')
        pass
    with pytest.raises(InvalidParameterError, match="The 'param1' parameter"):
        f(param1='wrong')

    class SomeType:
        pass
    f(param2=SomeType)
    f(param2=SomeType())

def test_pandas_na_constraint_with_pd_na():
    if False:
        while True:
            i = 10
    'Add a specific test for checking support for `pandas.NA`.'
    pd = pytest.importorskip('pandas')
    na_constraint = _PandasNAConstraint()
    assert na_constraint.is_satisfied_by(pd.NA)
    assert not na_constraint.is_satisfied_by(np.array([1, 2, 3]))

def test_iterable_not_string():
    if False:
        i = 10
        return i + 15
    'Check that a string does not satisfy the _IterableNotString constraint.'
    constraint = _IterablesNotString()
    assert constraint.is_satisfied_by([1, 2, 3])
    assert constraint.is_satisfied_by(range(10))
    assert not constraint.is_satisfied_by('some string')

def test_cv_objects():
    if False:
        print('Hello World!')
    'Check that the _CVObjects constraint accepts all current ways\n    to pass cv objects.'
    constraint = _CVObjects()
    assert constraint.is_satisfied_by(5)
    assert constraint.is_satisfied_by(LeaveOneOut())
    assert constraint.is_satisfied_by([([1, 2], [3, 4]), ([3, 4], [1, 2])])
    assert constraint.is_satisfied_by(None)
    assert not constraint.is_satisfied_by('not a CV object')

def test_third_party_estimator():
    if False:
        i = 10
        return i + 15
    'Check that the validation from a scikit-learn estimator inherited by a third\n    party estimator does not impose a match between the dict of constraints and the\n    parameters of the estimator.\n    '

    class ThirdPartyEstimator(_Estimator):

        def __init__(self, b):
            if False:
                while True:
                    i = 10
            self.b = b
            super().__init__(a=0)

        def fit(self, X=None, y=None):
            if False:
                while True:
                    i = 10
            super().fit(X, y)
    ThirdPartyEstimator(b=0).fit()

def test_interval_real_not_int():
    if False:
        for i in range(10):
            print('nop')
    'Check for the type RealNotInt in the Interval constraint.'
    constraint = Interval(RealNotInt, 0, 1, closed='both')
    assert constraint.is_satisfied_by(1.0)
    assert not constraint.is_satisfied_by(1)

def test_real_not_int():
    if False:
        for i in range(10):
            print('nop')
    'Check for the RealNotInt type.'
    assert isinstance(1.0, RealNotInt)
    assert not isinstance(1, RealNotInt)
    assert isinstance(np.float64(1), RealNotInt)
    assert not isinstance(np.int64(1), RealNotInt)

def test_skip_param_validation():
    if False:
        return 10
    'Check that param validation can be skipped using config_context.'

    @validate_params({'a': [int]}, prefer_skip_nested_validation=True)
    def f(a):
        if False:
            print('Hello World!')
        pass
    with pytest.raises(InvalidParameterError, match="The 'a' parameter"):
        f(a='1')
    with config_context(skip_parameter_validation=True):
        f(a='1')

@pytest.mark.parametrize('prefer_skip_nested_validation', [True, False])
def test_skip_nested_validation(prefer_skip_nested_validation):
    if False:
        while True:
            i = 10
    'Check that nested validation can be skipped.'

    @validate_params({'a': [int]}, prefer_skip_nested_validation=True)
    def f(a):
        if False:
            print('Hello World!')
        pass

    @validate_params({'b': [int]}, prefer_skip_nested_validation=prefer_skip_nested_validation)
    def g(b):
        if False:
            while True:
                i = 10
        return f(a='invalid_param_value')
    with pytest.raises(InvalidParameterError, match="The 'b' parameter"):
        g(b='invalid_param_value')
    if prefer_skip_nested_validation:
        g(b=1)
    else:
        with pytest.raises(InvalidParameterError, match="The 'a' parameter"):
            g(b=1)

@pytest.mark.parametrize('skip_parameter_validation, prefer_skip_nested_validation, expected_skipped', [(True, True, True), (True, False, True), (False, True, True), (False, False, False)])
def test_skip_nested_validation_and_config_context(skip_parameter_validation, prefer_skip_nested_validation, expected_skipped):
    if False:
        print('Hello World!')
    'Check interaction between global skip and local skip.'

    @validate_params({'a': [int]}, prefer_skip_nested_validation=prefer_skip_nested_validation)
    def g(a):
        if False:
            i = 10
            return i + 15
        return get_config()['skip_parameter_validation']
    with config_context(skip_parameter_validation=skip_parameter_validation):
        actual_skipped = g(1)
    assert actual_skipped == expected_skipped