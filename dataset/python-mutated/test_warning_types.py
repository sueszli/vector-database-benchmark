import inspect
import pytest
from _pytest import warning_types
from _pytest.pytester import Pytester

@pytest.mark.parametrize('warning_class', [w for (n, w) in vars(warning_types).items() if inspect.isclass(w) and issubclass(w, Warning)])
def test_warning_types(warning_class: UserWarning) -> None:
    if False:
        print('Hello World!')
    "Make sure all warnings declared in _pytest.warning_types are displayed as coming\n    from 'pytest' instead of the internal module (#5452).\n    "
    assert warning_class.__module__ == 'pytest'

@pytest.mark.filterwarnings('error::pytest.PytestWarning')
def test_pytest_warnings_repr_integration_test(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Small integration test to ensure our small hack of setting the __module__ attribute\n    of our warnings actually works (#5452).\n    '
    pytester.makepyfile('\n        import pytest\n        import warnings\n\n        def test():\n            warnings.warn(pytest.PytestWarning("some warning"))\n    ')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['E       pytest.PytestWarning: some warning'])

@pytest.mark.filterwarnings('error')
def test_warn_explicit_for_annotates_errors_with_location():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(Warning, match='(?m)test\n at .*python_api.py:\\d+'):
        warning_types.warn_explicit_for(pytest.raises, warning_types.PytestWarning('test'))