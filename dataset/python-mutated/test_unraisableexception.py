import sys
import pytest
from _pytest.pytester import Pytester
PYPY = hasattr(sys, 'pypy_version_info')

@pytest.mark.skipif(PYPY, reason='garbage-collection differences make this flaky')
@pytest.mark.filterwarnings('default::pytest.PytestUnraisableExceptionWarning')
def test_unraisable(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile(test_it='\n        class BrokenDel:\n            def __del__(self):\n                raise ValueError("del is broken")\n\n        def test_it():\n            obj = BrokenDel()\n            del obj\n\n        def test_2(): pass\n        ')
    result = pytester.runpytest()
    assert result.ret == 0
    assert result.parseoutcomes() == {'passed': 2, 'warnings': 1}
    result.stdout.fnmatch_lines(['*= warnings summary =*', 'test_it.py::test_it', '  * PytestUnraisableExceptionWarning: Exception ignored in: <function BrokenDel.__del__ at *>', '  ', '  Traceback (most recent call last):', '  ValueError: del is broken', '  ', '    warnings.warn(pytest.PytestUnraisableExceptionWarning(msg))'])

@pytest.mark.skipif(PYPY, reason='garbage-collection differences make this flaky')
@pytest.mark.filterwarnings('default::pytest.PytestUnraisableExceptionWarning')
def test_unraisable_in_setup(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile(test_it='\n        import pytest\n\n        class BrokenDel:\n            def __del__(self):\n                raise ValueError("del is broken")\n\n        @pytest.fixture\n        def broken_del():\n            obj = BrokenDel()\n            del obj\n\n        def test_it(broken_del): pass\n        def test_2(): pass\n        ')
    result = pytester.runpytest()
    assert result.ret == 0
    assert result.parseoutcomes() == {'passed': 2, 'warnings': 1}
    result.stdout.fnmatch_lines(['*= warnings summary =*', 'test_it.py::test_it', '  * PytestUnraisableExceptionWarning: Exception ignored in: <function BrokenDel.__del__ at *>', '  ', '  Traceback (most recent call last):', '  ValueError: del is broken', '  ', '    warnings.warn(pytest.PytestUnraisableExceptionWarning(msg))'])

@pytest.mark.skipif(PYPY, reason='garbage-collection differences make this flaky')
@pytest.mark.filterwarnings('default::pytest.PytestUnraisableExceptionWarning')
def test_unraisable_in_teardown(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    pytester.makepyfile(test_it='\n        import pytest\n\n        class BrokenDel:\n            def __del__(self):\n                raise ValueError("del is broken")\n\n        @pytest.fixture\n        def broken_del():\n            yield\n            obj = BrokenDel()\n            del obj\n\n        def test_it(broken_del): pass\n        def test_2(): pass\n        ')
    result = pytester.runpytest()
    assert result.ret == 0
    assert result.parseoutcomes() == {'passed': 2, 'warnings': 1}
    result.stdout.fnmatch_lines(['*= warnings summary =*', 'test_it.py::test_it', '  * PytestUnraisableExceptionWarning: Exception ignored in: <function BrokenDel.__del__ at *>', '  ', '  Traceback (most recent call last):', '  ValueError: del is broken', '  ', '    warnings.warn(pytest.PytestUnraisableExceptionWarning(msg))'])

@pytest.mark.filterwarnings('error::pytest.PytestUnraisableExceptionWarning')
def test_unraisable_warning_error(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile(test_it=f"""\n        class BrokenDel:\n            def __del__(self) -> None:\n                raise ValueError("del is broken")\n\n        def test_it() -> None:\n            obj = BrokenDel()\n            del obj\n            {'import gc; gc.collect()' * PYPY}\n\n        def test_2(): pass\n        """)
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert result.parseoutcomes() == {'passed': 1, 'failed': 1}