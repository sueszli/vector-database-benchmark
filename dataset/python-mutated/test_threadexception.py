import pytest
from _pytest.pytester import Pytester

@pytest.mark.filterwarnings('default::pytest.PytestUnhandledThreadExceptionWarning')
def test_unhandled_thread_exception(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile(test_it='\n        import threading\n\n        def test_it():\n            def oops():\n                raise ValueError("Oops")\n\n            t = threading.Thread(target=oops, name="MyThread")\n            t.start()\n            t.join()\n\n        def test_2(): pass\n        ')
    result = pytester.runpytest()
    assert result.ret == 0
    assert result.parseoutcomes() == {'passed': 2, 'warnings': 1}
    result.stdout.fnmatch_lines(['*= warnings summary =*', 'test_it.py::test_it', '  * PytestUnhandledThreadExceptionWarning: Exception in thread MyThread', '  ', '  Traceback (most recent call last):', '  ValueError: Oops', '  ', '    warnings.warn(pytest.PytestUnhandledThreadExceptionWarning(msg))'])

@pytest.mark.filterwarnings('default::pytest.PytestUnhandledThreadExceptionWarning')
def test_unhandled_thread_exception_in_setup(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile(test_it='\n        import threading\n        import pytest\n\n        @pytest.fixture\n        def threadexc():\n            def oops():\n                raise ValueError("Oops")\n            t = threading.Thread(target=oops, name="MyThread")\n            t.start()\n            t.join()\n\n        def test_it(threadexc): pass\n        def test_2(): pass\n        ')
    result = pytester.runpytest()
    assert result.ret == 0
    assert result.parseoutcomes() == {'passed': 2, 'warnings': 1}
    result.stdout.fnmatch_lines(['*= warnings summary =*', 'test_it.py::test_it', '  * PytestUnhandledThreadExceptionWarning: Exception in thread MyThread', '  ', '  Traceback (most recent call last):', '  ValueError: Oops', '  ', '    warnings.warn(pytest.PytestUnhandledThreadExceptionWarning(msg))'])

@pytest.mark.filterwarnings('default::pytest.PytestUnhandledThreadExceptionWarning')
def test_unhandled_thread_exception_in_teardown(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile(test_it='\n        import threading\n        import pytest\n\n        @pytest.fixture\n        def threadexc():\n            def oops():\n                raise ValueError("Oops")\n            yield\n            t = threading.Thread(target=oops, name="MyThread")\n            t.start()\n            t.join()\n\n        def test_it(threadexc): pass\n        def test_2(): pass\n        ')
    result = pytester.runpytest()
    assert result.ret == 0
    assert result.parseoutcomes() == {'passed': 2, 'warnings': 1}
    result.stdout.fnmatch_lines(['*= warnings summary =*', 'test_it.py::test_it', '  * PytestUnhandledThreadExceptionWarning: Exception in thread MyThread', '  ', '  Traceback (most recent call last):', '  ValueError: Oops', '  ', '    warnings.warn(pytest.PytestUnhandledThreadExceptionWarning(msg))'])

@pytest.mark.filterwarnings('error::pytest.PytestUnhandledThreadExceptionWarning')
def test_unhandled_thread_exception_warning_error(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile(test_it='\n        import threading\n        import pytest\n\n        def test_it():\n            def oops():\n                raise ValueError("Oops")\n            t = threading.Thread(target=oops, name="MyThread")\n            t.start()\n            t.join()\n\n        def test_2(): pass\n        ')
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert result.parseoutcomes() == {'passed': 1, 'failed': 1}