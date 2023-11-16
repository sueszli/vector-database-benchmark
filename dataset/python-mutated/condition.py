import functools
import unittest
import six
try:
    import _pytest.outcomes
    _error = None
except ImportError as e:
    _error = e

class QuietTestRunner(object):

    def run(self, suite):
        if False:
            while True:
                i = 10
        result = unittest.TestResult()
        suite(result)
        return result

def repeat_with_success_at_least(times, min_success):
    if False:
        while True:
            i = 10
    'Decorator for multiple trial of the test case.\n\n    The decorated test case is launched multiple times.\n    The case is judged as passed at least specified number of trials.\n    If the number of successful trials exceeds `min_success`,\n    the remaining trials are skipped.\n\n    Args:\n        times(int): The number of trials.\n        min_success(int): Threshold that the decorated test\n            case is regarded as passed.\n\n    '
    assert times >= min_success

    def _repeat_with_success_at_least(f):
        if False:
            print('Hello World!')

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            assert len(args) > 0
            instance = args[0]
            assert isinstance(instance, unittest.TestCase)
            success_counter = 0
            failure_counter = 0
            results = []

            def fail():
                if False:
                    i = 10
                    return i + 15
                msg = '\nFail: {0}, Success: {1}'.format(failure_counter, success_counter)
                if results:
                    first = results[0]
                    errs = first.failures + first.errors
                    if errs:
                        err_msg = '\n'.join((fail[1] for fail in errs))
                        msg += '\n\nThe first error message:\n' + err_msg
                instance.fail(msg)

            def f_wrap(ins, args, kwargs):
                if False:
                    i = 10
                    return i + 15
                try:
                    f(ins, *args[1:], **kwargs)
                except _pytest.outcomes.Skipped as e:
                    ins.skipTest(e.msg)
            for _ in six.moves.range(times):
                suite = unittest.TestSuite()
                ins = type(instance)(instance._testMethodName)
                suite.addTest(unittest.FunctionTestCase(lambda : f_wrap(ins, args, kwargs), setUp=ins.setUp, tearDown=ins.tearDown))
                result = QuietTestRunner().run(suite)
                if len(result.skipped) == 1:
                    instance.skipTest(result.skipped[0][1])
                elif result.wasSuccessful():
                    success_counter += 1
                else:
                    results.append(result)
                    failure_counter += 1
                if success_counter >= min_success:
                    instance.assertTrue(True)
                    return
                if failure_counter > times - min_success:
                    fail()
                    return
            fail()
        return wrapper
    return _repeat_with_success_at_least

def repeat(times):
    if False:
        print('Hello World!')
    'Decorator that imposes the test to be successful in a row.\n\n    Decorated test case is launched multiple times.\n    The case is regarded as passed only if it is successful\n    specified times in a row.\n\n    .. note::\n        In current implementation, this decorator grasps the\n        failure information of each trial.\n\n    Args:\n        times(int): The number of trials.\n    '
    return repeat_with_success_at_least(times, times)

def retry(times):
    if False:
        return 10
    'Decorator that imposes the test to be successful at least once.\n\n    Decorated test case is launched multiple times.\n    The case is regarded as passed if it is successful\n    at least once.\n\n    .. note::\n        In current implementation, this decorator grasps the\n        failure information of each trial.\n\n    Args:\n        times(int): The number of trials.\n    '
    return repeat_with_success_at_least(times, 1)