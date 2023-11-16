"""
GNU radio specific extension of unittest.
"""
import time
import unittest

class TestCase(unittest.TestCase):
    """A subclass of unittest.TestCase that adds additional assertions

    Adds new methods assertComplexAlmostEqual,
    assertComplexTuplesAlmostEqual and assertFloatTuplesAlmostEqual
    """

    def assertComplexAlmostEqual(self, first, second, places=7, msg=None):
        if False:
            i = 10
            return i + 15
        'Fail if the two complex objects are unequal as determined by their\n           difference rounded to the given number of decimal places\n           (default 7) and comparing to zero.\n\n           Note that decimal places (from zero) is usually not the same\n           as significant digits (measured from the most significant digit).\n       '
        if round(second.real - first.real, places) != 0:
            raise self.failureException(msg or '%r != %r within %r places' % (first, second, places))
        if round(second.imag - first.imag, places) != 0:
            raise self.failureException(msg or '%r != %r within %r places' % (first, second, places))

    def assertComplexAlmostEqual2(self, ref, x, abs_eps=1e-12, rel_eps=1e-06, msg=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fail if the two complex objects are unequal as determined by both\n        absolute delta (abs_eps) and relative delta (rel_eps).\n        '
        if abs(ref - x) < abs_eps:
            return
        if abs(ref) > abs_eps:
            if abs(ref - x) / abs(ref) > rel_eps:
                raise self.failureException(msg or '%r != %r rel_error = %r rel_limit = %r' % (ref, x, abs(ref - x) / abs(ref), rel_eps))
        else:
            raise self.failureException(msg or '%r != %r rel_error = %r rel_limit = %r' % (ref, x, abs(ref - x) / abs(ref), rel_eps))

    def assertComplexTuplesAlmostEqual(self, a, b, places=7, msg=None):
        if False:
            print('Hello World!')
        '\n        Fail if the two complex tuples are not approximately equal.\n        Approximate equality is determined by specifying the number of decimal\n        places.0\n        '
        self.assertEqual(len(a), len(b))
        return all([self.assertComplexAlmostEqual(x, y, places, msg) for (x, y) in zip(a, b)])

    def assertComplexTuplesAlmostEqual2(self, a, b, abs_eps=1e-12, rel_eps=1e-06, msg=None):
        if False:
            i = 10
            return i + 15
        '\n        Fail if the two complex tuples are not approximately equal.\n        Approximate equality is determined by calling assertComplexAlmostEqual().\n        '
        self.assertEqual(len(a), len(b))
        return all([self.assertComplexAlmostEqual2(x, y, abs_eps, rel_eps, msg) for (x, y) in zip(a, b)])

    def assertFloatTuplesAlmostEqual(self, a, b, places=7, msg=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fail if the two real-valued tuples are not approximately equal.\n        Approximate equality is determined by specifying the number of decimal\n        places.\n        '
        self.assertEqual(len(a), len(b))
        return all([self.assertAlmostEqual(x, y, places, msg) for (x, y) in zip(a, b)])

    def assertFloatTuplesAlmostEqual2(self, a, b, abs_eps=1e-12, rel_eps=1e-06, msg=None):
        if False:
            while True:
                i = 10
        self.assertEqual(len(a), len(b))
        return all([self.assertComplexAlmostEqual2(x, y, abs_eps, rel_eps, msg) for (x, y) in zip(a, b)])

    def assertSequenceEqualGR(self, data_in, data_out):
        if False:
            return 10
        '\n        Note this function exists because of this bug: https://bugs.python.org/issue19217\n        Calling self.assertEqual(seqA, seqB) can hang if seqA and seqB are not equal.\n        '
        self.assertEqual(len(data_in), len(data_out), msg='Lengths do not match')
        miscompares = []
        for (idx, item) in enumerate(zip(data_in, data_out)):
            if item[0] != item[1]:
                miscompares.append(f'Miscompare at: {idx} ({item[0]} -- {item[1]})')
        self.assertEqual(len(miscompares), 0, msg=f'Total miscompares: {len(miscompares)}\n' + '\n'.join(miscompares))

    def waitFor(self, condition, timeout=5.0, poll_interval=0.2, fail_on_timeout=True, fail_msg=None):
        if False:
            return 10
        "\n        Helper function: Wait for a callable to return True within a given\n        timeout.\n\n        This is useful for running tests where an exact wait time is not known.\n\n        Arguments:\n        - condition: A callable. Must return True when a 'good' condition is met.\n        - timeout: Timeout in seconds. `condition` must return True within this\n                   timeout.\n        - poll_interval: Time between calls to condition() in seconds\n        - fail_on_timeout: If True, the test case will fail when the timeout\n                           occurs. If False, this function will return False in\n                           that case.\n        - fail_msg: The message that is printed when a timeout occurs and\n                    fail_on_timeout is true.\n        "
        if not callable(condition):
            self.fail('Invalid condition provided to waitFor()!')
        stop_time = time.monotonic() + timeout
        while time.monotonic() <= stop_time:
            if condition():
                return True
            time.sleep(poll_interval)
        if fail_on_timeout:
            fail_msg = fail_msg or 'Timeout exceeded during call to waitFor()!'
            self.fail(fail_msg)
        return False
TestResult = unittest.TestResult
TestSuite = unittest.TestSuite
FunctionTestCase = unittest.FunctionTestCase
TestLoader = unittest.TestLoader
TextTestRunner = unittest.TextTestRunner
TestProgram = unittest.TestProgram
main = TestProgram

def run(PUT, filename=None, verbosity=1):
    if False:
        while True:
            i = 10
    '\n    Runs the unittest on a TestCase\n    PUT:      the program under test and should be a gr_unittest.TestCase\n    filename: This argument is here for historical reasons.\n    '
    if filename:
        print('DEPRECATED: Using filename with gr_unittest does no longer have any effect.')
    main(verbosity=verbosity)
if __name__ == '__main__':
    main(module=None)