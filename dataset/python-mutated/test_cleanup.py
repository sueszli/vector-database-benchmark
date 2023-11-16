from cStringIO import StringIO
import re
from bzrlib.cleanup import _do_with_cleanups, _run_cleanup, ObjectWithCleanups, OperationWithCleanups
from bzrlib.tests import TestCase
from bzrlib import debug, trace

class CleanupsTestCase(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(CleanupsTestCase, self).setUp()
        self.call_log = []

    def no_op_cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        self.call_log.append('no_op_cleanup')

    def assertLogContains(self, regex):
        if False:
            while True:
                i = 10
        self.assertContainsRe(self.get_log(), regex, re.DOTALL)

    def failing_cleanup(self):
        if False:
            print('Hello World!')
        self.call_log.append('failing_cleanup')
        raise Exception('failing_cleanup goes boom!')

class TestRunCleanup(CleanupsTestCase):

    def test_no_errors(self):
        if False:
            return 10
        'The function passed to _run_cleanup is run.'
        self.assertTrue(_run_cleanup(self.no_op_cleanup))
        self.assertEqual(['no_op_cleanup'], self.call_log)

    def test_cleanup_with_args_kwargs(self):
        if False:
            while True:
                i = 10

        def func_taking_args_kwargs(*args, **kwargs):
            if False:
                print('Hello World!')
            self.call_log.append(('func', args, kwargs))
        _run_cleanup(func_taking_args_kwargs, 'an arg', kwarg='foo')
        self.assertEqual([('func', ('an arg',), {'kwarg': 'foo'})], self.call_log)

    def test_cleanup_error(self):
        if False:
            i = 10
            return i + 15
        "An error from the cleanup function is logged by _run_cleanup, but not\n        propagated.\n\n        This is there's no way for _run_cleanup to know if there's an existing\n        exception in this situation::\n            try:\n              some_func()\n            finally:\n              _run_cleanup(cleanup_func)\n        So, the best _run_cleanup can do is always log errors but never raise\n        them.\n        "
        self.assertFalse(_run_cleanup(self.failing_cleanup))
        self.assertLogContains('Cleanup failed:.*failing_cleanup goes boom')

    def test_cleanup_error_debug_flag(self):
        if False:
            for i in range(10):
                print('nop')
        'The -Dcleanup debug flag causes cleanup errors to be reported to the\n        user.\n        '
        log = StringIO()
        trace.push_log_file(log)
        debug.debug_flags.add('cleanup')
        self.assertFalse(_run_cleanup(self.failing_cleanup))
        self.assertContainsRe(log.getvalue(), 'bzr: warning: Cleanup failed:.*failing_cleanup goes boom')

    def test_prior_error_cleanup_succeeds(self):
        if False:
            for i in range(10):
                print('nop')
        'Calling _run_cleanup from a finally block will not interfere with an\n        exception from the try block.\n        '

        def failing_operation():
            if False:
                print('Hello World!')
            try:
                1 / 0
            finally:
                _run_cleanup(self.no_op_cleanup)
        self.assertRaises(ZeroDivisionError, failing_operation)
        self.assertEqual(['no_op_cleanup'], self.call_log)

    def test_prior_error_cleanup_fails(self):
        if False:
            for i in range(10):
                print('nop')
        'Calling _run_cleanup from a finally block will not interfere with an\n        exception from the try block even when the cleanup itself raises an\n        exception.\n\n        The cleanup exception will be logged.\n        '

        def failing_operation():
            if False:
                while True:
                    i = 10
            try:
                1 / 0
            finally:
                _run_cleanup(self.failing_cleanup)
        self.assertRaises(ZeroDivisionError, failing_operation)
        self.assertLogContains('Cleanup failed:.*failing_cleanup goes boom')

class TestDoWithCleanups(CleanupsTestCase):

    def trivial_func(self):
        if False:
            i = 10
            return i + 15
        self.call_log.append('trivial_func')
        return 'trivial result'

    def test_runs_func(self):
        if False:
            for i in range(10):
                print('nop')
        '_do_with_cleanups runs the function it is given, and returns the\n        result.\n        '
        result = _do_with_cleanups([], self.trivial_func)
        self.assertEqual('trivial result', result)

    def test_runs_cleanups(self):
        if False:
            print('Hello World!')
        'Cleanup functions are run (in the given order).'
        cleanup_func_1 = (self.call_log.append, ('cleanup 1',), {})
        cleanup_func_2 = (self.call_log.append, ('cleanup 2',), {})
        _do_with_cleanups([cleanup_func_1, cleanup_func_2], self.trivial_func)
        self.assertEqual(['trivial_func', 'cleanup 1', 'cleanup 2'], self.call_log)

    def failing_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.call_log.append('failing_func')
        1 / 0

    def test_func_error_propagates(self):
        if False:
            i = 10
            return i + 15
        'Errors from the main function are propagated (after running\n        cleanups).\n        '
        self.assertRaises(ZeroDivisionError, _do_with_cleanups, [(self.no_op_cleanup, (), {})], self.failing_func)
        self.assertEqual(['failing_func', 'no_op_cleanup'], self.call_log)

    def test_func_error_trumps_cleanup_error(self):
        if False:
            i = 10
            return i + 15
        'Errors from the main function a propagated even if a cleanup raises\n        an error.\n\n        The cleanup error is be logged.\n        '
        self.assertRaises(ZeroDivisionError, _do_with_cleanups, [(self.failing_cleanup, (), {})], self.failing_func)
        self.assertLogContains('Cleanup failed:.*failing_cleanup goes boom')

    def test_func_passes_and_error_from_cleanup(self):
        if False:
            while True:
                i = 10
        "An error from a cleanup is propagated when the main function doesn't\n        raise an error.  Later cleanups are still executed.\n        "
        exc = self.assertRaises(Exception, _do_with_cleanups, [(self.failing_cleanup, (), {}), (self.no_op_cleanup, (), {})], self.trivial_func)
        self.assertEqual('failing_cleanup goes boom!', exc.args[0])
        self.assertEqual(['trivial_func', 'failing_cleanup', 'no_op_cleanup'], self.call_log)

    def test_multiple_cleanup_failures(self):
        if False:
            for i in range(10):
                print('nop')
        'When multiple cleanups fail (as tends to happen when something has\n        gone wrong), the first error is propagated, and subsequent errors are\n        logged.\n        '
        cleanups = self.make_two_failing_cleanup_funcs()
        self.assertRaises(ErrorA, _do_with_cleanups, cleanups, self.trivial_func)
        self.assertLogContains('Cleanup failed:.*ErrorB')
        self.assertFalse('ErrorA' in self.get_log())

    def make_two_failing_cleanup_funcs(self):
        if False:
            print('Hello World!')

        def raise_a():
            if False:
                return 10
            raise ErrorA('Error A')

        def raise_b():
            if False:
                i = 10
                return i + 15
            raise ErrorB('Error B')
        return [(raise_a, (), {}), (raise_b, (), {})]

    def test_multiple_cleanup_failures_debug_flag(self):
        if False:
            for i in range(10):
                print('nop')
        log = StringIO()
        trace.push_log_file(log)
        debug.debug_flags.add('cleanup')
        cleanups = self.make_two_failing_cleanup_funcs()
        self.assertRaises(ErrorA, _do_with_cleanups, cleanups, self.trivial_func)
        self.assertContainsRe(log.getvalue(), 'bzr: warning: Cleanup failed:.*Error B\n')
        self.assertEqual(1, log.getvalue().count('bzr: warning:'), log.getvalue())

    def test_func_and_cleanup_errors_debug_flag(self):
        if False:
            while True:
                i = 10
        log = StringIO()
        trace.push_log_file(log)
        debug.debug_flags.add('cleanup')
        cleanups = self.make_two_failing_cleanup_funcs()
        self.assertRaises(ZeroDivisionError, _do_with_cleanups, cleanups, self.failing_func)
        self.assertContainsRe(log.getvalue(), 'bzr: warning: Cleanup failed:.*Error A\n')
        self.assertContainsRe(log.getvalue(), 'bzr: warning: Cleanup failed:.*Error B\n')
        self.assertEqual(2, log.getvalue().count('bzr: warning:'))

    def test_func_may_mutate_cleanups(self):
        if False:
            for i in range(10):
                print('nop')
        'The main func may mutate the cleanups before it returns.\n        \n        This allows a function to gradually add cleanups as it acquires\n        resources, rather than planning all the cleanups up-front.  The\n        OperationWithCleanups helper relies on this working.\n        '
        cleanups_list = []

        def func_that_adds_cleanups():
            if False:
                print('Hello World!')
            self.call_log.append('func_that_adds_cleanups')
            cleanups_list.append((self.no_op_cleanup, (), {}))
            return 'result'
        result = _do_with_cleanups(cleanups_list, func_that_adds_cleanups)
        self.assertEqual('result', result)
        self.assertEqual(['func_that_adds_cleanups', 'no_op_cleanup'], self.call_log)

    def test_cleanup_error_debug_flag(self):
        if False:
            while True:
                i = 10
        'The -Dcleanup debug flag causes cleanup errors to be reported to the\n        user.\n        '
        log = StringIO()
        trace.push_log_file(log)
        debug.debug_flags.add('cleanup')
        self.assertRaises(ZeroDivisionError, _do_with_cleanups, [(self.failing_cleanup, (), {})], self.failing_func)
        self.assertContainsRe(log.getvalue(), 'bzr: warning: Cleanup failed:.*failing_cleanup goes boom')
        self.assertEqual(1, log.getvalue().count('bzr: warning:'))

class ErrorA(Exception):
    pass

class ErrorB(Exception):
    pass

class TestOperationWithCleanups(CleanupsTestCase):

    def test_cleanup_ordering(self):
        if False:
            for i in range(10):
                print('nop')
        'Cleanups are added in LIFO order.\n\n        So cleanups added before run is called are run last, and the last\n        cleanup added during the func is run first.\n        '
        call_log = []

        def func(op, foo):
            if False:
                return 10
            call_log.append(('func called', foo))
            op.add_cleanup(call_log.append, 'cleanup 2')
            op.add_cleanup(call_log.append, 'cleanup 1')
            return 'result'
        owc = OperationWithCleanups(func)
        owc.add_cleanup(call_log.append, 'cleanup 4')
        owc.add_cleanup(call_log.append, 'cleanup 3')
        result = owc.run('foo')
        self.assertEqual('result', result)
        self.assertEqual([('func called', 'foo'), 'cleanup 1', 'cleanup 2', 'cleanup 3', 'cleanup 4'], call_log)

class SampleWithCleanups(ObjectWithCleanups):
    pass

class TestObjectWithCleanups(TestCase):

    def test_object_with_cleanups(self):
        if False:
            while True:
                i = 10
        a = []
        s = SampleWithCleanups()
        s.add_cleanup(a.append, 42)
        s.cleanup_now()
        self.assertEqual(a, [42])