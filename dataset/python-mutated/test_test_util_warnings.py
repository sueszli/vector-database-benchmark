import warnings
from twisted.trial import unittest
from buildbot.test.util.warnings import assertNotProducesWarnings
from buildbot.test.util.warnings import assertProducesWarning
from buildbot.test.util.warnings import assertProducesWarnings
from buildbot.test.util.warnings import ignoreWarning

class SomeWarning(Warning):
    pass

class OtherWarning(Warning):
    pass

class TestWarningsFilter(unittest.TestCase):

    def test_warnigs_caught(self):
        if False:
            while True:
                i = 10
        with assertProducesWarning(SomeWarning):
            warnings.warn('test', SomeWarning)

    def test_warnigs_caught_num_check(self):
        if False:
            i = 10
            return i + 15
        with assertProducesWarnings(SomeWarning, num_warnings=3):
            warnings.warn('1', SomeWarning)
            warnings.warn('2', SomeWarning)
            warnings.warn('3', SomeWarning)

    def test_warnigs_caught_num_check_fail(self):
        if False:
            i = 10
            return i + 15

        def f1():
            if False:
                while True:
                    i = 10
            with assertProducesWarnings(SomeWarning, num_warnings=2):
                pass
        with self.assertRaises(AssertionError):
            f1()

        def f2():
            if False:
                while True:
                    i = 10
            with assertProducesWarnings(SomeWarning, num_warnings=2):
                warnings.warn('1', SomeWarning)
        with self.assertRaises(AssertionError):
            f2()

        def f3():
            if False:
                while True:
                    i = 10
            with assertProducesWarnings(SomeWarning, num_warnings=2):
                warnings.warn('1', SomeWarning)
                warnings.warn('2', SomeWarning)
                warnings.warn('3', SomeWarning)
        with self.assertRaises(AssertionError):
            f3()

    def test_warnigs_caught_pattern_check(self):
        if False:
            for i in range(10):
                print('nop')
        with assertProducesWarning(SomeWarning, message_pattern='t.st'):
            warnings.warn('The test', SomeWarning)

    def test_warnigs_caught_pattern_check_fail(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                i = 10
                return i + 15
            with assertProducesWarning(SomeWarning, message_pattern='other'):
                warnings.warn('The test', SomeWarning)
        with self.assertRaises(AssertionError):
            f()

    def test_warnigs_caught_patterns_check(self):
        if False:
            while True:
                i = 10
        with assertProducesWarnings(SomeWarning, messages_patterns=['1', '2', '3']):
            warnings.warn('log 1 message', SomeWarning)
            warnings.warn('log 2 message', SomeWarning)
            warnings.warn('log 3 message', SomeWarning)

    def test_warnigs_caught_patterns_check_fails(self):
        if False:
            for i in range(10):
                print('nop')

        def f1():
            if False:
                for i in range(10):
                    print('nop')
            with assertProducesWarnings(SomeWarning, messages_patterns=['1', '2']):
                warnings.warn('msg 1', SomeWarning)
        with self.assertRaises(AssertionError):
            f1()

        def f2():
            if False:
                print('Hello World!')
            with assertProducesWarnings(SomeWarning, messages_patterns=['1', '2']):
                warnings.warn('msg 2', SomeWarning)
                warnings.warn('msg 1', SomeWarning)
        with self.assertRaises(AssertionError):
            f2()

        def f3():
            if False:
                while True:
                    i = 10
            with assertProducesWarnings(SomeWarning, messages_patterns=['1', '2']):
                warnings.warn('msg 1', SomeWarning)
                warnings.warn('msg 2', SomeWarning)
                warnings.warn('msg 3', SomeWarning)
        with self.assertRaises(AssertionError):
            f3()

    def test_no_warnigs_check(self):
        if False:
            i = 10
            return i + 15
        with assertNotProducesWarnings(SomeWarning):
            pass
        with ignoreWarning(OtherWarning):
            with assertNotProducesWarnings(SomeWarning):
                warnings.warn('msg 3', OtherWarning)

    def test_warnigs_filter(self):
        if False:
            while True:
                i = 10
        with ignoreWarning(OtherWarning):
            with assertProducesWarnings(SomeWarning, messages_patterns=['1', '2', '3']):
                warnings.warn('other', OtherWarning)
                warnings.warn('log 1 message', SomeWarning)
                warnings.warn('other', OtherWarning)
                warnings.warn('log 2 message', SomeWarning)
                warnings.warn('other', OtherWarning)
                warnings.warn('log 3 message', SomeWarning)
                warnings.warn('other', OtherWarning)

    def test_nested_filters(self):
        if False:
            while True:
                i = 10
        with assertProducesWarnings(SomeWarning, messages_patterns=['some 1']):
            with assertProducesWarnings(OtherWarning, messages_patterns=['other 1']):
                warnings.warn('other 1', OtherWarning)
                warnings.warn('some 1', SomeWarning)

    def test_ignore_warnings(self):
        if False:
            print('Hello World!')
        with assertNotProducesWarnings(SomeWarning):
            with ignoreWarning(SomeWarning):
                warnings.warn('some 1', SomeWarning)