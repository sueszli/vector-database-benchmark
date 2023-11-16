import unittest
from cupy.testing import _condition
SKIP_REASON = 'test skip reason'

class MockUnitTest(unittest.TestCase):
    failure_case_counter = 0
    success_case_counter = 0
    skip_case_counter = 0
    probabilistic_case_counter = 0
    probabilistic_case_success_counter = 0
    probabilistic_case_failure_counter = 0

    @staticmethod
    def clear_counter():
        if False:
            while True:
                i = 10
        MockUnitTest.failure_case_counter = 0
        MockUnitTest.success_case_counter = 0
        MockUnitTest.skip_case_counter = 0
        MockUnitTest.probabilistic_case_counter = 0
        MockUnitTest.probabilistic_case_success_counter = 0
        MockUnitTest.probabilistic_case_failure_counter = 0

    def failure_case(self):
        if False:
            for i in range(10):
                print('nop')
        MockUnitTest.failure_case_counter += 1
        self.fail()

    def success_case(self):
        if False:
            while True:
                i = 10
        MockUnitTest.success_case_counter += 1
        assert True

    def skip_case(self):
        if False:
            return 10
        MockUnitTest.skip_case_counter += 1
        self.skipTest(SKIP_REASON)

    def error_case(self):
        if False:
            print('Hello World!')
        raise Exception()

    def probabilistic_case(self):
        if False:
            while True:
                i = 10
        MockUnitTest.probabilistic_case_counter += 1
        if MockUnitTest.probabilistic_case_counter % 2 == 0:
            MockUnitTest.probabilistic_case_success_counter += 1
            assert True
        else:
            MockUnitTest.probabilistic_case_failure_counter += 1
            self.fail()

    def runTest(self):
        if False:
            return 10
        pass

def _should_fail(self, f):
    if False:
        i = 10
        return i + 15
    try:
        f(self.unit_test)
        self.fail('AssertionError is expected to be raised, but none is raised')
    except AssertionError as e:
        assert 'first error message:' in str(e)

def _should_pass(self, f):
    if False:
        return 10
    f(self.unit_test)

def _should_skip(self, f):
    if False:
        for i in range(10):
            print('nop')
    try:
        f(self.unit_test)
        self.fail('SkipTest is expected to be raised, but none is raised')
    except unittest.SkipTest as e:
        assert SKIP_REASON in str(e)

class TestRepeatWithSuccessAtLeast(unittest.TestCase):

    def _decorate(self, f, times, min_success):
        if False:
            for i in range(10):
                print('nop')
        return _condition.repeat_with_success_at_least(times, min_success)(f)

    def setUp(self):
        if False:
            print('Hello World!')
        self.unit_test = MockUnitTest()
        MockUnitTest.clear_counter()

    def test_all_trials_fail(self):
        if False:
            return 10
        f = self._decorate(MockUnitTest.failure_case, 10, 1)
        _should_fail(self, f)
        assert self.unit_test.failure_case_counter == 10

    def test_all_trials_fail2(self):
        if False:
            print('Hello World!')
        f = self._decorate(MockUnitTest.failure_case, 10, 0)
        _should_pass(self, f)
        assert self.unit_test.failure_case_counter <= 10

    def test_all_trials_error(self):
        if False:
            return 10
        f = self._decorate(MockUnitTest.error_case, 10, 1)
        _should_fail(self, f)

    def test_all_trials_succeed(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._decorate(MockUnitTest.success_case, 10, 10)
        _should_pass(self, f)
        assert self.unit_test.success_case_counter == 10

    def test_all_trials_succeed2(self):
        if False:
            return 10
        self.assertRaises(AssertionError, _condition.repeat_with_success_at_least, 10, 11)

    def test_half_of_trials_succeed(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._decorate(MockUnitTest.probabilistic_case, 10, 5)
        _should_pass(self, f)
        assert self.unit_test.probabilistic_case_counter <= 10
        assert self.unit_test.probabilistic_case_success_counter >= 5
        assert self.unit_test.probabilistic_case_failure_counter <= 5

    def test_half_of_trials_succeed2(self):
        if False:
            while True:
                i = 10
        f = self._decorate(MockUnitTest.probabilistic_case, 10, 6)
        _should_fail(self, f)
        assert self.unit_test.probabilistic_case_counter <= 10
        assert self.unit_test.probabilistic_case_success_counter < 6
        assert self.unit_test.probabilistic_case_failure_counter >= 5

class TestRepeat(unittest.TestCase):

    def _decorate(self, f, times):
        if False:
            while True:
                i = 10
        return _condition.repeat(times)(f)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.unit_test = MockUnitTest()
        MockUnitTest.clear_counter()

    def test_failure_case(self):
        if False:
            while True:
                i = 10
        f = self._decorate(MockUnitTest.failure_case, 10)
        _should_fail(self, f)
        assert self.unit_test.failure_case_counter <= 10

    def test_success_case(self):
        if False:
            while True:
                i = 10
        f = self._decorate(MockUnitTest.success_case, 10)
        _should_pass(self, f)
        assert self.unit_test.success_case_counter == 10

    def test_skip_case(self):
        if False:
            print('Hello World!')
        f = self._decorate(MockUnitTest.skip_case, 10)
        _should_skip(self, f)
        assert self.unit_test.skip_case_counter == 1

    def test_probabilistic_case(self):
        if False:
            i = 10
            return i + 15
        f = self._decorate(MockUnitTest.probabilistic_case, 10)
        _should_fail(self, f)
        assert self.unit_test.probabilistic_case_counter <= 10
        assert self.unit_test.probabilistic_case_success_counter < 10
        assert self.unit_test.probabilistic_case_failure_counter > 0

class TestRetry(unittest.TestCase):

    def _decorate(self, f, times):
        if False:
            for i in range(10):
                print('nop')
        return _condition.retry(times)(f)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.unit_test = MockUnitTest()
        MockUnitTest.clear_counter()

    def test_failure_case(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._decorate(MockUnitTest.failure_case, 10)
        _should_fail(self, f)
        assert self.unit_test.failure_case_counter == 10

    def test_success_case(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._decorate(MockUnitTest.success_case, 10)
        _should_pass(self, f)
        assert self.unit_test.success_case_counter <= 10

    def test_skip_case(self):
        if False:
            i = 10
            return i + 15
        f = self._decorate(MockUnitTest.skip_case, 10)
        _should_skip(self, f)
        assert self.unit_test.skip_case_counter == 1

    def test_probabilistic_case(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._decorate(MockUnitTest.probabilistic_case, 10)
        _should_pass(self, f)
        assert self.unit_test.probabilistic_case_counter <= 10
        assert self.unit_test.probabilistic_case_success_counter > 0
        assert self.unit_test.probabilistic_case_failure_counter < 10