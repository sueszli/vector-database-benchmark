import io
import re
import sys
from twisted.trial import unittest
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS
from buildbot.steps import subunit
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import TestBuildStepMixin
try:
    from subunit import TestProtocolClient
except ImportError:
    TestProtocolClient = None

class FakeTest:

    def __init__(self, id):
        if False:
            i = 10
            return i + 15
        self._id = id

    def id(self):
        if False:
            return 10
        return self._id

def create_error(name):
    if False:
        i = 10
        return i + 15
    try:
        int('_' + name)
        return None
    except ValueError:
        (exctype, value, _) = sys.exc_info()
        return (exctype, value, None)

class TestSubUnit(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        if TestProtocolClient is None:
            raise unittest.SkipTest('Need to install python-subunit to test subunit step')
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tear_down_test_build_step()

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(subunit.SubunitShellCommand(command='test'))
        self.expect_commands(ExpectShell(workdir='wkdir', command='test').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='shell no tests run')
        return self.run_step()

    def test_empty_error(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(subunit.SubunitShellCommand(command='test', failureOnNoTests=True))
        self.expect_commands(ExpectShell(workdir='wkdir', command='test').exit(0))
        self.expect_outcome(result=FAILURE, state_string='shell no tests run (failure)')
        return self.run_step()

    def test_success(self):
        if False:
            return 10
        stream = io.BytesIO()
        client = TestProtocolClient(stream)
        test = FakeTest(id='test1')
        client.startTest(test)
        client.stopTest(test)
        self.setup_step(subunit.SubunitShellCommand(command='test'))
        self.expect_commands(ExpectShell(workdir='wkdir', command='test').stdout(stream.getvalue()).exit(0))
        self.expect_outcome(result=SUCCESS, state_string='shell 1 test passed')
        return self.run_step()

    def test_error(self):
        if False:
            print('Hello World!')
        stream = io.BytesIO()
        client = TestProtocolClient(stream)
        test = FakeTest(id='test1')
        client.startTest(test)
        client.addError(test, create_error('error1'))
        client.stopTest(test)
        self.setup_step(subunit.SubunitShellCommand(command='test'))
        self.expect_commands(ExpectShell(workdir='wkdir', command='test').stdout(stream.getvalue()).exit(0))
        self.expect_outcome(result=FAILURE, state_string='shell Total 1 test(s) 1 error (failure)')
        self.expect_log_file('problems', re.compile("test1\ntesttools.testresult.real._StringException:.*ValueError: invalid literal for int\\(\\) with base 10: '_error1'\n.*", re.MULTILINE | re.DOTALL))
        return self.run_step()

    def test_multiple_errors(self):
        if False:
            i = 10
            return i + 15
        stream = io.BytesIO()
        client = TestProtocolClient(stream)
        test1 = FakeTest(id='test1')
        test2 = FakeTest(id='test2')
        client.startTest(test1)
        client.addError(test1, create_error('error1'))
        client.stopTest(test1)
        client.startTest(test2)
        client.addError(test2, create_error('error2'))
        client.stopTest(test2)
        self.setup_step(subunit.SubunitShellCommand(command='test'))
        self.expect_commands(ExpectShell(workdir='wkdir', command='test').stdout(stream.getvalue()).exit(0))
        self.expect_outcome(result=FAILURE, state_string='shell Total 2 test(s) 2 errors (failure)')
        self.expect_log_file('problems', re.compile("test1\ntesttools.testresult.real._StringException:.*ValueError: invalid literal for int\\(\\) with base 10: '_error1'\n\ntest2\ntesttools.testresult.real._StringException:.*ValueError: invalid literal for int\\(\\) with base 10: '_error2'\n.*", re.MULTILINE | re.DOTALL))
        return self.run_step()

    def test_warnings(self):
        if False:
            return 10
        stream = io.BytesIO()
        client = TestProtocolClient(stream)
        test1 = FakeTest(id='test1')
        test2 = FakeTest(id='test2')
        client.startTest(test1)
        client.stopTest(test1)
        client.addError(test2, create_error('error2'))
        client.stopTest(test2)
        self.setup_step(subunit.SubunitShellCommand(command='test'))
        self.expect_commands(ExpectShell(workdir='wkdir', command='test').stdout(stream.getvalue()).exit(0))
        self.expect_outcome(result=SUCCESS, state_string='shell 1 test passed')
        self.expect_log_file('warnings', re.compile("error: test2 \\[.*\nValueError: invalid literal for int\\(\\) with base 10: '_error2'\n\\]\n", re.MULTILINE | re.DOTALL))
        return self.run_step()