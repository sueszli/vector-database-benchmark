from twisted.internet import defer
from twisted.trial import unittest
from buildbot.process.results import EXCEPTION
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS
from buildbot.process.results import WARNINGS
from buildbot.process.results import Results
from buildbot.steps import mswin
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import TestBuildStepMixin

class TestRobocopySimple(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):
    """
    Test L{Robocopy} command building.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tear_down_test_build_step()

    def _run_simple_test(self, source, destination, expected_args=None, expected_code=0, expected_res=SUCCESS, **kwargs):
        if False:
            return 10
        s = mswin.Robocopy(source, destination, **kwargs)
        self.setup_step(s)
        s.rendered = True
        command = ['robocopy', source, destination]
        if expected_args:
            command += expected_args
        command += ['/TEE', '/NP']
        self.expect_commands(ExpectShell(workdir='wkdir', command=command).exit(expected_code))
        state_string = f"'robocopy {source} ...'"
        if expected_res != SUCCESS:
            state_string += f' ({Results[expected_res]})'
        self.expect_outcome(result=expected_res, state_string=state_string)
        return self.run_step()

    def test_copy(self):
        if False:
            return 10
        return self._run_simple_test('D:\\source', 'E:\\dest')

    def test_copy_files(self):
        if False:
            for i in range(10):
                print('nop')
        return self._run_simple_test('D:\\source', 'E:\\dest', files=['a.txt', 'b.txt', '*.log'], expected_args=['a.txt', 'b.txt', '*.log'])

    def test_copy_recursive(self):
        if False:
            for i in range(10):
                print('nop')
        return self._run_simple_test('D:\\source', 'E:\\dest', recursive=True, expected_args=['/E'])

    def test_mirror_files(self):
        if False:
            i = 10
            return i + 15
        return self._run_simple_test('D:\\source', 'E:\\dest', files=['*.foo'], mirror=True, expected_args=['*.foo', '/MIR'])

    def test_move_files(self):
        if False:
            for i in range(10):
                print('nop')
        return self._run_simple_test('D:\\source', 'E:\\dest', files=['*.foo'], move=True, expected_args=['*.foo', '/MOVE'])

    def test_exclude(self):
        if False:
            return 10
        return self._run_simple_test('D:\\source', 'E:\\dest', files=['blah*'], exclude=['*.foo', '*.bar'], expected_args=['blah*', '/XF', '*.foo', '*.bar'])

    def test_exclude_files(self):
        if False:
            for i in range(10):
                print('nop')
        return self._run_simple_test('D:\\source', 'E:\\dest', files=['blah*'], exclude_files=['*.foo', '*.bar'], expected_args=['blah*', '/XF', '*.foo', '*.bar'])

    def test_exclude_dirs(self):
        if False:
            return 10
        return self._run_simple_test('D:\\source', 'E:\\dest', files=['blah*'], exclude_dirs=['foo', 'bar'], expected_args=['blah*', '/XD', 'foo', 'bar'])

    def test_custom_opts(self):
        if False:
            print('Hello World!')
        return self._run_simple_test('D:\\source', 'E:\\dest', files=['*.foo'], custom_opts=['/R:10', '/W:60'], expected_args=['*.foo', '/R:10', '/W:60'])

    def test_verbose_output(self):
        if False:
            for i in range(10):
                print('nop')
        return self._run_simple_test('D:\\source', 'E:\\dest', files=['*.foo'], verbose=True, expected_args=['*.foo', '/V', '/TS', '/FP'])

    @defer.inlineCallbacks
    def test_codes(self):
        if False:
            while True:
                i = 10
        for i in [0, 1]:
            yield self._run_simple_test('D:\\source', 'E:\\dest', expected_code=i, expected_res=SUCCESS)
        for i in range(2, 8):
            yield self._run_simple_test('D:\\source', 'E:\\dest', expected_code=i, expected_res=WARNINGS)
        for i in range(8, 32):
            yield self._run_simple_test('D:\\source', 'E:\\dest', expected_code=i, expected_res=FAILURE)
        yield self._run_simple_test('D:\\source', 'E:\\dest', expected_code=32, expected_res=EXCEPTION)