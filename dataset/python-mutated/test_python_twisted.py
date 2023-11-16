import textwrap
from twisted.trial import unittest
from buildbot.process.properties import Property
from buildbot.process.results import FAILURE
from buildbot.process.results import SKIPPED
from buildbot.process.results import SUCCESS
from buildbot.process.results import WARNINGS
from buildbot.steps import python_twisted
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import TestBuildStepMixin
failureLog = 'buildbot.test.unit.test_steps_python_twisted.Trial.testProperties ... [FAILURE]\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_env ... [FAILURE]\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_env_nodupe ... [FAILURE]/home/dustin/code/buildbot/t/buildbot/master/buildbot/test/fake/logfile.py:92: UserWarning: step uses removed LogFile method `getText`\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_env_supplement ... [FAILURE]/home/dustin/code/buildbot/t/buildbot/master/buildbot/test/fake/logfile.py:92: UserWarning: step uses removed LogFile method `getText`\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_jobs ... [FAILURE]/home/dustin/code/buildbot/t/buildbot/master/buildbot/test/fake/logfile.py:92: UserWarning: step uses removed LogFile method `getText`\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_jobsProperties ... [FAILURE]\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_plural ... [FAILURE]\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_singular ... [FAILURE]\n\n===============================================================================\n[FAIL]\nTraceback (most recent call last):\n  File "/home/dustin/code/buildbot/t/buildbot/master/buildbot/test/util/steps.py", line 244, in check\n    "expected step outcome")\n  File "/home/dustin/code/buildbot/t/buildbot/sandbox/lib/python2.7/site-packages/twisted/trial/_synctest.py", line 356, in assertEqual\n    % (msg, pformat(first), pformat(second)))\ntwisted.trial.unittest.FailTest: expected step outcome\nnot equal:\na = {\'result\': 3, \'status_text\': [\'2 tests\', \'passed\']}\nb = {\'result\': 0, \'status_text\': [\'2 tests\', \'passed\']}\n\n\nbuildbot.test.unit.test_steps_python_twisted.Trial.testProperties\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_plural\n===============================================================================\n[FAIL]\nTraceback (most recent call last):\n  File "/home/dustin/code/buildbot/t/buildbot/master/buildbot/test/util/steps.py", line 244, in check\n    "expected step outcome")\n  File "/home/dustin/code/buildbot/t/buildbot/sandbox/lib/python2.7/site-packages/twisted/trial/_synctest.py", line 356, in assertEqual\n    % (msg, pformat(first), pformat(second)))\ntwisted.trial.unittest.FailTest: expected step outcome\nnot equal:\na = {\'result\': 3, \'status_text\': [\'no tests\', \'run\']}\nb = {\'result\': 0, \'status_text\': [\'no tests\', \'run\']}\n\n\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_env\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_env_nodupe\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_env_supplement\n===============================================================================\n[FAIL]\nTraceback (most recent call last):\n  File "/home/dustin/code/buildbot/t/buildbot/master/buildbot/test/util/steps.py", line 244, in check\n    "expected step outcome")\n  File "/home/dustin/code/buildbot/t/buildbot/sandbox/lib/python2.7/site-packages/twisted/trial/_synctest.py", line 356, in assertEqual\n    % (msg, pformat(first), pformat(second)))\ntwisted.trial.unittest.FailTest: expected step outcome\nnot equal:\na = {\'result\': 3, \'status_text\': [\'1 test\', \'passed\']}\nb = {\'result\': 0, \'status_text\': [\'1 test\', \'passed\']}\n\n\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_jobs\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_jobsProperties\nbuildbot.test.unit.test_steps_python_twisted.Trial.test_run_singular\n-------------------------------------------------------------------------------\nRan 8 tests in 0.101s\n\nFAILED (failures=8)\n'

class Trial(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            print('Hello World!')
        return self.tear_down_test_build_step()

    def test_run_env(self):
        if False:
            return 10
        self.setup_step(python_twisted.Trial(workdir='build', tests='testname', testpath=None, env={'PYTHONPATH': 'somepath'}))
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}, env={'PYTHONPATH': 'somepath'}).stdout('Ran 0 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='no tests run')
        return self.run_step()

    def test_run_env_supplement(self):
        if False:
            return 10
        self.setup_step(python_twisted.Trial(workdir='build', tests='testname', testpath='path1', env={'PYTHONPATH': ['path2', 'path3']}))
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}, env={'PYTHONPATH': ['path1', 'path2', 'path3']}).stdout('Ran 0 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='no tests run')
        return self.run_step()

    def test_run_env_nodupe(self):
        if False:
            while True:
                i = 10
        self.setup_step(python_twisted.Trial(workdir='build', tests='testname', testpath='path2', env={'PYTHONPATH': ['path1', 'path2']}))
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}, env={'PYTHONPATH': ['path1', 'path2']}).stdout('Ran 0 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='no tests run')
        return self.run_step()

    def test_run_singular(self):
        if False:
            print('Hello World!')
        self.setup_step(python_twisted.Trial(workdir='build', tests='testname', testpath=None))
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}).stdout('Ran 1 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='1 test passed')
        return self.run_step()

    def test_run_plural(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(python_twisted.Trial(workdir='build', tests='testname', testpath=None))
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}).stdout('Ran 2 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='2 tests passed')
        return self.run_step()

    def test_run_failure(self):
        if False:
            while True:
                i = 10
        self.setup_step(python_twisted.Trial(workdir='build', tests='testname', testpath=None))
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}).stdout(failureLog).exit(1))
        self.expect_outcome(result=FAILURE, state_string='tests 8 failures (failure)')
        self.expect_log_file('problems', failureLog.split('\n\n', 1)[1][:-1] + '\nprogram finished with exit code 1')
        self.expect_log_file('warnings', textwrap.dedent('                buildbot.test.unit.test_steps_python_twisted.Trial.test_run_env_nodupe ... [FAILURE]/home/dustin/code/buildbot/t/buildbot/master/buildbot/test/fake/logfile.py:92: UserWarning: step uses removed LogFile method `getText`\n                buildbot.test.unit.test_steps_python_twisted.Trial.test_run_env_supplement ... [FAILURE]/home/dustin/code/buildbot/t/buildbot/master/buildbot/test/fake/logfile.py:92: UserWarning: step uses removed LogFile method `getText`\n                buildbot.test.unit.test_steps_python_twisted.Trial.test_run_jobs ... [FAILURE]/home/dustin/code/buildbot/t/buildbot/master/buildbot/test/fake/logfile.py:92: UserWarning: step uses removed LogFile method `getText`\n                buildbot.test.unit.test_steps_python_twisted.Trial.test_run_jobsProperties ... [FAILURE]\n                '))
        return self.run_step()

    def test_renderable_properties(self):
        if False:
            print('Hello World!')
        self.setup_step(python_twisted.Trial(workdir='build', tests=Property('test_list'), testpath=None))
        self.properties.setProperty('test_list', ['testname'], 'Test')
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}).stdout('Ran 2 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='2 tests passed')
        return self.run_step()

    def test_build_changed_files(self):
        if False:
            return 10
        self.setup_step(python_twisted.Trial(workdir='build', testChanges=True, testpath=None), build_files=['my/test/file.py', 'my/test/file2.py'])
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', '--testmodule=my/test/file.py', '--testmodule=my/test/file2.py'], logfiles={'test.log': '_trial_temp/test.log'}).stdout('Ran 2 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='2 tests passed')
        return self.run_step()

    def test_test_path_env_python_path(self):
        if False:
            return 10
        self.setup_step(python_twisted.Trial(workdir='build', tests='testname', testpath='custom/test/path', env={'PYTHONPATH': '/existing/pypath'}))
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}, env={'PYTHONPATH': ['custom/test/path', '/existing/pypath']}).stdout('Ran 2 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='2 tests passed')
        return self.run_step()

    def test_custom_reactor(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(python_twisted.Trial(workdir='build', reactor='customreactor', tests='testname', testpath=None))
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', '--reactor=customreactor', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}).stdout('Ran 2 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='2 tests passed (custom)')
        return self.run_step()

    def test_custom_python(self):
        if False:
            return 10
        self.setup_step(python_twisted.Trial(workdir='build', tests='testname', python='/bin/mypython', testpath=None))
        self.expect_commands(ExpectShell(workdir='build', command=['/bin/mypython', 'trial', '--reporter=bwverbose', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}).stdout('Ran 2 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='2 tests passed')
        return self.run_step()

    def test_randomly(self):
        if False:
            return 10
        self.setup_step(python_twisted.Trial(workdir='build', randomly=True, tests='testname', testpath=None))
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', '--random=0', 'testname'], logfiles={'test.log': '_trial_temp/test.log'}).stdout('Ran 2 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='2 tests passed')
        return self.run_step()

    def test_run_jobs(self):
        if False:
            print('Hello World!')
        "\n        The C{jobs} kwarg should correspond to trial's -j option (\n        included since Twisted 12.3.0), and make corresponding changes to\n        logfiles.\n        "
        self.setup_step(python_twisted.Trial(workdir='build', tests='testname', testpath=None, jobs=2))
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', '--jobs=2', 'testname'], logfiles={'test.0.log': '_trial_temp/0/test.log', 'err.0.log': '_trial_temp/0/err.log', 'out.0.log': '_trial_temp/0/out.log', 'test.1.log': '_trial_temp/1/test.log', 'err.1.log': '_trial_temp/1/err.log', 'out.1.log': '_trial_temp/1/out.log'}).stdout('Ran 1 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='1 test passed')
        return self.run_step()

    def test_run_jobsProperties(self):
        if False:
            i = 10
            return i + 15
        '\n        C{jobs} should accept Properties\n        '
        self.setup_step(python_twisted.Trial(workdir='build', tests='testname', jobs=Property('jobs_count'), testpath=None))
        self.properties.setProperty('jobs_count', '2', 'Test')
        self.expect_commands(ExpectShell(workdir='build', command=['trial', '--reporter=bwverbose', '--jobs=2', 'testname'], logfiles={'test.0.log': '_trial_temp/0/test.log', 'err.0.log': '_trial_temp/0/err.log', 'out.0.log': '_trial_temp/0/out.log', 'test.1.log': '_trial_temp/1/test.log', 'err.1.log': '_trial_temp/1/err.log', 'out.1.log': '_trial_temp/1/out.log'}).stdout('Ran 1 tests\n').exit(0))
        self.expect_outcome(result=SUCCESS, state_string='1 test passed')
        return self.run_step()

class HLint(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        return self.tear_down_test_build_step()

    def test_run_ok(self):
        if False:
            return 10
        self.setup_step(python_twisted.HLint(workdir='build'), build_files=['foo.xhtml'])
        self.expect_commands(ExpectShell(workdir='build', command=['bin/lore', '-p', '--output', 'lint', 'foo.xhtml']).stdout('dunno what hlint output looks like..\n').exit(0))
        self.expect_log_file('files', 'foo.xhtml\n')
        self.expect_outcome(result=SUCCESS, state_string='0 hlints')
        return self.run_step()

    def test_custom_python(self):
        if False:
            while True:
                i = 10
        self.setup_step(python_twisted.HLint(workdir='build', python='/bin/mypython'), build_files=['foo.xhtml'])
        self.expect_commands(ExpectShell(workdir='build', command=['/bin/mypython', 'bin/lore', '-p', '--output', 'lint', 'foo.xhtml']).exit(0))
        self.expect_log_file('files', 'foo.xhtml\n')
        self.expect_outcome(result=SUCCESS, state_string='0 hlints')
        return self.run_step()

    def test_command_failure(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(python_twisted.HLint(workdir='build'), build_files=['foo.xhtml'])
        self.expect_commands(ExpectShell(workdir='build', command=['bin/lore', '-p', '--output', 'lint', 'foo.xhtml']).exit(1))
        self.expect_log_file('files', 'foo.xhtml\n')
        self.expect_outcome(result=FAILURE, state_string='hlint (failure)')
        return self.run_step()

    def test_no_build_files(self):
        if False:
            print('Hello World!')
        self.setup_step(python_twisted.HLint(workdir='build'))
        self.expect_outcome(result=SKIPPED, state_string='hlint (skipped)')
        return self.run_step()

    def test_run_warnings(self):
        if False:
            while True:
                i = 10
        self.setup_step(python_twisted.HLint(workdir='build'), build_files=['foo.xhtml'])
        self.expect_commands(ExpectShell(workdir='build', command=['bin/lore', '-p', '--output', 'lint', 'foo.xhtml']).stdout('colon: meaning warning\n').exit(0))
        self.expect_log_file('warnings', 'colon: meaning warning')
        self.expect_outcome(result=WARNINGS, state_string='1 hlint (warnings)')
        return self.run_step()

class RemovePYCs(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tear_down_test_build_step()

    def test_run_ok(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(python_twisted.RemovePYCs())
        self.expect_commands(ExpectShell(workdir='wkdir', command=['find', '.', '-name', "'*.pyc'", '-exec', 'rm', '{}', ';']).exit(0))
        self.expect_outcome(result=SUCCESS, state_string='remove .pycs')
        return self.run_step()