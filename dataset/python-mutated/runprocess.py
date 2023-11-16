from buildbot.test.steps import ExpectMasterShell
from buildbot.test.steps import _check_env_is_expected
from buildbot.util import runprocess

class MasterRunProcessMixin:
    long_message = True

    def setup_master_run_process(self):
        if False:
            for i in range(10):
                print('nop')
        self._master_run_process_patched = False
        self._expected_master_commands = []
        self._master_run_process_expect_env = {}

    def assert_all_commands_ran(self):
        if False:
            return 10
        self.assertEqual(self._expected_master_commands, [], 'assert all expected commands were run')

    def patched_run_process(self, reactor, command, workdir=None, env=None, collect_stdout=True, collect_stderr=True, stderr_is_error=False, io_timeout=300, runtime_timeout=3600, sigterm_timeout=5, initial_stdin=None, use_pty=False):
        if False:
            print('Hello World!')
        _check_env_is_expected(self, self._master_run_process_expect_env, env)
        if not self._expected_master_commands:
            self.fail(f'got command {command} when no further commands were expected')
        expect = self._expected_master_commands.pop(0)
        (rc, stdout, stderr) = expect._check(self, command, workdir, env)
        if not collect_stderr and stderr_is_error and stderr:
            rc = -1
        if collect_stdout and collect_stderr:
            return (rc, stdout, stderr)
        if collect_stdout:
            return (rc, stdout)
        if collect_stderr:
            return (rc, stderr)
        return rc

    def _patch_runprocess(self):
        if False:
            while True:
                i = 10
        if not self._master_run_process_patched:
            self.patch(runprocess, 'run_process', self.patched_run_process)
            self._master_run_process_patched = True

    def add_run_process_expect_env(self, d):
        if False:
            i = 10
            return i + 15
        self._master_run_process_expect_env.update(d)

    def expect_commands(self, *exp):
        if False:
            i = 10
            return i + 15
        for e in exp:
            if not isinstance(e, ExpectMasterShell):
                raise RuntimeError('All expectation must be an instance of ExpectMasterShell')
        self._patch_runprocess()
        self._expected_master_commands.extend(exp)