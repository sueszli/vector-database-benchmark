from __future__ import absolute_import
import os
import sys
import glob
from st2tests.base import IntegrationTestCase
from st2common.util.shell import run_command
from st2tests import config as test_config
from st2tests.fixtures.packs.all_packs_glob import PACKS_PATH
from st2tests.fixtures.packs.dummy_pack_1.fixture import PACK_PATH as DUMMY_PACK_1_PATH
from st2tests.fixtures.packs.dummy_pack_4.fixture import PACK_PATH as DUMMY_PACK_4_PATH
from st2tests.fixtures.packs.runners.fixture import FIXTURE_PATH as RUNNER_DIRS
from st2tests.fixtures.packs_1.dummy_pack_4.fixture import PACK_PATH as EMPTY_PACK_PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, '../../bin/st2-register-content')
SCRIPT_PATH = os.path.abspath(SCRIPT_PATH)
BASE_CMD_ARGS = [sys.executable, SCRIPT_PATH, '--config-file=conf/st2.tests.conf', '-v']
BASE_REGISTER_ACTIONS_CMD_ARGS = BASE_CMD_ARGS + ['--register-actions']
PACKS_COUNT = len(glob.glob(f'{PACKS_PATH}/*/pack.yaml'))
assert PACKS_COUNT >= 2

class ContentRegisterScriptTestCase(IntegrationTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(ContentRegisterScriptTestCase, self).setUp()
        test_config.parse_args()

    def test_register_from_pack_success(self):
        if False:
            return 10
        pack_dir = DUMMY_PACK_1_PATH
        runner_dirs = RUNNER_DIRS
        opts = ['--register-pack=%s' % pack_dir, '--register-runner-dir=%s' % runner_dirs]
        cmd = BASE_REGISTER_ACTIONS_CMD_ARGS + opts
        (exit_code, _, stderr) = run_command(cmd=cmd)
        self.assertIn('Registered 3 actions.', stderr)
        self.assertEqual(exit_code, 0)

    def test_register_from_pack_fail_on_failure_pack_dir_doesnt_exist(self):
        if False:
            while True:
                i = 10
        pack_dir = 'doesntexistblah'
        runner_dirs = RUNNER_DIRS
        opts = ['--register-pack=%s' % pack_dir, '--register-runner-dir=%s' % runner_dirs, '--register-no-fail-on-failure']
        cmd = BASE_REGISTER_ACTIONS_CMD_ARGS + opts
        (exit_code, _, _) = run_command(cmd=cmd)
        self.assertEqual(exit_code, 0)
        opts = ['--register-pack=%s' % pack_dir, '--register-runner-dir=%s' % runner_dirs, '--register-fail-on-failure']
        cmd = BASE_REGISTER_ACTIONS_CMD_ARGS + opts
        (exit_code, _, stderr) = run_command(cmd=cmd)
        self.assertIn('Directory "doesntexistblah" doesn\'t exist', stderr)
        self.assertEqual(exit_code, 1)

    def test_register_from_pack_action_metadata_fails_validation(self):
        if False:
            return 10
        pack_dir = DUMMY_PACK_4_PATH
        runner_dirs = RUNNER_DIRS
        opts = ['--register-pack=%s' % pack_dir, '--register-no-fail-on-failure', '--register-runner-dir=%s' % runner_dirs]
        cmd = BASE_REGISTER_ACTIONS_CMD_ARGS + opts
        (exit_code, _, stderr) = run_command(cmd=cmd)
        self.assertIn('Registered 0 actions.', stderr)
        self.assertEqual(exit_code, 0)
        pack_dir = DUMMY_PACK_4_PATH
        opts = ['--register-pack=%s' % pack_dir, '--register-fail-on-failure', '--register-runner-dir=%s' % runner_dirs]
        cmd = BASE_REGISTER_ACTIONS_CMD_ARGS + opts
        (exit_code, _, stderr) = run_command(cmd=cmd)
        self.assertIn("object has no attribute 'get'", stderr)
        self.assertEqual(exit_code, 1)

    def test_register_from_packs_doesnt_throw_on_missing_pack_resource_folder(self):
        if False:
            while True:
                i = 10
        self.assertIn('fixtures/packs_1/', EMPTY_PACK_PATH)
        cmd = [sys.executable, SCRIPT_PATH, '--config-file=conf/st2.tests1.conf', '-v', '--register-sensors']
        (exit_code, _, stderr) = run_command(cmd=cmd)
        self.assertIn('Registered 0 sensors.', stderr, 'Actual stderr: %s' % stderr)
        self.assertEqual(exit_code, 0)
        cmd = [sys.executable, SCRIPT_PATH, '--config-file=conf/st2.tests1.conf', '-v', '--register-all', '--register-no-fail-on-failure']
        (exit_code, _, stderr) = run_command(cmd=cmd)
        self.assertIn('Registered 0 actions.', stderr)
        self.assertIn('Registered 0 sensors.', stderr)
        self.assertIn('Registered 0 rules.', stderr)
        self.assertEqual(exit_code, 0)

    def test_register_all_and_register_setup_virtualenvs(self):
        if False:
            for i in range(10):
                print('nop')
        pack_dir = DUMMY_PACK_1_PATH
        cmd = BASE_CMD_ARGS + ['--register-pack=%s' % pack_dir, '--register-all', '--register-setup-virtualenvs', '--register-no-fail-on-failure']
        (exit_code, stdout, stderr) = run_command(cmd=cmd)
        self.assertIn('Registering actions', stderr, 'Actual stderr: %s' % stderr)
        self.assertIn('Registering rules', stderr)
        self.assertIn('Setup virtualenv for %s pack(s)' % '1', stderr)
        self.assertEqual(exit_code, 0)

    def test_register_setup_virtualenvs(self):
        if False:
            for i in range(10):
                print('nop')
        pack_dir = DUMMY_PACK_1_PATH
        cmd = BASE_CMD_ARGS + ['--register-pack=%s' % pack_dir, '--register-setup-virtualenvs', '--register-no-fail-on-failure']
        (exit_code, stdout, stderr) = run_command(cmd=cmd)
        self.assertIn('Setting up virtualenv for pack "dummy_pack_1"', stderr)
        self.assertIn('Setup virtualenv for 1 pack(s)', stderr)
        self.assertEqual(exit_code, 0)

    def test_register_recreate_virtualenvs(self):
        if False:
            i = 10
            return i + 15
        pack_dir = DUMMY_PACK_1_PATH
        cmd = BASE_CMD_ARGS + ['--register-pack=%s' % pack_dir, '--register-setup-virtualenvs', '--register-no-fail-on-failure']
        (exit_code, stdout, stderr) = run_command(cmd=cmd)
        self.assertIn('Setting up virtualenv for pack "dummy_pack_1"', stderr)
        self.assertIn('Setup virtualenv for 1 pack(s)', stderr)
        self.assertEqual(exit_code, 0)
        pack_dir = DUMMY_PACK_1_PATH
        cmd = BASE_CMD_ARGS + ['--register-pack=%s' % pack_dir, '--register-recreate-virtualenvs', '--register-no-fail-on-failure']
        (exit_code, stdout, stderr) = run_command(cmd=cmd)
        self.assertIn('Setting up virtualenv for pack "dummy_pack_1"', stderr)
        self.assertIn('Virtualenv successfully removed.', stderr)
        self.assertIn('Setup virtualenv for 1 pack(s)', stderr)
        self.assertEqual(exit_code, 0)