"""Unit tests for scripts/run_acceptance_tests.py."""
from __future__ import annotations
import contextlib
import subprocess
import sys
from core.constants import constants
from core.tests import test_utils
from scripts import build
from scripts import common
from scripts import run_acceptance_tests
from scripts import scripts_test_utils
from scripts import servers
from typing import ContextManager, Optional

def mock_managed_long_lived_process(*unused_args: str, **unused_kwargs: str) -> ContextManager[scripts_test_utils.PopenStub]:
    if False:
        return 10
    'Mock method for replacing the managed_process() functions to simulate a\n    long-lived process. This process stays alive for 10 poll() calls, and\n    then terminates thereafter.\n\n    Returns:\n        Context manager. A context manager that always yields a mock\n        process.\n    '
    stub = scripts_test_utils.PopenStub(alive=True)

    def mock_poll(stub: scripts_test_utils.PopenStub) -> Optional[int]:
        if False:
            print('Hello World!')
        stub.poll_count += 1
        if stub.poll_count >= 10:
            stub.alive = False
        return None if stub.alive else stub.returncode
    stub.poll = lambda : mock_poll(stub)
    return contextlib.nullcontext(enter_result=stub)

def mock_managed_process(*unused_args: str, **unused_kwargs: str) -> ContextManager[scripts_test_utils.PopenStub]:
    if False:
        print('Hello World!')
    'Mock method for replacing the managed_process() functions.\n\n    Returns:\n        Context manager. A context manager that always yields a mock\n        process.\n    '
    return contextlib.nullcontext(enter_result=scripts_test_utils.PopenStub(alive=False))

class RunAcceptanceTestsTests(test_utils.GenericTestBase):
    """Test the run_acceptance_tests methods."""

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        self.exit_stack = contextlib.ExitStack()

        def mock_constants() -> None:
            if False:
                for i in range(10):
                    print('nop')
            print('mock_set_constants_to_default')
        self.swap_mock_set_constants_to_default = self.swap(common, 'set_constants_to_default', mock_constants)

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        try:
            self.exit_stack.close()
        finally:
            super().tearDown()

    def test_start_tests_when_other_instances_not_stopped(self) -> None:
        if False:
            print('Hello World!')
        self.exit_stack.enter_context(self.swap_with_checks(common, 'is_oppia_server_already_running', lambda *_: True))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_portserver', mock_managed_process))
        with self.assertRaisesRegex(SystemExit, '\n            Oppia server is already running. Try shutting all the servers down\n            before running the script.\n        '):
            run_acceptance_tests.main(args=['--suite', 'testSuite'])

    def test_start_tests_when_no_other_instance_running(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.exit_stack.enter_context(self.swap_with_checks(common, 'is_oppia_server_already_running', lambda *_: False))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'build_js_files', lambda *_, **__: None, expected_args=[(True,)]))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_elasticsearch_dev_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_firebase_auth_emulator', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_dev_appserver', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_redis_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_portserver', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_cloud_datastore_emulator', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_acceptance_tests_server', mock_managed_process, expected_kwargs=[{'suite_name': 'testSuite', 'stdout': subprocess.PIPE}]))
        self.exit_stack.enter_context(self.swap_with_checks(sys, 'exit', lambda _: None, expected_args=[(0,)]))
        with self.swap_mock_set_constants_to_default:
            run_acceptance_tests.main(args=['--suite', 'testSuite'])

    def test_work_with_non_ascii_chars(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def mock_managed_acceptance_tests_server(**unused_kwargs: str) -> ContextManager[scripts_test_utils.PopenStub]:
            if False:
                print('Hello World!')
            return contextlib.nullcontext(enter_result=scripts_test_utils.PopenStub(stdout='sample\n✓\noutput\n'.encode(encoding='utf-8'), alive=False))
        self.exit_stack.enter_context(self.swap_with_checks(common, 'is_oppia_server_already_running', lambda *_: False))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'build_js_files', lambda *_, **__: None, expected_args=[(True,)]))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_elasticsearch_dev_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_firebase_auth_emulator', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_dev_appserver', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_redis_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_cloud_datastore_emulator', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_acceptance_tests_server', mock_managed_acceptance_tests_server, expected_kwargs=[{'suite_name': 'testSuite', 'stdout': subprocess.PIPE}]))
        args = run_acceptance_tests._PARSER.parse_args(args=['--suite', 'testSuite'])
        with self.swap_mock_set_constants_to_default:
            (lines, _) = run_acceptance_tests.run_tests(args)
        self.assertEqual([line.decode('utf-8') for line in lines], ['sample', u'✓', 'output'])

    def test_start_tests_skip_build(self) -> None:
        if False:
            while True:
                i = 10
        self.exit_stack.enter_context(self.swap_with_checks(common, 'is_oppia_server_already_running', lambda *_: False))
        self.exit_stack.enter_context(self.swap_with_checks(common, 'modify_constants', lambda *_, **__: None, expected_kwargs=[{'prod_env': False}]))
        self.exit_stack.enter_context(self.swap_with_checks(common, 'set_constants_to_default', lambda : None))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_elasticsearch_dev_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_firebase_auth_emulator', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_dev_appserver', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_redis_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_webpack_compiler', mock_managed_process, called=False))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_portserver', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_cloud_datastore_emulator', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_acceptance_tests_server', mock_managed_process, expected_kwargs=[{'suite_name': 'testSuite', 'stdout': subprocess.PIPE}]))
        self.exit_stack.enter_context(self.swap_with_checks(sys, 'exit', lambda _: None, expected_args=[(0,)]))
        run_acceptance_tests.main(args=['--suite', 'testSuite', '--skip-build'])

    def test_start_tests_in_jasmine(self) -> None:
        if False:
            return 10
        self.exit_stack.enter_context(self.swap_with_checks(common, 'is_oppia_server_already_running', lambda *_: False))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'build_js_files', lambda *_, **__: None, expected_args=[(True,)]))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_elasticsearch_dev_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_firebase_auth_emulator', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_dev_appserver', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_redis_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_portserver', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_cloud_datastore_emulator', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_acceptance_tests_server', mock_managed_process, expected_kwargs=[{'suite_name': 'testSuite', 'stdout': subprocess.PIPE}]))
        self.exit_stack.enter_context(self.swap_with_checks(sys, 'exit', lambda _: None, expected_args=[(0,)]))
        with self.swap_mock_set_constants_to_default:
            run_acceptance_tests.main(args=['--suite', 'testSuite'])

    def test_start_tests_with_emulator_mode_false(self) -> None:
        if False:
            i = 10
            return i + 15
        self.exit_stack.enter_context(self.swap_with_checks(common, 'is_oppia_server_already_running', lambda *_: False))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'build_js_files', lambda *_, **__: None, expected_args=[(True,)]))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_elasticsearch_dev_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_firebase_auth_emulator', mock_managed_process, called=False))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_dev_appserver', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_redis_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_portserver', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_cloud_datastore_emulator', mock_managed_process, called=False))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_acceptance_tests_server', mock_managed_process, expected_kwargs=[{'suite_name': 'testSuite', 'stdout': subprocess.PIPE}]))
        self.exit_stack.enter_context(self.swap_with_checks(sys, 'exit', lambda _: None, expected_args=[(0,)]))
        with self.swap_mock_set_constants_to_default:
            with self.swap(constants, 'EMULATOR_MODE', False):
                run_acceptance_tests.main(args=['--suite', 'testSuite'])

    def test_start_tests_for_long_lived_process(self) -> None:
        if False:
            i = 10
            return i + 15
        self.exit_stack.enter_context(self.swap_with_checks(common, 'is_oppia_server_already_running', lambda *_: False))
        self.exit_stack.enter_context(self.swap_with_checks(build, 'build_js_files', lambda *_, **__: None, expected_args=[(True,)]))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_elasticsearch_dev_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_firebase_auth_emulator', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_dev_appserver', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_redis_server', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_cloud_datastore_emulator', mock_managed_process))
        self.exit_stack.enter_context(self.swap_with_checks(servers, 'managed_acceptance_tests_server', mock_managed_long_lived_process, expected_kwargs=[{'suite_name': 'testSuite', 'stdout': subprocess.PIPE}]))
        self.exit_stack.enter_context(self.swap_with_checks(sys, 'exit', lambda _: None, expected_args=[(0,)]))
        with self.swap_mock_set_constants_to_default:
            with self.swap(constants, 'EMULATOR_MODE', True):
                run_acceptance_tests.main(args=['--suite', 'testSuite'])