"""Unit tests for scripts/run_tests.py."""
from __future__ import annotations
import builtins
import subprocess
from core.tests import test_utils
from scripts import install_third_party_libs
from scripts import run_frontend_tests
from scripts import setup
from scripts import setup_gae

class RunTestsTests(test_utils.GenericTestBase):
    """Unit tests for scripts/run_tests.py."""

    def test_all_tests_are_run_correctly(self) -> None:
        if False:
            return 10
        print_arr: list[str] = []

        def mock_print(msg: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            print_arr.append(msg)
        print_swap = self.swap(builtins, 'print', mock_print)
        scripts_called = {'setup': False, 'setup_gae': False, 'run_frontend_tests': False, 'run_backend_tests': False, 'run_e2e_tests': False}

        def mock_setup(args: list[str]) -> None:
            if False:
                while True:
                    i = 10
            scripts_called['setup'] = True

        def mock_setup_gae(args: list[str]) -> None:
            if False:
                while True:
                    i = 10
            scripts_called['setup_gae'] = True

        def mock_frontend_tests(args: list[str]) -> None:
            if False:
                while True:
                    i = 10
            scripts_called['run_frontend_tests'] = True

        def mock_backend_tests(args: list[str]) -> None:
            if False:
                i = 10
                return i + 15
            scripts_called['run_backend_tests'] = True

        def mock_popen(cmd: str, shell: bool) -> None:
            if False:
                for i in range(10):
                    print('nop')
            if cmd == 'bash scripts/run_e2e_tests.sh' and shell:
                scripts_called['run_e2e_tests'] = True

        def mock_install_third_party_libs() -> None:
            if False:
                while True:
                    i = 10
            pass
        swap_install_third_party_libs = self.swap(install_third_party_libs, 'main', mock_install_third_party_libs)
        swap_setup = self.swap(setup, 'main', mock_setup)
        swap_setup_gae = self.swap(setup_gae, 'main', mock_setup_gae)
        swap_frontend_tests = self.swap(run_frontend_tests, 'main', mock_frontend_tests)
        swap_popen = self.swap(subprocess, 'Popen', mock_popen)
        with swap_install_third_party_libs:
            from scripts import run_backend_tests
            from scripts import run_tests
            swap_backend_tests = self.swap(run_backend_tests, 'main', mock_backend_tests)
            with print_swap, swap_setup, swap_setup_gae, swap_popen:
                with swap_frontend_tests, swap_backend_tests:
                    run_tests.main(args=[])
        for script in scripts_called:
            self.assertTrue(script)
        self.assertIn('SUCCESS    All frontend, backend and end-to-end tests passed!', print_arr)