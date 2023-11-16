"""Unit tests for scripts/typescript_checks.py."""
from __future__ import annotations
import json
import os
import subprocess
from core import utils
from core.tests import test_utils
from . import typescript_checks
TEST_SOURCE_DIR = os.path.join('core', 'tests', 'build_sources')
MOCK_COMPILED_JS_DIR = os.path.join(TEST_SOURCE_DIR, 'compiled_js_dir', '')

class TypescriptChecksTests(test_utils.GenericTestBase):
    """Test the typescript checks."""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        process = subprocess.Popen(['test'], stdout=subprocess.PIPE, encoding='utf-8')

        def mock_popen(unused_cmd: str, stdout: str, encoding: str) -> subprocess.Popen[str]:
            if False:
                for i in range(10):
                    print('nop')
            return process
        self.popen_swap = self.swap(subprocess, 'Popen', mock_popen)

    def test_compiled_js_dir_validation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that typescript_checks.COMPILED_JS_DIR is validated correctly\n        with outDir in typescript_checks.TSCONFIG_FILEPATH.\n        '
        with self.popen_swap:
            typescript_checks.compile_and_check_typescript(typescript_checks.TSCONFIG_FILEPATH)
            out_dir = ''
            with utils.open_file(typescript_checks.TSCONFIG_FILEPATH, 'r') as f:
                config_data = json.load(f)
                out_dir = os.path.join(config_data['compilerOptions']['outDir'], '')
            compiled_js_dir_swap = self.swap(typescript_checks, 'COMPILED_JS_DIR', MOCK_COMPILED_JS_DIR)
            with compiled_js_dir_swap, self.assertRaisesRegex(Exception, 'COMPILED_JS_DIR: %s does not match the output directory in %s: %s' % (MOCK_COMPILED_JS_DIR, typescript_checks.TSCONFIG_FILEPATH, out_dir)):
                typescript_checks.compile_and_check_typescript(typescript_checks.TSCONFIG_FILEPATH)

    def test_compiled_js_dir_is_deleted_before_compilation(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that compiled_js_dir is deleted before a fresh compilation.'

        def mock_validate_compiled_js_dir() -> None:
            if False:
                while True:
                    i = 10
            pass
        compiled_js_dir_swap = self.swap(typescript_checks, 'COMPILED_JS_DIR', MOCK_COMPILED_JS_DIR)
        validate_swap = self.swap(typescript_checks, 'validate_compiled_js_dir', mock_validate_compiled_js_dir)
        with self.popen_swap, compiled_js_dir_swap, validate_swap:
            if not os.path.exists(os.path.dirname(MOCK_COMPILED_JS_DIR)):
                os.mkdir(os.path.dirname(MOCK_COMPILED_JS_DIR))
            typescript_checks.compile_and_check_typescript(typescript_checks.STRICT_TSCONFIG_FILEPATH)
            self.assertFalse(os.path.exists(os.path.dirname(MOCK_COMPILED_JS_DIR)))

    def test_no_error_for_valid_compilation_of_tsconfig(self) -> None:
        if False:
            return 10
        'Test that no error is produced if stdout is empty.'
        with self.popen_swap:
            typescript_checks.compile_and_check_typescript(typescript_checks.TSCONFIG_FILEPATH)

    def test_no_error_for_valid_compilation_of_strict_tsconfig(self) -> None:
        if False:
            print('Hello World!')
        'Test that no error is produced if stdout is empty.'
        with self.popen_swap:
            typescript_checks.compile_and_check_typescript(typescript_checks.STRICT_TSCONFIG_FILEPATH)

    def test_error_is_raised_for_invalid_compilation_of_tsconfig(self) -> None:
        if False:
            print('Hello World!')
        'Test that error is produced if stdout is not empty.'
        process = subprocess.Popen(['echo', 'test'], stdout=subprocess.PIPE, encoding='utf-8')

        def mock_popen_for_errors(unused_cmd: str, stdout: str, encoding: str) -> subprocess.Popen[str]:
            if False:
                while True:
                    i = 10
            return process
        with self.swap(subprocess, 'Popen', mock_popen_for_errors):
            with self.assertRaisesRegex(SystemExit, '1'):
                typescript_checks.compile_and_check_typescript(typescript_checks.TSCONFIG_FILEPATH)

    def test_error_is_raised_for_invalid_compilation_of_strict_tsconfig(self) -> None:
        if False:
            print('Hello World!')
        'Test that error is produced if stdout is not empty.'
        with self.swap(typescript_checks, 'TS_STRICT_EXCLUDE_PATHS', []):
            with self.assertRaisesRegex(SystemExit, '1'):
                typescript_checks.compile_and_check_typescript(typescript_checks.STRICT_TSCONFIG_FILEPATH)

    def test_error_is_raised_for_invalid_compilation_of_temp_strict_tsconfig(self) -> None:
        if False:
            while True:
                i = 10
        'Test that error is produced if stdout is not empty.'

        class MockOutput:
            """This class simulates a process stdout."""

            def __init__(self, call_counter: int=0) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.call_counter = call_counter

            def readline(self) -> str:
                if False:
                    i = 10
                    return i + 15
                'This mocks the readline() method which reads and returns\n                a single line. It stops when it hits the EOF or an empty\n                string.\n\n                Returns:\n                    str. A single line of process output.\n                '
                self.call_counter = self.call_counter + 1
                return_values = {1: 'core/templates/App.ts', 2: 'core/new_directory/new_file.ts', 3: ''}
                return return_values[self.call_counter]

        class MockProcess:
            stdout = MockOutput()

        def mock_popen_for_errors(unused_cmd: str, stdout: str, encoding: str) -> MockProcess:
            if False:
                i = 10
                return i + 15
            return MockProcess()
        swap_path_exists = self.swap(os.path, 'exists', lambda _: False)
        with self.swap(subprocess, 'Popen', mock_popen_for_errors):
            with self.assertRaisesRegex(SystemExit, '1'), swap_path_exists:
                typescript_checks.compile_temp_strict_tsconfig(typescript_checks.STRICT_TSCONFIG_FILEPATH, ['core/templates/App.ts', 'core/new_directory/new_file.ts'])

    def test_config_path_when_no_arg_is_used(self) -> None:
        if False:
            print('Hello World!')
        'Test if the config path is correct when no arg is used.'

        def mock_compile_and_check_typescript(config_path: str) -> None:
            if False:
                i = 10
                return i + 15
            self.assertEqual(config_path, typescript_checks.TSCONFIG_FILEPATH)
        compile_and_check_typescript_swap = self.swap(typescript_checks, 'compile_and_check_typescript', mock_compile_and_check_typescript)
        with compile_and_check_typescript_swap:
            typescript_checks.main(args=[])

    def test_config_path_when_strict_checks_arg_is_used(self) -> None:
        if False:
            return 10
        'Test if the config path is correct when strict checks arg is used.'

        def mock_compile_and_check_typescript(config_path: str) -> None:
            if False:
                while True:
                    i = 10
            self.assertEqual(config_path, typescript_checks.STRICT_TSCONFIG_FILEPATH)
        compile_and_check_typescript_swap = self.swap(typescript_checks, 'compile_and_check_typescript', mock_compile_and_check_typescript)
        with compile_and_check_typescript_swap:
            typescript_checks.main(args=['--strict_checks'])