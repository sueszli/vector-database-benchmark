"""Unit tests for scripts/create_expression_parser.py."""
from __future__ import annotations
import os
import subprocess
from core.tests import test_utils
from . import common
from . import create_expression_parser
from . import setup

class CreateExpressionParserTests(test_utils.GenericTestBase):
    """Unit tests for scripts/create_expression_parser.py."""

    def test_expression_parser_is_produced_correctly(self) -> None:
        if False:
            return 10
        cmd_token_list = []

        def mock_check_call(cmd_tokens: list[str], **unused_kwargs: str) -> None:
            if False:
                i = 10
                return i + 15
            cmd_token_list.append(cmd_tokens)
        setup_script_args = []

        def mock_setup(args: list[str]) -> None:
            if False:
                print('Hello World!')
            setup_script_args.append(args)
        libraries_installed = []

        def mock_install_npm_library(library_name: str, library_version: str, path: str) -> None:
            if False:
                while True:
                    i = 10
            libraries_installed.append([library_name, library_version, path])
        swap_check_call = self.swap(subprocess, 'check_call', mock_check_call)
        swap_setup = self.swap(setup, 'main', mock_setup)
        swap_install_npm_library = self.swap(common, 'install_npm_library', mock_install_npm_library)
        expression_parser_definition = os.path.join('core', 'templates', 'expressions', 'parser.pegjs')
        expression_parser_js = os.path.join('core', 'templates', 'expressions', 'parser.js')
        cmd = [os.path.join(common.NODE_MODULES_PATH, 'pegjs', 'bin', 'pegjs'), expression_parser_definition, expression_parser_js]
        with swap_check_call, swap_setup, swap_install_npm_library:
            create_expression_parser.main(args=[])
        self.assertIsNot(len(setup_script_args), 0)
        self.assertEqual(libraries_installed[0][0], 'pegjs')
        self.assertIn(cmd, cmd_token_list)