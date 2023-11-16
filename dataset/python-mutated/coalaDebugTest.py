import os
import sys
import unittest
from unittest.mock import patch
from coalib.coala_main import run_coala
from coalib.output.printers.LogPrinter import LogPrinter
from coalib import coala
from pyprint.ConsolePrinter import ConsolePrinter
from coala_utils.ContextManagers import prepare_file
from coalib.output.Logging import configure_logging
from tests.TestUtilities import execute_coala, bear_test_module

class coalaDebugTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.old_argv = sys.argv

    def tearDown(self):
        if False:
            return 10
        sys.argv = self.old_argv

    def test_coala_main_bear__init__raises(self):
        if False:
            while True:
                i = 10
        with bear_test_module():
            with prepare_file(['#fixme  '], None) as (lines, filename):
                with self.assertRaisesRegex(RuntimeError, "^The bear ErrorTestBear does not fulfill all requirements\\. 'I_do_not_exist' is not installed\\.$"):
                    execute_coala(coala.main, 'coala', '-c', os.devnull, '-f', filename, '-b', 'ErrorTestBear', debug=True)

    def test_run_coala_bear__init__raises(self):
        if False:
            for i in range(10):
                print('nop')
        configure_logging()
        with bear_test_module():
            with prepare_file(['#fixme  '], None) as (lines, filename):
                with self.assertRaisesRegex(RuntimeError, "^The bear ErrorTestBear does not fulfill all requirements\\. 'I_do_not_exist' is not installed\\.$"):
                    run_coala(console_printer=ConsolePrinter(), log_printer=LogPrinter(), arg_list=('-c', os.devnull, '-f', filename, '-b', 'ErrorTestBear'), debug=True)

    def test_coala_main_bear_run_raises(self):
        if False:
            i = 10
            return i + 15
        with bear_test_module():
            with prepare_file(['#fixme  '], None) as (lines, filename):
                with self.assertRaisesRegex(RuntimeError, "^That's all the RaiseTestBear can do\\.$"):
                    execute_coala(coala.main, 'coala', '-c', os.devnull, '-f', filename, '-b', 'RaiseTestBear', debug=True)

    def test_run_coala_bear_run_raises(self):
        if False:
            while True:
                i = 10
        configure_logging()
        with bear_test_module():
            with prepare_file(['#fixme  '], None) as (lines, filename):
                with self.assertRaisesRegex(RuntimeError, "^That's all the RaiseTestBear can do\\.$"):
                    run_coala(console_printer=ConsolePrinter(), log_printer=LogPrinter(), arg_list=('-c', os.devnull, '-f', filename, '-b', 'RaiseTestBear'), debug=True)

    @patch('coalib.coala_modes.mode_json')
    def test_coala_main_mode_json_raises(self, mocked_mode_json):
        if False:
            while True:
                i = 10
        mocked_mode_json.side_effect = RuntimeError('Mocked mode_json fails.')
        with bear_test_module():
            with prepare_file(['#fixme  '], None) as (lines, filename):
                with self.assertRaisesRegex(RuntimeError, '^Mocked mode_json fails\\.$'):
                    execute_coala(coala.main, 'coala', '--json', '-c', os.devnull, '-f', filename, '-b', 'RaiseTestBear', debug=True)