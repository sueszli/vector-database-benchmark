import os
import locale
import sys
from io import StringIO
from pathlib import Path
from parameterized import parameterized
import pytest
from tests.integration.local.invoke.layer_utils import LayerUtils
from .invoke_integ_base import InvokeIntegBase
from tests.testing_utils import IS_WINDOWS, RUNNING_ON_CI, RUNNING_TEST_FOR_MASTER_ON_CI, RUN_BY_CANARY
from samcli.commands.local.invoke.cli import cli
from samcli.cli.context import Context
from click.testing import CliRunner

class TestSamPython36HelloWorldNonUTF8Integration(InvokeIntegBase):
    template = Path('template.yml')

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()

        def mock_getpreferredencoding(do_setlocale=True):
            if False:
                return 10
            return 'cp932'
        cls._original_getpreferredencoding = locale.getpreferredencoding
        locale.getpreferredencoding = mock_getpreferredencoding

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        locale.getpreferredencoding = cls._original_getpreferredencoding

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        if False:
            for i in range(10):
                print('nop')
        self._caplog = caplog

    @parameterized.expand([('MyReallyCoolFunction',), ('HelloWorldServerlessFunction',), ('HelloWorldServerlessWithFunctionNameRefFunction',)])
    def test_invoke_returns_execpted_results(self, function_name):
        if False:
            print('Hello World!')
        runner = CliRunner(mix_stderr=False)
        self._caplog.set_level(100000)
        result = runner.invoke(cli, [function_name, '-t', self.template_path, '-e', self.event_path])
        cli_stdout_lines = result.stdout.strip().split('\n')
        self.assertIn('"Hello world"', cli_stdout_lines)