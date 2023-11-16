import os
import tempfile
import pytest
from pathlib import Path
from unittest import skipIf
from click.testing import CliRunner
from samcli.commands.init import cli as init_cmd
from tests.integration.init.schemas.schemas_test_data_setup import SchemaTestDataSetup
from tests.testing_utils import RUNNING_ON_CI, RUNNING_TEST_FOR_MASTER_ON_CI, RUN_BY_CANARY
SKIP_SCHEMA_TESTS = RUNNING_ON_CI and RUNNING_TEST_FOR_MASTER_ON_CI and (not RUN_BY_CANARY)

@skipIf(SKIP_SCHEMA_TESTS, 'Skip schema test')
@pytest.mark.xdist_group(name='sam_init')
class TestBasicInitWithEventBridgeCommand(SchemaTestDataSetup):

    def test_init_interactive_with_event_bridge_app_aws_registry(self):
        if False:
            print('Hello World!')
        user_input = '\n1\n8\n4\n2\n2\nN\nN\neb-app-maven\nY\n1\n4\n9\n        '
        with tempfile.TemporaryDirectory() as temp:
            runner = CliRunner()
            result = runner.invoke(init_cmd, ['--output-dir', temp, '--debug'], input=user_input)
            self.assertFalse(result.exception)
            expected_output_folder = Path(temp, 'eb-app-maven')
            self.assertTrue(expected_output_folder.exists)
            self.assertTrue(expected_output_folder.is_dir())
            self.assertTrue(Path(expected_output_folder, 'HelloWorldFunction', 'src', 'main', 'java', 'schema').is_dir())

    def test_init_interactive_with_event_bridge_app_partner_registry(self):
        if False:
            for i in range(10):
                print('nop')
        user_input = '\n1\n8\n4\n2\n2\nN\nN\neb-app-maven\nY\n3\n1\n        '
        with tempfile.TemporaryDirectory() as temp:
            runner = CliRunner()
            result = runner.invoke(init_cmd, ['--output-dir', temp], input=user_input)
            self.assertFalse(result.exception)
            expected_output_folder = Path(temp, 'eb-app-maven')
            self.assertTrue(expected_output_folder.exists)
            self.assertTrue(expected_output_folder.is_dir())
            self.assertTrue(Path(expected_output_folder, 'HelloWorldFunction', 'src', 'main', 'java', 'schema').is_dir())
            self.assertTrue(Path(expected_output_folder, 'HelloWorldFunction', 'src', 'main', 'java', 'schema', 'schema_test_0', 'TicketCreated.java').is_file())

    def test_init_interactive_with_event_bridge_app_pagination(self):
        if False:
            i = 10
            return i + 15
        user_input = '\n1\n8\n4\n2\n2\nN\nN\neb-app-maven\nY\n4\nN\nP\n2\n        '
        with tempfile.TemporaryDirectory() as temp:
            runner = CliRunner()
            result = runner.invoke(init_cmd, ['--output-dir', temp], input=user_input)
            self.assertFalse(result.exception)
            expected_output_folder = Path(temp, 'eb-app-maven')
            self.assertTrue(expected_output_folder.exists)
            self.assertTrue(expected_output_folder.is_dir())
            self.assertTrue(Path(expected_output_folder, 'HelloWorldFunction', 'src', 'main', 'java', 'schema').is_dir())

    def test_init_interactive_with_event_bridge_app_customer_registry(self):
        if False:
            i = 10
            return i + 15
        user_input = '\n1\n8\n4\n2\n2\nN\nN\neb-app-maven\nY\n2\n1\n                '
        with tempfile.TemporaryDirectory() as temp:
            runner = CliRunner()
            result = runner.invoke(init_cmd, ['--output-dir', temp], input=user_input)
            self.assertFalse(result.exception)
            expected_output_folder = Path(temp, 'eb-app-maven')
            self.assertTrue(expected_output_folder.exists)
            self.assertTrue(expected_output_folder.is_dir())
            self.assertTrue(Path(expected_output_folder, 'HelloWorldFunction', 'src', 'main', 'java', 'schema').is_dir())
            self.assertTrue(Path(expected_output_folder, 'HelloWorldFunction', 'src', 'main', 'java', 'schema', 'schema_test_0', 'Some_Awesome_Schema.java').is_file())

    def test_init_interactive_with_event_bridge_app_aws_schemas_python(self):
        if False:
            while True:
                i = 10
        user_input = '\n1\n8\n7\n2\nN\nN\neb-app-python38\nY\n1\n4\n1\n        '
        with tempfile.TemporaryDirectory() as temp:
            runner = CliRunner()
            result = runner.invoke(init_cmd, ['--output-dir', temp], input=user_input)
            self.assertFalse(result.exception)
            expected_output_folder = Path(temp, 'eb-app-python38')
            self.assertTrue(expected_output_folder.exists)
            self.assertTrue(expected_output_folder.is_dir())
            self.assertTrue(Path(expected_output_folder, 'hello_world_function', 'schema').is_dir())

    def test_init_interactive_with_event_bridge_app_aws_schemas_go(self):
        if False:
            while True:
                i = 10
        user_input = '\n1\n8\n1\n2\nN\nN\neb-app-go\nY\n4\n1\n        '
        with tempfile.TemporaryDirectory() as temp:
            runner = CliRunner()
            result = runner.invoke(init_cmd, ['--output-dir', temp], input=user_input)
            self.assertFalse(result.exception)
            expected_output_folder = Path(temp, 'eb-app-go')
            self.assertTrue(expected_output_folder.exists)
            self.assertTrue(expected_output_folder.is_dir())
            self.assertTrue(Path(expected_output_folder, 'HelloWorld', 'schema').is_dir())

    def test_init_interactive_with_event_bridge_app_non_default_profile_selection(self):
        if False:
            i = 10
            return i + 15
        self._init_custom_config('mynewprofile', 'us-west-2')
        user_input = '\n1\n8\n7\n2\nN\nN\neb-app-python38\n3\nN\n2\nus-east-1\n1\n1\n        '
        with tempfile.TemporaryDirectory() as temp:
            runner = CliRunner()
            result = runner.invoke(init_cmd, ['--output-dir', temp], input=user_input)
            self.assertFalse(result.exception)
            expected_output_folder = Path(temp, 'eb-app-python38')
            self.assertTrue(expected_output_folder.exists)
            self.assertTrue(expected_output_folder.is_dir())
            self.assertTrue(Path(expected_output_folder, 'hello_world_function', 'schema').is_dir())

    def test_init_interactive_with_event_bridge_app_non_supported_schemas_region(self):
        if False:
            print('Hello World!')
        self._init_custom_config('default', 'cn-north-1')
        user_input = '\n1\n8\n7\n2\nN\nN\neb-app-python38\nY\n1\n1\n        '
        with tempfile.TemporaryDirectory() as temp:
            runner = CliRunner()
            result = runner.invoke(init_cmd, ['--output-dir', temp], input=user_input)
            self.assertTrue(result.exception)