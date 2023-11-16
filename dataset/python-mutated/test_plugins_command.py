from __future__ import annotations
import json
import textwrap
from contextlib import redirect_stdout
from io import StringIO
from airflow.cli import cli_parser
from airflow.cli.commands import plugins_command
from airflow.hooks.base import BaseHook
from airflow.listeners.listener import get_listener_manager
from airflow.plugins_manager import AirflowPlugin
from tests.plugins.test_plugin import AirflowTestPlugin as ComplexAirflowPlugin
from tests.test_utils.mock_plugins import mock_plugin_manager

class PluginHook(BaseHook):
    pass

class TestPlugin(AirflowPlugin):
    name = 'test-plugin-cli'
    hooks = [PluginHook]

class TestPluginsCommand:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.parser = cli_parser.get_parser()

    @mock_plugin_manager(plugins=[])
    def test_should_display_no_plugins(self):
        if False:
            i = 10
            return i + 15
        with redirect_stdout(StringIO()) as temp_stdout:
            plugins_command.dump_plugins(self.parser.parse_args(['plugins', '--output=json']))
            stdout = temp_stdout.getvalue()
        assert 'No plugins loaded' in stdout

    @mock_plugin_manager(plugins=[ComplexAirflowPlugin])
    def test_should_display_one_plugins(self):
        if False:
            while True:
                i = 10
        with redirect_stdout(StringIO()) as temp_stdout:
            plugins_command.dump_plugins(self.parser.parse_args(['plugins', '--output=json']))
            stdout = temp_stdout.getvalue()
        print(stdout)
        info = json.loads(stdout)
        assert info == [{'name': 'test_plugin', 'admin_views': [], 'macros': ['tests.plugins.test_plugin.plugin_macro'], 'menu_links': [], 'executors': ['tests.plugins.test_plugin.PluginExecutor'], 'flask_blueprints': ["<flask.blueprints.Blueprint: name='test_plugin' import_name='tests.plugins.test_plugin'>"], 'appbuilder_views': [{'name': 'Test View', 'category': 'Test Plugin', 'view': 'tests.plugins.test_plugin.PluginTestAppBuilderBaseView'}], 'global_operator_extra_links': ['<tests.test_utils.mock_operators.AirflowLink object>', '<tests.test_utils.mock_operators.GithubLink object>'], 'timetables': ['tests.plugins.test_plugin.CustomCronDataIntervalTimetable'], 'operator_extra_links': ['<tests.test_utils.mock_operators.GoogleLink object>', '<tests.test_utils.mock_operators.AirflowLink2 object>', '<tests.test_utils.mock_operators.CustomOpLink object>', '<tests.test_utils.mock_operators.CustomBaseIndexOpLink object>'], 'hooks': ['tests.plugins.test_plugin.PluginHook'], 'listeners': ['tests.listeners.empty_listener', 'tests.listeners.class_listener.ClassBasedListener'], 'source': None, 'appbuilder_menu_items': [{'name': 'Google', 'href': 'https://www.google.com', 'category': 'Search'}, {'name': 'apache', 'href': 'https://www.apache.org/', 'label': 'The Apache Software Foundation'}], 'ti_deps': ['<TIDep(CustomTestTriggerRule)>']}]
        get_listener_manager().clear()

    @mock_plugin_manager(plugins=[TestPlugin])
    def test_should_display_one_plugins_as_table(self):
        if False:
            while True:
                i = 10
        with redirect_stdout(StringIO()) as temp_stdout:
            plugins_command.dump_plugins(self.parser.parse_args(['plugins', '--output=table']))
            stdout = temp_stdout.getvalue()
        stdout = '\n'.join((line.rstrip(' ') for line in stdout.splitlines()))
        expected_output = textwrap.dedent('            name            | hooks\n            ================+===================================================\n            test-plugin-cli | tests.cli.commands.test_plugins_command.PluginHook\n            ')
        assert stdout == expected_output