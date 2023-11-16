import contextlib
import datetime
import unittest
from collections import Counter
from typing import Any, Callable
from unittest.mock import MagicMock, mock_open, patch
import pandas as pd
from parameterized import parameterized
import streamlit as st
import streamlit.components.v1 as components
from streamlit.connections import SnowparkConnection, SQLConnection
from streamlit.runtime import metrics_util
from streamlit.runtime.caching import cache_data_api, cache_resource_api
from streamlit.runtime.legacy_caching import caching
from streamlit.runtime.scriptrunner import get_script_run_ctx, magic_funcs
from streamlit.web.server import websocket_headers
from tests.delta_generator_test_case import DeltaGeneratorTestCase
MAC = 'mac'
UUID = 'uuid'
FILENAME = '/some/id/file'
mock_get_path = MagicMock(return_value=FILENAME)

class MetricsUtilTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.patch1 = patch('streamlit.file_util.os.stat')
        self.os_stat = self.patch1.start()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.patch1.stop()

    def test_machine_id_v3_from_etc(self):
        if False:
            for i in range(10):
                print('nop')
        'Test getting the machine id from /etc'
        file_data = 'etc'
        with patch('streamlit.runtime.metrics_util.uuid.getnode', return_value=MAC), patch('streamlit.runtime.metrics_util.open', mock_open(read_data=file_data), create=True), patch('streamlit.runtime.metrics_util.os.path.isfile', side_effect=lambda path: path == '/etc/machine-id'):
            machine_id = metrics_util._get_machine_id_v3()
        self.assertEqual(machine_id, file_data)

    def test_machine_id_v3_from_dbus(self):
        if False:
            print('Hello World!')
        'Test getting the machine id from /var/lib/dbus'
        file_data = 'dbus'
        with patch('streamlit.runtime.metrics_util.uuid.getnode', return_value=MAC), patch('streamlit.runtime.metrics_util.open', mock_open(read_data=file_data), create=True), patch('streamlit.runtime.metrics_util.os.path.isfile', side_effect=lambda path: path == '/var/lib/dbus/machine-id'):
            machine_id = metrics_util._get_machine_id_v3()
        self.assertEqual(machine_id, file_data)

    def test_machine_id_v3_from_node(self):
        if False:
            for i in range(10):
                print('nop')
        'Test getting the machine id as the mac address'
        with patch('streamlit.runtime.metrics_util.uuid.getnode', return_value=MAC), patch('streamlit.runtime.metrics_util.os.path.isfile', return_value=False):
            machine_id = metrics_util._get_machine_id_v3()
        self.assertEqual(machine_id, MAC)

class PageTelemetryTest(DeltaGeneratorTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        ctx = get_script_run_ctx()
        assert ctx is not None
        ctx.reset()
        ctx.gather_usage_stats = True

    @parameterized.expand([(10, 'int'), (0.01, 'float'), (True, 'bool'), (None, 'NoneType'), (['1'], 'list'), ({'foo': 'bar'}, 'dict'), ('foo', 'str'), (datetime.date.today(), 'datetime.date'), (datetime.datetime.today().time(), 'datetime.time'), (pd.DataFrame(), 'DataFrame'), (pd.Series(), 'PandasSeries'), (datetime.date, 'datetime.date'), (pd.DataFrame, 'DataFrame'), (SnowparkConnection, 'SnowparkConnection'), (SQLConnection, 'SQLConnection')])
    def test_get_type_name(self, obj: object, expected_type: str):
        if False:
            while True:
                i = 10
        'Test getting the type name via _get_type_name'
        self.assertEqual(metrics_util._get_type_name(obj), expected_type)

    def test_get_command_telemetry(self):
        if False:
            i = 10
            return i + 15
        'Test getting command telemetry via _get_command_telemetry.'
        command_metadata = metrics_util._get_command_telemetry(st.dataframe, 'dataframe', pd.DataFrame(), width=250)
        self.assertEqual(command_metadata.name, 'dataframe')
        self.assertEqual(len(command_metadata.args), 2)
        self.assertEqual(str(command_metadata.args[0]).strip(), 'k: "data"\nt: "DataFrame"\nm: "len:0"')
        self.assertEqual(str(command_metadata.args[1]).strip(), 'k: "width"\nt: "int"')
        command_metadata = metrics_util._get_command_telemetry(st.text_input, 'text_input', label='text input', value='foo', disabled=True)
        self.assertEqual(command_metadata.name, 'text_input')
        self.assertEqual(len(command_metadata.args), 3)
        self.assertEqual(str(command_metadata.args[0]).strip(), 'k: "label"\nt: "str"\nm: "len:10"')
        self.assertEqual(str(command_metadata.args[1]).strip(), 'k: "value"\nt: "str"\nm: "len:3"')
        self.assertEqual(str(command_metadata.args[2]).strip(), 'k: "disabled"\nt: "bool"\nm: "val:True"')

    def test_create_page_profile_message(self):
        if False:
            for i in range(10):
                print('nop')
        'Test creating the page profile message via create_page_profile_message.'
        forward_msg = metrics_util.create_page_profile_message(commands=[metrics_util._get_command_telemetry(st.dataframe, 'dataframe', pd.DataFrame(), width=250)], exec_time=1000, prep_time=2000)
        self.assertEqual(len(forward_msg.page_profile.commands), 1)
        self.assertEqual(forward_msg.page_profile.exec_time, 1000)
        self.assertEqual(forward_msg.page_profile.prep_time, 2000)
        self.assertEqual(forward_msg.page_profile.commands[0].name, 'dataframe')

    def test_gather_metrics_decorator(self):
        if False:
            while True:
                i = 10
        'The gather_metrics decorator works as expected.'
        ctx = get_script_run_ctx()
        assert ctx is not None

        @metrics_util.gather_metrics('test_function')
        def test_function(param1: int, param2: str, param3: float=0.1) -> str:
            if False:
                print('Hello World!')
            st.markdown('This command should not be tracked')
            return 'foo'
        test_function(param1=10, param2='foobar')
        self.assertEqual(len(ctx.tracked_commands), 1)
        self.assertTrue(ctx.tracked_commands[0].name.endswith('test_function'))
        self.assertTrue(ctx.tracked_commands[0].name.startswith('external:'))
        st.markdown('This function should be tracked')
        self.assertEqual(len(ctx.tracked_commands), 2)
        self.assertTrue(ctx.tracked_commands[0].name.endswith('test_function'))
        self.assertTrue(ctx.tracked_commands[0].name.startswith('external:'))
        self.assertEqual(ctx.tracked_commands[1].name, 'markdown')
        ctx.reset()
        ctx.gather_usage_stats = False
        self.assertEqual(len(ctx.tracked_commands), 0)
        test_function(param1=10, param2='foobar')
        self.assertEqual(len(ctx.tracked_commands), 0)

    @parameterized.expand([(magic_funcs.transparent_write, 'magic'), (st.cache_data.clear, 'clear_data_caches'), (st.cache_resource.clear, 'clear_resource_caches'), (st.session_state.__setattr__, 'session_state.set_attr'), (st.session_state.__setitem__, 'session_state.set_item'), (cache_data_api.DataCache.write_result, '_cache_data_object'), (cache_resource_api.ResourceCache.write_result, '_cache_resource_object'), (caching._write_to_cache, '_cache_object'), (websocket_headers._get_websocket_headers, '_get_websocket_headers'), (components.html, '_html'), (components.iframe, '_iframe')])
    def test_internal_api_commands(self, command: Callable[..., Any], expected_name: str):
        if False:
            print('Hello World!')
        'Some internal functions are also tracked and should use the correct name.'
        ctx = get_script_run_ctx()
        assert ctx is not None
        with contextlib.suppress(Exception):
            command()
        self.assertGreater(len(ctx.tracked_commands), 0, f'No command tracked for {expected_name}')
        self.assertIn(expected_name, [tracked_commands.name for tracked_commands in ctx.tracked_commands], f'Command {expected_name} was not tracked.')

    def test_public_api_commands(self):
        if False:
            return 10
        'All commands of the public API should be tracked with the correct name.'
        ignored_commands = {'connection', 'experimental_connection', 'spinner', 'empty', 'progress', 'get_option'}
        public_api_names = sorted([k for (k, v) in st.__dict__.items() if not k.startswith('_') and (not isinstance(v, type(st))) and (k not in ignored_commands)])
        for api_name in public_api_names:
            st_func = getattr(st, api_name)
            if not callable(st_func):
                continue
            ctx = get_script_run_ctx()
            assert ctx is not None
            ctx.reset()
            ctx.gather_usage_stats = True
            with contextlib.suppress(Exception):
                st_func()
            self.assertIn(api_name, [cmd.name for cmd in ctx.tracked_commands], (f'When executing `st.{api_name}()`, we expect the string "{api_name}" to be in the list of tracked commands.',))

    def test_column_config_commands(self):
        if False:
            while True:
                i = 10
        'All commands of the public column config API should be tracked with the correct name.'
        public_api_names = sorted([k for (k, v) in st.column_config.__dict__.items() if not k.startswith('_') and (not isinstance(v, type(st.column_config)))])
        for api_name in public_api_names:
            st_func = getattr(st.column_config, api_name)
            if not callable(st_func):
                continue
            ctx = get_script_run_ctx()
            assert ctx is not None
            ctx.reset()
            ctx.gather_usage_stats = True
            with contextlib.suppress(Exception):
                st_func()
            self.assertIn('column_config.' + api_name, [cmd.name for cmd in ctx.tracked_commands], (f'When executing `st.{api_name}()`, we expect the string "{api_name}" to be in the list of tracked commands.',))

    def test_command_tracking_limits(self):
        if False:
            print('Hello World!')
        'Command tracking limits should be respected.\n\n        Current limits are 25 per unique command and 200 in total.\n        '
        ctx = get_script_run_ctx()
        assert ctx is not None
        ctx.reset()
        ctx.gather_usage_stats = True
        funcs = []
        for i in range(10):

            def test_function() -> str:
                if False:
                    i = 10
                    return i + 15
                return 'foo'
            funcs.append(metrics_util.gather_metrics(f'test_function_{i}', test_function))
        for _ in range(metrics_util._MAX_TRACKED_PER_COMMAND + 1):
            for func in funcs:
                func()
        self.assertLessEqual(len(ctx.tracked_commands), metrics_util._MAX_TRACKED_COMMANDS)
        command_counts = Counter([command.name for command in ctx.tracked_commands]).most_common()
        self.assertLessEqual(command_counts[0][1], metrics_util._MAX_TRACKED_PER_COMMAND)