"""Tests ScriptRunner functionality"""
import os
import sys
import time
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch
import pytest
from parameterized import parameterized
from tornado.testing import AsyncTestCase
from streamlit import source_util
from streamlit.elements.exception import _GENERIC_UNCAUGHT_EXCEPTION_TEXT
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.proto.Delta_pb2 import Delta
from streamlit.proto.Element_pb2 import Element
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.WidgetStates_pb2 import WidgetState, WidgetStates
from streamlit.runtime import Runtime
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.legacy_caching import caching
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.scriptrunner import RerunData, RerunException, ScriptRunner, ScriptRunnerEvent, StopException
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.scriptrunner.script_requests import ScriptRequest, ScriptRequests, ScriptRequestType
from streamlit.runtime.state.session_state import SessionState
from tests import testutil
text_utf = 'complete! 👨\u200d🎤'
text_utf2 = 'complete2! 👨\u200d🎤'
text_no_encoding = text_utf
text_latin = 'complete! ð\x9f\x91¨â\x80\x8dð\x9f\x8e¤'

def _create_widget(id: str, states: WidgetStates) -> WidgetState:
    if False:
        print('Hello World!')
    '\n    Returns\n    -------\n    streamlit.proto.WidgetStates_pb2.WidgetState\n\n    '
    states.widgets.add().id = id
    return states.widgets[-1]

def _is_control_event(event: ScriptRunnerEvent) -> bool:
    if False:
        i = 10
        return i + 15
    "True if the given ScriptRunnerEvent is a 'control' event, as opposed\n    to a 'data' event.\n    "
    return event != ScriptRunnerEvent.ENQUEUE_FORWARD_MSG

@patch('streamlit.source_util._cached_pages', new=None)
class ScriptRunnerTest(AsyncTestCase):

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        mock_runtime = MagicMock(spec=Runtime)
        mock_runtime.media_file_mgr = MediaFileManager(MemoryMediaFileStorage('/mock/media'))
        Runtime._instance = mock_runtime

    def tearDown(self) -> None:
        if False:
            return 10
        super().tearDown()
        Runtime._instance = None

    def test_startup_shutdown(self):
        if False:
            print('Hello World!')
        'Test that we can create and shut down a ScriptRunner.'
        scriptrunner = TestScriptRunner('good_script.py')
        scriptrunner.request_stop()
        scriptrunner.start()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_control_events(scriptrunner, [ScriptRunnerEvent.SHUTDOWN])
        self._assert_text_deltas(scriptrunner, [])

    @parameterized.expand([('installTracer=False', False), ('installTracer=True', True)])
    def test_yield_on_enqueue(self, _, install_tracer: bool):
        if False:
            print('Hello World!')
        'Make sure we try to handle execution control requests whenever\n        our _enqueue_forward_msg function is called, unless "runner.installTracer" is set.\n        '
        with testutil.patch_config_options({'runner.installTracer': install_tracer}):
            runner = TestScriptRunner('not_a_script.py')
            runner._is_in_script_thread = MagicMock(return_value=True)
            maybe_handle_execution_control_request_mock = MagicMock()
            runner._maybe_handle_execution_control_request = maybe_handle_execution_control_request_mock
            mock_msg = MagicMock()
            runner._enqueue_forward_msg(mock_msg)
            self._assert_forward_msgs(runner, [mock_msg])
            expected_call_count = 0 if install_tracer else 1
            self.assertEqual(expected_call_count, maybe_handle_execution_control_request_mock.call_count)

    def test_dont_enqueue_with_pending_script_request(self):
        if False:
            for i in range(10):
                print('nop')
        'No ForwardMsgs are enqueued when the ScriptRunner has\n        a STOP or RERUN request.\n        '
        runner = TestScriptRunner('not_a_script.py')
        runner._is_in_script_thread = MagicMock(return_value=True)
        runner._execing = True
        runner._requests._state = ScriptRequestType.CONTINUE
        mock_msg = MagicMock()
        runner._enqueue_forward_msg(mock_msg)
        self._assert_forward_msgs(runner, [mock_msg])
        runner.clear_forward_msgs()
        runner._requests.request_stop()
        with self.assertRaises(StopException):
            runner._enqueue_forward_msg(MagicMock())
        self._assert_forward_msgs(runner, [])
        runner._requests = ScriptRequests()
        runner.request_rerun(RerunData())
        with self.assertRaises(RerunException):
            runner._enqueue_forward_msg(MagicMock())
        self._assert_forward_msgs(runner, [])

    def test_maybe_handle_execution_control_request(self):
        if False:
            return 10
        'maybe_handle_execution_control_request should no-op if called\n        from another thread.\n        '
        runner = TestScriptRunner('not_a_script.py')
        runner._execing = True
        requests_mock = MagicMock()
        requests_mock.on_scriptrunner_yield = MagicMock(return_value=ScriptRequest(ScriptRequestType.RERUN, RerunData()))
        runner._requests = requests_mock
        runner._is_in_script_thread = MagicMock(return_value=False)
        runner._maybe_handle_execution_control_request()
        requests_mock.on_scriptrunner_yield.assert_not_called()
        runner._is_in_script_thread = MagicMock(return_value=True)
        with self.assertRaises(RerunException):
            runner._maybe_handle_execution_control_request()
        requests_mock.on_scriptrunner_yield.assert_called_once()

    def test_run_script_in_loop(self):
        if False:
            i = 10
            return i + 15
        '_run_script_thread should continue re-running its script\n        while it has pending rerun requests.'
        scriptrunner = TestScriptRunner('not_a_script.py')
        on_scriptrunner_ready_mock = MagicMock()
        on_scriptrunner_ready_mock.side_effect = [ScriptRequest(ScriptRequestType.RERUN, RerunData()), ScriptRequest(ScriptRequestType.RERUN, RerunData()), ScriptRequest(ScriptRequestType.RERUN, RerunData()), ScriptRequest(ScriptRequestType.STOP)]
        scriptrunner._requests.on_scriptrunner_ready = on_scriptrunner_ready_mock
        run_script_mock = MagicMock()
        scriptrunner._run_script = run_script_mock
        scriptrunner.start()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self.assertEqual(3, run_script_mock.call_count)

    @parameterized.expand([('good_script.py', text_utf), ('good_script_no_encoding.py.txt', text_no_encoding), ('good_script_latin_encoding.py.txt', text_latin)])
    def test_run_script(self, filename, text):
        if False:
            while True:
                i = 10
        'Tests that we can run a script to completion.'
        scriptrunner = TestScriptRunner(filename)
        scriptrunner.request_rerun(RerunData())
        scriptrunner.start()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])
        self._assert_text_deltas(scriptrunner, [text])
        self.assertEqual(scriptrunner._main_script_path, sys.modules['__main__'].__file__, ' ScriptRunner should set the __main__.__file__attribute correctly')

    def test_compile_error(self):
        if False:
            for i in range(10):
                print('nop')
        "Tests that we get an exception event when a script can't compile."
        scriptrunner = TestScriptRunner('compile_error.py.txt')
        scriptrunner.request_rerun(RerunData())
        scriptrunner.start()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_COMPILE_ERROR, ScriptRunnerEvent.SHUTDOWN])
        self._assert_text_deltas(scriptrunner, [])

    @patch('streamlit.runtime.state.session_state.SessionState._call_callbacks')
    def test_calls_widget_callbacks(self, patched_call_callbacks):
        if False:
            for i in range(10):
                print('nop')
        'Before a script is rerun, we call callbacks for any widgets\n        whose value has changed.\n        '
        scriptrunner = TestScriptRunner('widgets_script.py')
        scriptrunner.request_rerun(RerunData())
        scriptrunner.start()
        require_widgets_deltas([scriptrunner])
        self._assert_text_deltas(scriptrunner, ['False', 'ahoy!', '0', 'False', 'loop_forever'])
        patched_call_callbacks.assert_not_called()
        states = WidgetStates()
        w1_id = scriptrunner.get_widget_id('checkbox', 'checkbox')
        _create_widget(w1_id, states).bool_value = True
        w2_id = scriptrunner.get_widget_id('text_area', 'text_area')
        _create_widget(w2_id, states).string_value = 'matey!'
        w3_id = scriptrunner.get_widget_id('radio', 'radio')
        _create_widget(w3_id, states).int_value = 2
        w4_id = scriptrunner.get_widget_id('button', 'button')
        _create_widget(w4_id, states).trigger_value = True
        scriptrunner.clear_forward_msgs()
        scriptrunner.request_rerun(RerunData(widget_states=states))
        require_widgets_deltas([scriptrunner])
        patched_call_callbacks.assert_called_once()
        self._assert_text_deltas(scriptrunner, ['True', 'matey!', '2', 'True', 'loop_forever'])
        scriptrunner.request_stop()
        scriptrunner.join()

    @patch('streamlit.runtime.state.session_state.SessionState._call_callbacks')
    def test_calls_widget_callbacks_on_new_scriptrunner_instance(self, patched_call_callbacks):
        if False:
            return 10
        'A new ScriptRunner instance will call widget callbacks\n        if widget values have changed. (This differs slightly from\n        `test_calls_widget_callbacks`, which tests that an *already-running*\n        ScriptRunner calls its callbacks on rerun).\n        '
        scriptrunner = TestScriptRunner('widgets_script.py')
        scriptrunner.request_rerun(RerunData())
        scriptrunner.start()
        require_widgets_deltas([scriptrunner])
        scriptrunner.request_stop()
        scriptrunner.join()
        patched_call_callbacks.assert_not_called()
        states = WidgetStates()
        checkbox_id = scriptrunner.get_widget_id('checkbox', 'checkbox')
        _create_widget(checkbox_id, states).bool_value = True
        scriptrunner = TestScriptRunner('widgets_script.py')
        scriptrunner.request_rerun(RerunData(widget_states=states))
        scriptrunner.start()
        require_widgets_deltas([scriptrunner])
        scriptrunner.request_stop()
        scriptrunner.join()
        patched_call_callbacks.assert_called_once()

    @patch('streamlit.exception')
    @patch('streamlit.runtime.state.session_state.SessionState._call_callbacks')
    def test_calls_widget_callbacks_error(self, patched_call_callbacks, patched_st_exception):
        if False:
            i = 10
            return i + 15
        'If an exception is raised from a callback function,\n        it should result in a call to `streamlit.exception`.\n        '
        patched_call_callbacks.side_effect = RuntimeError('Random Error')
        scriptrunner = TestScriptRunner('widgets_script.py')
        scriptrunner.request_rerun(RerunData())
        scriptrunner.start()
        require_widgets_deltas([scriptrunner])
        self._assert_text_deltas(scriptrunner, ['False', 'ahoy!', '0', 'False', 'loop_forever'])
        patched_call_callbacks.assert_not_called()
        states = WidgetStates()
        w1_id = scriptrunner.get_widget_id('checkbox', 'checkbox')
        _create_widget(w1_id, states).bool_value = True
        w2_id = scriptrunner.get_widget_id('text_area', 'text_area')
        _create_widget(w2_id, states).string_value = 'matey!'
        w3_id = scriptrunner.get_widget_id('radio', 'radio')
        _create_widget(w3_id, states).int_value = 2
        w4_id = scriptrunner.get_widget_id('button', 'button')
        _create_widget(w4_id, states).trigger_value = True
        scriptrunner.clear_forward_msgs()
        scriptrunner.request_rerun(RerunData(widget_states=states))
        scriptrunner.join()
        patched_call_callbacks.assert_called_once()
        self._assert_control_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.SCRIPT_STOPPED_FOR_RERUN, ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])
        patched_st_exception.assert_called_once()

    def test_missing_script(self):
        if False:
            for i in range(10):
                print('nop')
        "Tests that we get an exception event when a script doesn't exist."
        scriptrunner = TestScriptRunner('i_do_not_exist.py')
        scriptrunner.request_rerun(RerunData())
        scriptrunner.start()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_COMPILE_ERROR, ScriptRunnerEvent.SHUTDOWN])
        self._assert_text_deltas(scriptrunner, [])

    @parameterized.expand([(True,), (False,)])
    def test_runtime_error(self, show_error_details: bool):
        if False:
            i = 10
            return i + 15
        'Tests that we correctly handle scripts with runtime errors.'
        with testutil.patch_config_options({'client.showErrorDetails': show_error_details}):
            scriptrunner = TestScriptRunner('runtime_error.py')
            scriptrunner.request_rerun(RerunData())
            scriptrunner.start()
            scriptrunner.join()
            self._assert_no_exceptions(scriptrunner)
            self._assert_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])
            elts = scriptrunner.elements()
            self.assertEqual(elts[0].WhichOneof('type'), 'text')
            if show_error_details:
                self._assert_num_deltas(scriptrunner, 2)
                self.assertEqual(elts[1].WhichOneof('type'), 'exception')
            else:
                self._assert_num_deltas(scriptrunner, 2)
                self.assertEqual(elts[1].WhichOneof('type'), 'exception')
                exc_msg = elts[1].exception.message
                self.assertTrue(_GENERIC_UNCAUGHT_EXCEPTION_TEXT == exc_msg)

    @pytest.mark.slow
    def test_stop_script(self):
        if False:
            print('Hello World!')
        "Tests that we can stop a script while it's running."
        scriptrunner = TestScriptRunner('infinite_loop.py')
        scriptrunner.request_rerun(RerunData())
        scriptrunner.start()
        time.sleep(0.1)
        scriptrunner.request_rerun(RerunData())
        time.sleep(1)
        scriptrunner.request_stop()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_control_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.SCRIPT_STOPPED_FOR_RERUN, ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])
        self._assert_text_deltas(scriptrunner, ['loop_forever'])

    def test_shutdown(self):
        if False:
            i = 10
            return i + 15
        'Test that we can shutdown while a script is running.'
        scriptrunner = TestScriptRunner('infinite_loop.py')
        scriptrunner.request_rerun(RerunData())
        scriptrunner.start()
        time.sleep(0.1)
        scriptrunner.request_stop()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_control_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])
        self._assert_text_deltas(scriptrunner, ['loop_forever'])

    def test_widgets(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that widget values behave as expected.'
        scriptrunner = TestScriptRunner('widgets_script.py')
        try:
            scriptrunner.request_rerun(RerunData())
            scriptrunner.start()
            require_widgets_deltas([scriptrunner])
            self._assert_text_deltas(scriptrunner, ['False', 'ahoy!', '0', 'False', 'loop_forever'])
            states = WidgetStates()
            w1_id = scriptrunner.get_widget_id('checkbox', 'checkbox')
            _create_widget(w1_id, states).bool_value = True
            w2_id = scriptrunner.get_widget_id('text_area', 'text_area')
            _create_widget(w2_id, states).string_value = 'matey!'
            w3_id = scriptrunner.get_widget_id('radio', 'radio')
            _create_widget(w3_id, states).int_value = 2
            w4_id = scriptrunner.get_widget_id('button', 'button')
            _create_widget(w4_id, states).trigger_value = True
            scriptrunner.clear_forward_msgs()
            scriptrunner.request_rerun(RerunData(widget_states=states))
            require_widgets_deltas([scriptrunner])
            self._assert_text_deltas(scriptrunner, ['True', 'matey!', '2', 'True', 'loop_forever'])
            scriptrunner.clear_forward_msgs()
            scriptrunner.request_rerun(RerunData())
            require_widgets_deltas([scriptrunner])
            self._assert_text_deltas(scriptrunner, ['True', 'matey!', '2', 'False', 'loop_forever'])
        finally:
            scriptrunner.request_stop()
            scriptrunner.join()
            self._assert_no_exceptions(scriptrunner)

    @patch('streamlit.source_util.get_pages', MagicMock(return_value={'hash1': {'page_script_hash': 'hash1', 'script_path': os.path.join(os.path.dirname(__file__), 'test_data', 'good_script.py')}}))
    def test_query_string_and_page_script_hash_saved(self):
        if False:
            for i in range(10):
                print('nop')
        scriptrunner = TestScriptRunner('good_script.py')
        scriptrunner.request_rerun(RerunData(query_string='foo=bar', page_script_hash='hash1'))
        scriptrunner.start()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])
        shutdown_data = scriptrunner.event_data[-1]
        self.assertEqual(shutdown_data['client_state'].query_string, 'foo=bar')
        self.assertEqual(shutdown_data['client_state'].page_script_hash, 'hash1')

    def test_coalesce_rerun(self):
        if False:
            while True:
                i = 10
        'Tests that multiple pending rerun requests get coalesced.'
        scriptrunner = TestScriptRunner('good_script.py')
        scriptrunner.request_rerun(RerunData())
        scriptrunner.request_rerun(RerunData())
        scriptrunner.request_rerun(RerunData())
        scriptrunner.start()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])
        self._assert_text_deltas(scriptrunner, [text_utf])

    def test_remove_nonexistent_elements(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that nonexistent elements are removed from widget cache after script run.'
        widget_id = 'nonexistent_widget_id'
        scriptrunner = TestScriptRunner('good_script.py')
        states = WidgetStates()
        _create_widget(widget_id, states).string_value = 'streamlit'
        scriptrunner.request_rerun(RerunData(widget_states=states))
        scriptrunner.start()
        self.assertRaises(KeyError, lambda : scriptrunner._session_state[widget_id])

    def off_test_multiple_scriptrunners(self):
        if False:
            return 10
        'Tests that multiple scriptrunners can run simultaneously.'
        scriptrunner = TestScriptRunner('widgets_script.py')
        scriptrunner.request_rerun(RerunData())
        scriptrunner.start()
        require_widgets_deltas([scriptrunner])
        radio_widget_id = scriptrunner.get_widget_id('radio', 'radio')
        scriptrunner.request_stop()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        runners = []
        for ii in range(3):
            runner = TestScriptRunner('widgets_script.py')
            runners.append(runner)
            states = WidgetStates()
            _create_widget(radio_widget_id, states).int_value = ii
            runner.request_rerun(RerunData(widget_states=states))
        for runner in runners:
            runner.start()
        require_widgets_deltas(runners)
        for (ii, runner) in enumerate(runners):
            self._assert_text_deltas(runner, ['False', 'ahoy!', '%s' % ii, 'False', 'loop_forever'])
            runner.request_stop()
        time.sleep(0.1)
        for runner in runners:
            runner.join()
        for runner in runners:
            self._assert_no_exceptions(runner)
            self._assert_control_events(runner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])

    def test_invalidating_cache(self):
        if False:
            i = 10
            return i + 15
        'Test that st.caches are cleared when a dependency changes.'
        caching._mem_caches.clear()
        runner = TestScriptRunner('st_cache_script.py')
        runner.request_rerun(RerunData())
        runner.start()
        runner.join()
        self._assert_text_deltas(runner, ['cached function called', 'cached function called', 'cached function called', 'cached function called', 'cached_depending_on_not_yet_defined called'])
        source_util._cached_pages = None
        runner = TestScriptRunner('st_cache_script_changed.py')
        runner.request_rerun(RerunData())
        runner.start()
        runner.join()
        self._assert_text_deltas(runner, ['cached_depending_on_not_yet_defined called'])

    @patch('streamlit.source_util.get_pages', MagicMock(return_value={'hash2': {'page_script_hash': 'hash2', 'page_name': 'good_script2', 'script_path': os.path.join(os.path.dirname(__file__), 'test_data', 'good_script2.py')}}))
    def test_page_script_hash_to_script_path(self):
        if False:
            return 10
        scriptrunner = TestScriptRunner('good_script.py')
        scriptrunner.request_rerun(RerunData(page_name='good_script2'))
        scriptrunner.start()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])
        self._assert_text_deltas(scriptrunner, [text_utf2])
        self.assertEqual(os.path.join(os.path.dirname(__file__), 'test_data', 'good_script2.py'), sys.modules['__main__'].__file__, ' ScriptRunner should set the __main__.__file__attribute correctly')
        shutdown_data = scriptrunner.event_data[-1]
        self.assertEqual(shutdown_data['client_state'].page_script_hash, 'hash2')

    @patch('streamlit.source_util.get_pages', MagicMock(return_value={'hash2': {'page_script_hash': 'hash2', 'script_path': 'script2'}}))
    def test_404_hash_not_found(self):
        if False:
            i = 10
            return i + 15
        scriptrunner = TestScriptRunner('good_script.py')
        scriptrunner.request_rerun(RerunData(page_script_hash='hash3'))
        scriptrunner.start()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])
        self._assert_text_deltas(scriptrunner, [text_utf])
        page_not_found_msg = scriptrunner.forward_msg_queue._queue[0].page_not_found
        self.assertEqual(page_not_found_msg.page_name, '')
        self.assertEqual(scriptrunner._main_script_path, sys.modules['__main__'].__file__, ' ScriptRunner should set the __main__.__file__attribute correctly')

    @patch('streamlit.source_util.get_pages', MagicMock(return_value={'hash2': {'page_script_hash': 'hash2', 'script_path': 'script2', 'page_name': 'page2'}}))
    def test_404_page_name_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        scriptrunner = TestScriptRunner('good_script.py')
        scriptrunner.request_rerun(RerunData(page_name='nonexistent'))
        scriptrunner.start()
        scriptrunner.join()
        self._assert_no_exceptions(scriptrunner)
        self._assert_events(scriptrunner, [ScriptRunnerEvent.SCRIPT_STARTED, ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, ScriptRunnerEvent.SHUTDOWN])
        self._assert_text_deltas(scriptrunner, [text_utf])
        page_not_found_msg = scriptrunner.forward_msg_queue._queue[0].page_not_found
        self.assertEqual(page_not_found_msg.page_name, 'nonexistent')
        self.assertEqual(scriptrunner._main_script_path, sys.modules['__main__'].__file__, ' ScriptRunner should set the __main__.__file__attribute correctly')

    def _assert_no_exceptions(self, scriptrunner: 'TestScriptRunner') -> None:
        if False:
            print('Hello World!')
        "Assert that no uncaught exceptions were thrown in the\n        scriptrunner's run thread.\n        "
        self.assertEqual([], scriptrunner.script_thread_exceptions)

    def _assert_events(self, scriptrunner: 'TestScriptRunner', expected_events: List[ScriptRunnerEvent]) -> None:
        if False:
            while True:
                i = 10
        'Assert that the ScriptRunnerEvents emitted by a TestScriptRunner\n        are what we expect.'
        self.assertEqual(expected_events, scriptrunner.events)

    def _assert_control_events(self, scriptrunner: 'TestScriptRunner', expected_events: List[ScriptRunnerEvent]) -> None:
        if False:
            i = 10
            return i + 15
        'Assert the non-data ScriptRunnerEvents emitted by a TestScriptRunner\n        are what we expect. ("Non-data" refers to all events except\n        ENQUEUE_FORWARD_MSG.)\n        '
        control_events = [event for event in scriptrunner.events if _is_control_event(event)]
        self.assertEqual(expected_events, control_events)

    def _assert_forward_msgs(self, scriptrunner: 'TestScriptRunner', messages: List[ForwardMsg]) -> None:
        if False:
            print('Hello World!')
        "Assert that the ScriptRunner's ForwardMsgQueue contains the\n        given list of ForwardMsgs.\n        "
        self.assertEqual(messages, scriptrunner.forward_msgs())

    def _assert_num_deltas(self, scriptrunner: 'TestScriptRunner', num_deltas: int) -> None:
        if False:
            return 10
        'Assert that the given number of delta ForwardMsgs were enqueued\n        during script execution.\n\n        Parameters\n        ----------\n        scriptrunner : TestScriptRunner\n        num_deltas : int\n\n        '
        self.assertEqual(num_deltas, len(scriptrunner.deltas()))

    def _assert_text_deltas(self, scriptrunner: 'TestScriptRunner', text_deltas: List[str]) -> None:
        if False:
            print('Hello World!')
        "Assert that the scriptrunner's ForwardMsgQueue contains text deltas\n        with the given contents.\n        "
        self.assertEqual(text_deltas, scriptrunner.text_deltas())

class TestScriptRunner(ScriptRunner):
    """Subclasses ScriptRunner to provide some testing features."""
    __test__ = False

    def __init__(self, script_name: str):
        if False:
            return 10
        'Initializes the ScriptRunner for the given script_name'
        self.forward_msg_queue = ForwardMsgQueue()
        main_script_path = os.path.join(os.path.dirname(__file__), 'test_data', script_name)
        super().__init__(session_id='test session id', main_script_path=main_script_path, session_state=SessionState(), uploaded_file_mgr=MemoryUploadedFileManager('/mock/upload'), script_cache=ScriptCache(), initial_rerun_data=RerunData(), user_info={'email': 'test@test.com'})
        self.script_thread_exceptions: List[BaseException] = []
        self.events: List[ScriptRunnerEvent] = []
        self.event_data: List[Any] = []

        def record_event(sender: Optional[ScriptRunner], event: ScriptRunnerEvent, **kwargs) -> None:
            if False:
                return 10
            assert sender is None or sender == self, 'Unexpected ScriptRunnerEvent sender!'
            self.events.append(event)
            self.event_data.append(kwargs)
            if event == ScriptRunnerEvent.ENQUEUE_FORWARD_MSG:
                forward_msg = kwargs['forward_msg']
                self.forward_msg_queue.enqueue(forward_msg)
        self.on_event.connect(record_event, weak=False)

    def _run_script_thread(self) -> None:
        if False:
            while True:
                i = 10
        try:
            super()._run_script_thread()
        except BaseException as e:
            self.script_thread_exceptions.append(e)

    def _run_script(self, rerun_data: RerunData) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.forward_msg_queue.clear()
        super()._run_script(rerun_data)

    def join(self) -> None:
        if False:
            print('Hello World!')
        "Join the script_thread if it's running."
        if self._script_thread is not None:
            self._script_thread.join()

    def clear_forward_msgs(self) -> None:
        if False:
            return 10
        'Clear all messages from our ForwardMsgQueue.'
        self.forward_msg_queue.clear()

    def forward_msgs(self) -> List[ForwardMsg]:
        if False:
            i = 10
            return i + 15
        'Return all messages in our ForwardMsgQueue.'
        return self.forward_msg_queue._queue

    def deltas(self) -> List[Delta]:
        if False:
            print('Hello World!')
        'Return the delta messages in our ForwardMsgQueue.'
        return [msg.delta for msg in self.forward_msg_queue._queue if msg.HasField('delta')]

    def elements(self) -> List[Element]:
        if False:
            i = 10
            return i + 15
        'Return the delta.new_element messages in our ForwardMsgQueue.'
        return [delta.new_element for delta in self.deltas()]

    def text_deltas(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Return the string contents of text deltas in our ForwardMsgQueue'
        return [element.text.body for element in self.elements() if element.WhichOneof('type') == 'text']

    def get_widget_id(self, widget_type: str, label: str) -> Optional[str]:
        if False:
            print('Hello World!')
        'Returns the id of the widget with the specified type and label'
        for delta in self.deltas():
            new_element = getattr(delta, 'new_element', None)
            widget = getattr(new_element, widget_type, None)
            widget_label = getattr(widget, 'label', None)
            if widget_label == label:
                return widget.id
        return None

def require_widgets_deltas(runners: List[TestScriptRunner], timeout: float=15) -> None:
    if False:
        i = 10
        return i + 15
    'Wait for the given ScriptRunners to each produce the appropriate\n    number of deltas for widgets_script.py before a timeout. If the timeout\n    is reached, the runners will all be shutdown and an error will be thrown.\n    '
    NUM_DELTAS = 9
    t0 = time.time()
    num_complete = 0
    while time.time() - t0 < timeout:
        time.sleep(0.1)
        num_complete = sum((1 for runner in runners if len(runner.deltas()) >= NUM_DELTAS))
        if num_complete == len(runners):
            return
    err_string = f'require_widgets_deltas() timed out after {timeout}s ({num_complete}/{len(runners)} runners complete)'
    for runner in runners:
        if len(runner.deltas()) < NUM_DELTAS:
            err_string += f'\n- incomplete deltas: {runner.text_deltas()}'
    for runner in runners:
        runner.request_stop()
    for runner in runners:
        runner.join()
    raise RuntimeError(err_string)