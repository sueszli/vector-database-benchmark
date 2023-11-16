import asyncio
import gc
import threading
import unittest
from asyncio import AbstractEventLoop
from typing import Any, Callable, List, Optional, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
import pytest
import streamlit.runtime.app_session as app_session
from streamlit import config
from streamlit.proto.AppPage_pb2 import AppPage
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.Common_pb2 import FileURLs, FileURLsRequest, FileURLsResponse
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime import Runtime
from streamlit.runtime.app_session import AppSession, AppSessionState
from streamlit.runtime.caching.storage.dummy_cache_storage import MemoryCacheStorageManager
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner import RerunData, ScriptRunContext, ScriptRunner, ScriptRunnerEvent, add_script_run_ctx, get_script_run_ctx
from streamlit.runtime.state import SessionState
from streamlit.runtime.uploaded_file_manager import UploadedFileManager, UploadFileUrlInfo
from streamlit.watcher.local_sources_watcher import LocalSourcesWatcher
from tests.testutil import patch_config_options

@pytest.fixture
def del_path(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setenv('PATH', '')

def _create_test_session(event_loop: Optional[AbstractEventLoop]=None) -> AppSession:
    if False:
        i = 10
        return i + 15
    'Create an AppSession instance with some default mocked data.'
    if event_loop is None:
        event_loop = MagicMock()
    with patch('streamlit.runtime.app_session.asyncio.get_running_loop', return_value=event_loop):
        return AppSession(script_data=ScriptData('/fake/script_path.py', 'fake_command_line'), uploaded_file_manager=MagicMock(), script_cache=MagicMock(), message_enqueued_callback=None, local_sources_watcher=MagicMock(), user_info={'email': 'test@test.com'})

@patch('streamlit.runtime.app_session.LocalSourcesWatcher', MagicMock(spec=LocalSourcesWatcher))
class AppSessionTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        mock_runtime = MagicMock(spec=Runtime)
        mock_runtime.media_file_mgr = MediaFileManager(MemoryMediaFileStorage('/mock/media'))
        mock_runtime.cache_storage_manager = MemoryCacheStorageManager()
        Runtime._instance = mock_runtime

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        super().tearDown()
        Runtime._instance = None

    @patch('streamlit.runtime.app_session.secrets_singleton.file_change_listener.disconnect')
    def test_shutdown(self, patched_disconnect):
        if False:
            for i in range(10):
                print('nop')
        'Test that AppSession.shutdown behaves sanely.'
        session = _create_test_session()
        mock_file_mgr = MagicMock(spec=UploadedFileManager)
        session._uploaded_file_mgr = mock_file_mgr
        session.shutdown()
        self.assertEqual(AppSessionState.SHUTDOWN_REQUESTED, session._state)
        mock_file_mgr.remove_session_files.assert_called_once_with(session.id)
        patched_disconnect.assert_called_once_with(session._on_secrets_file_changed)
        session.shutdown()
        self.assertEqual(AppSessionState.SHUTDOWN_REQUESTED, session._state)
        mock_file_mgr.remove_session_files.assert_called_once_with(session.id)

    def test_shutdown_with_running_scriptrunner(self):
        if False:
            print('Hello World!')
        'If we have a running ScriptRunner, shutting down should stop it.'
        session = _create_test_session()
        mock_scriptrunner = MagicMock(spec=ScriptRunner)
        session._scriptrunner = mock_scriptrunner
        session.shutdown()
        mock_scriptrunner.request_stop.assert_called_once()
        mock_scriptrunner.reset_mock()
        session.shutdown()
        mock_scriptrunner.request_stop.assert_not_called()

    def test_request_script_stop(self):
        if False:
            i = 10
            return i + 15
        'Verify that request_script_stop forwards the request to the scriptrunner.'
        session = _create_test_session()
        mock_scriptrunner = MagicMock(spec=ScriptRunner)
        session._scriptrunner = mock_scriptrunner
        session.request_script_stop()
        mock_scriptrunner.request_stop.assert_called()

    def test_request_script_stop_no_scriptrunner(self):
        if False:
            return 10
        "Test that calling request_script_stop when there is no scriptrunner doesn't\n        result in an error.\n        "
        session = _create_test_session()
        session._scriptrunner = None
        session.request_script_stop()

    def test_unique_id(self):
        if False:
            for i in range(10):
                print('nop')
        'Each AppSession should have a unique ID'
        session1 = _create_test_session()
        session2 = _create_test_session()
        self.assertNotEqual(session1.id, session2.id)

    def test_creates_session_state_on_init(self):
        if False:
            while True:
                i = 10
        session = _create_test_session()
        self.assertTrue(isinstance(session.session_state, SessionState))

    def test_clear_cache_resets_session_state(self):
        if False:
            for i in range(10):
                print('nop')
        session = _create_test_session()
        session._session_state['foo'] = 'bar'
        session._handle_clear_cache_request()
        self.assertTrue('foo' not in session._session_state)

    @patch('streamlit.runtime.legacy_caching.clear_cache')
    @patch('streamlit.runtime.caching.cache_data.clear')
    @patch('streamlit.runtime.caching.cache_resource.clear')
    def test_clear_cache_all_caches(self, clear_resource_caches, clear_data_caches, clear_legacy_cache):
        if False:
            print('Hello World!')
        session = _create_test_session()
        session._handle_clear_cache_request()
        clear_resource_caches.assert_called_once()
        clear_data_caches.assert_called_once()
        clear_legacy_cache.assert_called_once()

    @patch('streamlit.runtime.app_session.secrets_singleton.file_change_listener.connect')
    def test_request_rerun_on_secrets_file_change(self, patched_connect):
        if False:
            for i in range(10):
                print('nop')
        'AppSession should add a secrets listener on creation.'
        session = _create_test_session()
        patched_connect.assert_called_once_with(session._on_secrets_file_changed)

    @patch_config_options({'runner.fastReruns': False})
    @patch('streamlit.runtime.app_session.AppSession._create_scriptrunner')
    def test_rerun_with_no_scriptrunner(self, mock_create_scriptrunner: MagicMock):
        if False:
            while True:
                i = 10
        "If we don't have a ScriptRunner, a rerun request will result in\n        one being created."
        session = _create_test_session()
        session.request_rerun(None)
        mock_create_scriptrunner.assert_called_once_with(RerunData())

    @patch_config_options({'runner.fastReruns': False})
    @patch('streamlit.runtime.app_session.AppSession._create_scriptrunner')
    def test_rerun_with_active_scriptrunner(self, mock_create_scriptrunner: MagicMock):
        if False:
            print('Hello World!')
        'If we have an active ScriptRunner, it receives rerun requests.'
        session = _create_test_session()
        mock_active_scriptrunner = MagicMock(spec=ScriptRunner)
        mock_active_scriptrunner.request_rerun = MagicMock(return_value=True)
        session._scriptrunner = mock_active_scriptrunner
        session.request_rerun(None)
        mock_active_scriptrunner.request_rerun.assert_called_once_with(RerunData())
        mock_create_scriptrunner.assert_not_called()

    @patch_config_options({'runner.fastReruns': False})
    @patch('streamlit.runtime.app_session.AppSession._create_scriptrunner')
    def test_rerun_with_stopped_scriptrunner(self, mock_create_scriptrunner: MagicMock):
        if False:
            while True:
                i = 10
        "If have a ScriptRunner but it's shutting down and cannot handle\n        new rerun requests, we'll create a new ScriptRunner."
        session = _create_test_session()
        mock_stopped_scriptrunner = MagicMock(spec=ScriptRunner)
        mock_stopped_scriptrunner.request_rerun = MagicMock(return_value=False)
        session._scriptrunner = mock_stopped_scriptrunner
        session.request_rerun(None)
        mock_stopped_scriptrunner.request_rerun.assert_called_once_with(RerunData())
        mock_create_scriptrunner.assert_called_once_with(RerunData())

    @patch_config_options({'runner.fastReruns': True})
    @patch('streamlit.runtime.app_session.AppSession._create_scriptrunner')
    def test_fast_rerun(self, mock_create_scriptrunner: MagicMock):
        if False:
            i = 10
            return i + 15
        'If runner.fastReruns is enabled, a rerun request will stop the\n        existing ScriptRunner and immediately create a new one.\n        '
        session = _create_test_session()
        mock_active_scriptrunner = MagicMock(spec=ScriptRunner)
        session._scriptrunner = mock_active_scriptrunner
        session.request_rerun(None)
        mock_active_scriptrunner.request_rerun.assert_not_called()
        mock_active_scriptrunner.request_stop.assert_called_once()
        mock_create_scriptrunner.assert_called_once()

    @patch('streamlit.runtime.app_session.ScriptRunner')
    def test_create_scriptrunner(self, mock_scriptrunner: MagicMock):
        if False:
            while True:
                i = 10
        'Test that _create_scriptrunner does what it should.'
        session = _create_test_session()
        self.assertIsNone(session._scriptrunner)
        session._create_scriptrunner(initial_rerun_data=RerunData())
        mock_scriptrunner.assert_called_once_with(session_id=session.id, main_script_path=session._script_data.main_script_path, session_state=session._session_state, uploaded_file_mgr=session._uploaded_file_mgr, script_cache=session._script_cache, initial_rerun_data=RerunData(), user_info={'email': 'test@test.com'})
        self.assertIsNotNone(session._scriptrunner)
        scriptrunner: MagicMock = cast(MagicMock, session._scriptrunner)
        scriptrunner.on_event.connect.assert_called_once_with(session._on_scriptrunner_event)
        scriptrunner.start.assert_called_once()

    @patch('streamlit.runtime.app_session.ScriptRunner', MagicMock(spec=ScriptRunner))
    @patch('streamlit.runtime.app_session.AppSession._enqueue_forward_msg')
    def test_ignore_events_from_noncurrent_scriptrunner(self, mock_enqueue: MagicMock):
        if False:
            i = 10
            return i + 15
        'If we receive ScriptRunnerEvents from anything other than our\n        current ScriptRunner, we should silently ignore them.\n        '
        session = _create_test_session()
        session._create_scriptrunner(initial_rerun_data=RerunData())
        with patch('streamlit.runtime.app_session.asyncio.get_running_loop', return_value=session._event_loop):
            session._handle_scriptrunner_event_on_event_loop(sender=session._scriptrunner, event=ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, forward_msg=ForwardMsg())
            mock_enqueue.assert_called_once_with(ForwardMsg())
            mock_enqueue.reset_mock()
            non_current_scriptrunner = MagicMock(spec=ScriptRunner)
            session._handle_scriptrunner_event_on_event_loop(sender=non_current_scriptrunner, event=ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, forward_msg=ForwardMsg())
            mock_enqueue.assert_not_called()

    @patch('streamlit.runtime.app_session.ScriptRunner', MagicMock(spec=ScriptRunner))
    @patch('streamlit.runtime.app_session.AppSession._enqueue_forward_msg', MagicMock())
    def test_resets_debug_last_backmsg_id_on_script_finished(self):
        if False:
            return 10
        session = _create_test_session()
        session._create_scriptrunner(initial_rerun_data=RerunData())
        session._debug_last_backmsg_id = 'some_backmsg_id'
        with patch('streamlit.runtime.app_session.asyncio.get_running_loop', return_value=session._event_loop):
            session._handle_scriptrunner_event_on_event_loop(sender=session._scriptrunner, event=ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS, forward_msg=ForwardMsg())
            self.assertIsNone(session._debug_last_backmsg_id)

    def test_passes_client_state_on_run_on_save(self):
        if False:
            print('Hello World!')
        session = _create_test_session()
        session._run_on_save = True
        session.request_rerun = MagicMock()
        session._on_source_file_changed()
        session._script_cache.clear.assert_called_once()
        session.request_rerun.assert_called_once_with(session._client_state)

    @patch('streamlit.runtime.app_session.AppSession._should_rerun_on_file_change', MagicMock(return_value=False))
    def test_does_not_rerun_if_not_current_page(self):
        if False:
            for i in range(10):
                print('nop')
        session = _create_test_session()
        session._run_on_save = True
        session.request_rerun = MagicMock()
        session._on_source_file_changed('/fake/script_path.py')
        session._script_cache.clear.assert_called_once()
        self.assertEqual(session.request_rerun.called, False)

    @patch('streamlit.runtime.app_session.source_util.get_pages', MagicMock(return_value={'hash1': {'page_name': 'page1', 'icon': '', 'script_path': 'script1'}, 'hash2': {'page_name': 'page2', 'icon': '🎉', 'script_path': 'script2'}}))
    @patch('streamlit.runtime.app_session.AppSession._enqueue_forward_msg')
    def test_on_pages_changed(self, mock_enqueue: MagicMock):
        if False:
            for i in range(10):
                print('nop')
        session = _create_test_session()
        session._on_pages_changed('/foo/pages')
        expected_msg = ForwardMsg()
        expected_msg.pages_changed.app_pages.extend([AppPage(page_script_hash='hash1', page_name='page1', icon=''), AppPage(page_script_hash='hash2', page_name='page2', icon='🎉')])
        mock_enqueue.assert_called_once_with(expected_msg)

    @patch('streamlit.runtime.app_session.source_util.register_pages_changed_callback')
    def test_installs_pages_watcher_on_init(self, patched_register_callback):
        if False:
            while True:
                i = 10
        session = _create_test_session()
        patched_register_callback.assert_called_once_with(session._on_pages_changed)

    @patch('streamlit.runtime.app_session.source_util._on_pages_changed')
    def test_deregisters_pages_watcher_on_shutdown(self, patched_on_pages_changed):
        if False:
            while True:
                i = 10
        session = _create_test_session()
        session.shutdown()
        patched_on_pages_changed.disconnect.assert_called_once_with(session._on_pages_changed)

    def test_tags_fwd_msgs_with_last_backmsg_id_if_set(self):
        if False:
            for i in range(10):
                print('nop')
        session = _create_test_session()
        session._debug_last_backmsg_id = 'some backmsg id'
        msg = ForwardMsg()
        session._enqueue_forward_msg(msg)
        self.assertEqual(msg.debug_last_backmsg_id, 'some backmsg id')

    @patch('streamlit.runtime.app_session.config.on_config_parsed')
    @patch('streamlit.runtime.app_session.source_util.register_pages_changed_callback')
    @patch('streamlit.runtime.app_session.secrets_singleton.file_change_listener.connect')
    def test_registers_file_watchers(self, patched_secrets_connect, patched_register_pages_changed_callback, patched_on_config_parsed):
        if False:
            return 10
        session = _create_test_session()
        session._local_sources_watcher.register_file_change_callback.assert_called_once_with(session._on_source_file_changed)
        patched_on_config_parsed.assert_called_once_with(session._on_source_file_changed, force_connect=True)
        patched_register_pages_changed_callback.assert_called_once_with(session._on_pages_changed)
        patched_secrets_connect.assert_called_once_with(session._on_secrets_file_changed)

    def test_recreates_local_sources_watcher_if_none(self):
        if False:
            print('Hello World!')
        session = _create_test_session()
        session._local_sources_watcher = None
        session.register_file_watchers()
        self.assertIsNotNone(session._local_sources_watcher)

    @patch('streamlit.runtime.app_session.secrets_singleton.file_change_listener.disconnect')
    def test_disconnect_file_watchers(self, patched_secrets_disconnect):
        if False:
            i = 10
            return i + 15
        session = _create_test_session()
        with patch.object(session._local_sources_watcher, 'close') as patched_close_local_sources_watcher, patch.object(session, '_stop_config_listener') as patched_stop_config_listener, patch.object(session, '_stop_pages_listener') as patched_stop_pages_listener:
            session.disconnect_file_watchers()
            patched_close_local_sources_watcher.assert_called_once()
            patched_stop_config_listener.assert_called_once()
            patched_stop_pages_listener.assert_called_once()
            patched_secrets_disconnect.assert_called_once_with(session._on_secrets_file_changed)
            self.assertIsNone(session._local_sources_watcher)
            self.assertIsNone(session._stop_config_listener)
            self.assertIsNone(session._stop_pages_listener)

    def test_disconnect_file_watchers_removes_refs(self):
        if False:
            return 10
        'Test that calling disconnect_file_watchers on the AppSession\n        removes references to it so it is eligible to be garbage collected after the\n        method is called.\n        '
        session = _create_test_session()
        self.assertGreater(len(gc.get_referrers(session)), 0)
        session.disconnect_file_watchers()
        gc.collect(2)
        self.assertEqual(len(gc.get_referrers(session)), 0)

    @patch('streamlit.runtime.app_session.AppSession._enqueue_forward_msg')
    def test_handle_file_urls_request(self, mock_enqueue):
        if False:
            while True:
                i = 10
        session = _create_test_session()
        upload_file_urls = [UploadFileUrlInfo(file_id='file_1', upload_url='upload_file_url_1', delete_url='delete_file_url_1'), UploadFileUrlInfo(file_id='file_2', upload_url='upload_file_url_2', delete_url='delete_file_url_2'), UploadFileUrlInfo(file_id='file_3', upload_url='upload_file_url_3', delete_url='delete_file_url_3')]
        session._uploaded_file_mgr.get_upload_urls.return_value = upload_file_urls
        session._handle_file_urls_request(FileURLsRequest(request_id='my_id', file_names=['file_1', 'file_2', 'file_3'], session_id=session.id))
        session._uploaded_file_mgr.get_upload_urls.assert_called_once_with(session.id, ['file_1', 'file_2', 'file_3'])
        expected_msg = ForwardMsg(file_urls_response=FileURLsResponse(response_id='my_id', file_urls=[FileURLs(file_id=url.file_id, upload_url=url.upload_url, delete_url=url.delete_url) for url in upload_file_urls]))
        mock_enqueue.assert_called_once_with(expected_msg)

def _mock_get_options_for_section(overrides=None) -> Callable[..., Any]:
    if False:
        return 10
    if not overrides:
        overrides = {}
    theme_opts = {'base': 'dark', 'primaryColor': 'coral', 'backgroundColor': 'white', 'secondaryBackgroundColor': 'blue', 'textColor': 'black', 'font': 'serif'}
    for (k, v) in overrides.items():
        theme_opts[k] = v

    def get_options_for_section(section):
        if False:
            return 10
        if section == 'theme':
            return theme_opts
        return config.get_options_for_section(section)
    return get_options_for_section

class AppSessionScriptEventTest(IsolatedAsyncioTestCase):
    """Tests for AppSession's ScriptRunner event handling."""

    @patch('streamlit.runtime.app_session.config.get_options_for_section', MagicMock(side_effect=_mock_get_options_for_section()))
    @patch('streamlit.runtime.app_session.source_util.get_pages', MagicMock(return_value={'hash1': {'page_name': 'page1', 'icon': '', 'script_path': 'script1'}, 'hash2': {'page_name': 'page2', 'icon': '🎉', 'script_path': 'script2'}}))
    @patch('streamlit.runtime.app_session._generate_scriptrun_id', MagicMock(return_value='mock_scriptrun_id'))
    async def test_enqueue_new_session_message(self):
        """The SCRIPT_STARTED event should enqueue a 'new_session' message."""
        session = _create_test_session(asyncio.get_running_loop())
        orig_ctx = get_script_run_ctx()
        ctx = ScriptRunContext(session_id='TestSessionID', _enqueue=session._enqueue_forward_msg, query_string='', session_state=MagicMock(), uploaded_file_mgr=MagicMock(), page_script_hash='', user_info={'email': 'test@test.com'})
        add_script_run_ctx(ctx=ctx)
        mock_scriptrunner = MagicMock(spec=ScriptRunner)
        session._scriptrunner = mock_scriptrunner
        session._on_scriptrunner_event(sender=mock_scriptrunner, event=ScriptRunnerEvent.SCRIPT_STARTED, page_script_hash='')
        await asyncio.sleep(0)
        sent_messages = session._browser_queue._queue
        self.assertEqual(2, len(sent_messages))
        new_session_msg = sent_messages[0].new_session
        self.assertEqual('mock_scriptrun_id', new_session_msg.script_run_id)
        self.assertTrue(new_session_msg.HasField('config'))
        self.assertEqual(config.get_option('server.allowRunOnSave'), new_session_msg.config.allow_run_on_save)
        self.assertTrue(new_session_msg.HasField('custom_theme'))
        self.assertEqual('black', new_session_msg.custom_theme.text_color)
        init_msg = new_session_msg.initialize
        self.assertTrue(init_msg.HasField('user_info'))
        self.assertEqual(list(new_session_msg.app_pages), [AppPage(page_script_hash='hash1', page_name='page1', icon=''), AppPage(page_script_hash='hash2', page_name='page2', icon='🎉')])
        add_script_run_ctx(ctx=orig_ctx)

    async def test_events_handled_on_event_loop(self):
        """ScriptRunner events should be handled on the main thread only."""
        session = _create_test_session(asyncio.get_running_loop())
        handle_event_spy = MagicMock(side_effect=session._handle_scriptrunner_event_on_event_loop)
        session._handle_scriptrunner_event_on_event_loop = handle_event_spy
        thread = threading.Thread(target=lambda : session._on_scriptrunner_event(sender=MagicMock(), event=ScriptRunnerEvent.SCRIPT_STARTED))
        thread.start()
        thread.join()
        handle_event_spy.assert_not_called()
        await asyncio.sleep(0)
        handle_event_spy.assert_called_once()

    async def test_event_handler_asserts_if_called_off_event_loop(self):
        """AppSession._handle_scriptrunner_event_on_event_loop will assert
        if it's called from another event loop (or no event loop).
        """
        event_loop = asyncio.get_running_loop()
        session = _create_test_session(event_loop)
        with patch('streamlit.runtime.app_session.asyncio.get_running_loop', return_value=MagicMock()):
            with self.assertRaises(AssertionError):
                session._handle_scriptrunner_event_on_event_loop(sender=MagicMock(), event=ScriptRunnerEvent.SCRIPT_STARTED)

    @patch('streamlit.runtime.app_session.config.get_options_for_section', MagicMock(side_effect=_mock_get_options_for_section()))
    @patch('streamlit.runtime.app_session._generate_scriptrun_id', MagicMock(return_value='mock_scriptrun_id'))
    async def test_handle_backmsg_exception(self):
        """handle_backmsg_exception is a bit of a hack. Test that it does
        what it says.
        """
        session = _create_test_session(asyncio.get_running_loop())
        forward_msg_queue_events: List[Any] = []
        CLEAR_QUEUE = object()
        mock_queue = MagicMock(spec=ForwardMsgQueue)
        mock_queue.enqueue = MagicMock(side_effect=lambda msg: forward_msg_queue_events.append(msg))
        mock_queue.clear = MagicMock(side_effect=lambda : forward_msg_queue_events.append(CLEAR_QUEUE))
        session._browser_queue = mock_queue
        FAKE_EXCEPTION = RuntimeError('I am error')
        session.handle_backmsg_exception(FAKE_EXCEPTION)
        self.assertEqual([], forward_msg_queue_events)
        await asyncio.sleep(0)
        expected_events = []
        with patch.object(session, '_state', new=AppSessionState.APP_IS_RUNNING):
            expected_events.extend([session._create_script_finished_message(ForwardMsg.FINISHED_SUCCESSFULLY), CLEAR_QUEUE, session._create_new_session_message(page_script_hash=''), session._create_session_status_changed_message()])
        with patch.object(session, '_state', new=AppSessionState.APP_NOT_RUNNING):
            expected_events.extend([session._create_script_finished_message(ForwardMsg.FINISHED_SUCCESSFULLY), session._create_session_status_changed_message(), session._create_exception_message(FAKE_EXCEPTION)])
        self.assertEqual(expected_events, forward_msg_queue_events)

    async def test_handle_backmsg_handles_exceptions(self):
        """Exceptions raised in handle_backmsg should be sent to
        handle_backmsg_exception.
        """
        session = _create_test_session(asyncio.get_running_loop())
        with patch.object(session, 'handle_backmsg_exception') as handle_backmsg_exception, patch.object(session, '_handle_clear_cache_request') as handle_clear_cache_request:
            error = Exception('explode!')
            handle_clear_cache_request.side_effect = error
            msg = BackMsg()
            msg.clear_cache = True
            session.handle_backmsg(msg)
            handle_clear_cache_request.assert_called_once()
            handle_backmsg_exception.assert_called_once_with(error)

    @patch('streamlit.runtime.app_session.AppSession._create_scriptrunner', MagicMock())
    async def test_handle_backmsg_handles_debug_ids(self):
        session = _create_test_session(asyncio.get_running_loop())
        msg = BackMsg(rerun_script=session._client_state, debug_last_backmsg_id='some backmsg')
        session.handle_backmsg(msg)
        self.assertEqual(session._debug_last_backmsg_id, 'some backmsg')

class PopulateCustomThemeMsgTest(unittest.TestCase):

    @patch('streamlit.runtime.app_session.config')
    def test_no_custom_theme_prop_if_no_theme(self, patched_config):
        if False:
            for i in range(10):
                print('nop')
        patched_config.get_options_for_section.side_effect = _mock_get_options_for_section({'base': None, 'primaryColor': None, 'backgroundColor': None, 'secondaryBackgroundColor': None, 'textColor': None, 'font': None})
        msg = ForwardMsg()
        new_session_msg = msg.new_session
        app_session._populate_theme_msg(new_session_msg.custom_theme)
        self.assertEqual(new_session_msg.HasField('custom_theme'), False)

    @patch('streamlit.runtime.app_session.config')
    def test_can_specify_some_options(self, patched_config):
        if False:
            return 10
        patched_config.get_options_for_section.side_effect = _mock_get_options_for_section({'backgroundColor': None, 'secondaryBackgroundColor': None, 'textColor': None})
        msg = ForwardMsg()
        new_session_msg = msg.new_session
        app_session._populate_theme_msg(new_session_msg.custom_theme)
        self.assertEqual(new_session_msg.HasField('custom_theme'), True)
        self.assertEqual(new_session_msg.custom_theme.primary_color, 'coral')
        self.assertEqual(new_session_msg.custom_theme.background_color, '')

    @patch('streamlit.runtime.app_session.config')
    def test_can_specify_all_options(self, patched_config):
        if False:
            for i in range(10):
                print('nop')
        patched_config.get_options_for_section.side_effect = _mock_get_options_for_section()
        msg = ForwardMsg()
        new_session_msg = msg.new_session
        app_session._populate_theme_msg(new_session_msg.custom_theme)
        self.assertEqual(new_session_msg.HasField('custom_theme'), True)
        self.assertEqual(new_session_msg.custom_theme.primary_color, 'coral')
        self.assertEqual(new_session_msg.custom_theme.background_color, 'white')

    @patch('streamlit.runtime.app_session.LOGGER')
    @patch('streamlit.runtime.app_session.config')
    def test_logs_warning_if_base_invalid(self, patched_config, patched_logger):
        if False:
            i = 10
            return i + 15
        patched_config.get_options_for_section.side_effect = _mock_get_options_for_section({'base': 'blah'})
        msg = ForwardMsg()
        new_session_msg = msg.new_session
        app_session._populate_theme_msg(new_session_msg.custom_theme)
        patched_logger.warning.assert_called_once_with('"blah" is an invalid value for theme.base. Allowed values include [\'light\', \'dark\']. Setting theme.base to "light".')

    @patch('streamlit.runtime.app_session.LOGGER')
    @patch('streamlit.runtime.app_session.config')
    def test_logs_warning_if_font_invalid(self, patched_config, patched_logger):
        if False:
            for i in range(10):
                print('nop')
        patched_config.get_options_for_section.side_effect = _mock_get_options_for_section({'font': 'comic sans'})
        msg = ForwardMsg()
        new_session_msg = msg.new_session
        app_session._populate_theme_msg(new_session_msg.custom_theme)
        patched_logger.warning.assert_called_once_with('"comic sans" is an invalid value for theme.font. Allowed values include [\'sans serif\', \'serif\', \'monospace\']. Setting theme.font to "sans serif".')

@patch('streamlit.runtime.app_session.source_util.get_pages', MagicMock(return_value={'hash1': {'page_name': 'page1', 'script_path': 'page1.py'}, 'hash2': {'page_name': 'page2', 'script_path': 'page2.py'}}))
class ShouldRerunOnFileChangeTest(unittest.TestCase):

    def test_returns_true_if_current_page_changed(self):
        if False:
            print('Hello World!')
        session = _create_test_session()
        session._client_state.page_script_hash = 'hash2'
        self.assertEqual(session._should_rerun_on_file_change('page2.py'), True)

    def test_returns_true_if_changed_file_is_not_page(self):
        if False:
            i = 10
            return i + 15
        session = _create_test_session()
        session._client_state.page_script_hash = 'hash1'
        self.assertEqual(session._should_rerun_on_file_change('some_other_file.py'), True)

    def test_returns_false_if_different_page_changed(self):
        if False:
            for i in range(10):
                print('nop')
        session = _create_test_session()
        session._client_state.page_script_hash = 'hash2'
        self.assertEqual(session._should_rerun_on_file_change('page1.py'), False)