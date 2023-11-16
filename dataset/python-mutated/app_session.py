import asyncio
import sys
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union
import streamlit.elements.exception as exception_utils
from streamlit import config, runtime, source_util
from streamlit.case_converters import to_snake_case
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.proto.Common_pb2 import FileURLs, FileURLsRequest
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.GitInfo_pb2 import GitInfo
from streamlit.proto.NewSession_pb2 import Config, CustomThemeConfig, NewSession, UserInfo
from streamlit.proto.PagesChanged_pb2 import PagesChanged
from streamlit.runtime import caching, legacy_caching
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.metrics_util import Installation
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner import RerunData, ScriptRunner, ScriptRunnerEvent
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.secrets import secrets_singleton
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.version import STREAMLIT_VERSION_STRING
from streamlit.watcher import LocalSourcesWatcher
LOGGER = get_logger(__name__)
if TYPE_CHECKING:
    from streamlit.runtime.state import SessionState

class AppSessionState(Enum):
    APP_NOT_RUNNING = 'APP_NOT_RUNNING'
    APP_IS_RUNNING = 'APP_IS_RUNNING'
    SHUTDOWN_REQUESTED = 'SHUTDOWN_REQUESTED'

def _generate_scriptrun_id() -> str:
    if False:
        for i in range(10):
            print('nop')
    'Randomly generate a unique ID for a script execution.'
    return str(uuid.uuid4())

class AppSession:
    """
    Contains session data for a single "user" of an active app
    (that is, a connected browser tab).

    Each AppSession has its own ScriptData, root DeltaGenerator, ScriptRunner,
    and widget state.

    An AppSession is attached to each thread involved in running its script.

    """

    def __init__(self, script_data: ScriptData, uploaded_file_manager: UploadedFileManager, script_cache: ScriptCache, message_enqueued_callback: Optional[Callable[[], None]], local_sources_watcher: LocalSourcesWatcher, user_info: Dict[str, Optional[str]]) -> None:
        if False:
            return 10
        'Initialize the AppSession.\n\n        Parameters\n        ----------\n        script_data\n            Object storing parameters related to running a script\n\n        uploaded_file_manager\n            Used to manage files uploaded by users via the Streamlit web client.\n\n        script_cache\n            The app\'s ScriptCache instance. Stores cached user scripts. ScriptRunner\n            uses the ScriptCache to avoid having to reload user scripts from disk\n            on each rerun.\n\n        message_enqueued_callback\n            After enqueuing a message, this callable notification will be invoked.\n\n        local_sources_watcher\n            The file watcher that lets the session know local files have changed.\n\n        user_info\n            A dict that contains information about the current user. For now,\n            it only contains the user\'s email address.\n\n            {\n                "email": "example@example.com"\n            }\n\n            Information about the current user is optionally provided when a\n            websocket connection is initialized via the "X-Streamlit-User" header.\n\n        '
        self.id = str(uuid.uuid4())
        self._event_loop = asyncio.get_running_loop()
        self._script_data = script_data
        self._uploaded_file_mgr = uploaded_file_manager
        self._script_cache = script_cache
        self._browser_queue = ForwardMsgQueue()
        self._message_enqueued_callback = message_enqueued_callback
        self._state = AppSessionState.APP_NOT_RUNNING
        self._client_state = ClientState()
        self._local_sources_watcher: Optional[LocalSourcesWatcher] = local_sources_watcher
        self._stop_config_listener: Optional[Callable[[], bool]] = None
        self._stop_pages_listener: Optional[Callable[[], bool]] = None
        self.register_file_watchers()
        self._run_on_save = config.get_option('server.runOnSave')
        self._scriptrunner: Optional[ScriptRunner] = None
        from streamlit.runtime.state import SessionState
        self._session_state = SessionState()
        self._user_info = user_info
        self._debug_last_backmsg_id: Optional[str] = None
        LOGGER.debug('AppSession initialized (id=%s)', self.id)

    def __del__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Ensure that we call shutdown() when an AppSession is garbage collected.'
        self.shutdown()

    def register_file_watchers(self) -> None:
        if False:
            return 10
        "Register handlers to be called when various files are changed.\n\n        Files that we watch include:\n          * source files that already exist (for edits)\n          * `.py` files in the the main script's `pages/` directory (for file additions\n            and deletions)\n          * project and user-level config.toml files\n          * the project-level secrets.toml files\n\n        This method is called automatically on AppSession construction, but it may be\n        called again in the case when a session is disconnected and is being reconnect\n        to.\n        "
        if self._local_sources_watcher is None:
            self._local_sources_watcher = LocalSourcesWatcher(self._script_data.main_script_path)
        self._local_sources_watcher.register_file_change_callback(self._on_source_file_changed)
        self._stop_config_listener = config.on_config_parsed(self._on_source_file_changed, force_connect=True)
        self._stop_pages_listener = source_util.register_pages_changed_callback(self._on_pages_changed)
        secrets_singleton.file_change_listener.connect(self._on_secrets_file_changed)

    def disconnect_file_watchers(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Disconnect the file watcher handlers registered by register_file_watchers.'
        if self._local_sources_watcher is not None:
            self._local_sources_watcher.close()
        if self._stop_config_listener is not None:
            self._stop_config_listener()
        if self._stop_pages_listener is not None:
            self._stop_pages_listener()
        secrets_singleton.file_change_listener.disconnect(self._on_secrets_file_changed)
        self._local_sources_watcher = None
        self._stop_config_listener = None
        self._stop_pages_listener = None

    def flush_browser_queue(self) -> List[ForwardMsg]:
        if False:
            for i in range(10):
                print('nop')
        'Clear the forward message queue and return the messages it contained.\n\n        The Server calls this periodically to deliver new messages\n        to the browser connected to this app.\n\n        Returns\n        -------\n        list[ForwardMsg]\n            The messages that were removed from the queue and should\n            be delivered to the browser.\n\n        '
        return self._browser_queue.flush()

    def shutdown(self) -> None:
        if False:
            return 10
        "Shut down the AppSession.\n\n        It's an error to use a AppSession after it's been shut down.\n\n        "
        if self._state != AppSessionState.SHUTDOWN_REQUESTED:
            LOGGER.debug('Shutting down (id=%s)', self.id)
            self._uploaded_file_mgr.remove_session_files(self.id)
            if runtime.exists():
                rt = runtime.get_instance()
                rt.media_file_mgr.clear_session_refs(self.id)
                rt.media_file_mgr.remove_orphaned_files()
            self.request_script_stop()
            self._state = AppSessionState.SHUTDOWN_REQUESTED
            self.disconnect_file_watchers()

    def _enqueue_forward_msg(self, msg: ForwardMsg) -> None:
        if False:
            while True:
                i = 10
        'Enqueue a new ForwardMsg to our browser queue.\n\n        This can be called on both the main thread and a ScriptRunner\n        run thread.\n\n        Parameters\n        ----------\n        msg : ForwardMsg\n            The message to enqueue\n\n        '
        if not config.get_option('client.displayEnabled'):
            return
        if self._debug_last_backmsg_id:
            msg.debug_last_backmsg_id = self._debug_last_backmsg_id
        self._browser_queue.enqueue(msg)
        if self._message_enqueued_callback:
            self._message_enqueued_callback()

    def handle_backmsg(self, msg: BackMsg) -> None:
        if False:
            while True:
                i = 10
        'Process a BackMsg.'
        try:
            msg_type = msg.WhichOneof('type')
            if msg_type == 'rerun_script':
                if msg.debug_last_backmsg_id:
                    self._debug_last_backmsg_id = msg.debug_last_backmsg_id
                self._handle_rerun_script_request(msg.rerun_script)
            elif msg_type == 'load_git_info':
                self._handle_git_information_request()
            elif msg_type == 'clear_cache':
                self._handle_clear_cache_request()
            elif msg_type == 'set_run_on_save':
                self._handle_set_run_on_save_request(msg.set_run_on_save)
            elif msg_type == 'stop_script':
                self._handle_stop_script_request()
            elif msg_type == 'file_urls_request':
                self._handle_file_urls_request(msg.file_urls_request)
            else:
                LOGGER.warning('No handler for "%s"', msg_type)
        except Exception as ex:
            LOGGER.error(ex)
            self.handle_backmsg_exception(ex)

    def handle_backmsg_exception(self, e: BaseException) -> None:
        if False:
            while True:
                i = 10
        'Handle an Exception raised while processing a BackMsg from the browser.'
        self._on_scriptrunner_event(self._scriptrunner, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS)
        self._on_scriptrunner_event(self._scriptrunner, ScriptRunnerEvent.SCRIPT_STARTED, page_script_hash='')
        self._on_scriptrunner_event(self._scriptrunner, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS)
        self._event_loop.call_soon_threadsafe(lambda : self._enqueue_forward_msg(self._create_exception_message(e)))

    def request_rerun(self, client_state: Optional[ClientState]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Signal that we're interested in running the script.\n\n        If the script is not already running, it will be started immediately.\n        Otherwise, a rerun will be requested.\n\n        Parameters\n        ----------\n        client_state : streamlit.proto.ClientState_pb2.ClientState | None\n            The ClientState protobuf to run the script with, or None\n            to use previous client state.\n\n        "
        if self._state == AppSessionState.SHUTDOWN_REQUESTED:
            LOGGER.warning('Discarding rerun request after shutdown')
            return
        if client_state:
            rerun_data = RerunData(client_state.query_string, client_state.widget_states, client_state.page_script_hash, client_state.page_name)
        else:
            rerun_data = RerunData()
        if self._scriptrunner is not None:
            if bool(config.get_option('runner.fastReruns')):
                self._scriptrunner.request_stop()
                self._scriptrunner = None
            else:
                success = self._scriptrunner.request_rerun(rerun_data)
                if success:
                    return
        self._create_scriptrunner(rerun_data)

    def request_script_stop(self) -> None:
        if False:
            i = 10
            return i + 15
        'Request that the scriptrunner stop execution.\n\n        Does nothing if no scriptrunner exists.\n        '
        if self._scriptrunner is not None:
            self._scriptrunner.request_stop()

    def _create_scriptrunner(self, initial_rerun_data: RerunData) -> None:
        if False:
            i = 10
            return i + 15
        'Create and run a new ScriptRunner with the given RerunData.'
        self._scriptrunner = ScriptRunner(session_id=self.id, main_script_path=self._script_data.main_script_path, session_state=self._session_state, uploaded_file_mgr=self._uploaded_file_mgr, script_cache=self._script_cache, initial_rerun_data=initial_rerun_data, user_info=self._user_info)
        self._scriptrunner.on_event.connect(self._on_scriptrunner_event)
        self._scriptrunner.start()

    @property
    def session_state(self) -> 'SessionState':
        if False:
            for i in range(10):
                print('nop')
        return self._session_state

    def _should_rerun_on_file_change(self, filepath: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        main_script_path = self._script_data.main_script_path
        pages = source_util.get_pages(main_script_path)
        changed_page_script_hash = next(filter(lambda k: pages[k]['script_path'] == filepath, pages), None)
        if changed_page_script_hash is not None:
            current_page_script_hash = self._client_state.page_script_hash
            return changed_page_script_hash == current_page_script_hash
        return True

    def _on_source_file_changed(self, filepath: Optional[str]=None) -> None:
        if False:
            return 10
        'One of our source files changed. Clear the cache and schedule a rerun if appropriate.'
        self._script_cache.clear()
        if filepath is not None and (not self._should_rerun_on_file_change(filepath)):
            return
        if self._run_on_save:
            self.request_rerun(self._client_state)
        else:
            self._enqueue_forward_msg(self._create_file_change_message())

    def _on_secrets_file_changed(self, _) -> None:
        if False:
            print('Hello World!')
        'Called when `secrets.file_change_listener` emits a Signal.'
        self._on_source_file_changed()

    def _on_pages_changed(self, _) -> None:
        if False:
            while True:
                i = 10
        msg = ForwardMsg()
        _populate_app_pages(msg.pages_changed, self._script_data.main_script_path)
        self._enqueue_forward_msg(msg)

    def _clear_queue(self) -> None:
        if False:
            return 10
        self._browser_queue.clear()

    def _on_scriptrunner_event(self, sender: Optional[ScriptRunner], event: ScriptRunnerEvent, forward_msg: Optional[ForwardMsg]=None, exception: Optional[BaseException]=None, client_state: Optional[ClientState]=None, page_script_hash: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        "Called when our ScriptRunner emits an event.\n\n        This is generally called from the sender ScriptRunner's script thread.\n        We forward the event on to _handle_scriptrunner_event_on_event_loop,\n        which will be called on the main thread.\n        "
        self._event_loop.call_soon_threadsafe(lambda : self._handle_scriptrunner_event_on_event_loop(sender, event, forward_msg, exception, client_state, page_script_hash))

    def _handle_scriptrunner_event_on_event_loop(self, sender: Optional[ScriptRunner], event: ScriptRunnerEvent, forward_msg: Optional[ForwardMsg]=None, exception: Optional[BaseException]=None, client_state: Optional[ClientState]=None, page_script_hash: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        "Handle a ScriptRunner event.\n\n        This function must only be called on our eventloop thread.\n\n        Parameters\n        ----------\n        sender : ScriptRunner | None\n            The ScriptRunner that emitted the event. (This may be set to\n            None when called from `handle_backmsg_exception`, if no\n            ScriptRunner was active when the backmsg exception was raised.)\n\n        event : ScriptRunnerEvent\n            The event type.\n\n        forward_msg : ForwardMsg | None\n            The ForwardMsg to send to the frontend. Set only for the\n            ENQUEUE_FORWARD_MSG event.\n\n        exception : BaseException | None\n            An exception thrown during compilation. Set only for the\n            SCRIPT_STOPPED_WITH_COMPILE_ERROR event.\n\n        client_state : streamlit.proto.ClientState_pb2.ClientState | None\n            The ScriptRunner's final ClientState. Set only for the\n            SHUTDOWN event.\n\n        page_script_hash : str | None\n            A hash of the script path corresponding to the page currently being\n            run. Set only for the SCRIPT_STARTED event.\n        "
        assert self._event_loop == asyncio.get_running_loop(), 'This function must only be called on the eventloop thread the AppSession was created on.'
        if sender is not self._scriptrunner:
            LOGGER.debug('Ignoring event from non-current ScriptRunner: %s', event)
            return
        prev_state = self._state
        if event == ScriptRunnerEvent.SCRIPT_STARTED:
            if self._state != AppSessionState.SHUTDOWN_REQUESTED:
                self._state = AppSessionState.APP_IS_RUNNING
            assert page_script_hash is not None, 'page_script_hash must be set for the SCRIPT_STARTED event'
            self._clear_queue()
            self._enqueue_forward_msg(self._create_new_session_message(page_script_hash))
        elif event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS or event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_COMPILE_ERROR:
            if self._state != AppSessionState.SHUTDOWN_REQUESTED:
                self._state = AppSessionState.APP_NOT_RUNNING
            script_succeeded = event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS
            script_finished_msg = self._create_script_finished_message(ForwardMsg.FINISHED_SUCCESSFULLY if script_succeeded else ForwardMsg.FINISHED_WITH_COMPILE_ERROR)
            self._enqueue_forward_msg(script_finished_msg)
            self._debug_last_backmsg_id = None
            if script_succeeded:
                if self._local_sources_watcher:
                    self._local_sources_watcher.update_watched_modules()
            else:
                assert exception is not None, 'exception must be set for the SCRIPT_STOPPED_WITH_COMPILE_ERROR event'
                msg = ForwardMsg()
                exception_utils.marshall(msg.session_event.script_compilation_exception, exception)
                self._enqueue_forward_msg(msg)
        elif event == ScriptRunnerEvent.SCRIPT_STOPPED_FOR_RERUN:
            script_finished_msg = self._create_script_finished_message(ForwardMsg.FINISHED_EARLY_FOR_RERUN)
            self._enqueue_forward_msg(script_finished_msg)
            if self._local_sources_watcher:
                self._local_sources_watcher.update_watched_modules()
        elif event == ScriptRunnerEvent.SHUTDOWN:
            assert client_state is not None, 'client_state must be set for the SHUTDOWN event'
            if self._state == AppSessionState.SHUTDOWN_REQUESTED:
                runtime.get_instance().media_file_mgr.clear_session_refs(self.id)
            self._client_state = client_state
            self._scriptrunner = None
        elif event == ScriptRunnerEvent.ENQUEUE_FORWARD_MSG:
            assert forward_msg is not None, 'null forward_msg in ENQUEUE_FORWARD_MSG event'
            self._enqueue_forward_msg(forward_msg)
        app_was_running = prev_state == AppSessionState.APP_IS_RUNNING
        app_is_running = self._state == AppSessionState.APP_IS_RUNNING
        if app_is_running != app_was_running:
            self._enqueue_forward_msg(self._create_session_status_changed_message())

    def _create_session_status_changed_message(self) -> ForwardMsg:
        if False:
            for i in range(10):
                print('nop')
        'Create and return a session_status_changed ForwardMsg.'
        msg = ForwardMsg()
        msg.session_status_changed.run_on_save = self._run_on_save
        msg.session_status_changed.script_is_running = self._state == AppSessionState.APP_IS_RUNNING
        return msg

    def _create_file_change_message(self) -> ForwardMsg:
        if False:
            while True:
                i = 10
        "Create and return a 'script_changed_on_disk' ForwardMsg."
        msg = ForwardMsg()
        msg.session_event.script_changed_on_disk = True
        return msg

    def _create_new_session_message(self, page_script_hash: str) -> ForwardMsg:
        if False:
            return 10
        'Create and return a new_session ForwardMsg.'
        msg = ForwardMsg()
        msg.new_session.script_run_id = _generate_scriptrun_id()
        msg.new_session.name = self._script_data.name
        msg.new_session.main_script_path = self._script_data.main_script_path
        msg.new_session.page_script_hash = page_script_hash
        _populate_app_pages(msg.new_session, self._script_data.main_script_path)
        _populate_config_msg(msg.new_session.config)
        _populate_theme_msg(msg.new_session.custom_theme)
        imsg = msg.new_session.initialize
        _populate_user_info_msg(imsg.user_info)
        imsg.environment_info.streamlit_version = STREAMLIT_VERSION_STRING
        imsg.environment_info.python_version = '.'.join(map(str, sys.version_info))
        imsg.session_status.run_on_save = self._run_on_save
        imsg.session_status.script_is_running = self._state == AppSessionState.APP_IS_RUNNING
        imsg.command_line = self._script_data.command_line
        imsg.session_id = self.id
        return msg

    def _create_script_finished_message(self, status: 'ForwardMsg.ScriptFinishedStatus.ValueType') -> ForwardMsg:
        if False:
            while True:
                i = 10
        'Create and return a script_finished ForwardMsg.'
        msg = ForwardMsg()
        msg.script_finished = status
        return msg

    def _create_exception_message(self, e: BaseException) -> ForwardMsg:
        if False:
            i = 10
            return i + 15
        'Create and return an Exception ForwardMsg.'
        msg = ForwardMsg()
        exception_utils.marshall(msg.delta.new_element.exception, e)
        return msg

    def _handle_git_information_request(self) -> None:
        if False:
            i = 10
            return i + 15
        msg = ForwardMsg()
        try:
            from streamlit.git_util import GitRepo
            repo = GitRepo(self._script_data.main_script_path)
            repo_info = repo.get_repo_info()
            if repo_info is None:
                return
            (repository_name, branch, module) = repo_info
            if repository_name.endswith('.git'):
                repository_name = repository_name[:-4]
            msg.git_info_changed.repository = repository_name
            msg.git_info_changed.branch = branch
            msg.git_info_changed.module = module
            msg.git_info_changed.untracked_files[:] = repo.untracked_files
            msg.git_info_changed.uncommitted_files[:] = repo.uncommitted_files
            if repo.is_head_detached:
                msg.git_info_changed.state = GitInfo.GitStates.HEAD_DETACHED
            elif len(repo.ahead_commits) > 0:
                msg.git_info_changed.state = GitInfo.GitStates.AHEAD_OF_REMOTE
            else:
                msg.git_info_changed.state = GitInfo.GitStates.DEFAULT
            self._enqueue_forward_msg(msg)
        except Exception as ex:
            LOGGER.debug('Obtaining Git information produced an error', exc_info=ex)

    def _handle_rerun_script_request(self, client_state: Optional[ClientState]=None) -> None:
        if False:
            print('Hello World!')
        'Tell the ScriptRunner to re-run its script.\n\n        Parameters\n        ----------\n        client_state : streamlit.proto.ClientState_pb2.ClientState | None\n            The ClientState protobuf to run the script with, or None\n            to use previous client state.\n\n        '
        self.request_rerun(client_state)

    def _handle_stop_script_request(self) -> None:
        if False:
            while True:
                i = 10
        'Tell the ScriptRunner to stop running its script.'
        self.request_script_stop()

    def _handle_clear_cache_request(self) -> None:
        if False:
            return 10
        "Clear this app's cache.\n\n        Because this cache is global, it will be cleared for all users.\n\n        "
        legacy_caching.clear_cache()
        caching.cache_data.clear()
        caching.cache_resource.clear()
        self._session_state.clear()

    def _handle_set_run_on_save_request(self, new_value: bool) -> None:
        if False:
            while True:
                i = 10
        'Change our run_on_save flag to the given value.\n\n        The browser will be notified of the change.\n\n        Parameters\n        ----------\n        new_value : bool\n            New run_on_save value\n\n        '
        self._run_on_save = new_value
        self._enqueue_forward_msg(self._create_session_status_changed_message())

    def _handle_file_urls_request(self, file_urls_request: FileURLsRequest) -> None:
        if False:
            while True:
                i = 10
        'Handle a file_urls_request BackMsg sent by the client.'
        msg = ForwardMsg()
        msg.file_urls_response.response_id = file_urls_request.request_id
        upload_url_infos = self._uploaded_file_mgr.get_upload_urls(self.id, file_urls_request.file_names)
        for upload_url_info in upload_url_infos:
            msg.file_urls_response.file_urls.append(FileURLs(file_id=upload_url_info.file_id, upload_url=upload_url_info.upload_url, delete_url=upload_url_info.delete_url))
        self._enqueue_forward_msg(msg)

def _get_toolbar_mode() -> 'Config.ToolbarMode.ValueType':
    if False:
        i = 10
        return i + 15
    config_key = 'client.toolbarMode'
    config_value = config.get_option(config_key)
    enum_value: Optional['Config.ToolbarMode.ValueType'] = getattr(Config.ToolbarMode, config_value.upper())
    if enum_value is None:
        allowed_values = ', '.join((k.lower() for k in Config.ToolbarMode.keys()))
        raise ValueError(f'Config {config_key!r} expects to have one of the following values: {allowed_values}. Current value: {config_value}')
    return enum_value

def _populate_config_msg(msg: Config) -> None:
    if False:
        while True:
            i = 10
    msg.gather_usage_stats = config.get_option('browser.gatherUsageStats')
    msg.max_cached_message_age = config.get_option('global.maxCachedMessageAge')
    msg.allow_run_on_save = config.get_option('server.allowRunOnSave')
    msg.hide_top_bar = config.get_option('ui.hideTopBar')
    msg.hide_sidebar_nav = config.get_option('ui.hideSidebarNav')
    msg.toolbar_mode = _get_toolbar_mode()

def _populate_theme_msg(msg: CustomThemeConfig) -> None:
    if False:
        while True:
            i = 10
    enum_encoded_options = {'base', 'font'}
    theme_opts = config.get_options_for_section('theme')
    if not any(theme_opts.values()):
        return
    for (option_name, option_val) in theme_opts.items():
        if option_name not in enum_encoded_options and option_val is not None:
            setattr(msg, to_snake_case(option_name), option_val)
    base_map = {'light': msg.BaseTheme.LIGHT, 'dark': msg.BaseTheme.DARK}
    base = theme_opts['base']
    if base is not None:
        if base not in base_map:
            LOGGER.warning(f'"{base}" is an invalid value for theme.base. Allowed values include {list(base_map.keys())}. Setting theme.base to "light".')
        else:
            msg.base = base_map[base]
    font_map = {'sans serif': msg.FontFamily.SANS_SERIF, 'serif': msg.FontFamily.SERIF, 'monospace': msg.FontFamily.MONOSPACE}
    font = theme_opts['font']
    if font is not None:
        if font not in font_map:
            LOGGER.warning(f'"{font}" is an invalid value for theme.font. Allowed values include {list(font_map.keys())}. Setting theme.font to "sans serif".')
        else:
            msg.font = font_map[font]

def _populate_user_info_msg(msg: UserInfo) -> None:
    if False:
        for i in range(10):
            print('nop')
    msg.installation_id = Installation.instance().installation_id
    msg.installation_id_v3 = Installation.instance().installation_id_v3

def _populate_app_pages(msg: Union[NewSession, PagesChanged], main_script_path: str) -> None:
    if False:
        while True:
            i = 10
    for (page_script_hash, page_info) in source_util.get_pages(main_script_path).items():
        page_proto = msg.app_pages.add()
        page_proto.page_script_hash = page_script_hash
        page_proto.page_name = page_info['page_name']
        page_proto.icon = page_info['icon']