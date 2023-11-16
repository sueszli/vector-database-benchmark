from typing import Callable, Dict, List, Optional, cast
from typing_extensions import Final
from streamlit.logger import get_logger
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.session_manager import ActiveSessionInfo, SessionClient, SessionInfo, SessionManager, SessionStorage
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.watcher import LocalSourcesWatcher
LOGGER: Final = get_logger(__name__)

class WebsocketSessionManager(SessionManager):
    """A SessionManager used to manage sessions with lifecycles tied to those of a
    browser tab's websocket connection.

    WebsocketSessionManagers differentiate between "active" and "inactive" sessions.
    Active sessions are those with a currently active websocket connection. Inactive
    sessions are sessions without. Eventual cleanup of inactive sessions is a detail left
    to the specific SessionStorage that a WebsocketSessionManager is instantiated with.
    """

    def __init__(self, session_storage: SessionStorage, uploaded_file_manager: UploadedFileManager, script_cache: ScriptCache, message_enqueued_callback: Optional[Callable[[], None]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._session_storage = session_storage
        self._uploaded_file_mgr = uploaded_file_manager
        self._script_cache = script_cache
        self._message_enqueued_callback = message_enqueued_callback
        self._active_session_info_by_id: Dict[str, ActiveSessionInfo] = {}

    def connect_session(self, client: SessionClient, script_data: ScriptData, user_info: Dict[str, Optional[str]], existing_session_id: Optional[str]=None) -> str:
        if False:
            return 10
        if existing_session_id in self._active_session_info_by_id:
            LOGGER.warning('Session with id %s is already connected! Connecting to a new session.', existing_session_id)
        session_info = existing_session_id and existing_session_id not in self._active_session_info_by_id and self._session_storage.get(existing_session_id)
        if session_info:
            existing_session = session_info.session
            existing_session.register_file_watchers()
            self._active_session_info_by_id[existing_session.id] = ActiveSessionInfo(client, existing_session, session_info.script_run_count)
            self._session_storage.delete(existing_session.id)
            return existing_session.id
        session = AppSession(script_data=script_data, uploaded_file_manager=self._uploaded_file_mgr, script_cache=self._script_cache, message_enqueued_callback=self._message_enqueued_callback, local_sources_watcher=LocalSourcesWatcher(script_data.main_script_path), user_info=user_info)
        LOGGER.debug('Created new session for client %s. Session ID: %s', id(client), session.id)
        assert session.id not in self._active_session_info_by_id, f"session.id '{session.id}' registered multiple times!"
        self._active_session_info_by_id[session.id] = ActiveSessionInfo(client, session)
        return session.id

    def disconnect_session(self, session_id: str) -> None:
        if False:
            while True:
                i = 10
        if session_id in self._active_session_info_by_id:
            active_session_info = self._active_session_info_by_id[session_id]
            session = active_session_info.session
            session.request_script_stop()
            session.disconnect_file_watchers()
            self._session_storage.save(SessionInfo(client=None, session=session, script_run_count=active_session_info.script_run_count))
            del self._active_session_info_by_id[session_id]

    def get_active_session_info(self, session_id: str) -> Optional[ActiveSessionInfo]:
        if False:
            return 10
        return self._active_session_info_by_id.get(session_id)

    def is_active_session(self, session_id: str) -> bool:
        if False:
            return 10
        return session_id in self._active_session_info_by_id

    def list_active_sessions(self) -> List[ActiveSessionInfo]:
        if False:
            return 10
        return list(self._active_session_info_by_id.values())

    def close_session(self, session_id: str) -> None:
        if False:
            while True:
                i = 10
        if session_id in self._active_session_info_by_id:
            active_session_info = self._active_session_info_by_id[session_id]
            del self._active_session_info_by_id[session_id]
            active_session_info.session.shutdown()
            return
        session_info = self._session_storage.get(session_id)
        if session_info:
            self._session_storage.delete(session_id)
            session_info.session.shutdown()

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        if False:
            while True:
                i = 10
        session_info = self.get_active_session_info(session_id)
        if session_info:
            return cast(SessionInfo, session_info)
        return self._session_storage.get(session_id)

    def list_sessions(self) -> List[SessionInfo]:
        if False:
            while True:
                i = 10
        return cast(List[SessionInfo], self.list_active_sessions()) + self._session_storage.list()