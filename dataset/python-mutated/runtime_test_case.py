import asyncio
from typing import Callable, Dict, List, Optional
from unittest import IsolatedAsyncioTestCase, mock
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime import Runtime, RuntimeConfig, RuntimeState
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.caching.storage.dummy_cache_storage import MemoryCacheStorageManager
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.session_manager import SessionClient, SessionInfo, SessionManager, SessionStorage
from streamlit.runtime.uploaded_file_manager import UploadedFileManager

class MockSessionManager(SessionManager):
    """A MockSessionManager used for runtime tests.

    This is done so that our runtime tests don't rely on a specific SessionManager
    implementation.
    """

    def __init__(self, session_storage: SessionStorage, uploaded_file_manager: UploadedFileManager, script_cache: ScriptCache, message_enqueued_callback: Optional[Callable[[], None]]) -> None:
        if False:
            while True:
                i = 10
        self._uploaded_file_mgr = uploaded_file_manager
        self._script_cache = script_cache
        self._message_enqueued_callback = message_enqueued_callback
        self._session_info_by_id: Dict[str, SessionInfo] = {}

    def connect_session(self, client: SessionClient, script_data: ScriptData, user_info: Dict[str, Optional[str]], existing_session_id: Optional[str]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('streamlit.runtime.scriptrunner.ScriptRunner', new=mock.MagicMock()):
            session = AppSession(script_data=script_data, uploaded_file_manager=self._uploaded_file_mgr, script_cache=self._script_cache, message_enqueued_callback=self._message_enqueued_callback, local_sources_watcher=mock.MagicMock(), user_info=user_info)
        assert session.id not in self._session_info_by_id, f"session.id '{session.id}' registered multiple times!"
        self._session_info_by_id[session.id] = SessionInfo(client, session)
        return session.id

    def close_session(self, session_id: str) -> None:
        if False:
            return 10
        if session_id in self._session_info_by_id:
            session_info = self._session_info_by_id[session_id]
            del self._session_info_by_id[session_id]
            session_info.session.shutdown()

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        if False:
            for i in range(10):
                print('nop')
        return self._session_info_by_id.get(session_id, None)

    def list_sessions(self) -> List[SessionInfo]:
        if False:
            for i in range(10):
                print('nop')
        return list(self._session_info_by_id.values())

class RuntimeTestCase(IsolatedAsyncioTestCase):
    """Base class for tests that use streamlit.Runtime directly."""
    _next_session_id = 0

    async def asyncSetUp(self):
        config = RuntimeConfig(script_path='mock/script/path.py', command_line='', media_file_storage=MemoryMediaFileStorage('/mock/media'), uploaded_file_manager=MemoryUploadedFileManager('/mock/upload'), session_manager_class=MockSessionManager, session_storage=mock.MagicMock(), cache_storage_manager=MemoryCacheStorageManager())
        self.runtime = Runtime(config)

    async def asyncTearDown(self):
        if self.runtime.state != RuntimeState.INITIAL:
            self.runtime.stop()
            await self.runtime.stopped
        Runtime._instance = None

    @staticmethod
    async def tick_runtime_loop() -> None:
        """Sleep just long enough to guarantee that the Runtime's loop
        has a chance to run.
        """
        await asyncio.sleep(0.03)

    def enqueue_forward_msg(self, session_id: str, msg: ForwardMsg) -> None:
        if False:
            return 10
        'Enqueue a ForwardMsg to a given session_id. It will be sent\n        to the client on the next iteration through the run loop. (You can\n        use `await self.tick_runtime_loop()` to tick the run loop.)\n        '
        session_info = self.runtime._session_mgr.get_active_session_info(session_id)
        if session_info is None:
            return
        session_info.session._enqueue_forward_msg(msg)