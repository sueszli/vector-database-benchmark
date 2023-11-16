"""Base class for DeltaGenerator-related unit tests."""
import threading
import unittest
from typing import List
from unittest.mock import MagicMock
from streamlit.proto.Delta_pb2 import Delta
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime import Runtime
from streamlit.runtime.caching.storage.dummy_cache_storage import MemoryCacheStorageManager
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.scriptrunner import ScriptRunContext, add_script_run_ctx, get_script_run_ctx
from streamlit.runtime.scriptrunner.script_requests import ScriptRequests
from streamlit.runtime.state import SafeSessionState, SessionState
from streamlit.web.server.server import MEDIA_ENDPOINT, UPLOAD_FILE_ENDPOINT

class DeltaGeneratorTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.forward_msg_queue = ForwardMsgQueue()
        self.orig_report_ctx = get_script_run_ctx()
        self.script_run_ctx = ScriptRunContext(session_id='test session id', _enqueue=self.forward_msg_queue.enqueue, query_string='', session_state=SafeSessionState(SessionState(), lambda : None), uploaded_file_mgr=MemoryUploadedFileManager(UPLOAD_FILE_ENDPOINT), page_script_hash='', user_info={'email': 'test@test.com'}, script_requests=ScriptRequests())
        add_script_run_ctx(threading.current_thread(), self.script_run_ctx)
        self.media_file_storage = MemoryMediaFileStorage(MEDIA_ENDPOINT)
        mock_runtime = MagicMock(spec=Runtime)
        mock_runtime.cache_storage_manager = MemoryCacheStorageManager()
        mock_runtime.media_file_mgr = MediaFileManager(self.media_file_storage)
        mock_runtime.uploaded_file_mgr = self.script_run_ctx.uploaded_file_mgr
        Runtime._instance = mock_runtime

    def tearDown(self):
        if False:
            return 10
        self.clear_queue()
        add_script_run_ctx(threading.current_thread(), self.orig_report_ctx)
        Runtime._instance = None

    def get_message_from_queue(self, index=-1) -> ForwardMsg:
        if False:
            for i in range(10):
                print('nop')
        'Get a ForwardMsg proto from the queue, by index.'
        return self.forward_msg_queue._queue[index]

    def get_delta_from_queue(self, index=-1) -> Delta:
        if False:
            return 10
        'Get a Delta proto from the queue, by index.'
        deltas = self.get_all_deltas_from_queue()
        return deltas[index]

    def get_all_deltas_from_queue(self) -> List[Delta]:
        if False:
            return 10
        'Return all the delta messages in our ForwardMsgQueue'
        return [msg.delta for msg in self.forward_msg_queue._queue if msg.HasField('delta')]

    def clear_queue(self) -> None:
        if False:
            return 10
        self.forward_msg_queue.clear()