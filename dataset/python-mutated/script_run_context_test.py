import unittest
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.scriptrunner import ScriptRunContext
from streamlit.runtime.state import SafeSessionState, SessionState

class ScriptRunContextTest(unittest.TestCase):

    def test_set_page_config_immutable(self):
        if False:
            while True:
                i = 10
        'st.set_page_config must be called at most once'
        fake_enqueue = lambda msg: None
        ctx = ScriptRunContext(session_id='TestSessionID', _enqueue=fake_enqueue, query_string='', session_state=SafeSessionState(SessionState(), lambda : None), uploaded_file_mgr=MemoryUploadedFileManager('mock/upload'), page_script_hash='', user_info={'email': 'test@test.com'})
        msg = ForwardMsg()
        msg.page_config_changed.title = 'foo'
        ctx.enqueue(msg)
        with self.assertRaises(StreamlitAPIException):
            ctx.enqueue(msg)

    def test_set_page_config_first(self):
        if False:
            print('Hello World!')
        'st.set_page_config must be called before other st commands\n        when the script has been marked as started'
        fake_enqueue = lambda msg: None
        ctx = ScriptRunContext(session_id='TestSessionID', _enqueue=fake_enqueue, query_string='', session_state=SafeSessionState(SessionState(), lambda : None), uploaded_file_mgr=MemoryUploadedFileManager('/mock/upload'), page_script_hash='', user_info={'email': 'test@test.com'})
        ctx.on_script_start()
        markdown_msg = ForwardMsg()
        markdown_msg.delta.new_element.markdown.body = 'foo'
        msg = ForwardMsg()
        msg.page_config_changed.title = 'foo'
        ctx.enqueue(markdown_msg)
        with self.assertRaises(StreamlitAPIException):
            ctx.enqueue(msg)

    def test_disallow_set_page_config_twice(self):
        if False:
            i = 10
            return i + 15
        'st.set_page_config cannot be called twice'
        fake_enqueue = lambda msg: None
        ctx = ScriptRunContext(session_id='TestSessionID', _enqueue=fake_enqueue, query_string='', session_state=SafeSessionState(SessionState(), lambda : None), uploaded_file_mgr=MemoryUploadedFileManager('/mock/upload'), page_script_hash='', user_info={'email': 'test@test.com'})
        ctx.on_script_start()
        msg = ForwardMsg()
        msg.page_config_changed.title = 'foo'
        ctx.enqueue(msg)
        with self.assertRaises(StreamlitAPIException):
            same_msg = ForwardMsg()
            same_msg.page_config_changed.title = 'bar'
            ctx.enqueue(same_msg)

    def test_set_page_config_reset(self):
        if False:
            for i in range(10):
                print('nop')
        'st.set_page_config should be allowed after a rerun'
        fake_enqueue = lambda msg: None
        ctx = ScriptRunContext(session_id='TestSessionID', _enqueue=fake_enqueue, query_string='', session_state=SafeSessionState(SessionState(), lambda : None), uploaded_file_mgr=MemoryUploadedFileManager('/mock/upload'), page_script_hash='', user_info={'email': 'test@test.com'})
        ctx.on_script_start()
        msg = ForwardMsg()
        msg.page_config_changed.title = 'foo'
        ctx.enqueue(msg)
        ctx.reset()
        try:
            ctx.on_script_start()
            ctx.enqueue(msg)
        except StreamlitAPIException:
            self.fail('set_page_config should have succeeded after reset!')