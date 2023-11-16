import threading
import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.scriptrunner import ScriptRunContext, add_script_run_ctx, get_script_run_ctx
from streamlit.runtime.state import SafeSessionState, SessionState
from tests.delta_generator_test_case import DeltaGeneratorTestCase

class UserInfoProxyTest(DeltaGeneratorTestCase):
    """Test UserInfoProxy."""

    def test_user_email_attr(self):
        if False:
            i = 10
            return i + 15
        'Test that `st.user.email` returns user info from ScriptRunContext'
        self.assertEqual(st.experimental_user.email, 'test@test.com')

    def test_user_email_key(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(st.experimental_user['email'], 'test@test.com')

    def test_user_non_existing_attr(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that an error is raised when called non existed attr.'
        with self.assertRaises(AttributeError):
            st.write(st.experimental_user.attribute)

    def test_user_non_existing_key(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that an error is raised when called non existed key.'
        with self.assertRaises(KeyError):
            st.write(st.experimental_user['key'])

    def test_user_cannot_be_modified_existing_key(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that an error is raised when try to assign new value to existing key.\n        '
        with self.assertRaises(StreamlitAPIException) as e:
            st.experimental_user['email'] = 'NEW_VALUE'
        self.assertEqual(str(e.exception), 'st.experimental_user cannot be modified')

    def test_user_cannot_be_modified_new_key(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that an error is raised when try to assign new value to new key.\n        '
        with self.assertRaises(StreamlitAPIException) as e:
            st.experimental_user['foo'] = 'bar'
        self.assertEqual(str(e.exception), 'st.experimental_user cannot be modified')

    def test_user_cannot_be_modified_existing_attr(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that an error is raised when try to assign new value to existing attr.\n        '
        with self.assertRaises(StreamlitAPIException) as e:
            st.experimental_user.email = 'bar'
        self.assertEqual(str(e.exception), 'st.experimental_user cannot be modified')

    def test_user_cannot_be_modified_new_attr(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that an error is raised when try to assign new value to new attr.\n        '
        with self.assertRaises(StreamlitAPIException) as e:
            st.experimental_user.foo = 'bar'
        self.assertEqual(str(e.exception), 'st.experimental_user cannot be modified')

    def test_user_len(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(st.experimental_user), 1)

    def test_st_user_reads_from_context_(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that st.user reads information from current ScriptRunContext\n        And after ScriptRunContext changed, it returns new email\n        '
        orig_report_ctx = get_script_run_ctx()
        forward_msg_queue = ForwardMsgQueue()
        try:
            add_script_run_ctx(threading.current_thread(), ScriptRunContext(session_id='test session id', _enqueue=forward_msg_queue.enqueue, query_string='', session_state=SafeSessionState(SessionState(), lambda : None), uploaded_file_mgr=None, page_script_hash='', user_info={'email': 'something@else.com'}))
            self.assertEqual(st.experimental_user.email, 'something@else.com')
        except Exception as e:
            raise e
        finally:
            add_script_run_ctx(threading.current_thread(), orig_report_ctx)