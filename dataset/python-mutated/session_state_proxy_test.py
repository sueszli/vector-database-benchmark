"""SessionStateProxy unit tests."""
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.state import SafeSessionState, SessionState, SessionStateProxy
from streamlit.runtime.state.common import GENERATED_WIDGET_ID_PREFIX, require_valid_user_key

def _create_mock_session_state(initial_state_values: Dict[str, Any]) -> SafeSessionState:
    if False:
        for i in range(10):
            print('nop')
    'Return a new SafeSessionState instance populated with the\n    given state values.\n    '
    session_state = SessionState()
    for (key, value) in initial_state_values.items():
        session_state[key] = value
    return SafeSessionState(session_state, lambda : None)

@patch('streamlit.runtime.state.session_state_proxy.get_session_state', MagicMock(return_value=_create_mock_session_state({'foo': 'bar'})))
class SessionStateProxyTests(unittest.TestCase):
    reserved_key = f'{GENERATED_WIDGET_ID_PREFIX}-some_key'

    def setUp(self):
        if False:
            print('Hello World!')
        self.session_state_proxy = SessionStateProxy()

    def test_iter(self):
        if False:
            while True:
                i = 10
        state_iter = iter(self.session_state_proxy)
        assert next(state_iter) == 'foo'
        with pytest.raises(StopIteration):
            next(state_iter)

    def test_len(self):
        if False:
            while True:
                i = 10
        assert len(self.session_state_proxy) == 1

    def test_validate_key(self):
        if False:
            print('Hello World!')
        with pytest.raises(StreamlitAPIException) as e:
            require_valid_user_key(self.reserved_key)
        assert 'are reserved' in str(e.value)

    def test_to_dict(self):
        if False:
            while True:
                i = 10
        assert self.session_state_proxy.to_dict() == {'foo': 'bar'}

    def test_getitem_reserved_key(self):
        if False:
            return 10
        with pytest.raises(StreamlitAPIException):
            _ = self.session_state_proxy[self.reserved_key]

    def test_setitem_reserved_key(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(StreamlitAPIException):
            self.session_state_proxy[self.reserved_key] = 'foo'

    def test_delitem_reserved_key(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(StreamlitAPIException):
            del self.session_state_proxy[self.reserved_key]

    def test_getattr_reserved_key(self):
        if False:
            while True:
                i = 10
        with pytest.raises(StreamlitAPIException):
            getattr(self.session_state_proxy, self.reserved_key)

    def test_setattr_reserved_key(self):
        if False:
            while True:
                i = 10
        with pytest.raises(StreamlitAPIException):
            setattr(self.session_state_proxy, self.reserved_key, 'foo')

    def test_delattr_reserved_key(self):
        if False:
            while True:
                i = 10
        with pytest.raises(StreamlitAPIException):
            delattr(self.session_state_proxy, self.reserved_key)

class SessionStateProxyAttributeTests(unittest.TestCase):
    """Tests of SessionStateProxy attribute methods.

    Separate from the others to change patching. Test methods are individually
    patched to avoid issues with mutability.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.session_state_proxy = SessionStateProxy()

    @patch('streamlit.runtime.state.session_state_proxy.get_session_state', MagicMock(return_value=SessionState(_new_session_state={'foo': 'bar'})))
    def test_delattr(self):
        if False:
            while True:
                i = 10
        del self.session_state_proxy.foo
        assert 'foo' not in self.session_state_proxy

    @patch('streamlit.runtime.state.session_state_proxy.get_session_state', MagicMock(return_value=SessionState(_new_session_state={'foo': 'bar'})))
    def test_getattr(self):
        if False:
            return 10
        assert self.session_state_proxy.foo == 'bar'

    @patch('streamlit.runtime.state.session_state_proxy.get_session_state', MagicMock(return_value=SessionState(_new_session_state={'foo': 'bar'})))
    def test_getattr_error(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(AttributeError):
            del self.session_state_proxy.nonexistent

    @patch('streamlit.runtime.state.session_state_proxy.get_session_state', MagicMock(return_value=SessionState(_new_session_state={'foo': 'bar'})))
    def test_setattr(self):
        if False:
            return 10
        self.session_state_proxy.corge = 'grault2'
        assert self.session_state_proxy.corge == 'grault2'