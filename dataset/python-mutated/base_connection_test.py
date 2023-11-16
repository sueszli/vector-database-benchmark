import os
import unittest
from unittest.mock import PropertyMock, mock_open, patch
import pytest
import streamlit as st
from streamlit.connections import BaseConnection
from streamlit.runtime.secrets import AttrDict
MOCK_TOML = '\n[connections.my_mock_connection]\nfoo="bar"\n'

class MockRawConnection:

    def some_raw_connection_method(self):
        if False:
            print('Hello World!')
        return 'some raw connection method'

class MockConnection(BaseConnection[str]):

    def _connect(self, **kwargs) -> str:
        if False:
            for i in range(10):
                print('nop')
        return MockRawConnection()

    def some_method(self):
        if False:
            while True:
                i = 10
        return 'some method'

class BaseConnectionDefaultMethodTests(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self._prev_environ = dict(os.environ)

    def tearDown(self) -> None:
        if False:
            return 10
        os.environ.clear()
        os.environ.update(self._prev_environ)
        st.secrets._reset()

    def test_instance_set_to_connect_return_value(self):
        if False:
            print('Hello World!')
        assert isinstance(MockConnection('my_mock_connection')._instance, MockRawConnection)

    def test_getattr_works_with_methods_on_connection(self):
        if False:
            print('Hello World!')
        assert MockConnection('my_mock_connection').some_method() == 'some method'

    def test_getattr_friendly_error_message(self):
        if False:
            print('Hello World!')
        with pytest.raises(AttributeError) as e:
            MockConnection('my_mock_connection').some_raw_connection_method()
        assert str(e.value) == "`some_raw_connection_method` doesn't exist here, but you can call `._instance.some_raw_connection_method` instead"
        assert MockConnection('my_mock_connection')._instance.some_raw_connection_method() == 'some raw connection method'

    def test_getattr_totally_nonexistent_attr(self):
        if False:
            print('Hello World!')
        with pytest.raises(AttributeError) as e:
            MockConnection('my_mock_connection').totally_nonexistent_method()
        assert str(e.value) == "'MockConnection' object has no attribute 'totally_nonexistent_method'"

    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_secrets_property(self, _):
        if False:
            i = 10
            return i + 15
        conn = MockConnection('my_mock_connection')
        assert conn._secrets.foo == 'bar'

    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_secrets_property_no_matching_section(self, _):
        if False:
            print('Hello World!')
        conn = MockConnection('nonexistent')
        assert conn._secrets == {}

    def test_secrets_property_no_secrets(self):
        if False:
            print('Hello World!')
        conn = MockConnection('my_mock_connection')
        assert conn._secrets == {}

    def test_instance_prop_caches_raw_instance(self):
        if False:
            print('Hello World!')
        conn = MockConnection('my_mock_connection')
        conn._raw_instance = 'some other value'
        assert conn._instance == 'some other value'

    def test_instance_prop_reinitializes_if_reset(self):
        if False:
            print('Hello World!')
        conn = MockConnection('my_mock_connection')
        conn._raw_instance = None
        assert isinstance(conn._instance, MockRawConnection)

    def test_on_secrets_changed_when_nothing_changed(self):
        if False:
            print('Hello World!')
        conn = MockConnection('my_mock_connection')
        with patch('streamlit.connections.base_connection.BaseConnection.reset') as patched_reset:
            conn._on_secrets_changed('unused_arg')
            patched_reset.assert_not_called()

    def test_on_secrets_changed(self):
        if False:
            for i in range(10):
                print('nop')
        conn = MockConnection('my_mock_connection')
        with patch('streamlit.connections.base_connection.BaseConnection.reset') as patched_reset, patch('streamlit.connections.base_connection.BaseConnection._secrets', PropertyMock(return_value=AttrDict({'mock_connection': {'new': 'secret'}}))):
            conn._on_secrets_changed('unused_arg')
            patched_reset.assert_called_once()

    def test_repr_html_(self):
        if False:
            for i in range(10):
                print('nop')
        repr_ = MockConnection('my_mock_connection')._repr_html_()
        assert 'st.connection my_mock_connection built from `tests.streamlit.connections.base_connection_test.MockConnection`' in repr_

    @patch('builtins.open', new_callable=mock_open, read_data=MOCK_TOML)
    def test_repr_html_with_secrets(self, _):
        if False:
            for i in range(10):
                print('nop')
        repr_ = MockConnection('my_mock_connection')._repr_html_()
        assert 'st.connection my_mock_connection built from `tests.streamlit.connections.base_connection_test.MockConnection`' in repr_
        assert 'Configured from `[connections.my_mock_connection]`' in repr_