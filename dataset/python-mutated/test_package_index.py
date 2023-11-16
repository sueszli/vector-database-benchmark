"""Test for Package Index Hook."""
from __future__ import annotations
import pytest
from airflow.hooks.package_index import PackageIndexHook
from airflow.models.connection import Connection

class MockConnection(Connection):
    """Mock for the Connection class."""

    def __init__(self, host: str | None, login: str | None, password: str | None):
        if False:
            print('Hello World!')
        super().__init__()
        self.host = host
        self.login = login
        self.password = password
PI_MOCK_TESTDATA = {'missing-url': {}, 'anonymous-https': {'host': 'https://site/path', 'expected_result': 'https://site/path'}, 'no_password-http': {'host': 'http://site/path', 'login': 'any_user', 'expected_result': 'http://any_user@site/path'}, 'with_password-http': {'host': 'http://site/path', 'login': 'any_user', 'password': 'secret@_%1234!', 'expected_result': 'http://any_user:secret%40_%251234%21@site/path'}, 'with_password-https': {'host': 'https://old_user:pass@site/path', 'login': 'any_user', 'password': 'secret@_%1234!', 'expected_result': 'https://any_user:secret%40_%251234%21@site/path'}}

@pytest.fixture(params=list(PI_MOCK_TESTDATA.values()), ids=list(PI_MOCK_TESTDATA.keys()))
def mock_get_connection(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> str | None:
    if False:
        i = 10
        return i + 15
    'Pytest Fixture.'
    testdata: dict[str, str | None] = request.param
    host: str | None = testdata.get('host', None)
    login: str | None = testdata.get('login', None)
    password: str | None = testdata.get('password', None)
    expected_result: str | None = testdata.get('expected_result', None)
    monkeypatch.setattr('airflow.hooks.package_index.PackageIndexHook.get_connection', lambda *_: MockConnection(host, login, password))
    return expected_result

def test_get_connection_url(mock_get_connection: str | None):
    if False:
        print('Hello World!')
    'Test if connection url is assembled correctly from credentials and index_url.'
    expected_result = mock_get_connection
    hook_instance = PackageIndexHook()
    if expected_result:
        connection_url = hook_instance.get_connection_url()
        assert connection_url == expected_result
    else:
        with pytest.raises(Exception):
            hook_instance.get_connection_url()

@pytest.mark.parametrize('success', [0, 1])
def test_test_connection(monkeypatch: pytest.MonkeyPatch, mock_get_connection: str | None, success: int):
    if False:
        return 10
    'Test if connection test responds correctly to return code.'

    def mock_run(*_, **__):
        if False:
            return 10

        class MockProc:
            """Mock class."""
            returncode = success
            stderr = 'some error text'
        return MockProc()
    monkeypatch.setattr('airflow.hooks.package_index.subprocess.run', mock_run)
    hook_instance = PackageIndexHook()
    if mock_get_connection:
        result = hook_instance.test_connection()
        assert result[0] == (success == 0)
    else:
        with pytest.raises(Exception):
            hook_instance.test_connection()

def test_get_ui_field_behaviour():
    if False:
        while True:
            i = 10
    'Tests UI field result structure'
    ui_field_behavior = PackageIndexHook.get_ui_field_behaviour()
    assert 'hidden_fields' in ui_field_behavior
    assert 'relabeling' in ui_field_behavior
    assert 'placeholders' in ui_field_behavior