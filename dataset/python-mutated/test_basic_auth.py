from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
from flask import Response
from flask_appbuilder.const import AUTH_LDAP
from airflow.api.auth.backend.basic_auth import requires_authentication
from airflow.www import app as application

@pytest.fixture
def app():
    if False:
        return 10
    return application.create_app(testing=True)

@pytest.fixture
def mock_sm():
    if False:
        print('Hello World!')
    return Mock()

@pytest.fixture
def mock_appbuilder(mock_sm):
    if False:
        i = 10
        return i + 15
    appbuilder = Mock()
    appbuilder.sm = mock_sm
    return appbuilder

@pytest.fixture
def mock_app(mock_appbuilder):
    if False:
        while True:
            i = 10
    app = Mock()
    app.appbuilder = mock_appbuilder
    return app

@pytest.fixture
def mock_authorization():
    if False:
        i = 10
        return i + 15
    authorization = Mock()
    authorization.username = 'username'
    authorization.password = 'password'
    return authorization
mock_call = Mock()

@requires_authentication
def function_decorated():
    if False:
        i = 10
        return i + 15
    mock_call()

@pytest.mark.db_test
class TestBasicAuth:

    def setup_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        mock_call.reset_mock()

    def test_requires_authentication_with_no_header(self, app):
        if False:
            for i in range(10):
                print('nop')
        with app.test_request_context() as mock_context:
            mock_context.request.authorization = None
            result = function_decorated()
        assert type(result) is Response
        assert result.status_code == 401

    @patch('airflow.auth.managers.fab.api.auth.backend.basic_auth.get_airflow_app')
    @patch('airflow.auth.managers.fab.api.auth.backend.basic_auth.login_user')
    def test_requires_authentication_with_ldap(self, mock_login_user, mock_get_airflow_app, app, mock_app, mock_sm, mock_authorization):
        if False:
            print('Hello World!')
        mock_get_airflow_app.return_value = mock_app
        mock_sm.auth_type = AUTH_LDAP
        user = Mock()
        mock_sm.auth_user_ldap.return_value = user
        with app.test_request_context() as mock_context:
            mock_context.request.authorization = mock_authorization
            function_decorated()
        mock_sm.auth_user_ldap.assert_called_once_with(mock_authorization.username, mock_authorization.password)
        mock_login_user.assert_called_once_with(user, remember=False)
        mock_call.assert_called_once()

    @patch('airflow.auth.managers.fab.api.auth.backend.basic_auth.get_airflow_app')
    @patch('airflow.auth.managers.fab.api.auth.backend.basic_auth.login_user')
    def test_requires_authentication_with_db(self, mock_login_user, mock_get_airflow_app, app, mock_app, mock_sm, mock_authorization):
        if False:
            for i in range(10):
                print('nop')
        mock_get_airflow_app.return_value = mock_app
        user = Mock()
        mock_sm.auth_user_db.return_value = user
        with app.test_request_context() as mock_context:
            mock_context.request.authorization = mock_authorization
            function_decorated()
        mock_sm.auth_user_db.assert_called_once_with(mock_authorization.username, mock_authorization.password)
        mock_login_user.assert_called_once_with(user, remember=False)
        mock_call.assert_called_once()