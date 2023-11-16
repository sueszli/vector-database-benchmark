from __future__ import annotations
from itertools import chain
from unittest import mock
from unittest.mock import Mock
import pytest
from flask import Flask
from airflow.auth.managers.fab.fab_auth_manager import FabAuthManager
from airflow.auth.managers.fab.models import User
from airflow.auth.managers.fab.security_manager.override import FabAirflowSecurityManagerOverride
from airflow.auth.managers.models.resource_details import AccessView, DagAccessEntity, DagDetails
from airflow.exceptions import AirflowException
from airflow.security.permissions import ACTION_CAN_ACCESS_MENU, ACTION_CAN_CREATE, ACTION_CAN_DELETE, ACTION_CAN_EDIT, ACTION_CAN_READ, RESOURCE_CLUSTER_ACTIVITY, RESOURCE_CONFIG, RESOURCE_CONNECTION, RESOURCE_DAG, RESOURCE_DAG_RUN, RESOURCE_DATASET, RESOURCE_JOB, RESOURCE_PLUGIN, RESOURCE_PROVIDER, RESOURCE_TASK_INSTANCE, RESOURCE_TRIGGER, RESOURCE_VARIABLE, RESOURCE_WEBSITE
from airflow.www.extensions.init_appbuilder import init_appbuilder
IS_AUTHORIZED_METHODS_SIMPLE = {'is_authorized_configuration': RESOURCE_CONFIG, 'is_authorized_cluster_activity': RESOURCE_CLUSTER_ACTIVITY, 'is_authorized_connection': RESOURCE_CONNECTION, 'is_authorized_dataset': RESOURCE_DATASET, 'is_authorized_variable': RESOURCE_VARIABLE}

@pytest.fixture
def auth_manager():
    if False:
        for i in range(10):
            print('nop')
    return FabAuthManager(None, None)

@pytest.fixture
def auth_manager_with_appbuilder():
    if False:
        i = 10
        return i + 15
    flask_app = Flask(__name__)
    appbuilder = init_appbuilder(flask_app)
    return FabAuthManager(flask_app, appbuilder)

@pytest.mark.db_test
class TestFabAuthManager:

    @pytest.mark.parametrize('id,first_name,last_name,username,email,expected', [(1, 'First', 'Last', None, None, '1'), (1, None, None, None, None, '1'), (1, 'First', 'Last', 'user', None, 'user'), (1, 'First', 'Last', 'user', 'email', 'user'), (1, None, None, None, 'email', 'email'), (1, 'First', 'Last', None, 'email', 'email')])
    @mock.patch.object(FabAuthManager, 'get_user')
    def test_get_user_name(self, mock_get_user, id, first_name, last_name, username, email, expected, auth_manager):
        if False:
            while True:
                i = 10
        user = User()
        user.id = id
        user.first_name = first_name
        user.last_name = last_name
        user.username = username
        user.email = email
        mock_get_user.return_value = user
        assert auth_manager.get_user_name() == expected

    @pytest.mark.parametrize('id,first_name,last_name,username,email,expected', [(1, 'First', 'Last', None, None, 'First Last'), (1, 'First', None, 'user', None, 'First'), (1, None, 'Last', 'user', 'email', 'Last'), (1, None, None, None, 'email', ''), (1, None, None, None, 'email', '')])
    @mock.patch.object(FabAuthManager, 'get_user')
    def test_get_user_display_name(self, mock_get_user, id, first_name, last_name, username, email, expected, auth_manager):
        if False:
            for i in range(10):
                print('nop')
        user = User()
        user.id = id
        user.first_name = first_name
        user.last_name = last_name
        user.username = username
        user.email = email
        mock_get_user.return_value = user
        assert auth_manager.get_user_display_name() == expected

    @mock.patch('flask_login.utils._get_user')
    def test_get_user(self, mock_current_user, auth_manager):
        if False:
            return 10
        user = Mock()
        user.is_anonymous.return_value = True
        mock_current_user.return_value = user
        assert auth_manager.get_user() == user

    @mock.patch.object(FabAuthManager, 'get_user')
    def test_get_user_id(self, mock_get_user, auth_manager):
        if False:
            i = 10
            return i + 15
        user_id = 'test'
        user = Mock()
        user.get_id.return_value = user_id
        mock_get_user.return_value = user
        assert auth_manager.get_user_id() == user_id

    @mock.patch.object(FabAuthManager, 'get_user')
    def test_is_logged_in(self, mock_get_user, auth_manager):
        if False:
            for i in range(10):
                print('nop')
        user = Mock()
        user.is_anonymous.return_value = True
        mock_get_user.return_value = user
        assert auth_manager.is_logged_in() is False

    @pytest.mark.parametrize('api_name, method, user_permissions, expected_result', chain(*[((api_name, 'POST', [(ACTION_CAN_CREATE, resource_type)], True), (api_name, 'GET', [(ACTION_CAN_READ, resource_type)], True), (api_name, 'DELETE', [(ACTION_CAN_DELETE, resource_type), (ACTION_CAN_CREATE, 'resource_test')], True), (api_name, 'GET', [(ACTION_CAN_ACCESS_MENU, resource_type)], True), (api_name, 'POST', [(ACTION_CAN_READ, resource_type), (ACTION_CAN_CREATE, 'resource_test')], False)) for (api_name, resource_type) in IS_AUTHORIZED_METHODS_SIMPLE.items()]))
    def test_is_authorized(self, api_name, method, user_permissions, expected_result, auth_manager):
        if False:
            return 10
        user = Mock()
        user.perms = user_permissions
        result = getattr(auth_manager, api_name)(method=method, user=user)
        assert result == expected_result

    @pytest.mark.parametrize('method, dag_access_entity, dag_details, user_permissions, expected_result', [('GET', None, None, [(ACTION_CAN_READ, RESOURCE_DAG)], True), ('GET', None, DagDetails(id='test_dag_id'), [(ACTION_CAN_READ, RESOURCE_DAG)], True), ('GET', None, DagDetails(id='test_dag_id'), [(ACTION_CAN_READ, 'DAG:test_dag_id'), (ACTION_CAN_READ, 'DAG:test_dag_id2')], True), ('POST', None, DagDetails(id='test_dag_id'), [(ACTION_CAN_READ, 'DAG:test_dag_id')], False), ('GET', None, DagDetails(id='test_dag_id2'), [(ACTION_CAN_READ, 'DAG:test_dag_id')], False), ('GET', None, None, [(ACTION_CAN_READ, 'resource_test')], False), ('GET', DagAccessEntity.RUN, DagDetails(id='test_dag_id'), [(ACTION_CAN_READ, RESOURCE_DAG), (ACTION_CAN_READ, RESOURCE_DAG_RUN)], True), ('GET', DagAccessEntity.TASK_INSTANCE, DagDetails(id='test_dag_id'), [(ACTION_CAN_READ, RESOURCE_TASK_INSTANCE)], False), ('GET', DagAccessEntity.TASK_INSTANCE, DagDetails(id='test_dag_id'), [(ACTION_CAN_READ, 'DAG:test_dag_id'), (ACTION_CAN_READ, RESOURCE_TASK_INSTANCE)], False), ('GET', DagAccessEntity.TASK_INSTANCE, DagDetails(id='test_dag_id'), [(ACTION_CAN_READ, 'DAG:test_dag_id'), (ACTION_CAN_READ, RESOURCE_TASK_INSTANCE), (ACTION_CAN_READ, RESOURCE_DAG_RUN)], True), ('DELETE', DagAccessEntity.TASK, DagDetails(id='test_dag_id'), [(ACTION_CAN_EDIT, 'DAG:test_dag_id'), (ACTION_CAN_DELETE, RESOURCE_TASK_INSTANCE)], True), ('POST', DagAccessEntity.RUN, DagDetails(id='test_dag_id'), [(ACTION_CAN_EDIT, 'DAG:test_dag_id'), (ACTION_CAN_CREATE, RESOURCE_DAG_RUN)], True), ('POST', DagAccessEntity.RUN, DagDetails(id='test_dag_id'), [(ACTION_CAN_CREATE, RESOURCE_DAG_RUN)], False), ('GET', DagAccessEntity.TASK_LOGS, DagDetails(id='test_dag_id'), [(ACTION_CAN_READ, RESOURCE_TASK_INSTANCE)], False)])
    def test_is_authorized_dag(self, method, dag_access_entity, dag_details, user_permissions, expected_result, auth_manager):
        if False:
            while True:
                i = 10
        user = Mock()
        user.perms = user_permissions
        result = auth_manager.is_authorized_dag(method=method, access_entity=dag_access_entity, details=dag_details, user=user)
        assert result == expected_result

    @pytest.mark.parametrize('access_view, user_permissions, expected_result', [(AccessView.JOBS, [(ACTION_CAN_READ, RESOURCE_JOB)], True), (AccessView.PLUGINS, [(ACTION_CAN_READ, RESOURCE_PLUGIN)], True), (AccessView.PROVIDERS, [(ACTION_CAN_READ, RESOURCE_PROVIDER)], True), (AccessView.TRIGGERS, [(ACTION_CAN_READ, RESOURCE_TRIGGER)], True), (AccessView.WEBSITE, [(ACTION_CAN_READ, RESOURCE_WEBSITE)], True), (AccessView.WEBSITE, [(ACTION_CAN_READ, 'resource_test'), (ACTION_CAN_CREATE, RESOURCE_WEBSITE)], False), (AccessView.WEBSITE, [(ACTION_CAN_READ, RESOURCE_TRIGGER)], False)])
    def test_is_authorized_view(self, access_view, user_permissions, expected_result, auth_manager):
        if False:
            print('Hello World!')
        user = Mock()
        user.perms = user_permissions
        result = auth_manager.is_authorized_view(access_view=access_view, user=user)
        assert result == expected_result

    @pytest.mark.db_test
    def test_security_manager_return_fab_security_manager_override(self, auth_manager_with_appbuilder):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(auth_manager_with_appbuilder.security_manager, FabAirflowSecurityManagerOverride)

    @pytest.mark.db_test
    def test_get_url_login_when_auth_view_not_defined(self, auth_manager_with_appbuilder):
        if False:
            print('Hello World!')
        with pytest.raises(AirflowException, match='`auth_view` not defined in the security manager.'):
            auth_manager_with_appbuilder.get_url_login()

    @pytest.mark.db_test
    @mock.patch('airflow.auth.managers.fab.fab_auth_manager.url_for')
    def test_get_url_login(self, mock_url_for, auth_manager_with_appbuilder):
        if False:
            i = 10
            return i + 15
        auth_manager_with_appbuilder.security_manager.auth_view = Mock()
        auth_manager_with_appbuilder.security_manager.auth_view.endpoint = 'test_endpoint'
        auth_manager_with_appbuilder.get_url_login()
        mock_url_for.assert_called_once_with('test_endpoint.login')

    @pytest.mark.db_test
    @mock.patch('airflow.auth.managers.fab.fab_auth_manager.url_for')
    def test_get_url_login_with_next(self, mock_url_for, auth_manager_with_appbuilder):
        if False:
            print('Hello World!')
        auth_manager_with_appbuilder.security_manager.auth_view = Mock()
        auth_manager_with_appbuilder.security_manager.auth_view.endpoint = 'test_endpoint'
        auth_manager_with_appbuilder.get_url_login(next_url='next_url')
        mock_url_for.assert_called_once_with('test_endpoint.login', next='next_url')

    @pytest.mark.db_test
    def test_get_url_logout_when_auth_view_not_defined(self, auth_manager_with_appbuilder):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(AirflowException, match='`auth_view` not defined in the security manager.'):
            auth_manager_with_appbuilder.get_url_logout()

    @pytest.mark.db_test
    @mock.patch('airflow.auth.managers.fab.fab_auth_manager.url_for')
    def test_get_url_logout(self, mock_url_for, auth_manager_with_appbuilder):
        if False:
            for i in range(10):
                print('nop')
        auth_manager_with_appbuilder.security_manager.auth_view = Mock()
        auth_manager_with_appbuilder.security_manager.auth_view.endpoint = 'test_endpoint'
        auth_manager_with_appbuilder.get_url_logout()
        mock_url_for.assert_called_once_with('test_endpoint.logout')

    @pytest.mark.db_test
    def test_get_url_user_profile_when_auth_view_not_defined(self, auth_manager_with_appbuilder):
        if False:
            while True:
                i = 10
        assert auth_manager_with_appbuilder.get_url_user_profile() is None

    @pytest.mark.db_test
    @mock.patch('airflow.auth.managers.fab.fab_auth_manager.url_for')
    def test_get_url_user_profile(self, mock_url_for, auth_manager_with_appbuilder):
        if False:
            i = 10
            return i + 15
        expected_url = 'test_url'
        mock_url_for.return_value = expected_url
        auth_manager_with_appbuilder.security_manager.user_view = Mock()
        auth_manager_with_appbuilder.security_manager.user_view.endpoint = 'test_endpoint'
        actual_url = auth_manager_with_appbuilder.get_url_user_profile()
        mock_url_for.assert_called_once_with('test_endpoint.userinfo')
        assert actual_url == expected_url