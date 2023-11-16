from __future__ import annotations
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock
import pytest
from flask import Flask
from airflow.auth.managers.base_auth_manager import BaseAuthManager, ResourceMethod
from airflow.exceptions import AirflowException
from airflow.security import permissions
from airflow.www.extensions.init_appbuilder import init_appbuilder
from airflow.www.security_manager import AirflowSecurityManagerV2
if TYPE_CHECKING:
    from airflow.auth.managers.models.base_user import BaseUser
    from airflow.auth.managers.models.resource_details import AccessView, ConfigurationDetails, ConnectionDetails, DagAccessEntity, DagDetails, DatasetDetails, PoolDetails, VariableDetails

class EmptyAuthManager(BaseAuthManager):

    def get_user_display_name(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def get_user_name(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def get_user(self) -> BaseUser:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def get_user_id(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def is_authorized_configuration(self, *, method: ResourceMethod, details: ConfigurationDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            return 10
        raise NotImplementedError()

    def is_authorized_cluster_activity(self, *, method: ResourceMethod, user: BaseUser | None=None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def is_authorized_connection(self, *, method: ResourceMethod, details: ConnectionDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def is_authorized_dag(self, *, method: ResourceMethod, access_entity: DagAccessEntity | None=None, details: DagDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def is_authorized_dataset(self, *, method: ResourceMethod, details: DatasetDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def is_authorized_pool(self, *, method: ResourceMethod, details: PoolDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            return 10
        raise NotImplementedError()

    def is_authorized_variable(self, *, method: ResourceMethod, details: VariableDetails | None=None, user: BaseUser | None=None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def is_authorized_view(self, *, access_view: AccessView, user: BaseUser | None=None) -> bool:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def is_logged_in(self) -> bool:
        if False:
            return 10
        raise NotImplementedError()

    def get_url_login(self, **kwargs) -> str:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def get_url_logout(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def get_url_user_profile(self) -> str | None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

@pytest.fixture
def auth_manager():
    if False:
        while True:
            i = 10
    return EmptyAuthManager(None, None)

@pytest.fixture
def auth_manager_with_appbuilder():
    if False:
        return 10
    flask_app = Flask(__name__)
    appbuilder = init_appbuilder(flask_app)
    return EmptyAuthManager(flask_app, appbuilder)

class TestBaseAuthManager:

    def test_get_cli_commands_return_empty_list(self, auth_manager):
        if False:
            print('Hello World!')
        assert auth_manager.get_cli_commands() == []

    def test_get_api_endpoints_return_none(self, auth_manager):
        if False:
            i = 10
            return i + 15
        assert auth_manager.get_api_endpoints() is None

    def test_is_authorized_custom_view_throws_exception(self, auth_manager):
        if False:
            return 10
        with pytest.raises(AirflowException, match='The resource `.*` does not exist in the environment.'):
            auth_manager.is_authorized_custom_view(fab_action_name=permissions.ACTION_CAN_READ, fab_resource_name=permissions.RESOURCE_MY_PASSWORD)

    @pytest.mark.db_test
    def test_security_manager_return_default_security_manager(self, auth_manager_with_appbuilder):
        if False:
            i = 10
            return i + 15
        assert isinstance(auth_manager_with_appbuilder.security_manager, AirflowSecurityManagerV2)

    @pytest.mark.parametrize('access_all, access_per_dag, dag_ids, expected', [(True, {}, ['dag1', 'dag2'], {'dag1', 'dag2'}), (False, {}, ['dag1', 'dag2'], set()), (False, {'dag1': True}, ['dag1', 'dag2'], {'dag1'})])
    def test_get_permitted_dag_ids(self, auth_manager, access_all: bool, access_per_dag: dict, dag_ids: list, expected: set):
        if False:
            for i in range(10):
                print('nop')

        def side_effect_func(*, method: ResourceMethod, access_entity: DagAccessEntity | None=None, details: DagDetails | None=None, user: BaseUser | None=None):
            if False:
                i = 10
                return i + 15
            if not details:
                return access_all
            else:
                return access_per_dag.get(details.id, False)
        auth_manager.is_authorized_dag = MagicMock(side_effect=side_effect_func)
        user = Mock()
        session = Mock()
        dags = []
        for dag_id in dag_ids:
            mock = Mock()
            mock.dag_id = dag_id
            dags.append(mock)
        session.execute.return_value = dags
        result = auth_manager.get_permitted_dag_ids(user=user, session=session)
        assert result == expected