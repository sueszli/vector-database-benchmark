from __future__ import annotations
import pytest
from airflow.api_connexion.schemas.role_and_permission_schema import RoleCollection, role_collection_schema, role_schema
from airflow.security import permissions
from tests.test_utils.api_connexion_utils import create_role, delete_role
pytestmark = pytest.mark.db_test

class TestRoleCollectionItemSchema:

    @pytest.fixture(scope='class')
    def role(self, minimal_app_for_api):
        if False:
            for i in range(10):
                print('nop')
        yield create_role(minimal_app_for_api, name='Test', permissions=[(permissions.ACTION_CAN_CREATE, permissions.RESOURCE_CONNECTION)])
        delete_role(minimal_app_for_api, 'Test')

    @pytest.fixture(autouse=True)
    def _set_attrs(self, minimal_app_for_api, role):
        if False:
            return 10
        self.app = minimal_app_for_api
        self.role = role

    def test_serialize(self):
        if False:
            return 10
        deserialized_role = role_schema.dump(self.role)
        assert deserialized_role == {'name': 'Test', 'actions': [{'resource': {'name': 'Connections'}, 'action': {'name': 'can_create'}}]}

    def test_deserialize(self):
        if False:
            print('Hello World!')
        role = {'name': 'Test', 'actions': [{'resource': {'name': 'Connections'}, 'action': {'name': 'can_create'}}]}
        role_obj = role_schema.load(role)
        assert role_obj == {'name': 'Test', 'permissions': [{'resource': {'name': 'Connections'}, 'action': {'name': 'can_create'}}]}

class TestRoleCollectionSchema:

    @pytest.fixture(scope='class')
    def role1(self, minimal_app_for_api):
        if False:
            for i in range(10):
                print('nop')
        yield create_role(minimal_app_for_api, name='Test1', permissions=[(permissions.ACTION_CAN_CREATE, permissions.RESOURCE_CONNECTION)])
        delete_role(minimal_app_for_api, 'Test1')

    @pytest.fixture(scope='class')
    def role2(self, minimal_app_for_api):
        if False:
            return 10
        yield create_role(minimal_app_for_api, name='Test2', permissions=[(permissions.ACTION_CAN_EDIT, permissions.RESOURCE_DAG)])
        delete_role(minimal_app_for_api, 'Test2')

    def test_serialize(self, role1, role2):
        if False:
            return 10
        instance = RoleCollection([role1, role2], total_entries=2)
        deserialized = role_collection_schema.dump(instance)
        assert deserialized == {'roles': [{'name': 'Test1', 'actions': [{'resource': {'name': 'Connections'}, 'action': {'name': 'can_create'}}]}, {'name': 'Test2', 'actions': [{'resource': {'name': 'DAGs'}, 'action': {'name': 'can_edit'}}]}], 'total_entries': 2}