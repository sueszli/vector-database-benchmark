from __future__ import annotations
from unittest.mock import ANY
import pytest
from airflow.models.dag import DagModel
from airflow.models.dagwarning import DagWarning
from airflow.security import permissions
from airflow.utils.session import create_session
from tests.test_utils.api_connexion_utils import assert_401, create_user, delete_user
from tests.test_utils.db import clear_db_dag_warnings, clear_db_dags
pytestmark = pytest.mark.db_test

@pytest.fixture(scope='module')
def configured_app(minimal_app_for_api):
    if False:
        while True:
            i = 10
    app = minimal_app_for_api
    create_user(app, username='test', role_name='Test', permissions=[(permissions.ACTION_CAN_READ, permissions.RESOURCE_DAG_WARNING), (permissions.ACTION_CAN_READ, permissions.RESOURCE_DAG)])
    create_user(app, username='test_no_permissions', role_name='TestNoPermissions')
    create_user(app, username='test_with_dag2_read', role_name='TestWithDag2Read', permissions=[(permissions.ACTION_CAN_READ, permissions.RESOURCE_DAG_WARNING), (permissions.ACTION_CAN_READ, f'{permissions.RESOURCE_DAG_PREFIX}dag2')])
    yield minimal_app_for_api
    delete_user(app, username='test')
    delete_user(app, username='test_no_permissions')
    delete_user(app, username='test_with_dag2_read')

class TestBaseDagWarning:
    timestamp = '2020-06-10T12:00'

    @pytest.fixture(autouse=True)
    def setup_attrs(self, configured_app) -> None:
        if False:
            return 10
        self.app = configured_app
        self.client = self.app.test_client()

    def teardown_method(self) -> None:
        if False:
            i = 10
            return i + 15
        clear_db_dag_warnings()
        clear_db_dags()

    @staticmethod
    def _normalize_dag_warnings(dag_warnings):
        if False:
            return 10
        for (i, dag_warning) in enumerate(dag_warnings, 1):
            dag_warning['dag_warning_id'] = i

class TestGetDagWarningEndpoint(TestBaseDagWarning):

    def setup_class(self):
        if False:
            print('Hello World!')
        clear_db_dag_warnings()
        clear_db_dags()

    def setup_method(self):
        if False:
            while True:
                i = 10
        with create_session() as session:
            session.add(DagModel(dag_id='dag1'))
            session.add(DagModel(dag_id='dag2'))
            session.add(DagModel(dag_id='dag3'))
            session.add(DagWarning('dag1', 'non-existent pool', 'test message'))
            session.add(DagWarning('dag2', 'non-existent pool', 'test message'))
            session.commit()

    def test_response_one(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get('/api/v1/dagWarnings', environ_overrides={'REMOTE_USER': 'test'}, query_string={'dag_id': 'dag1', 'warning_type': 'non-existent pool'})
        assert response.status_code == 200
        response_data = response.json
        assert response_data == {'dag_warnings': [{'dag_id': 'dag1', 'message': 'test message', 'timestamp': ANY, 'warning_type': 'non-existent pool'}], 'total_entries': 1}

    def test_response_some(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get('/api/v1/dagWarnings', environ_overrides={'REMOTE_USER': 'test'}, query_string={'warning_type': 'non-existent pool'})
        assert response.status_code == 200
        response_data = response.json
        assert len(response_data['dag_warnings']) == 2
        assert response_data == {'dag_warnings': ANY, 'total_entries': 2}

    def test_response_none(self, session):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get('/api/v1/dagWarnings', environ_overrides={'REMOTE_USER': 'test'}, query_string={'dag_id': 'missing_dag'})
        assert response.status_code == 200
        response_data = response.json
        assert response_data == {'dag_warnings': [], 'total_entries': 0}

    def test_response_all(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get('/api/v1/dagWarnings', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        response_data = response.json
        assert len(response_data['dag_warnings']) == 2
        assert response_data == {'dag_warnings': ANY, 'total_entries': 2}

    def test_should_raises_401_unauthenticated(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get('/api/v1/dagWarnings')
        assert_401(response)

    def test_should_raise_403_forbidden(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get('/api/v1/dagWarnings', environ_overrides={'REMOTE_USER': 'test_no_permissions'})
        assert response.status_code == 403

    def test_should_raise_403_forbidden_when_user_has_no_dag_read_permission(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get('/api/v1/dagWarnings', environ_overrides={'REMOTE_USER': 'test_with_dag2_read'}, query_string={'dag_id': 'dag1'})
        assert response.status_code == 403