from __future__ import annotations
import ast
import os
from typing import TYPE_CHECKING
import pytest
from airflow.models import DagBag
from airflow.security import permissions
from tests.test_utils.api_connexion_utils import assert_401, create_user, delete_user
from tests.test_utils.db import clear_db_dag_code, clear_db_dags, clear_db_serialized_dags
pytestmark = pytest.mark.db_test
if TYPE_CHECKING:
    from airflow.models.dag import DAG
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
EXAMPLE_DAG_FILE = os.path.join('airflow', 'example_dags', 'example_bash_operator.py')

@pytest.fixture(scope='module')
def configured_app(minimal_app_for_api):
    if False:
        while True:
            i = 10
    app = minimal_app_for_api
    create_user(app, username='test', role_name='Test', permissions=[(permissions.ACTION_CAN_READ, permissions.RESOURCE_DAG_CODE)])
    create_user(app, username='test_no_permissions', role_name='TestNoPermissions')
    yield app
    delete_user(app, username='test')
    delete_user(app, username='test_no_permissions')

class TestGetSource:

    @pytest.fixture(autouse=True)
    def setup_attrs(self, configured_app) -> None:
        if False:
            while True:
                i = 10
        self.app = configured_app
        self.client = self.app.test_client()
        self.clear_db()

    def teardown_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.clear_db()

    @staticmethod
    def clear_db():
        if False:
            for i in range(10):
                print('nop')
        clear_db_dags()
        clear_db_serialized_dags()
        clear_db_dag_code()

    @staticmethod
    def _get_dag_file_docstring(fileloc: str) -> str | None:
        if False:
            return 10
        with open(fileloc) as f:
            file_contents = f.read()
        module = ast.parse(file_contents)
        docstring = ast.get_docstring(module)
        return docstring

    def test_should_respond_200_text(self, url_safe_serializer):
        if False:
            print('Hello World!')
        dagbag = DagBag(dag_folder=EXAMPLE_DAG_FILE)
        dagbag.sync_to_db()
        first_dag: DAG = next(iter(dagbag.dags.values()))
        dag_docstring = self._get_dag_file_docstring(first_dag.fileloc)
        url = f'/api/v1/dagSources/{url_safe_serializer.dumps(first_dag.fileloc)}'
        response = self.client.get(url, headers={'Accept': 'text/plain'}, environ_overrides={'REMOTE_USER': 'test'})
        assert 200 == response.status_code
        assert dag_docstring in response.data.decode()
        assert 'text/plain' == response.headers['Content-Type']

    def test_should_respond_200_json(self, url_safe_serializer):
        if False:
            for i in range(10):
                print('nop')
        dagbag = DagBag(dag_folder=EXAMPLE_DAG_FILE)
        dagbag.sync_to_db()
        first_dag: DAG = next(iter(dagbag.dags.values()))
        dag_docstring = self._get_dag_file_docstring(first_dag.fileloc)
        url = f'/api/v1/dagSources/{url_safe_serializer.dumps(first_dag.fileloc)}'
        response = self.client.get(url, headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test'})
        assert 200 == response.status_code
        assert dag_docstring in response.json['content']
        assert 'application/json' == response.headers['Content-Type']

    def test_should_respond_406(self, url_safe_serializer):
        if False:
            return 10
        dagbag = DagBag(dag_folder=EXAMPLE_DAG_FILE)
        dagbag.sync_to_db()
        first_dag: DAG = next(iter(dagbag.dags.values()))
        url = f'/api/v1/dagSources/{url_safe_serializer.dumps(first_dag.fileloc)}'
        response = self.client.get(url, headers={'Accept': 'image/webp'}, environ_overrides={'REMOTE_USER': 'test'})
        assert 406 == response.status_code

    def test_should_respond_404(self):
        if False:
            print('Hello World!')
        wrong_fileloc = 'abcd1234'
        url = f'/api/v1/dagSources/{wrong_fileloc}'
        response = self.client.get(url, headers={'Accept': 'application/json'}, environ_overrides={'REMOTE_USER': 'test'})
        assert 404 == response.status_code

    def test_should_raises_401_unauthenticated(self, url_safe_serializer):
        if False:
            i = 10
            return i + 15
        dagbag = DagBag(dag_folder=EXAMPLE_DAG_FILE)
        dagbag.sync_to_db()
        first_dag: DAG = next(iter(dagbag.dags.values()))
        response = self.client.get(f'/api/v1/dagSources/{url_safe_serializer.dumps(first_dag.fileloc)}', headers={'Accept': 'text/plain'})
        assert_401(response)

    def test_should_raise_403_forbidden(self, url_safe_serializer):
        if False:
            print('Hello World!')
        dagbag = DagBag(dag_folder=EXAMPLE_DAG_FILE)
        dagbag.sync_to_db()
        first_dag: DAG = next(iter(dagbag.dags.values()))
        response = self.client.get(f'/api/v1/dagSources/{url_safe_serializer.dumps(first_dag.fileloc)}', headers={'Accept': 'text/plain'}, environ_overrides={'REMOTE_USER': 'test_no_permissions'})
        assert response.status_code == 403