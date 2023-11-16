from __future__ import annotations
import os
import unittest.mock
from datetime import datetime
import pytest
from airflow.models import DagBag
from airflow.models.dag import DAG
from airflow.models.expandinput import EXPAND_INPUT_EMPTY
from airflow.models.serialized_dag import SerializedDagModel
from airflow.operators.empty import EmptyOperator
from airflow.security import permissions
from tests.test_utils.api_connexion_utils import assert_401, create_user, delete_user
from tests.test_utils.db import clear_db_dags, clear_db_runs, clear_db_serialized_dags
pytestmark = pytest.mark.db_test

@pytest.fixture(scope='module')
def configured_app(minimal_app_for_api):
    if False:
        i = 10
        return i + 15
    app = minimal_app_for_api
    create_user(app, username='test', role_name='Test', permissions=[(permissions.ACTION_CAN_READ, permissions.RESOURCE_DAG), (permissions.ACTION_CAN_READ, permissions.RESOURCE_DAG_RUN), (permissions.ACTION_CAN_READ, permissions.RESOURCE_TASK_INSTANCE)])
    create_user(app, username='test_no_permissions', role_name='TestNoPermissions')
    yield app
    delete_user(app, username='test')
    delete_user(app, username='test_no_permissions')

class TestTaskEndpoint:
    dag_id = 'test_dag'
    mapped_dag_id = 'test_mapped_task'
    task_id = 'op1'
    task_id2 = 'op2'
    task_id3 = 'op3'
    mapped_task_id = 'mapped_task'
    task1_start_date = datetime(2020, 6, 15)
    task2_start_date = datetime(2020, 6, 16)

    @pytest.fixture(scope='class')
    def setup_dag(self, configured_app):
        if False:
            while True:
                i = 10
        with DAG(self.dag_id, start_date=self.task1_start_date, doc_md='details') as dag:
            task1 = EmptyOperator(task_id=self.task_id, params={'foo': 'bar'})
            task2 = EmptyOperator(task_id=self.task_id2, start_date=self.task2_start_date)
        with DAG(self.mapped_dag_id, start_date=self.task1_start_date) as mapped_dag:
            EmptyOperator(task_id=self.task_id3)
            EmptyOperator.partial(task_id=self.mapped_task_id)._expand(EXPAND_INPUT_EMPTY, strict=False)
        task1 >> task2
        dag_bag = DagBag(os.devnull, include_examples=False)
        dag_bag.dags = {dag.dag_id: dag, mapped_dag.dag_id: mapped_dag}
        configured_app.dag_bag = dag_bag

    @staticmethod
    def clean_db():
        if False:
            print('Hello World!')
        clear_db_runs()
        clear_db_dags()
        clear_db_serialized_dags()

    @pytest.fixture(autouse=True)
    def setup_attrs(self, configured_app, setup_dag) -> None:
        if False:
            return 10
        self.clean_db()
        self.app = configured_app
        self.client = self.app.test_client()

    def teardown_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.clean_db()

class TestGetTask(TestTaskEndpoint):

    def test_should_respond_200(self):
        if False:
            i = 10
            return i + 15
        expected = {'class_ref': {'class_name': 'EmptyOperator', 'module_path': 'airflow.operators.empty'}, 'depends_on_past': False, 'downstream_task_ids': [self.task_id2], 'end_date': None, 'execution_timeout': None, 'extra_links': [], 'operator_name': 'EmptyOperator', 'owner': 'airflow', 'params': {'foo': {'__class': 'airflow.models.param.Param', 'value': 'bar', 'description': None, 'schema': {}}}, 'pool': 'default_pool', 'pool_slots': 1.0, 'priority_weight': 1.0, 'queue': 'default', 'retries': 0.0, 'retry_delay': {'__type': 'TimeDelta', 'days': 0, 'seconds': 300, 'microseconds': 0}, 'retry_exponential_backoff': False, 'start_date': '2020-06-15T00:00:00+00:00', 'task_id': 'op1', 'template_fields': [], 'trigger_rule': 'all_success', 'ui_color': '#e8f7e4', 'ui_fgcolor': '#000', 'wait_for_downstream': False, 'weight_rule': 'downstream', 'is_mapped': False}
        response = self.client.get(f'/api/v1/dags/{self.dag_id}/tasks/{self.task_id}', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert response.json == expected

    def test_mapped_task(self):
        if False:
            print('Hello World!')
        expected = {'class_ref': {'class_name': 'EmptyOperator', 'module_path': 'airflow.operators.empty'}, 'depends_on_past': False, 'downstream_task_ids': [], 'end_date': None, 'execution_timeout': None, 'extra_links': [], 'is_mapped': True, 'operator_name': 'EmptyOperator', 'owner': 'airflow', 'params': {}, 'pool': 'default_pool', 'pool_slots': 1.0, 'priority_weight': 1.0, 'queue': 'default', 'retries': 0.0, 'retry_delay': {'__type': 'TimeDelta', 'days': 0, 'microseconds': 0, 'seconds': 300}, 'retry_exponential_backoff': False, 'start_date': '2020-06-15T00:00:00+00:00', 'task_id': 'mapped_task', 'template_fields': [], 'trigger_rule': 'all_success', 'ui_color': '#e8f7e4', 'ui_fgcolor': '#000', 'wait_for_downstream': False, 'weight_rule': 'downstream'}
        response = self.client.get(f'/api/v1/dags/{self.mapped_dag_id}/tasks/{self.mapped_task_id}', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert response.json == expected

    def test_should_respond_200_serialized(self):
        if False:
            i = 10
            return i + 15
        SerializedDagModel.write_dag(self.app.dag_bag.get_dag(self.dag_id))
        dag_bag = DagBag(os.devnull, include_examples=False, read_dags_from_db=True)
        patcher = unittest.mock.patch.object(self.app, 'dag_bag', dag_bag)
        patcher.start()
        expected = {'class_ref': {'class_name': 'EmptyOperator', 'module_path': 'airflow.operators.empty'}, 'depends_on_past': False, 'downstream_task_ids': [self.task_id2], 'end_date': None, 'execution_timeout': None, 'extra_links': [], 'operator_name': 'EmptyOperator', 'owner': 'airflow', 'params': {'foo': {'__class': 'airflow.models.param.Param', 'value': 'bar', 'description': None, 'schema': {}}}, 'pool': 'default_pool', 'pool_slots': 1.0, 'priority_weight': 1.0, 'queue': 'default', 'retries': 0.0, 'retry_delay': {'__type': 'TimeDelta', 'days': 0, 'seconds': 300, 'microseconds': 0}, 'retry_exponential_backoff': False, 'start_date': '2020-06-15T00:00:00+00:00', 'task_id': 'op1', 'template_fields': [], 'trigger_rule': 'all_success', 'ui_color': '#e8f7e4', 'ui_fgcolor': '#000', 'wait_for_downstream': False, 'weight_rule': 'downstream', 'is_mapped': False}
        response = self.client.get(f'/api/v1/dags/{self.dag_id}/tasks/{self.task_id}', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert response.json == expected
        patcher.stop()

    def test_should_respond_404(self):
        if False:
            return 10
        task_id = 'xxxx_not_existing'
        response = self.client.get(f'/api/v1/dags/{self.dag_id}/tasks/{task_id}', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 404

    def test_should_respond_404_when_dag_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        dag_id = 'xxxx_not_existing'
        response = self.client.get(f'/api/v1/dags/{dag_id}/tasks/{self.task_id}', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 404
        assert response.json['title'] == 'DAG not found'

    def test_should_raises_401_unauthenticated(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(f'/api/v1/dags/{self.dag_id}/tasks/{self.task_id}')
        assert_401(response)

    def test_should_raise_403_forbidden(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(f'/api/v1/dags/{self.dag_id}/tasks', environ_overrides={'REMOTE_USER': 'test_no_permissions'})
        assert response.status_code == 403

class TestGetTasks(TestTaskEndpoint):

    def test_should_respond_200(self):
        if False:
            for i in range(10):
                print('nop')
        expected = {'tasks': [{'class_ref': {'class_name': 'EmptyOperator', 'module_path': 'airflow.operators.empty'}, 'depends_on_past': False, 'downstream_task_ids': [self.task_id2], 'end_date': None, 'execution_timeout': None, 'extra_links': [], 'operator_name': 'EmptyOperator', 'owner': 'airflow', 'params': {'foo': {'__class': 'airflow.models.param.Param', 'value': 'bar', 'description': None, 'schema': {}}}, 'pool': 'default_pool', 'pool_slots': 1.0, 'priority_weight': 1.0, 'queue': 'default', 'retries': 0.0, 'retry_delay': {'__type': 'TimeDelta', 'days': 0, 'seconds': 300, 'microseconds': 0}, 'retry_exponential_backoff': False, 'start_date': '2020-06-15T00:00:00+00:00', 'task_id': 'op1', 'template_fields': [], 'trigger_rule': 'all_success', 'ui_color': '#e8f7e4', 'ui_fgcolor': '#000', 'wait_for_downstream': False, 'weight_rule': 'downstream', 'is_mapped': False}, {'class_ref': {'class_name': 'EmptyOperator', 'module_path': 'airflow.operators.empty'}, 'depends_on_past': False, 'downstream_task_ids': [], 'end_date': None, 'execution_timeout': None, 'extra_links': [], 'operator_name': 'EmptyOperator', 'owner': 'airflow', 'params': {}, 'pool': 'default_pool', 'pool_slots': 1.0, 'priority_weight': 1.0, 'queue': 'default', 'retries': 0.0, 'retry_delay': {'__type': 'TimeDelta', 'days': 0, 'seconds': 300, 'microseconds': 0}, 'retry_exponential_backoff': False, 'start_date': '2020-06-16T00:00:00+00:00', 'task_id': self.task_id2, 'template_fields': [], 'trigger_rule': 'all_success', 'ui_color': '#e8f7e4', 'ui_fgcolor': '#000', 'wait_for_downstream': False, 'weight_rule': 'downstream', 'is_mapped': False}], 'total_entries': 2}
        response = self.client.get(f'/api/v1/dags/{self.dag_id}/tasks', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert response.json == expected

    def test_get_tasks_mapped(self):
        if False:
            print('Hello World!')
        expected = {'tasks': [{'class_ref': {'class_name': 'EmptyOperator', 'module_path': 'airflow.operators.empty'}, 'depends_on_past': False, 'downstream_task_ids': [], 'end_date': None, 'execution_timeout': None, 'extra_links': [], 'is_mapped': True, 'operator_name': 'EmptyOperator', 'owner': 'airflow', 'params': {}, 'pool': 'default_pool', 'pool_slots': 1.0, 'priority_weight': 1.0, 'queue': 'default', 'retries': 0.0, 'retry_delay': {'__type': 'TimeDelta', 'days': 0, 'microseconds': 0, 'seconds': 300}, 'retry_exponential_backoff': False, 'start_date': '2020-06-15T00:00:00+00:00', 'task_id': 'mapped_task', 'template_fields': [], 'trigger_rule': 'all_success', 'ui_color': '#e8f7e4', 'ui_fgcolor': '#000', 'wait_for_downstream': False, 'weight_rule': 'downstream'}, {'class_ref': {'class_name': 'EmptyOperator', 'module_path': 'airflow.operators.empty'}, 'depends_on_past': False, 'downstream_task_ids': [], 'end_date': None, 'execution_timeout': None, 'extra_links': [], 'operator_name': 'EmptyOperator', 'owner': 'airflow', 'params': {}, 'pool': 'default_pool', 'pool_slots': 1.0, 'priority_weight': 1.0, 'queue': 'default', 'retries': 0.0, 'retry_delay': {'__type': 'TimeDelta', 'days': 0, 'seconds': 300, 'microseconds': 0}, 'retry_exponential_backoff': False, 'start_date': '2020-06-15T00:00:00+00:00', 'task_id': self.task_id3, 'template_fields': [], 'trigger_rule': 'all_success', 'ui_color': '#e8f7e4', 'ui_fgcolor': '#000', 'wait_for_downstream': False, 'weight_rule': 'downstream', 'is_mapped': False}], 'total_entries': 2}
        response = self.client.get(f'/api/v1/dags/{self.mapped_dag_id}/tasks', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert response.json == expected

    def test_should_respond_200_ascending_order_by_start_date(self):
        if False:
            i = 10
            return i + 15
        response = self.client.get(f'/api/v1/dags/{self.dag_id}/tasks?order_by=start_date', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert self.task1_start_date < self.task2_start_date
        assert response.json['tasks'][0]['task_id'] == self.task_id
        assert response.json['tasks'][1]['task_id'] == self.task_id2

    def test_should_respond_200_descending_order_by_start_date(self):
        if False:
            return 10
        response = self.client.get(f'/api/v1/dags/{self.dag_id}/tasks?order_by=-start_date', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert self.task1_start_date < self.task2_start_date
        assert response.json['tasks'][0]['task_id'] == self.task_id2
        assert response.json['tasks'][1]['task_id'] == self.task_id

    def test_should_raise_400_for_invalid_order_by_name(self):
        if False:
            return 10
        response = self.client.get(f'/api/v1/dags/{self.dag_id}/tasks?order_by=invalid_task_colume_name', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 400
        assert response.json['detail'] == "'EmptyOperator' object has no attribute 'invalid_task_colume_name'"

    def test_should_respond_404(self):
        if False:
            while True:
                i = 10
        dag_id = 'xxxx_not_existing'
        response = self.client.get(f'/api/v1/dags/{dag_id}/tasks', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 404

    def test_should_raises_401_unauthenticated(self):
        if False:
            while True:
                i = 10
        response = self.client.get(f'/api/v1/dags/{self.dag_id}/tasks')
        assert_401(response)