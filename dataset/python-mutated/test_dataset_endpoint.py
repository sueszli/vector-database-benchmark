from __future__ import annotations
import urllib
import pytest
from airflow.api_connexion.exceptions import EXCEPTIONS_LINK_MAP
from airflow.models.dagrun import DagRun
from airflow.models.dataset import DatasetEvent, DatasetModel
from airflow.security import permissions
from airflow.utils import timezone
from airflow.utils.session import provide_session
from airflow.utils.types import DagRunType
from tests.test_utils.api_connexion_utils import assert_401, create_user, delete_user
from tests.test_utils.asserts import assert_queries_count
from tests.test_utils.config import conf_vars
from tests.test_utils.db import clear_db_datasets, clear_db_runs
pytestmark = pytest.mark.db_test

@pytest.fixture(scope='module')
def configured_app(minimal_app_for_api):
    if False:
        return 10
    app = minimal_app_for_api
    create_user(app, username='test', role_name='Test', permissions=[(permissions.ACTION_CAN_READ, permissions.RESOURCE_DATASET)])
    create_user(app, username='test_no_permissions', role_name='TestNoPermissions')
    yield app
    delete_user(app, username='test')
    delete_user(app, username='test_no_permissions')

class TestDatasetEndpoint:
    default_time = '2020-06-11T18:00:00+00:00'

    @pytest.fixture(autouse=True)
    def setup_attrs(self, configured_app) -> None:
        if False:
            i = 10
            return i + 15
        self.app = configured_app
        self.client = self.app.test_client()
        clear_db_datasets()
        clear_db_runs()

    def teardown_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        clear_db_datasets()
        clear_db_runs()

    def _create_dataset(self, session):
        if False:
            return 10
        dataset_model = DatasetModel(id=1, uri='s3://bucket/key', extra={'foo': 'bar'}, created_at=timezone.parse(self.default_time), updated_at=timezone.parse(self.default_time))
        session.add(dataset_model)
        session.commit()
        return dataset_model

class TestGetDatasetEndpoint(TestDatasetEndpoint):

    def test_should_respond_200(self, session):
        if False:
            for i in range(10):
                print('nop')
        self._create_dataset(session)
        assert session.query(DatasetModel).count() == 1
        with assert_queries_count(5):
            response = self.client.get(f"/api/v1/datasets/{urllib.parse.quote('s3://bucket/key', safe='')}", environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert response.json == {'id': 1, 'uri': 's3://bucket/key', 'extra': {'foo': 'bar'}, 'created_at': self.default_time, 'updated_at': self.default_time, 'consuming_dags': [], 'producing_tasks': []}

    def test_should_respond_404(self):
        if False:
            while True:
                i = 10
        response = self.client.get(f"/api/v1/datasets/{urllib.parse.quote('s3://bucket/key', safe='')}", environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 404
        assert {'detail': 'The Dataset with uri: `s3://bucket/key` was not found', 'status': 404, 'title': 'Dataset not found', 'type': EXCEPTIONS_LINK_MAP[404]} == response.json

    def test_should_raises_401_unauthenticated(self, session):
        if False:
            for i in range(10):
                print('nop')
        self._create_dataset(session)
        response = self.client.get(f"/api/v1/datasets/{urllib.parse.quote('s3://bucket/key', safe='')}")
        assert_401(response)

class TestGetDatasets(TestDatasetEndpoint):

    def test_should_respond_200(self, session):
        if False:
            i = 10
            return i + 15
        datasets = [DatasetModel(id=i, uri=f's3://bucket/key/{i}', extra={'foo': 'bar'}, created_at=timezone.parse(self.default_time), updated_at=timezone.parse(self.default_time)) for i in [1, 2]]
        session.add_all(datasets)
        session.commit()
        assert session.query(DatasetModel).count() == 2
        with assert_queries_count(8):
            response = self.client.get('/api/v1/datasets', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        response_data = response.json
        assert response_data == {'datasets': [{'id': 1, 'uri': 's3://bucket/key/1', 'extra': {'foo': 'bar'}, 'created_at': self.default_time, 'updated_at': self.default_time, 'consuming_dags': [], 'producing_tasks': []}, {'id': 2, 'uri': 's3://bucket/key/2', 'extra': {'foo': 'bar'}, 'created_at': self.default_time, 'updated_at': self.default_time, 'consuming_dags': [], 'producing_tasks': []}], 'total_entries': 2}

    def test_order_by_raises_400_for_invalid_attr(self, session):
        if False:
            while True:
                i = 10
        datasets = [DatasetModel(uri=f's3://bucket/key/{i}', extra={'foo': 'bar'}, created_at=timezone.parse(self.default_time), updated_at=timezone.parse(self.default_time)) for i in [1, 2]]
        session.add_all(datasets)
        session.commit()
        assert session.query(DatasetModel).count() == 2
        response = self.client.get('/api/v1/datasets?order_by=fake', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 400
        msg = "Ordering with 'fake' is disallowed or the attribute does not exist on the model"
        assert response.json['detail'] == msg

    def test_should_raises_401_unauthenticated(self, session):
        if False:
            return 10
        datasets = [DatasetModel(uri=f's3://bucket/key/{i}', extra={'foo': 'bar'}, created_at=timezone.parse(self.default_time), updated_at=timezone.parse(self.default_time)) for i in [1, 2]]
        session.add_all(datasets)
        session.commit()
        assert session.query(DatasetModel).count() == 2
        response = self.client.get('/api/v1/datasets')
        assert_401(response)

    @pytest.mark.parametrize('url, expected_datasets', [('api/v1/datasets?uri_pattern=s3', {'s3://folder/key'}), ('api/v1/datasets?uri_pattern=bucket', {'gcp://bucket/key', 'wasb://some_dataset_bucket_/key'}), ('api/v1/datasets?uri_pattern=dataset', {'somescheme://dataset/key', 'wasb://some_dataset_bucket_/key'}), ('api/v1/datasets?uri_pattern=', {'gcp://bucket/key', 's3://folder/key', 'somescheme://dataset/key', 'wasb://some_dataset_bucket_/key'})])
    @provide_session
    def test_filter_datasets_by_uri_pattern_works(self, url, expected_datasets, session):
        if False:
            return 10
        dataset1 = DatasetModel('s3://folder/key')
        dataset2 = DatasetModel('gcp://bucket/key')
        dataset3 = DatasetModel('somescheme://dataset/key')
        dataset4 = DatasetModel('wasb://some_dataset_bucket_/key')
        session.add_all([dataset1, dataset2, dataset3, dataset4])
        session.commit()
        response = self.client.get(url, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        dataset_urls = {dataset['uri'] for dataset in response.json['datasets']}
        assert expected_datasets == dataset_urls

class TestGetDatasetsEndpointPagination(TestDatasetEndpoint):

    @pytest.mark.parametrize('url, expected_dataset_uris', [('/api/v1/datasets?limit=1', ['s3://bucket/key/1']), ('/api/v1/datasets?limit=100', [f's3://bucket/key/{i}' for i in range(1, 101)]), ('/api/v1/datasets?offset=1', [f's3://bucket/key/{i}' for i in range(2, 102)]), ('/api/v1/datasets?offset=3', [f's3://bucket/key/{i}' for i in range(4, 104)]), ('/api/v1/datasets?offset=3&limit=3', [f's3://bucket/key/{i}' for i in [4, 5, 6]])])
    @provide_session
    def test_limit_and_offset(self, url, expected_dataset_uris, session):
        if False:
            while True:
                i = 10
        datasets = [DatasetModel(uri=f's3://bucket/key/{i}', extra={'foo': 'bar'}, created_at=timezone.parse(self.default_time), updated_at=timezone.parse(self.default_time)) for i in range(1, 110)]
        session.add_all(datasets)
        session.commit()
        response = self.client.get(url, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        dataset_uris = [dataset['uri'] for dataset in response.json['datasets']]
        assert dataset_uris == expected_dataset_uris

    def test_should_respect_page_size_limit_default(self, session):
        if False:
            for i in range(10):
                print('nop')
        datasets = [DatasetModel(uri=f's3://bucket/key/{i}', extra={'foo': 'bar'}, created_at=timezone.parse(self.default_time), updated_at=timezone.parse(self.default_time)) for i in range(1, 110)]
        session.add_all(datasets)
        session.commit()
        response = self.client.get('/api/v1/datasets', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert len(response.json['datasets']) == 100

    @conf_vars({('api', 'maximum_page_limit'): '150'})
    def test_should_return_conf_max_if_req_max_above_conf(self, session):
        if False:
            for i in range(10):
                print('nop')
        datasets = [DatasetModel(uri=f's3://bucket/key/{i}', extra={'foo': 'bar'}, created_at=timezone.parse(self.default_time), updated_at=timezone.parse(self.default_time)) for i in range(1, 200)]
        session.add_all(datasets)
        session.commit()
        response = self.client.get('/api/v1/datasets?limit=180', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert len(response.json['datasets']) == 150

class TestGetDatasetEvents(TestDatasetEndpoint):

    def test_should_respond_200(self, session):
        if False:
            return 10
        d = self._create_dataset(session)
        common = {'dataset_id': 1, 'extra': {'foo': 'bar'}, 'source_dag_id': 'foo', 'source_task_id': 'bar', 'source_run_id': 'custom', 'source_map_index': -1, 'created_dagruns': []}
        events = [DatasetEvent(id=i, timestamp=timezone.parse(self.default_time), **common) for i in [1, 2]]
        session.add_all(events)
        session.commit()
        assert session.query(DatasetEvent).count() == 2
        response = self.client.get('/api/v1/datasets/events', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        response_data = response.json
        assert response_data == {'dataset_events': [{'id': 1, 'timestamp': self.default_time, **common, 'dataset_uri': d.uri}, {'id': 2, 'timestamp': self.default_time, **common, 'dataset_uri': d.uri}], 'total_entries': 2}

    @pytest.mark.parametrize('attr, value', [('dataset_id', '2'), ('source_dag_id', 'dag2'), ('source_task_id', 'task2'), ('source_run_id', 'run2'), ('source_map_index', '2')])
    @provide_session
    def test_filtering(self, attr, value, session):
        if False:
            i = 10
            return i + 15
        datasets = [DatasetModel(id=i, uri=f's3://bucket/key/{i}', extra={'foo': 'bar'}, created_at=timezone.parse(self.default_time), updated_at=timezone.parse(self.default_time)) for i in [1, 2, 3]]
        session.add_all(datasets)
        session.commit()
        events = [DatasetEvent(id=i, dataset_id=i, source_dag_id=f'dag{i}', source_task_id=f'task{i}', source_run_id=f'run{i}', source_map_index=i, timestamp=timezone.parse(self.default_time)) for i in [1, 2, 3]]
        session.add_all(events)
        session.commit()
        assert session.query(DatasetEvent).count() == 3
        response = self.client.get(f'/api/v1/datasets/events?{attr}={value}', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        response_data = response.json
        assert response_data == {'dataset_events': [{'id': 2, 'dataset_id': 2, 'dataset_uri': datasets[1].uri, 'extra': {}, 'source_dag_id': 'dag2', 'source_task_id': 'task2', 'source_run_id': 'run2', 'source_map_index': 2, 'timestamp': self.default_time, 'created_dagruns': []}], 'total_entries': 1}

    def test_order_by_raises_400_for_invalid_attr(self, session):
        if False:
            i = 10
            return i + 15
        self._create_dataset(session)
        events = [DatasetEvent(dataset_id=1, extra="{'foo': 'bar'}", source_dag_id='foo', source_task_id='bar', source_run_id='custom', source_map_index=-1, timestamp=timezone.parse(self.default_time)) for i in [1, 2]]
        session.add_all(events)
        session.commit()
        assert session.query(DatasetEvent).count() == 2
        response = self.client.get('/api/v1/datasets/events?order_by=fake', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 400
        msg = "Ordering with 'fake' is disallowed or the attribute does not exist on the model"
        assert response.json['detail'] == msg

    def test_should_raises_401_unauthenticated(self, session):
        if False:
            return 10
        response = self.client.get('/api/v1/datasets/events')
        assert_401(response)

    def test_includes_created_dagrun(self, session):
        if False:
            while True:
                i = 10
        self._create_dataset(session)
        event = DatasetEvent(id=1, dataset_id=1, timestamp=timezone.parse(self.default_time))
        session.add(event)
        session.commit()
        dagrun = DagRun(dag_id='TEST_DAG_ID', run_id='TEST_DAG_RUN_ID', run_type=DagRunType.DATASET_TRIGGERED, execution_date=timezone.parse(self.default_time), start_date=timezone.parse(self.default_time), external_trigger=True, state='success')
        dagrun.end_date = timezone.parse(self.default_time)
        session.add(dagrun)
        session.commit()
        event.created_dagruns.append(dagrun)
        session.commit()
        response = self.client.get('/api/v1/datasets/events', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        response_data = response.json
        assert response_data == {'dataset_events': [{'id': 1, 'dataset_id': 1, 'dataset_uri': 's3://bucket/key', 'extra': {}, 'source_dag_id': None, 'source_task_id': None, 'source_run_id': None, 'source_map_index': -1, 'timestamp': self.default_time, 'created_dagruns': [{'dag_id': 'TEST_DAG_ID', 'dag_run_id': 'TEST_DAG_RUN_ID', 'data_interval_end': None, 'data_interval_start': None, 'end_date': self.default_time, 'logical_date': self.default_time, 'start_date': self.default_time, 'state': 'success'}]}], 'total_entries': 1}

class TestGetDatasetEventsEndpointPagination(TestDatasetEndpoint):

    @pytest.mark.parametrize('url, expected_event_runids', [('/api/v1/datasets/events?limit=1&order_by=source_run_id', ['run1']), ('/api/v1/datasets/events?limit=3&order_by=source_run_id', [f'run{i}' for i in range(1, 4)]), ('/api/v1/datasets/events?offset=1&order_by=source_run_id', [f'run{i}' for i in range(2, 10)]), ('/api/v1/datasets/events?offset=3&order_by=source_run_id', [f'run{i}' for i in range(4, 10)]), ('/api/v1/datasets/events?offset=3&limit=3&order_by=source_run_id', [f'run{i}' for i in [4, 5, 6]])])
    @provide_session
    def test_limit_and_offset(self, url, expected_event_runids, session):
        if False:
            for i in range(10):
                print('nop')
        self._create_dataset(session)
        events = [DatasetEvent(dataset_id=1, source_dag_id='foo', source_task_id='bar', source_run_id=f'run{i}', source_map_index=-1, timestamp=timezone.parse(self.default_time)) for i in range(1, 10)]
        session.add_all(events)
        session.commit()
        response = self.client.get(url, environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        event_runids = [event['source_run_id'] for event in response.json['dataset_events']]
        assert event_runids == expected_event_runids

    def test_should_respect_page_size_limit_default(self, session):
        if False:
            return 10
        self._create_dataset(session)
        events = [DatasetEvent(dataset_id=1, source_dag_id='foo', source_task_id='bar', source_run_id=f'run{i}', source_map_index=-1, timestamp=timezone.parse(self.default_time)) for i in range(1, 110)]
        session.add_all(events)
        session.commit()
        response = self.client.get('/api/v1/datasets/events', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert len(response.json['dataset_events']) == 100

    @conf_vars({('api', 'maximum_page_limit'): '150'})
    def test_should_return_conf_max_if_req_max_above_conf(self, session):
        if False:
            return 10
        self._create_dataset(session)
        events = [DatasetEvent(dataset_id=1, source_dag_id='foo', source_task_id='bar', source_run_id=f'run{i}', source_map_index=-1, timestamp=timezone.parse(self.default_time)) for i in range(1, 200)]
        session.add_all(events)
        session.commit()
        response = self.client.get('/api/v1/datasets/events?limit=180', environ_overrides={'REMOTE_USER': 'test'})
        assert response.status_code == 200
        assert len(response.json['dataset_events']) == 150