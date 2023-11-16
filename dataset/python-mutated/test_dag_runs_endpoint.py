from __future__ import annotations
import json
import pytest
from airflow.api.common.trigger_dag import trigger_dag
from airflow.models import DagBag, DagRun
from airflow.models.serialized_dag import SerializedDagModel
from airflow.settings import Session
pytestmark = pytest.mark.db_test

class TestDagRunsEndpoint:

    @pytest.fixture(scope='class', autouse=True)
    def _setup_session(self):
        if False:
            print('Hello World!')
        session = Session()
        session.query(DagRun).delete()
        session.commit()
        session.close()
        dagbag = DagBag(include_examples=True)
        for dag in dagbag.dags.values():
            dag.sync_to_db()
            SerializedDagModel.write_dag(dag)

    @pytest.fixture(autouse=True)
    def _reset_test_session(self, experiemental_api_app):
        if False:
            i = 10
            return i + 15
        self.app = experiemental_api_app.test_client()
        yield
        session = Session()
        session.query(DagRun).delete()
        session.commit()
        session.close()

    def test_get_dag_runs_success(self):
        if False:
            print('Hello World!')
        url_template = '/api/experimental/dags/{}/dag_runs'
        dag_id = 'example_bash_operator'
        dag_run = trigger_dag(dag_id=dag_id, run_id='test_get_dag_runs_success')
        response = self.app.get(url_template.format(dag_id))
        assert 200 == response.status_code
        data = json.loads(response.data.decode('utf-8'))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]['dag_id'] == dag_id
        assert data[0]['id'] == dag_run.id

    def test_get_dag_runs_success_with_state_parameter(self):
        if False:
            i = 10
            return i + 15
        url_template = '/api/experimental/dags/{}/dag_runs?state=queued'
        dag_id = 'example_bash_operator'
        dag_run = trigger_dag(dag_id=dag_id, run_id='test_get_dag_runs_success')
        response = self.app.get(url_template.format(dag_id))
        assert 200 == response.status_code
        data = json.loads(response.data.decode('utf-8'))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]['dag_id'] == dag_id
        assert data[0]['id'] == dag_run.id

    def test_get_dag_runs_success_with_capital_state_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        url_template = '/api/experimental/dags/{}/dag_runs?state=QUEUED'
        dag_id = 'example_bash_operator'
        dag_run = trigger_dag(dag_id=dag_id, run_id='test_get_dag_runs_success')
        response = self.app.get(url_template.format(dag_id))
        assert 200 == response.status_code
        data = json.loads(response.data.decode('utf-8'))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]['dag_id'] == dag_id
        assert data[0]['id'] == dag_run.id

    def test_get_dag_runs_success_with_state_no_result(self):
        if False:
            i = 10
            return i + 15
        url_template = '/api/experimental/dags/{}/dag_runs?state=dummy'
        dag_id = 'example_bash_operator'
        trigger_dag(dag_id=dag_id, run_id='test_get_dag_runs_success')
        with pytest.raises(ValueError):
            self.app.get(url_template.format(dag_id))

    def test_get_dag_runs_invalid_dag_id(self):
        if False:
            for i in range(10):
                print('nop')
        url_template = '/api/experimental/dags/{}/dag_runs'
        dag_id = 'DUMMY_DAG'
        response = self.app.get(url_template.format(dag_id))
        assert 400 == response.status_code
        data = json.loads(response.data.decode('utf-8'))
        assert not isinstance(data, list)

    def test_get_dag_runs_no_runs(self):
        if False:
            i = 10
            return i + 15
        url_template = '/api/experimental/dags/{}/dag_runs'
        dag_id = 'example_bash_operator'
        response = self.app.get(url_template.format(dag_id))
        assert 200 == response.status_code
        data = json.loads(response.data.decode('utf-8'))
        assert isinstance(data, list)
        assert len(data) == 0