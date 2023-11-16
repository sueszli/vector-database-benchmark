from __future__ import annotations
import json
from unittest.mock import MagicMock, patch
import pytest
from airflow.providers.apache.druid.hooks.druid import IngestionType
from airflow.providers.apache.druid.operators.druid import DruidOperator
from airflow.utils import timezone
from airflow.utils.types import DagRunType
DEFAULT_DATE = timezone.datetime(2017, 1, 1)
JSON_INDEX_STR = '\n    {\n        "type": "{{ params.index_type }}",\n        "datasource": "{{ params.datasource }}",\n        "spec": {\n            "dataSchema": {\n                "granularitySpec": {\n                    "intervals": ["{{ ds }}/{{ macros.ds_add(ds, 1) }}"]\n                }\n            }\n        }\n    }\n'
RENDERED_INDEX = {'type': 'index_hadoop', 'datasource': 'datasource_prd', 'spec': {'dataSchema': {'granularitySpec': {'intervals': ['2017-01-01/2017-01-02']}}}}

@pytest.mark.db_test
def test_render_template(dag_maker):
    if False:
        for i in range(10):
            print('nop')
    with dag_maker('test_druid_render_template', default_args={'start_date': DEFAULT_DATE}):
        operator = DruidOperator(task_id='spark_submit_job', json_index_file=JSON_INDEX_STR, params={'index_type': 'index_hadoop', 'datasource': 'datasource_prd'})
    dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED).task_instances[0].render_templates()
    assert RENDERED_INDEX == json.loads(operator.json_index_file)

@pytest.mark.db_test
def test_render_template_from_file(tmp_path, dag_maker):
    if False:
        while True:
            i = 10
    json_index_file = tmp_path.joinpath('json_index.json')
    json_index_file.write_text(JSON_INDEX_STR)
    with dag_maker('test_druid_render_template_from_file', template_searchpath=[str(tmp_path)], default_args={'start_date': DEFAULT_DATE}):
        operator = DruidOperator(task_id='spark_submit_job', json_index_file=json_index_file.name, params={'index_type': 'index_hadoop', 'datasource': 'datasource_prd'})
    dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED).task_instances[0].render_templates()
    assert RENDERED_INDEX == json.loads(operator.json_index_file)

def test_init_with_timeout_and_max_ingestion_time():
    if False:
        for i in range(10):
            print('nop')
    operator = DruidOperator(task_id='spark_submit_job', json_index_file=JSON_INDEX_STR, timeout=60, max_ingestion_time=180, params={'index_type': 'index_hadoop', 'datasource': 'datasource_prd'})
    expected_values = {'task_id': 'spark_submit_job', 'timeout': 60, 'max_ingestion_time': 180}
    assert expected_values['task_id'] == operator.task_id
    assert expected_values['timeout'] == operator.timeout
    assert expected_values['max_ingestion_time'] == operator.max_ingestion_time

def test_init_default_timeout():
    if False:
        print('Hello World!')
    operator = DruidOperator(task_id='spark_submit_job', json_index_file=JSON_INDEX_STR, params={'index_type': 'index_hadoop', 'datasource': 'datasource_prd'})
    expected_default_timeout = 1
    assert expected_default_timeout == operator.timeout

@patch('airflow.providers.apache.druid.operators.druid.DruidHook')
def test_execute_calls_druid_hook_with_the_right_parameters(mock_druid_hook):
    if False:
        while True:
            i = 10
    mock_druid_hook_instance = MagicMock()
    mock_druid_hook.return_value = mock_druid_hook_instance
    json_index_file = 'sql.json'
    druid_ingest_conn_id = 'druid_ingest_default'
    max_ingestion_time = 5
    timeout = 5
    operator = DruidOperator(task_id='spark_submit_job', json_index_file=json_index_file, druid_ingest_conn_id=druid_ingest_conn_id, timeout=timeout, ingestion_type=IngestionType.MSQ, max_ingestion_time=max_ingestion_time)
    operator.execute(context={})
    mock_druid_hook.assert_called_once_with(druid_ingest_conn_id=druid_ingest_conn_id, timeout=timeout, max_ingestion_time=max_ingestion_time)
    mock_druid_hook_instance.submit_indexing_job.assert_called_once_with(json_index_file, IngestionType.MSQ)