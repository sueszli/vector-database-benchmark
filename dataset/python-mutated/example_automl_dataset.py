"""
Example Airflow DAG for Google AutoML service testing dataset operations.
"""
from __future__ import annotations
import os
from copy import deepcopy
from datetime import datetime
from airflow.models.dag import DAG
from airflow.providers.google.cloud.hooks.automl import CloudAutoMLHook
from airflow.providers.google.cloud.operators.automl import AutoMLCreateDatasetOperator, AutoMLDeleteDatasetOperator, AutoMLImportDataOperator, AutoMLListDatasetOperator, AutoMLTablesListColumnSpecsOperator, AutoMLTablesListTableSpecsOperator, AutoMLTablesUpdateDatasetOperator
from airflow.providers.google.cloud.operators.gcs import GCSCreateBucketOperator, GCSDeleteBucketOperator, GCSSynchronizeBucketsOperator
from airflow.utils.trigger_rule import TriggerRule
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID', 'default')
DAG_ID = 'example_automl_dataset'
GCP_PROJECT_ID = os.environ.get('SYSTEM_TESTS_GCP_PROJECT', 'default')
GCP_AUTOML_LOCATION = 'us-central1'
RESOURCE_DATA_BUCKET = 'airflow-system-tests-resources'
DATA_SAMPLE_GCS_BUCKET_NAME = f'bucket_{DAG_ID}_{ENV_ID}'.replace('_', '-')
DATASET_NAME = f'ds_tabular_{ENV_ID}'.replace('-', '_')
DATASET = {'display_name': DATASET_NAME, 'tables_dataset_metadata': {'target_column_spec_id': ''}}
AUTOML_DATASET_BUCKET = f'gs://{DATA_SAMPLE_GCS_BUCKET_NAME}/automl/tabular-classification.csv'
IMPORT_INPUT_CONFIG = {'gcs_source': {'input_uris': [AUTOML_DATASET_BUCKET]}}
extract_object_id = CloudAutoMLHook.extract_object_id

def get_target_column_spec(columns_specs: list[dict], column_name: str) -> str:
    if False:
        return 10
    '\n    Using column name returns spec of the column.\n    '
    for column in columns_specs:
        if column['display_name'] == column_name:
            return extract_object_id(column)
    raise Exception(f'Unknown target column: {column_name}')
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['example', 'automl', 'dataset'], user_defined_macros={'get_target_column_spec': get_target_column_spec, 'target': 'Class', 'extract_object_id': extract_object_id}) as dag:
    create_bucket = GCSCreateBucketOperator(task_id='create_bucket', bucket_name=DATA_SAMPLE_GCS_BUCKET_NAME, storage_class='REGIONAL', location=GCP_AUTOML_LOCATION)
    move_dataset_file = GCSSynchronizeBucketsOperator(task_id='move_dataset_to_bucket', source_bucket=RESOURCE_DATA_BUCKET, source_object='automl/datasets/tabular', destination_bucket=DATA_SAMPLE_GCS_BUCKET_NAME, destination_object='automl', recursive=True)
    create_dataset = AutoMLCreateDatasetOperator(task_id='create_dataset', dataset=DATASET, location=GCP_AUTOML_LOCATION, project_id=GCP_PROJECT_ID)
    dataset_id = create_dataset.output['dataset_id']
    import_dataset = AutoMLImportDataOperator(task_id='import_dataset', dataset_id=dataset_id, location=GCP_AUTOML_LOCATION, input_config=IMPORT_INPUT_CONFIG)
    list_tables_spec = AutoMLTablesListTableSpecsOperator(task_id='list_tables_spec', dataset_id=dataset_id, location=GCP_AUTOML_LOCATION, project_id=GCP_PROJECT_ID)
    list_columns_spec = AutoMLTablesListColumnSpecsOperator(task_id='list_columns_spec', dataset_id=dataset_id, table_spec_id="{{ extract_object_id(task_instance.xcom_pull('list_tables_spec_task')[0]) }}", location=GCP_AUTOML_LOCATION, project_id=GCP_PROJECT_ID)
    update = deepcopy(DATASET)
    update['name'] = '{{ task_instance.xcom_pull("create_dataset")["name"] }}'
    update['tables_dataset_metadata']['target_column_spec_id'] = "{{ get_target_column_spec(task_instance.xcom_pull('list_columns_spec_task'), target) }}"
    update_dataset = AutoMLTablesUpdateDatasetOperator(task_id='update_dataset', dataset=update, location=GCP_AUTOML_LOCATION)
    list_datasets = AutoMLListDatasetOperator(task_id='list_datasets', location=GCP_AUTOML_LOCATION, project_id=GCP_PROJECT_ID)
    delete_dataset = AutoMLDeleteDatasetOperator(task_id='delete_dataset', dataset_id=dataset_id, location=GCP_AUTOML_LOCATION, project_id=GCP_PROJECT_ID)
    delete_bucket = GCSDeleteBucketOperator(task_id='delete_bucket', bucket_name=DATA_SAMPLE_GCS_BUCKET_NAME, trigger_rule=TriggerRule.ALL_DONE)
    [create_bucket >> move_dataset_file, create_dataset] >> import_dataset >> list_tables_spec >> list_columns_spec >> update_dataset >> list_datasets >> delete_dataset >> delete_bucket
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)