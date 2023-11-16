"""
Example Airflow DAG that uses Google AutoML services.
"""
from __future__ import annotations
import os
from datetime import datetime
from typing import cast
from google.cloud import storage
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.models.xcom_arg import XComArg
from airflow.providers.google.cloud.hooks.automl import CloudAutoMLHook
from airflow.providers.google.cloud.operators.automl import AutoMLCreateDatasetOperator, AutoMLDeleteDatasetOperator, AutoMLDeleteModelOperator, AutoMLImportDataOperator, AutoMLTrainModelOperator
from airflow.providers.google.cloud.operators.gcs import GCSCreateBucketOperator, GCSDeleteBucketOperator
from airflow.providers.google.cloud.transfers.gcs_to_gcs import GCSToGCSOperator
from airflow.utils.trigger_rule import TriggerRule
DAG_ID = 'example_automl_translate'
GCP_PROJECT_ID = os.environ.get('SYSTEM_TESTS_GCP_PROJECT', 'default')
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID', 'default')
GCP_AUTOML_LOCATION = 'us-central1'
DATA_SAMPLE_GCS_BUCKET_NAME = f'bucket_{DAG_ID}_{ENV_ID}'.replace('_', '-')
RESOURCE_DATA_BUCKET = 'airflow-system-tests-resources'
MODEL_NAME = 'translate_test_model'
MODEL = {'display_name': MODEL_NAME, 'translation_model_metadata': {}}
DATASET_NAME = f'ds_translate_{ENV_ID}'.replace('-', '_')
DATASET = {'display_name': DATASET_NAME, 'translation_dataset_metadata': {'source_language_code': 'en', 'target_language_code': 'es'}}
CSV_FILE_NAME = 'en-es.csv'
TSV_FILE_NAME = 'en-es.tsv'
GCS_FILE_PATH = f'automl/datasets/translate/{CSV_FILE_NAME}'
AUTOML_DATASET_BUCKET = f'gs://{DATA_SAMPLE_GCS_BUCKET_NAME}/automl/{CSV_FILE_NAME}'
IMPORT_INPUT_CONFIG = {'gcs_source': {'input_uris': [AUTOML_DATASET_BUCKET]}}
extract_object_id = CloudAutoMLHook.extract_object_id
with DAG(DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, user_defined_macros={'extract_object_id': extract_object_id}, tags=['example', 'automl', 'translate']) as dag:
    create_bucket = GCSCreateBucketOperator(task_id='create_bucket', bucket_name=DATA_SAMPLE_GCS_BUCKET_NAME, storage_class='REGIONAL', location=GCP_AUTOML_LOCATION)

    @task
    def upload_csv_file_to_gcs():
        if False:
            for i in range(10):
                print('nop')
        storage_client = storage.Client()
        bucket = storage_client.bucket(RESOURCE_DATA_BUCKET)
        blob = bucket.blob(GCS_FILE_PATH)
        contents = blob.download_as_string().decode()
        updated_contents = contents.replace('template-bucket', DATA_SAMPLE_GCS_BUCKET_NAME)
        destination_bucket = storage_client.bucket(DATA_SAMPLE_GCS_BUCKET_NAME)
        destination_blob = destination_bucket.blob(f'automl/{CSV_FILE_NAME}')
        destination_blob.upload_from_string(updated_contents)
    upload_csv_file_to_gcs_task = upload_csv_file_to_gcs()
    copy_dataset_file = GCSToGCSOperator(task_id='copy_dataset_file', source_bucket=RESOURCE_DATA_BUCKET, source_object=f'automl/datasets/translate/{TSV_FILE_NAME}', destination_bucket=DATA_SAMPLE_GCS_BUCKET_NAME, destination_object=f'automl/{TSV_FILE_NAME}')
    create_dataset = AutoMLCreateDatasetOperator(task_id='create_dataset', dataset=DATASET, location=GCP_AUTOML_LOCATION)
    dataset_id = cast(str, XComArg(create_dataset, key='dataset_id'))
    import_dataset = AutoMLImportDataOperator(task_id='import_dataset', dataset_id=dataset_id, location=GCP_AUTOML_LOCATION, input_config=IMPORT_INPUT_CONFIG)
    MODEL['dataset_id'] = dataset_id
    create_model = AutoMLTrainModelOperator(task_id='create_model', model=MODEL, location=GCP_AUTOML_LOCATION)
    model_id = cast(str, XComArg(create_model, key='model_id'))
    delete_model = AutoMLDeleteModelOperator(task_id='delete_model', model_id=model_id, location=GCP_AUTOML_LOCATION, project_id=GCP_PROJECT_ID)
    delete_dataset = AutoMLDeleteDatasetOperator(task_id='delete_dataset', dataset_id=dataset_id, location=GCP_AUTOML_LOCATION, project_id=GCP_PROJECT_ID)
    delete_bucket = GCSDeleteBucketOperator(task_id='delete_bucket', bucket_name=DATA_SAMPLE_GCS_BUCKET_NAME, trigger_rule=TriggerRule.ALL_DONE)
    [create_bucket >> upload_csv_file_to_gcs_task >> copy_dataset_file] >> create_dataset >> import_dataset >> create_model >> delete_dataset >> delete_model >> delete_bucket
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)