"""
Example Airflow DAG for Google Vertex AI service testing Custom Jobs operations.
"""
from __future__ import annotations
import os
from datetime import datetime
from google.cloud.aiplatform import schema
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Value
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.gcs import GCSCreateBucketOperator, GCSDeleteBucketOperator, GCSSynchronizeBucketsOperator
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import CreateCustomContainerTrainingJobOperator, DeleteCustomTrainingJobOperator
from airflow.providers.google.cloud.operators.vertex_ai.dataset import CreateDatasetOperator, DeleteDatasetOperator
from airflow.utils.trigger_rule import TriggerRule
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID', 'default')
PROJECT_ID = os.environ.get('SYSTEM_TESTS_GCP_PROJECT', 'default')
DAG_ID = 'example_vertex_ai_custom_job_operations'
REGION = 'us-central1'
CONTAINER_DISPLAY_NAME = f'train-housing-container-{ENV_ID}'
MODEL_DISPLAY_NAME = f'container-housing-model-{ENV_ID}'
RESOURCE_DATA_BUCKET = 'airflow-system-tests-resources'
CUSTOM_CONTAINER_GCS_BUCKET_NAME = f'bucket_cont_{DAG_ID}_{ENV_ID}'.replace('_', '-')
DATA_SAMPLE_GCS_OBJECT_NAME = 'vertex-ai/california_housing_train.csv'

def TABULAR_DATASET(bucket_name):
    if False:
        print('Hello World!')
    return {'display_name': f'tabular-dataset-{ENV_ID}', 'metadata_schema_uri': schema.dataset.metadata.tabular, 'metadata': ParseDict({'input_config': {'gcs_source': {'uri': [f'gs://{bucket_name}/{DATA_SAMPLE_GCS_OBJECT_NAME}']}}}, Value())}
CONTAINER_URI = 'gcr.io/cloud-aiplatform/training/tf-cpu.2-2:latest'
CUSTOM_CONTAINER_URI = f'us-central1-docker.pkg.dev/{PROJECT_ID}/system-tests/housing:latest'
MODEL_SERVING_CONTAINER_URI = 'gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-2:latest'
REPLICA_COUNT = 1
MACHINE_TYPE = 'n1-standard-4'
ACCELERATOR_TYPE = 'ACCELERATOR_TYPE_UNSPECIFIED'
ACCELERATOR_COUNT = 0
TRAINING_FRACTION_SPLIT = 0.7
TEST_FRACTION_SPLIT = 0.15
VALIDATION_FRACTION_SPLIT = 0.15
with DAG(f'{DAG_ID}_custom_container', schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['example', 'vertex_ai', 'custom_job']) as dag:
    create_bucket = GCSCreateBucketOperator(task_id='create_bucket', bucket_name=CUSTOM_CONTAINER_GCS_BUCKET_NAME, storage_class='REGIONAL', location=REGION)
    move_data_files = GCSSynchronizeBucketsOperator(task_id='move_files_to_bucket', source_bucket=RESOURCE_DATA_BUCKET, source_object='vertex-ai/california-housing-data', destination_bucket=CUSTOM_CONTAINER_GCS_BUCKET_NAME, destination_object='vertex-ai', recursive=True)
    create_tabular_dataset = CreateDatasetOperator(task_id='tabular_dataset', dataset=TABULAR_DATASET(CUSTOM_CONTAINER_GCS_BUCKET_NAME), region=REGION, project_id=PROJECT_ID)
    tabular_dataset_id = create_tabular_dataset.output['dataset_id']
    create_custom_container_training_job = CreateCustomContainerTrainingJobOperator(task_id='custom_container_task', staging_bucket=f'gs://{CUSTOM_CONTAINER_GCS_BUCKET_NAME}', display_name=CONTAINER_DISPLAY_NAME, container_uri=CUSTOM_CONTAINER_URI, model_serving_container_image_uri=MODEL_SERVING_CONTAINER_URI, dataset_id=tabular_dataset_id, command=['python3', 'task.py'], model_display_name=MODEL_DISPLAY_NAME, replica_count=REPLICA_COUNT, machine_type=MACHINE_TYPE, accelerator_type=ACCELERATOR_TYPE, accelerator_count=ACCELERATOR_COUNT, training_fraction_split=TRAINING_FRACTION_SPLIT, validation_fraction_split=VALIDATION_FRACTION_SPLIT, test_fraction_split=TEST_FRACTION_SPLIT, region=REGION, project_id=PROJECT_ID)
    delete_custom_training_job = DeleteCustomTrainingJobOperator(task_id='delete_custom_training_job', training_pipeline_id="{{ task_instance.xcom_pull(task_ids='custom_container_task', key='training_id') }}", custom_job_id="{{ task_instance.xcom_pull(task_ids='custom_container_task', key='custom_job_id') }}", region=REGION, project_id=PROJECT_ID, trigger_rule=TriggerRule.ALL_DONE)
    delete_tabular_dataset = DeleteDatasetOperator(task_id='delete_tabular_dataset', dataset_id=tabular_dataset_id, region=REGION, project_id=PROJECT_ID, trigger_rule=TriggerRule.ALL_DONE)
    delete_bucket = GCSDeleteBucketOperator(task_id='delete_bucket', bucket_name=CUSTOM_CONTAINER_GCS_BUCKET_NAME, trigger_rule=TriggerRule.ALL_DONE)
    create_bucket >> move_data_files >> create_tabular_dataset >> create_custom_container_training_job >> delete_custom_training_job >> delete_tabular_dataset >> delete_bucket
from tests.system.utils import get_test_run
test_run = get_test_run(dag)