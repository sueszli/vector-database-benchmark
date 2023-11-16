"""
Example Airflow DAG for Google Cloud Storage sensors.
"""
from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.gcs import GCSCreateBucketOperator, GCSDeleteBucketOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceAsyncSensor, GCSObjectExistenceSensor, GCSObjectsWithPrefixExistenceSensor, GCSObjectUpdateSensor, GCSUploadSessionCompleteSensor
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.utils.trigger_rule import TriggerRule
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
PROJECT_ID = os.environ.get('SYSTEM_TESTS_GCP_PROJECT')
DAG_ID = 'gcs_sensor'
BUCKET_NAME = f'bucket_{DAG_ID}_{ENV_ID}'
FILE_NAME = 'example_upload.txt'
UPLOAD_FILE_PATH = str(Path(__file__).parent / 'resources' / FILE_NAME)

def workaround_in_debug_executor(cls):
    if False:
        i = 10
        return i + 15
    "\n    DebugExecutor change sensor mode from poke to reschedule. Some sensors don't work correctly\n    in reschedule mode. They are decorated with `poke_mode_only` decorator to fail when mode is changed.\n    This method creates dummy property to overwrite it and force poke method to always return True.\n    "
    cls.mode = dummy_mode_property()
    cls.poke = lambda self, context: True

def dummy_mode_property():
    if False:
        return 10

    def mode_getter(self):
        if False:
            print('Hello World!')
        return self._mode

    def mode_setter(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._mode = value
    return property(mode_getter, mode_setter)
with DAG(DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['gcs', 'example']) as dag:
    create_bucket = GCSCreateBucketOperator(task_id='create_bucket', bucket_name=BUCKET_NAME, project_id=PROJECT_ID)
    workaround_in_debug_executor(GCSUploadSessionCompleteSensor)
    gcs_upload_session_complete = GCSUploadSessionCompleteSensor(bucket=BUCKET_NAME, prefix=FILE_NAME, inactivity_period=15, min_objects=1, allow_delete=True, previous_objects=set(), task_id='gcs_upload_session_complete_task')
    gcs_upload_session_async_complete = GCSUploadSessionCompleteSensor(bucket=BUCKET_NAME, prefix=FILE_NAME, inactivity_period=15, min_objects=1, allow_delete=True, previous_objects=set(), task_id='gcs_upload_session_async_complete', deferrable=True)
    gcs_update_object_exists = GCSObjectUpdateSensor(bucket=BUCKET_NAME, object=FILE_NAME, task_id='gcs_object_update_sensor_task')
    gcs_update_object_exists_async = GCSObjectUpdateSensor(bucket=BUCKET_NAME, object=FILE_NAME, task_id='gcs_object_update_sensor_task_async', deferrable=True)
    upload_file = LocalFilesystemToGCSOperator(task_id='upload_file', src=UPLOAD_FILE_PATH, dst=FILE_NAME, bucket=BUCKET_NAME)
    gcs_object_exists = GCSObjectExistenceSensor(bucket=BUCKET_NAME, object=FILE_NAME, task_id='gcs_object_exists_task')
    gcs_object_exists_async = GCSObjectExistenceAsyncSensor(bucket=BUCKET_NAME, object=FILE_NAME, task_id='gcs_object_exists_task_async')
    gcs_object_exists_defered = GCSObjectExistenceSensor(bucket=BUCKET_NAME, object=FILE_NAME, task_id='gcs_object_exists_defered', deferrable=True)
    gcs_object_with_prefix_exists = GCSObjectsWithPrefixExistenceSensor(bucket=BUCKET_NAME, prefix=FILE_NAME[:5], task_id='gcs_object_with_prefix_exists_task')
    gcs_object_with_prefix_exists_async = GCSObjectsWithPrefixExistenceSensor(bucket=BUCKET_NAME, prefix=FILE_NAME[:5], task_id='gcs_object_with_prefix_exists_task_async', deferrable=True)
    delete_bucket = GCSDeleteBucketOperator(task_id='delete_bucket', bucket_name=BUCKET_NAME, trigger_rule=TriggerRule.ALL_DONE)
    chain(create_bucket, upload_file, [gcs_object_exists, gcs_object_exists_defered, gcs_object_exists_async, gcs_object_with_prefix_exists, gcs_object_with_prefix_exists_async], delete_bucket)
    chain(create_bucket, gcs_upload_session_complete, gcs_update_object_exists, gcs_update_object_exists_async, delete_bucket)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)