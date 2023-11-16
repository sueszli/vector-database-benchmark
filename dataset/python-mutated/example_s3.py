from __future__ import annotations
from datetime import datetime
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.providers.amazon.aws.operators.s3 import S3CopyObjectOperator, S3CreateBucketOperator, S3CreateObjectOperator, S3DeleteBucketOperator, S3DeleteBucketTaggingOperator, S3DeleteObjectsOperator, S3FileTransformOperator, S3GetBucketTaggingOperator, S3ListOperator, S3ListPrefixesOperator, S3PutBucketTaggingOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor, S3KeysUnchangedSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_s3'
sys_test_context_task = SystemTestContextBuilder().build()
DATA = '\n    apple,0.5\n    milk,2.5\n    bread,4.0\n'
PREFIX = ''
DELIMITER = '/'
TAG_KEY = 'test-s3-bucket-tagging-key'
TAG_VALUE = 'test-s3-bucket-tagging-value'
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['example']) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    bucket_name = f'{env_id}-s3-bucket'
    bucket_name_2 = f'{env_id}-s3-bucket-2'
    key = f'{env_id}-key'
    key_2 = f'{env_id}-key2'

    def check_fn(files: list) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Example of custom check: check if all files are bigger than ``20 bytes``\n\n        :param files: List of S3 object attributes.\n        :return: true if the criteria is met\n        '
        return all((f.get('Size', 0) > 20 for f in files))
    create_bucket = S3CreateBucketOperator(task_id='create_bucket', bucket_name=bucket_name)
    create_bucket_2 = S3CreateBucketOperator(task_id='create_bucket_2', bucket_name=bucket_name_2)
    put_tagging = S3PutBucketTaggingOperator(task_id='put_tagging', bucket_name=bucket_name, key=TAG_KEY, value=TAG_VALUE)
    get_tagging = S3GetBucketTaggingOperator(task_id='get_tagging', bucket_name=bucket_name)
    delete_tagging = S3DeleteBucketTaggingOperator(task_id='delete_tagging', bucket_name=bucket_name)
    create_object = S3CreateObjectOperator(task_id='create_object', s3_bucket=bucket_name, s3_key=key, data=DATA, replace=True)
    create_object_2 = S3CreateObjectOperator(task_id='create_object_2', s3_bucket=bucket_name, s3_key=key_2, data=DATA, replace=True)
    list_prefixes = S3ListPrefixesOperator(task_id='list_prefixes', bucket=bucket_name, prefix=PREFIX, delimiter=DELIMITER)
    list_keys = S3ListOperator(task_id='list_keys', bucket=bucket_name, prefix=PREFIX)
    sensor_one_key = S3KeySensor(task_id='sensor_one_key', bucket_name=bucket_name, bucket_key=key)
    sensor_two_keys = S3KeySensor(task_id='sensor_two_keys', bucket_name=bucket_name, bucket_key=[key, key_2])
    sensor_one_key_deferrable = S3KeySensor(task_id='sensor_one_key_deferrable', bucket_name=bucket_name, bucket_key=key, deferrable=True)
    sensor_two_keys_deferrable = S3KeySensor(task_id='sensor_two_keys_deferrable', bucket_name=bucket_name, bucket_key=[key, key_2], deferrable=True)
    sensor_key_with_function_deferrable = S3KeySensor(task_id='sensor_key_with_function_deferrable', bucket_name=bucket_name, bucket_key=key, check_fn=check_fn, deferrable=True)
    sensor_key_with_function = S3KeySensor(task_id='sensor_key_with_function', bucket_name=bucket_name, bucket_key=key, check_fn=check_fn)
    copy_object = S3CopyObjectOperator(task_id='copy_object', source_bucket_name=bucket_name, dest_bucket_name=bucket_name_2, source_bucket_key=key, dest_bucket_key=key_2)
    file_transform = S3FileTransformOperator(task_id='file_transform', source_s3_key=f's3://{bucket_name}/{key}', dest_s3_key=f's3://{bucket_name_2}/{key_2}', transform_script='cp', replace=True)
    branching = BranchPythonOperator(task_id='branch_to_delete_objects', python_callable=lambda : 'delete_objects')
    sensor_keys_unchanged = S3KeysUnchangedSensor(task_id='sensor_keys_unchanged', bucket_name=bucket_name_2, prefix=PREFIX, inactivity_period=10)
    delete_objects = S3DeleteObjectsOperator(task_id='delete_objects', bucket=bucket_name_2, keys=key_2)
    delete_objects.trigger_rule = TriggerRule.ALL_DONE
    delete_bucket = S3DeleteBucketOperator(task_id='delete_bucket', bucket_name=bucket_name, force_delete=True)
    delete_bucket.trigger_rule = TriggerRule.ALL_DONE
    delete_bucket_2 = S3DeleteBucketOperator(task_id='delete_bucket_2', bucket_name=bucket_name_2, force_delete=True)
    delete_bucket_2.trigger_rule = TriggerRule.ALL_DONE
    chain(test_context, create_bucket, create_bucket_2, put_tagging, get_tagging, delete_tagging, create_object, create_object_2, list_prefixes, list_keys, [sensor_one_key, sensor_two_keys, sensor_key_with_function], [sensor_one_key_deferrable, sensor_two_keys_deferrable, sensor_key_with_function_deferrable], copy_object, file_transform, branching, sensor_keys_unchanged, delete_objects, delete_bucket, delete_bucket_2)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)