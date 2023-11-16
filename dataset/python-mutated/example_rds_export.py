from __future__ import annotations
from datetime import datetime
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.rds import RdsHook
from airflow.providers.amazon.aws.operators.rds import RdsCancelExportTaskOperator, RdsCreateDbInstanceOperator, RdsCreateDbSnapshotOperator, RdsDeleteDbInstanceOperator, RdsDeleteDbSnapshotOperator, RdsStartExportTaskOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator, S3DeleteBucketOperator
from airflow.providers.amazon.aws.sensors.rds import RdsExportTaskExistenceSensor, RdsSnapshotExistenceSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_rds_export'
KMS_KEY_ID_KEY = 'KMS_KEY_ID'
ROLE_ARN_KEY = 'ROLE_ARN'
sys_test_context_task = SystemTestContextBuilder().add_variable(KMS_KEY_ID_KEY).add_variable(ROLE_ARN_KEY).build()

@task
def get_snapshot_arn(snapshot_name: str) -> str:
    if False:
        print('Hello World!')
    result = RdsHook().conn.describe_db_snapshots(DBSnapshotIdentifier=snapshot_name)
    return result['DBSnapshots'][0]['DBSnapshotArn']
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    bucket_name: str = f'{env_id}-bucket'
    rds_db_name: str = f'{env_id}_db'
    rds_instance_name: str = f'{env_id}-instance'
    rds_snapshot_name: str = f'{env_id}-snapshot'
    rds_export_task_id: str = f'{env_id}-export-task'
    create_bucket = S3CreateBucketOperator(task_id='create_bucket', bucket_name=bucket_name)
    create_db_instance = RdsCreateDbInstanceOperator(task_id='create_db_instance', db_instance_identifier=rds_instance_name, db_instance_class='db.t4g.micro', engine='postgres', rds_kwargs={'MasterUsername': 'rds_username', 'MasterUserPassword': 'rds_password', 'AllocatedStorage': 20, 'DBName': rds_db_name, 'PubliclyAccessible': False})
    create_snapshot = RdsCreateDbSnapshotOperator(task_id='create_snapshot', db_type='instance', db_identifier=rds_instance_name, db_snapshot_identifier=rds_snapshot_name)
    await_snapshot = RdsSnapshotExistenceSensor(task_id='snapshot_sensor', db_type='instance', db_snapshot_identifier=rds_snapshot_name, target_statuses=['available'])
    snapshot_arn = get_snapshot_arn(rds_snapshot_name)
    start_export = RdsStartExportTaskOperator(task_id='start_export', export_task_identifier=rds_export_task_id, source_arn=snapshot_arn, s3_bucket_name=bucket_name, s3_prefix='rds-test', iam_role_arn=test_context[ROLE_ARN_KEY], kms_key_id=test_context[KMS_KEY_ID_KEY])
    start_export.wait_for_completion = False
    cancel_export = RdsCancelExportTaskOperator(task_id='cancel_export', export_task_identifier=rds_export_task_id)
    cancel_export.check_interval = 10
    cancel_export.max_attempts = 120
    export_sensor = RdsExportTaskExistenceSensor(task_id='export_sensor', export_task_identifier=rds_export_task_id, target_statuses=['canceled'])
    delete_snapshot = RdsDeleteDbSnapshotOperator(task_id='delete_snapshot', db_type='instance', db_snapshot_identifier=rds_snapshot_name, trigger_rule=TriggerRule.ALL_DONE)
    delete_bucket = S3DeleteBucketOperator(task_id='delete_bucket', bucket_name=bucket_name, force_delete=True, trigger_rule=TriggerRule.ALL_DONE)
    delete_db_instance = RdsDeleteDbInstanceOperator(task_id='delete_db_instance', db_instance_identifier=rds_instance_name, rds_kwargs={'SkipFinalSnapshot': True}, trigger_rule=TriggerRule.ALL_DONE)
    chain(test_context, create_bucket, create_db_instance, create_snapshot, await_snapshot, snapshot_arn, start_export, cancel_export, export_sensor, delete_snapshot, delete_bucket, delete_db_instance)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)