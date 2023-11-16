"""
Example Airflow DAG for Google Cloud Storage GCSSynchronizeBucketsOperator and
GCSToGCSOperator operators.
"""
from __future__ import annotations
import os
import shutil
from datetime import datetime
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.google.cloud.operators.gcs import GCSCreateBucketOperator, GCSDeleteBucketOperator, GCSSynchronizeBucketsOperator
from airflow.providers.google.cloud.transfers.gcs_to_gcs import GCSToGCSOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.utils.trigger_rule import TriggerRule
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
PROJECT_ID = os.environ.get('SYSTEM_TESTS_GCP_PROJECT')
DAG_ID = 'gcs_to_gcs'
BUCKET_NAME_SRC = f'bucket_{DAG_ID}_{ENV_ID}'
BUCKET_NAME_DST = f'bucket_dst_{DAG_ID}_{ENV_ID}'
RANDOM_FILE_NAME = OBJECT_1 = OBJECT_2 = 'random.bin'
HOME = '/home/airflow/gcs'
PREFIX = f'{HOME}/data/{DAG_ID}_{ENV_ID}/'
with DAG(DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['gcs', 'example']) as dag:

    @task
    def create_workdir() -> str:
        if False:
            for i in range(10):
                print('nop')
        "\n        Task creates working directory. The logic behind this task is a workaround that provides sustainable\n        execution in Composer environment: local files can be safely shared among tasks if they are located\n        within '/home/airflow/gcs/data/' folder which is mounted to GCS bucket under the hood\n        (https://cloud.google.com/composer/docs/composer-2/cloud-storage).\n        Otherwise, worker nodes don't share local path and thus files created by one task aren't guaranteed\n        to be accessible be others.\n        "
        workdir = PREFIX if os.path.exists(HOME) else HOME
        os.makedirs(PREFIX)
        return workdir
    create_workdir_task = create_workdir()
    generate_random_file = BashOperator(task_id='generate_random_file', bash_command=f'cat /dev/urandom | head -c $((1 * 1024 * 1024)) > {PREFIX + RANDOM_FILE_NAME}')
    create_bucket_src = GCSCreateBucketOperator(task_id='create_bucket_src', bucket_name=BUCKET_NAME_SRC, project_id=PROJECT_ID)
    create_bucket_dst = GCSCreateBucketOperator(task_id='create_bucket_dst', bucket_name=BUCKET_NAME_DST, project_id=PROJECT_ID)
    upload_file_src = LocalFilesystemToGCSOperator(task_id='upload_file_src', src=PREFIX + RANDOM_FILE_NAME, dst=PREFIX + RANDOM_FILE_NAME, bucket=BUCKET_NAME_SRC)
    upload_file_src_sub = LocalFilesystemToGCSOperator(task_id='upload_file_src_sub', src=PREFIX + RANDOM_FILE_NAME, dst=f'{PREFIX}subdir/{RANDOM_FILE_NAME}', bucket=BUCKET_NAME_SRC)
    upload_file_dst = LocalFilesystemToGCSOperator(task_id='upload_file_dst', src=PREFIX + RANDOM_FILE_NAME, dst=PREFIX + RANDOM_FILE_NAME, bucket=BUCKET_NAME_DST)
    upload_file_dst_sub = LocalFilesystemToGCSOperator(task_id='upload_file_dst_sub', src=PREFIX + RANDOM_FILE_NAME, dst=f'{PREFIX}subdir/{RANDOM_FILE_NAME}', bucket=BUCKET_NAME_DST)
    sync_bucket = GCSSynchronizeBucketsOperator(task_id='sync_bucket', source_bucket=BUCKET_NAME_SRC, destination_bucket=BUCKET_NAME_DST)
    sync_full_bucket = GCSSynchronizeBucketsOperator(task_id='sync_full_bucket', source_bucket=BUCKET_NAME_SRC, destination_bucket=BUCKET_NAME_DST, delete_extra_files=True, allow_overwrite=True)
    sync_to_subdirectory = GCSSynchronizeBucketsOperator(task_id='sync_to_subdirectory', source_bucket=BUCKET_NAME_SRC, destination_bucket=BUCKET_NAME_DST, destination_object='subdir/')
    sync_from_subdirectory = GCSSynchronizeBucketsOperator(task_id='sync_from_subdirectory', source_bucket=BUCKET_NAME_SRC, source_object='subdir/', destination_bucket=BUCKET_NAME_DST)
    copy_single_file = GCSToGCSOperator(task_id='copy_single_gcs_file', source_bucket=BUCKET_NAME_SRC, source_object=OBJECT_1, destination_bucket=BUCKET_NAME_DST, destination_object='backup_' + OBJECT_1, exact_match=True)
    copy_files_with_wildcard = GCSToGCSOperator(task_id='copy_files_with_wildcard', source_bucket=BUCKET_NAME_SRC, source_object='data/*.txt', destination_bucket=BUCKET_NAME_DST, destination_object='backup/')
    copy_files_without_wildcard = GCSToGCSOperator(task_id='copy_files_without_wildcard', source_bucket=BUCKET_NAME_SRC, source_object='subdir/', destination_bucket=BUCKET_NAME_DST, destination_object='backup/')
    copy_files_with_delimiter = GCSToGCSOperator(task_id='copy_files_with_delimiter', source_bucket=BUCKET_NAME_SRC, source_object='data/', destination_bucket=BUCKET_NAME_DST, destination_object='backup/', delimiter='.txt')
    copy_files_with_match_glob = GCSToGCSOperator(task_id='copy_files_with_match_glob', source_bucket=BUCKET_NAME_SRC, source_object='data/', destination_bucket=BUCKET_NAME_DST, destination_object='backup/', match_glob='**/*.txt')
    copy_files_with_list = GCSToGCSOperator(task_id='copy_files_with_list', source_bucket=BUCKET_NAME_SRC, source_objects=[OBJECT_1, OBJECT_2], destination_bucket=BUCKET_NAME_DST, destination_object='backup/')
    move_single_file = GCSToGCSOperator(task_id='move_single_file', source_bucket=BUCKET_NAME_SRC, source_object=OBJECT_1, destination_bucket=BUCKET_NAME_DST, destination_object='backup_' + OBJECT_1, exact_match=True, move_object=True)
    move_files_with_list = GCSToGCSOperator(task_id='move_files_with_list', source_bucket=BUCKET_NAME_SRC, source_objects=[OBJECT_1, OBJECT_2], destination_bucket=BUCKET_NAME_DST, destination_object='backup/')
    delete_bucket_src = GCSDeleteBucketOperator(task_id='delete_bucket_src', bucket_name=BUCKET_NAME_SRC, trigger_rule=TriggerRule.ALL_DONE)
    delete_bucket_dst = GCSDeleteBucketOperator(task_id='delete_bucket_dst', bucket_name=BUCKET_NAME_DST, trigger_rule=TriggerRule.ALL_DONE)

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def delete_work_dir(create_workdir_result: str) -> None:
        if False:
            return 10
        shutil.rmtree(create_workdir_result)
    chain(create_workdir_task, generate_random_file, [create_bucket_src, create_bucket_dst], [upload_file_src, upload_file_src_sub], [upload_file_dst, upload_file_dst_sub], sync_bucket, sync_full_bucket, sync_to_subdirectory, sync_from_subdirectory, copy_single_file, copy_files_with_wildcard, copy_files_without_wildcard, copy_files_with_delimiter, copy_files_with_match_glob, copy_files_with_list, move_single_file, move_files_with_list, [delete_bucket_src, delete_bucket_dst, delete_work_dir(create_workdir_task)])
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)