from __future__ import annotations
import boto3
import pytest
from moto import mock_s3
from airflow.models import DAG
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.transfers.s3_to_sftp import S3ToSFTPOperator
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.utils.timezone import datetime
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test
TASK_ID = 'test_s3_to_sftp'
BUCKET = 'test-s3-bucket'
S3_KEY = 'test/test_1_file.csv'
SFTP_PATH = '/tmp/remote_path.txt'
SFTP_CONN_ID = 'ssh_default'
LOCAL_FILE_PATH = '/tmp/test_s3_upload'
SFTP_MOCK_FILE = 'test_sftp_file.csv'
S3_MOCK_FILES = 'test_1_file.csv'
TEST_DAG_ID = 'unit_tests_s3_to_sftp'
DEFAULT_DATE = datetime(2018, 1, 1)

class TestS3ToSFTPOperator:

    def setup_method(self):
        if False:
            print('Hello World!')
        hook = SSHHook(ssh_conn_id='ssh_default')
        hook.no_host_key_check = True
        dag = DAG(f'{TEST_DAG_ID}test_schedule_dag_once', start_date=DEFAULT_DATE, schedule='@once')
        self.hook = hook
        self.ssh_client = self.hook.get_conn()
        self.sftp_client = self.ssh_client.open_sftp()
        self.dag = dag
        self.s3_bucket = BUCKET
        self.sftp_path = SFTP_PATH
        self.s3_key = S3_KEY

    @mock_s3
    @conf_vars({('core', 'enable_xcom_pickling'): 'True'})
    def test_s3_to_sftp_operation(self):
        if False:
            i = 10
            return i + 15
        s3_hook = S3Hook(aws_conn_id=None)
        test_remote_file_content = 'This is remote file content \n which is also multiline another line here \n this is last line. EOF'
        conn = boto3.client('s3')
        conn.create_bucket(Bucket=self.s3_bucket)
        assert s3_hook.check_for_bucket(self.s3_bucket)
        with open(LOCAL_FILE_PATH, 'w') as file:
            file.write(test_remote_file_content)
        s3_hook.load_file(LOCAL_FILE_PATH, self.s3_key, bucket_name=BUCKET)
        objects_in_dest_bucket = conn.list_objects(Bucket=self.s3_bucket, Prefix=self.s3_key)
        assert len(objects_in_dest_bucket['Contents']) == 1
        assert objects_in_dest_bucket['Contents'][0]['Key'] == self.s3_key
        run_task = S3ToSFTPOperator(s3_bucket=BUCKET, s3_key=S3_KEY, sftp_path=SFTP_PATH, sftp_conn_id=SFTP_CONN_ID, task_id=TASK_ID, dag=self.dag)
        assert run_task is not None
        run_task.execute(None)
        check_file_task = SSHOperator(task_id='test_check_file', ssh_hook=self.hook, command=f'cat {self.sftp_path}', do_xcom_push=True, dag=self.dag)
        assert check_file_task is not None
        result = check_file_task.execute(None)
        assert result.strip() == test_remote_file_content.encode('utf-8')
        conn.delete_object(Bucket=self.s3_bucket, Key=self.s3_key)
        conn.delete_bucket(Bucket=self.s3_bucket)
        assert not s3_hook.check_for_bucket(self.s3_bucket)

    def delete_remote_resource(self):
        if False:
            i = 10
            return i + 15
        remove_file_task = SSHOperator(task_id='test_rm_file', ssh_hook=self.hook, command=f'rm {self.sftp_path}', do_xcom_push=True, dag=self.dag)
        assert remove_file_task is not None
        remove_file_task.execute(None)

    def teardown_method(self):
        if False:
            return 10
        self.delete_remote_resource()