from __future__ import annotations
import datetime
import os
import boto3
import pytest
from moto import mock_s3
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator
CONFIG = {'verify': False, 'replace': False, 'encrypt': False, 'gzip': False}

class TestFileToS3Operator:

    def setup_method(self):
        if False:
            print('Hello World!')
        args = {'owner': 'airflow', 'start_date': datetime.datetime(2017, 1, 1)}
        self.dag = DAG('test_dag_id', default_args=args)
        self.dest_key = 'test/test1.csv'
        self.dest_bucket = 'dummy'
        self.testfile1 = '/tmp/fake1.csv'
        with open(self.testfile1, 'wb') as f:
            f.write(b'x' * 393216)

    def teardown_method(self):
        if False:
            return 10
        os.remove(self.testfile1)

    def test_init(self):
        if False:
            while True:
                i = 10
        operator = LocalFilesystemToS3Operator(task_id='file_to_s3_operator', dag=self.dag, filename=self.testfile1, dest_key=self.dest_key, dest_bucket=self.dest_bucket, **CONFIG)
        assert operator.filename == self.testfile1
        assert operator.dest_key == self.dest_key
        assert operator.dest_bucket == self.dest_bucket
        assert operator.verify == CONFIG['verify']
        assert operator.replace == CONFIG['replace']
        assert operator.encrypt == CONFIG['encrypt']
        assert operator.gzip == CONFIG['gzip']

    def test_execute_exception(self):
        if False:
            i = 10
            return i + 15
        operator = LocalFilesystemToS3Operator(task_id='file_to_s3_operatro_exception', dag=self.dag, filename=self.testfile1, dest_key=f's3://dummy/{self.dest_key}', dest_bucket=self.dest_bucket, **CONFIG)
        with pytest.raises(TypeError):
            operator.execute(None)

    @mock_s3
    def test_execute(self):
        if False:
            while True:
                i = 10
        conn = boto3.client('s3')
        conn.create_bucket(Bucket=self.dest_bucket)
        operator = LocalFilesystemToS3Operator(task_id='s3_to_file_sensor', dag=self.dag, filename=self.testfile1, dest_key=self.dest_key, dest_bucket=self.dest_bucket, **CONFIG)
        operator.execute(None)
        objects_in_dest_bucket = conn.list_objects(Bucket=self.dest_bucket, Prefix=self.dest_key)
        assert len(objects_in_dest_bucket['Contents']) == 1
        assert objects_in_dest_bucket['Contents'][0]['Key'] == self.dest_key

    @mock_s3
    def test_execute_with_only_key(self):
        if False:
            print('Hello World!')
        conn = boto3.client('s3')
        conn.create_bucket(Bucket=self.dest_bucket)
        operator = LocalFilesystemToS3Operator(task_id='s3_to_file_sensor', dag=self.dag, filename=self.testfile1, dest_key=f's3://dummy/{self.dest_key}', **CONFIG)
        operator.execute(None)
        objects_in_dest_bucket = conn.list_objects(Bucket=self.dest_bucket, Prefix=self.dest_key)
        assert len(objects_in_dest_bucket['Contents']) == 1
        assert objects_in_dest_bucket['Contents'][0]['Key'] == self.dest_key