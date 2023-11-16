from __future__ import annotations
import datetime
from unittest import mock
import boto3
from moto import mock_s3
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.transfers.http_to_s3 import HttpToS3Operator
EXAMPLE_URL = 'http://www.example.com'

@mock.patch.dict('os.environ', AIRFLOW_CONN_HTTP_EXAMPLE=EXAMPLE_URL)
class TestHttpToS3Operator:

    def setup_method(self):
        if False:
            print('Hello World!')
        args = {'owner': 'airflow', 'start_date': datetime.datetime(2017, 1, 1)}
        self.dag = DAG('test_dag_id', default_args=args)
        self.http_conn_id = 'HTTP_EXAMPLE'
        self.response = b'Example.com fake response'
        self.endpoint = '/'
        self.s3_key = 'test/test1.csv'
        self.s3_bucket = 'dummy'

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        operator = HttpToS3Operator(task_id='http_to_s3_operator', http_conn_id=self.http_conn_id, endpoint=self.endpoint, s3_key=self.s3_key, s3_bucket=self.s3_bucket, dag=self.dag)
        assert operator.endpoint == self.endpoint
        assert operator.s3_key == self.s3_key
        assert operator.s3_bucket == self.s3_bucket
        assert operator.http_conn_id == self.http_conn_id

    @mock_s3
    def test_execute(self, requests_mock):
        if False:
            for i in range(10):
                print('nop')
        requests_mock.register_uri('GET', EXAMPLE_URL, content=self.response)
        conn = boto3.client('s3')
        conn.create_bucket(Bucket=self.s3_bucket)
        operator = HttpToS3Operator(task_id='s3_to_file_sensor', http_conn_id=self.http_conn_id, endpoint=self.endpoint, s3_key=self.s3_key, s3_bucket=self.s3_bucket, dag=self.dag)
        operator.execute(None)
        objects_in_bucket = conn.list_objects(Bucket=self.s3_bucket, Prefix=self.s3_key)
        assert len(objects_in_bucket['Contents']) == 1
        assert objects_in_bucket['Contents'][0]['Key'] == self.s3_key