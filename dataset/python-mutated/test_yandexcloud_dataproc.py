from __future__ import annotations
import json
from unittest.mock import patch
import pytest
from airflow.models import Connection
try:
    import yandexcloud
    from airflow.providers.yandex.hooks.yandexcloud_dataproc import DataprocHook
except ImportError:
    yandexcloud = None
CONNECTION_ID = 'yandexcloud_default'
AVAILABILITY_ZONE_ID = 'ru-central1-c'
CLUSTER_NAME = 'dataproc_cluster'
CLUSTER_IMAGE_VERSION = '1.4'
FOLDER_ID = 'my_folder_id'
SUBNET_ID = 'my_subnet_id'
S3_BUCKET_NAME_FOR_LOGS = 'my_bucket_name'
SERVICE_ACCOUNT_ID = 'my_service_account_id'
OAUTH_TOKEN = 'my_oauth_token'
SSH_PUBLIC_KEYS = ['ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAAYQCxO38tKAJXIs9ivPxt7AYdfybgtAR1ow3Qkb9GPQ6wkFHQqcFDe6faKCxH6iDRteo4D8L8BxwzN42uZSB0nfmjkIxFTcEU3mFSXEbWByg78aoddMrAAjatyrhH1pON6P0=']
HAS_CREDENTIALS = OAUTH_TOKEN != 'my_oauth_token'

@pytest.mark.skipif(yandexcloud is None, reason='Skipping Yandex.Cloud hook test: no yandexcloud module')
class TestYandexCloudDataprocHook:

    def _init_hook(self):
        if False:
            print('Hello World!')
        with patch('airflow.hooks.base.BaseHook.get_connection') as get_connection_mock:
            get_connection_mock.return_value = self.connection
            self.hook = DataprocHook()

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.connection = Connection(extra=json.dumps({'oauth': OAUTH_TOKEN}))
        self._init_hook()

    @patch('yandexcloud.SDK.create_operation_and_get_result')
    def test_create_dataproc_cluster_mocked(self, create_operation_mock):
        if False:
            return 10
        self._init_hook()
        self.hook.client.create_cluster(cluster_name=CLUSTER_NAME, ssh_public_keys=SSH_PUBLIC_KEYS, folder_id=FOLDER_ID, subnet_id=SUBNET_ID, zone=AVAILABILITY_ZONE_ID, s3_bucket=S3_BUCKET_NAME_FOR_LOGS, cluster_image_version=CLUSTER_IMAGE_VERSION, service_account_id=SERVICE_ACCOUNT_ID)
        assert create_operation_mock.called

    @patch('yandexcloud.SDK.create_operation_and_get_result')
    def test_delete_dataproc_cluster_mocked(self, create_operation_mock):
        if False:
            i = 10
            return i + 15
        self._init_hook()
        self.hook.client.delete_cluster('my_cluster_id')
        assert create_operation_mock.called

    @patch('yandexcloud.SDK.create_operation_and_get_result')
    def test_create_hive_job_hook(self, create_operation_mock):
        if False:
            print('Hello World!')
        self._init_hook()
        self.hook.client.create_hive_job(cluster_id='my_cluster_id', continue_on_failure=False, name='Hive job', properties=None, query='SELECT 1;', script_variables=None)
        assert create_operation_mock.called

    @patch('yandexcloud.SDK.create_operation_and_get_result')
    def test_create_mapreduce_job_hook(self, create_operation_mock):
        if False:
            print('Hello World!')
        self._init_hook()
        self.hook.client.create_mapreduce_job(archive_uris=None, args=['-mapper', 'mapper.py', '-reducer', 'reducer.py', '-numReduceTasks', '1', '-input', 's3a://some-in-bucket/jobs/sources/data/cities500.txt.bz2', '-output', 's3a://some-out-bucket/dataproc/job/results'], cluster_id='my_cluster_id', file_uris=['s3a://some-in-bucket/jobs/sources/mapreduce-001/mapper.py', 's3a://some-in-bucket/jobs/sources/mapreduce-001/reducer.py'], jar_file_uris=None, main_class='org.apache.hadoop.streaming.HadoopStreaming', main_jar_file_uri=None, name='Mapreduce job', properties={'yarn.app.mapreduce.am.resource.mb': '2048', 'yarn.app.mapreduce.am.command-opts': '-Xmx2048m', 'mapreduce.job.maps': '6'})
        assert create_operation_mock.called

    @patch('yandexcloud.SDK.create_operation_and_get_result')
    def test_create_spark_job_hook(self, create_operation_mock):
        if False:
            return 10
        self._init_hook()
        self.hook.client.create_spark_job(archive_uris=['s3a://some-in-bucket/jobs/sources/data/country-codes.csv.zip'], args=['s3a://some-in-bucket/jobs/sources/data/cities500.txt.bz2', 's3a://some-out-bucket/dataproc/job/results/${{JOB_ID}}'], cluster_id='my_cluster_id', file_uris=['s3a://some-in-bucket/jobs/sources/data/config.json'], jar_file_uris=['s3a://some-in-bucket/jobs/sources/java/icu4j-61.1.jar', 's3a://some-in-bucket/jobs/sources/java/commons-lang-2.6.jar', 's3a://some-in-bucket/jobs/sources/java/opencsv-4.1.jar', 's3a://some-in-bucket/jobs/sources/java/json-20190722.jar'], main_class='ru.yandex.cloud.dataproc.examples.PopulationSparkJob', main_jar_file_uri='s3a://data-proc-public/jobs/sources/java/dataproc-examples-1.0.jar', name='Spark job', properties={'spark.submit.deployMode': 'cluster'})
        assert create_operation_mock.called

    @patch('yandexcloud.SDK.create_operation_and_get_result')
    def test_create_pyspark_job_hook(self, create_operation_mock):
        if False:
            while True:
                i = 10
        self._init_hook()
        self.hook.client.create_pyspark_job(archive_uris=['s3a://some-in-bucket/jobs/sources/data/country-codes.csv.zip'], args=['s3a://some-in-bucket/jobs/sources/data/cities500.txt.bz2', 's3a://some-out-bucket/jobs/results/${{JOB_ID}}'], cluster_id='my_cluster_id', file_uris=['s3a://some-in-bucket/jobs/sources/data/config.json'], jar_file_uris=['s3a://some-in-bucket/jobs/sources/java/dataproc-examples-1.0.jar', 's3a://some-in-bucket/jobs/sources/java/icu4j-61.1.jar', 's3a://some-in-bucket/jobs/sources/java/commons-lang-2.6.jar'], main_python_file_uri='s3a://some-in-bucket/jobs/sources/pyspark-001/main.py', name='Pyspark job', properties={'spark.submit.deployMode': 'cluster'}, python_file_uris=['s3a://some-in-bucket/jobs/sources/pyspark-001/geonames.py'])
        assert create_operation_mock.called