from __future__ import annotations
from unittest import mock
from unittest.mock import PropertyMock
import pytest
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.providers.alibaba.cloud.sensors.oss_key import OSSKeySensor
MODULE_NAME = 'airflow.providers.alibaba.cloud.sensors.oss_key'
MOCK_TASK_ID = 'test-oss-operator'
MOCK_REGION = 'mock_region'
MOCK_BUCKET = 'mock_bucket_name'
MOCK_OSS_CONN_ID = 'mock_oss_conn_default'
MOCK_KEY = 'mock_key'
MOCK_KEYS = ['mock_key1', 'mock_key_2', 'mock_key3']
MOCK_CONTENT = 'mock_content'

@pytest.fixture
def oss_key_sensor():
    if False:
        while True:
            i = 10
    return OSSKeySensor(bucket_key=MOCK_KEY, oss_conn_id=MOCK_OSS_CONN_ID, region=MOCK_REGION, bucket_name=MOCK_BUCKET, task_id=MOCK_TASK_ID)

class TestOSSKeySensor:

    @mock.patch(f'{MODULE_NAME}.OSSHook')
    def test_get_hook(self, mock_service, oss_key_sensor):
        if False:
            return 10
        oss_key_sensor.hook
        mock_service.assert_called_once_with(oss_conn_id=MOCK_OSS_CONN_ID, region=MOCK_REGION)

    @mock.patch(f'{MODULE_NAME}.OSSKeySensor.hook', new_callable=PropertyMock)
    def test_poke_exsiting_key(self, mock_service, oss_key_sensor):
        if False:
            print('Hello World!')
        mock_service.return_value.object_exists.return_value = True
        res = oss_key_sensor.poke(None)
        assert res is True
        mock_service.return_value.object_exists.assert_called_once_with(key=MOCK_KEY, bucket_name=MOCK_BUCKET)

    @mock.patch(f'{MODULE_NAME}.OSSKeySensor.hook', new_callable=PropertyMock)
    def test_poke_non_exsiting_key(self, mock_service, oss_key_sensor):
        if False:
            for i in range(10):
                print('nop')
        mock_service.return_value.object_exists.return_value = False
        res = oss_key_sensor.poke(None)
        assert res is False
        mock_service.return_value.object_exists.assert_called_once_with(key=MOCK_KEY, bucket_name=MOCK_BUCKET)

    @pytest.mark.parametrize('soft_fail, expected_exception', ((False, AirflowException), (True, AirflowSkipException)))
    @mock.patch(f'{MODULE_NAME}.OSSKeySensor.hook', new_callable=PropertyMock)
    def test_poke_without_bucket_name(self, mock_service, oss_key_sensor, soft_fail: bool, expected_exception: AirflowException):
        if False:
            for i in range(10):
                print('nop')
        oss_key_sensor.soft_fail = soft_fail
        oss_key_sensor.bucket_name = None
        mock_service.return_value.object_exists.return_value = False
        with pytest.raises(expected_exception, match='If key is a relative path from root, please provide a bucket_name'):
            oss_key_sensor.poke(None)