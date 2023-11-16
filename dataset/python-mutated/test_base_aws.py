from __future__ import annotations
import contextlib
from unittest import mock
from unittest.mock import ANY
import pytest
from airflow.models.connection import Connection
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseAsyncHook
pytest.importorskip('aiobotocore')
with contextlib.suppress(ImportError):
    from aiobotocore.credentials import AioCredentials

class TestAwsBaseAsyncHook:

    @staticmethod
    def compare_aio_cred(first, second):
        if False:
            print('Hello World!')
        if type(first) != type(second):
            return False
        if first.access_key != second.access_key:
            return False
        if first.secret_key != second.secret_key:
            return False
        if first.method != second.method:
            return False
        if first.token != second.token:
            return False
        return True

    class Matcher:

        def __init__(self, compare, obj):
            if False:
                return 10
            self.compare = compare
            self.obj = obj

        def __eq__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            return self.compare(self.obj, other)

    @pytest.mark.asyncio
    @mock.patch('airflow.hooks.base.BaseHook.get_connection')
    @mock.patch('aiobotocore.session.AioClientCreator.create_client')
    async def test_get_client_async(self, mock_client, mock_get_connection):
        """Check the connection credential passed while creating client"""
        mock_get_connection.return_value = Connection(conn_id='aws_default1', extra='{\n                        "aws_secret_access_key": "mock_aws_access_key",\n                        "aws_access_key_id": "mock_aws_access_key_id",\n                        "region_name": "us-east-2"\n                    }')
        hook = AwsBaseAsyncHook(client_type='S3', aws_conn_id='aws_default1')
        cred = await hook.get_async_session().get_credentials()
        assert cred.__dict__ == {'access_key': 'mock_aws_access_key_id', 'method': 'explicit', 'secret_key': 'mock_aws_access_key', 'token': None}
        aio_cred = AioCredentials(access_key='mock_aws_access_key_id', method='explicit', secret_key='mock_aws_access_key')
        await (await hook.get_client_async())._coro
        mock_client.assert_called_once_with(service_name='S3', region_name='us-east-2', is_secure=True, endpoint_url=None, verify=None, credentials=self.Matcher(self.compare_aio_cred, aio_cred), scoped_config={}, client_config=ANY, api_version=None, auth_token=None)