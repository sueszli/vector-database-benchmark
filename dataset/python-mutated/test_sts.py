from __future__ import annotations
from airflow.providers.amazon.aws.hooks.sts import StsHook

class TestSTS:

    def test_get_conn_returns_a_boto3_connection(self):
        if False:
            i = 10
            return i + 15
        hook = StsHook(aws_conn_id='aws_default', region_name='us-east-1')
        assert hook.conn is not None