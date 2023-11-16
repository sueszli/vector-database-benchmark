from __future__ import annotations
from unittest import mock
import pytest
from airflow.providers.apache.pinot.hooks.pinot import PinotDbApiHook

@pytest.mark.integration('pinot')
class TestPinotDbApiHookIntegration:

    @pytest.mark.flaky(reruns=1, reruns_delay=30)
    @mock.patch.dict('os.environ', AIRFLOW_CONN_PINOT_BROKER_DEFAULT='pinot://pinot:8000/')
    def test_should_return_records(self):
        if False:
            while True:
                i = 10
        hook = PinotDbApiHook()
        sql = 'select playerName from baseballStats  ORDER BY playerName limit 5'
        records = hook.get_records(sql)
        assert [['A. Harry'], ['A. Harry'], ['Aaron'], ['Aaron Albert'], ['Aaron Albert']] == records