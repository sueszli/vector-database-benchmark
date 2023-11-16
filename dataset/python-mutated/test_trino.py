from __future__ import annotations
from unittest import mock
import pytest
from airflow.providers.trino.hooks.trino import TrinoHook
from airflow.providers.trino.operators.trino import TrinoOperator

@pytest.mark.integration('trino')
class TestTrinoHookIntegration:

    @mock.patch.dict('os.environ', AIRFLOW_CONN_TRINO_DEFAULT='trino://airflow@trino:8080/')
    def test_should_record_records(self):
        if False:
            for i in range(10):
                print('nop')
        hook = TrinoHook()
        sql = 'SELECT name FROM tpch.sf1.customer ORDER BY custkey ASC LIMIT 3'
        records = hook.get_records(sql)
        assert [['Customer#000000001'], ['Customer#000000002'], ['Customer#000000003']] == records

    @pytest.mark.integration('kerberos')
    def test_should_record_records_with_kerberos_auth(self):
        if False:
            return 10
        conn_url = 'trino://airflow@trino.example.com:7778/?auth=kerberos&kerberos__service_name=HTTP&verify=False&protocol=https'
        with mock.patch.dict('os.environ', AIRFLOW_CONN_TRINO_DEFAULT=conn_url):
            hook = TrinoHook()
            sql = 'SELECT name FROM tpch.sf1.customer ORDER BY custkey ASC LIMIT 3'
            records = hook.get_records(sql)
            assert [['Customer#000000001'], ['Customer#000000002'], ['Customer#000000003']] == records

    @mock.patch.dict('os.environ', AIRFLOW_CONN_TRINO_DEFAULT='trino://airflow@trino:8080/')
    def test_openlineage_methods(self):
        if False:
            i = 10
            return i + 15
        op = TrinoOperator(task_id='trino_test', sql='SELECT name FROM tpch.sf1.customer LIMIT 3')
        op.execute({})
        lineage = op.get_openlineage_facets_on_start()
        assert lineage.inputs[0].namespace == 'trino://trino:8080'
        assert lineage.inputs[0].name == 'tpch.sf1.customer'
        assert 'schema' in lineage.inputs[0].facets
        assert lineage.job_facets['sql'].query == 'SELECT name FROM tpch.sf1.customer LIMIT 3'