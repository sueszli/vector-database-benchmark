from __future__ import annotations
from unittest import mock
from airflow.providers.neo4j.operators.neo4j import Neo4jOperator
from airflow.utils import timezone
DEFAULT_DATE = timezone.datetime(2015, 1, 1)
DEFAULT_DATE_ISO = DEFAULT_DATE.isoformat()
DEFAULT_DATE_DS = DEFAULT_DATE_ISO[:10]
TEST_DAG_ID = 'unit_test_dag'

class TestNeo4jOperator:

    @mock.patch('airflow.providers.neo4j.operators.neo4j.Neo4jHook')
    def test_neo4j_operator_test(self, mock_hook):
        if False:
            while True:
                i = 10
        sql = '\n            MATCH (tom {name: "Tom Hanks"}) RETURN tom\n            '
        op = Neo4jOperator(task_id='basic_neo4j', sql=sql)
        op.execute(mock.MagicMock())
        mock_hook.assert_called_once_with(conn_id='neo4j_default')
        mock_hook.return_value.run.assert_called_once_with(sql)