from __future__ import annotations
import json
import os
import pytest
from airflow.models import Connection
from airflow.utils.session import create_session
from tests.test_utils.db import clear_db_connections
from tests.test_utils.gcp_system_helpers import CLOUD_DAG_FOLDER, GoogleSystemTest
TOKEN = os.environ.get('DATAPREP_TOKEN')
EXTRA = {'token': TOKEN}

@pytest.mark.skipif(TOKEN is None, reason='Dataprep token not present')
class TestDataprepExampleDagsSystem(GoogleSystemTest):
    """
    System tests for Dataprep operators.
    It uses a real service and requires real data for test.
    """

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        with create_session() as session:
            dataprep_conn_id = Connection(conn_id='dataprep_default', conn_type='dataprep', extra=json.dumps(EXTRA))
            session.add(dataprep_conn_id)

    def teardown_method(self):
        if False:
            print('Hello World!')
        clear_db_connections()

    def test_run_example_dag(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_dag(dag_id='example_dataprep', dag_folder=CLOUD_DAG_FOLDER)