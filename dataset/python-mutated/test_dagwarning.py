from __future__ import annotations
from unittest import mock
from unittest.mock import MagicMock
import pytest
from sqlalchemy.exc import OperationalError
from airflow.models import DagModel
from airflow.models.dagwarning import DagWarning
from tests.test_utils.db import clear_db_dags
pytestmark = pytest.mark.db_test

class TestDagWarning:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        clear_db_dags()

    def test_purge_inactive_dag_warnings(self, session):
        if False:
            print('Hello World!')
        '\n        Test that the purge_inactive_dag_warnings method deletes inactive dag warnings\n        '
        dags = [DagModel(dag_id='dag_1', is_active=False), DagModel(dag_id='dag_2', is_active=True)]
        session.add_all(dags)
        session.commit()
        dag_warnings = [DagWarning('dag_1', 'non-existent pool', 'non-existent pool'), DagWarning('dag_2', 'non-existent pool', 'non-existent pool')]
        session.add_all(dag_warnings)
        session.commit()
        DagWarning.purge_inactive_dag_warnings(session)
        remaining_dag_warnings = session.query(DagWarning).all()
        assert len(remaining_dag_warnings) == 1
        assert remaining_dag_warnings[0].dag_id == 'dag_2'

    @mock.patch('airflow.models.dagwarning.delete')
    def test_retry_purge_inactive_dag_warnings(self, delete_mock):
        if False:
            while True:
                i = 10
        '\n        Test that the purge_inactive_dag_warnings method calls the delete method twice\n        if the query throws an operationalError on the first call and works on the second attempt\n        '
        self.session_mock = MagicMock()
        self.session_mock.execute.side_effect = [OperationalError(None, None, 'database timeout'), None]
        DagWarning.purge_inactive_dag_warnings(self.session_mock)
        assert delete_mock.call_count == 2
        assert self.session_mock.execute.call_count == 2