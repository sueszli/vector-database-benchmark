from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
from airflow.models.dag import DAG
from airflow.models.dagrun import DagRun
from airflow.ti_deps.deps.dagrun_exists_dep import DagrunRunningDep
from airflow.utils.state import State
pytestmark = pytest.mark.db_test

class TestDagrunRunningDep:

    @patch('airflow.models.DagRun.find', return_value=())
    def test_dagrun_doesnt_exist(self, mock_dagrun_find):
        if False:
            i = 10
            return i + 15
        '\n        Task instances without dagruns should fail this dep\n        '
        dag = DAG('test_dag', max_active_runs=2)
        dagrun = DagRun(state=State.QUEUED)
        ti = Mock(task=Mock(dag=dag), get_dagrun=Mock(return_value=dagrun))
        assert not DagrunRunningDep().is_met(ti=ti)

    def test_dagrun_exists(self):
        if False:
            i = 10
            return i + 15
        '\n        Task instances with a dagrun should pass this dep\n        '
        dagrun = DagRun(state=State.RUNNING)
        ti = Mock(get_dagrun=Mock(return_value=dagrun))
        assert DagrunRunningDep().is_met(ti=ti)