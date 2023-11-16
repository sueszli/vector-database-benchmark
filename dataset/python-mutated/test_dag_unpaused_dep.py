from __future__ import annotations
from unittest.mock import Mock
import pytest
from airflow.models import TaskInstance
from airflow.ti_deps.deps.dag_unpaused_dep import DagUnpausedDep
pytestmark = pytest.mark.db_test

class TestDagUnpausedDep:

    def test_concurrency_reached(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test paused DAG should fail dependency\n        '
        dag = Mock(**{'get_is_paused.return_value': True})
        task = Mock(dag=dag)
        ti = TaskInstance(task=task, execution_date=None)
        assert not DagUnpausedDep().is_met(ti=ti)

    def test_all_conditions_met(self):
        if False:
            while True:
                i = 10
        '\n        Test all conditions met should pass dep\n        '
        dag = Mock(**{'get_is_paused.return_value': False})
        task = Mock(dag=dag)
        ti = TaskInstance(task=task, execution_date=None)
        assert DagUnpausedDep().is_met(ti=ti)