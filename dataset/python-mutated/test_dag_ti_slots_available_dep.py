from __future__ import annotations
from unittest.mock import Mock
import pytest
from airflow.models import TaskInstance
from airflow.ti_deps.deps.dag_ti_slots_available_dep import DagTISlotsAvailableDep
pytestmark = pytest.mark.db_test

class TestDagTISlotsAvailableDep:

    def test_concurrency_reached(self):
        if False:
            return 10
        '\n        Test max_active_tasks reached should fail dep\n        '
        dag = Mock(concurrency=1, get_concurrency_reached=Mock(return_value=True))
        task = Mock(dag=dag, pool_slots=1)
        ti = TaskInstance(task, execution_date=None)
        assert not DagTISlotsAvailableDep().is_met(ti=ti)

    def test_all_conditions_met(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test all conditions met should pass dep\n        '
        dag = Mock(concurrency=1, get_concurrency_reached=Mock(return_value=False))
        task = Mock(dag=dag, pool_slots=1)
        ti = TaskInstance(task, execution_date=None)
        assert DagTISlotsAvailableDep().is_met(ti=ti)