from __future__ import annotations
from datetime import datetime
from unittest.mock import Mock
import pytest
from airflow.models.baseoperator import BaseOperator
from airflow.models.dag import DAG
from airflow.ti_deps.dep_context import DepContext
from airflow.ti_deps.deps.task_concurrency_dep import TaskConcurrencyDep
pytestmark = pytest.mark.db_test

class TestTaskConcurrencyDep:

    def _get_task(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return BaseOperator(task_id='test_task', dag=DAG('test_dag'), **kwargs)

    @pytest.mark.parametrize('kwargs, num_running_tis, is_task_concurrency_dep_met', [({}, None, True), ({'max_active_tis_per_dag': 1}, 0, True), ({'max_active_tis_per_dag': 2}, 1, True), ({'max_active_tis_per_dag': 2}, 2, False), ({'max_active_tis_per_dagrun': 2}, 1, True), ({'max_active_tis_per_dagrun': 2}, 2, False), ({'max_active_tis_per_dag': 2, 'max_active_tis_per_dagrun': 2}, 1, True), ({'max_active_tis_per_dag': 1, 'max_active_tis_per_dagrun': 2}, 1, False), ({'max_active_tis_per_dag': 2, 'max_active_tis_per_dagrun': 1}, 1, False), ({'max_active_tis_per_dag': 1, 'max_active_tis_per_dagrun': 1}, 1, False)])
    def test_concurrency(self, kwargs, num_running_tis, is_task_concurrency_dep_met):
        if False:
            while True:
                i = 10
        task = self._get_task(start_date=datetime(2016, 1, 1), **kwargs)
        dep_context = DepContext()
        ti = Mock(task=task, execution_date=datetime(2016, 1, 1))
        if num_running_tis is not None:
            ti.get_num_running_task_instances.return_value = num_running_tis
        assert TaskConcurrencyDep().is_met(ti=ti, dep_context=dep_context) == is_task_concurrency_dep_met