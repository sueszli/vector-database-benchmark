from __future__ import annotations
from datetime import datetime
from unittest.mock import Mock
import pytest
from airflow.ti_deps.deps.task_not_running_dep import TaskNotRunningDep
from airflow.utils.state import State
pytestmark = pytest.mark.db_test

class TestTaskNotRunningDep:

    def test_not_running_state(self):
        if False:
            print('Hello World!')
        ti = Mock(state=State.QUEUED, end_date=datetime(2016, 1, 1))
        assert TaskNotRunningDep().is_met(ti=ti)

    def test_running_state(self):
        if False:
            i = 10
            return i + 15
        ti = Mock(state=State.RUNNING, end_date=datetime(2016, 1, 1))
        assert not TaskNotRunningDep().is_met(ti=ti)