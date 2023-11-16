from __future__ import annotations
from datetime import timedelta
from unittest.mock import Mock
import pytest
import time_machine
from airflow.models import TaskInstance
from airflow.ti_deps.deps.not_in_retry_period_dep import NotInRetryPeriodDep
from airflow.utils.state import State
from airflow.utils.timezone import datetime
pytestmark = pytest.mark.db_test

class TestNotInRetryPeriodDep:

    def _get_task_instance(self, state, end_date=None, retry_delay=timedelta(minutes=15)):
        if False:
            i = 10
            return i + 15
        task = Mock(retry_delay=retry_delay, retry_exponential_backoff=False)
        ti = TaskInstance(task=task, state=state, execution_date=None)
        ti.end_date = end_date
        return ti

    @time_machine.travel('2016-01-01 15:44')
    def test_still_in_retry_period(self):
        if False:
            return 10
        '\n        Task instances that are in their retry period should fail this dep\n        '
        ti = self._get_task_instance(State.UP_FOR_RETRY, end_date=datetime(2016, 1, 1, 15, 30))
        assert ti.is_premature
        assert not NotInRetryPeriodDep().is_met(ti=ti)

    @time_machine.travel('2016-01-01 15:46')
    def test_retry_period_finished(self):
        if False:
            i = 10
            return i + 15
        "\n        Task instance's that have had their retry period elapse should pass this dep\n        "
        ti = self._get_task_instance(State.UP_FOR_RETRY, end_date=datetime(2016, 1, 1))
        assert not ti.is_premature
        assert NotInRetryPeriodDep().is_met(ti=ti)

    def test_not_in_retry_period(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Task instance's that are not up for retry can not be in their retry period\n        "
        ti = self._get_task_instance(State.SUCCESS)
        assert NotInRetryPeriodDep().is_met(ti=ti)