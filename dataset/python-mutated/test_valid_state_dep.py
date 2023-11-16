from __future__ import annotations
from datetime import datetime
from unittest.mock import Mock
import pytest
from airflow.exceptions import AirflowException
from airflow.ti_deps.deps.valid_state_dep import ValidStateDep
from airflow.utils.state import State
pytestmark = pytest.mark.db_test

class TestValidStateDep:

    def test_valid_state(self):
        if False:
            i = 10
            return i + 15
        '\n        Valid state should pass this dep\n        '
        ti = Mock(state=State.QUEUED, end_date=datetime(2016, 1, 1))
        assert ValidStateDep({State.QUEUED}).is_met(ti=ti)

    def test_invalid_state(self):
        if False:
            while True:
                i = 10
        '\n        Invalid state should fail this dep\n        '
        ti = Mock(state=State.SUCCESS, end_date=datetime(2016, 1, 1))
        assert not ValidStateDep({State.FAILED}).is_met(ti=ti)

    def test_no_valid_states(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If there are no valid states the dependency should throw\n        '
        ti = Mock(state=State.SUCCESS, end_date=datetime(2016, 1, 1))
        with pytest.raises(AirflowException):
            ValidStateDep({}).is_met(ti=ti)