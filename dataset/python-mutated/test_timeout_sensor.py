from __future__ import annotations
import time
from datetime import timedelta
from typing import TYPE_CHECKING
import pytest
from airflow.exceptions import AirflowSensorTimeout, AirflowSkipException
from airflow.models.dag import DAG
from airflow.sensors.base import BaseSensorOperator
from airflow.utils import timezone
from airflow.utils.timezone import datetime
pytestmark = pytest.mark.db_test
if TYPE_CHECKING:
    from airflow.utils.context import Context
DEFAULT_DATE = datetime(2015, 1, 1)
TEST_DAG_ID = 'unit_test_dag'

class TimeoutTestSensor(BaseSensorOperator):
    """
    Sensor that always returns the return_value provided

    :param return_value: Set to true to mark the task as SKIPPED on failure
    """

    def __init__(self, return_value=False, **kwargs):
        if False:
            while True:
                i = 10
        self.return_value = return_value
        super().__init__(**kwargs)

    def poke(self, context: Context):
        if False:
            print('Hello World!')
        return self.return_value

    def execute(self, context: Context):
        if False:
            print('Hello World!')
        started_at = timezone.utcnow()
        time_jump = self.params['time_jump']
        while not self.poke(context):
            if time_jump:
                started_at -= time_jump
            if (timezone.utcnow() - started_at).total_seconds() > self.timeout:
                if self.soft_fail:
                    raise AirflowSkipException('timeout')
                else:
                    raise AirflowSensorTimeout('timeout')
            time.sleep(self.poke_interval)
        self.log.info('Success criteria met. Exiting.')

class TestSensorTimeout:

    def setup_method(self):
        if False:
            return 10
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        self.dag = DAG(TEST_DAG_ID, default_args=args)

    def test_timeout(self):
        if False:
            print('Hello World!')
        op = TimeoutTestSensor(task_id='test_timeout', execution_timeout=timedelta(days=2), return_value=False, poke_interval=5, params={'time_jump': timedelta(days=2, seconds=1)}, dag=self.dag)
        with pytest.raises(AirflowSensorTimeout):
            op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)