from __future__ import annotations
from datetime import timedelta
import pytest
from airflow.models import DagBag
from airflow.models.dag import DAG
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.utils.timezone import datetime
pytestmark = pytest.mark.db_test
DEFAULT_DATE = datetime(2015, 1, 1)
DEV_NULL = '/dev/null'
TEST_DAG_ID = 'unit_tests'

class TestTimedeltaSensor:

    def setup_method(self):
        if False:
            return 10
        self.dagbag = DagBag(dag_folder=DEV_NULL, include_examples=True)
        self.args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        self.dag = DAG(TEST_DAG_ID, default_args=self.args)

    def test_timedelta_sensor(self):
        if False:
            for i in range(10):
                print('nop')
        op = TimeDeltaSensor(task_id='timedelta_sensor_check', delta=timedelta(seconds=2), dag=self.dag)
        op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)