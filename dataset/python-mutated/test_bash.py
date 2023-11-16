from __future__ import annotations
import datetime
import pytest
from airflow.exceptions import AirflowFailException, AirflowSensorTimeout
from airflow.models.dag import DAG
from airflow.sensors.bash import BashSensor

class TestBashSensor:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        args = {'owner': 'airflow', 'start_date': datetime.datetime(2017, 1, 1)}
        dag = DAG('test_dag_id', default_args=args)
        self.dag = dag

    def test_true_condition(self):
        if False:
            return 10
        op = BashSensor(task_id='test_true_condition', bash_command='freturn() { return "$1"; }; freturn 0', output_encoding='utf-8', poke_interval=1, timeout=2, dag=self.dag)
        op.execute(None)

    def test_false_condition(self):
        if False:
            for i in range(10):
                print('nop')
        op = BashSensor(task_id='test_false_condition', bash_command='freturn() { return "$1"; }; freturn 1', output_encoding='utf-8', poke_interval=1, timeout=2, dag=self.dag)
        with pytest.raises(AirflowSensorTimeout):
            op.execute(None)

    def test_retry_code_retries(self):
        if False:
            i = 10
            return i + 15
        op = BashSensor(task_id='test_false_condition', bash_command='freturn() { return "$1"; }; freturn 99', output_encoding='utf-8', poke_interval=1, timeout=2, retry_exit_code=99, dag=self.dag)
        with pytest.raises(AirflowSensorTimeout):
            op.execute(None)

    def test_retry_code_fails(self):
        if False:
            print('Hello World!')
        op = BashSensor(task_id='test_false_condition', bash_command='freturn() { return "$1"; }; freturn 1', output_encoding='utf-8', poke_interval=1, timeout=2, retry_exit_code=99, dag=self.dag)
        with pytest.raises(AirflowFailException):
            op.execute(None)