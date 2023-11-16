from __future__ import annotations
import datetime
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.bash import BashSensor
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.python import PythonSensor
from airflow.sensors.time_delta import TimeDeltaSensor, TimeDeltaSensorAsync
from airflow.sensors.time_sensor import TimeSensor, TimeSensorAsync
from airflow.sensors.weekday import DayOfWeekSensor
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.weekday import WeekDay

def success_callable():
    if False:
        print('Hello World!')
    return True

def failure_callable():
    if False:
        for i in range(10):
            print('nop')
    return False
with DAG(dag_id='example_sensors', schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example']) as dag:
    t0 = TimeDeltaSensor(task_id='wait_some_seconds', delta=datetime.timedelta(seconds=2))
    t0a = TimeDeltaSensorAsync(task_id='wait_some_seconds_async', delta=datetime.timedelta(seconds=2))
    t1 = TimeSensor(task_id='fire_immediately', target_time=datetime.datetime.now(tz=datetime.timezone.utc).time())
    t2 = TimeSensor(task_id='timeout_after_second_date_in_the_future', timeout=1, soft_fail=True, target_time=(datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(hours=1)).time())
    t1a = TimeSensorAsync(task_id='fire_immediately_async', target_time=datetime.datetime.now(tz=datetime.timezone.utc).time())
    t2a = TimeSensorAsync(task_id='timeout_after_second_date_in_the_future_async', timeout=1, soft_fail=True, target_time=(datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(hours=1)).time())
    t3 = BashSensor(task_id='Sensor_succeeds', bash_command='exit 0')
    t4 = BashSensor(task_id='Sensor_fails_after_3_seconds', timeout=3, soft_fail=True, bash_command='exit 1')
    t5 = BashOperator(task_id='remove_file', bash_command='rm -rf /tmp/temporary_file_for_testing')
    t6 = FileSensor(task_id='wait_for_file', filepath='/tmp/temporary_file_for_testing')
    t7 = BashOperator(task_id='create_file_after_3_seconds', bash_command='sleep 3; touch /tmp/temporary_file_for_testing')
    t8 = PythonSensor(task_id='success_sensor_python', python_callable=success_callable)
    t9 = PythonSensor(task_id='failure_timeout_sensor_python', timeout=3, soft_fail=True, python_callable=failure_callable)
    t10 = DayOfWeekSensor(task_id='week_day_sensor_failing_on_timeout', timeout=3, soft_fail=True, week_day=WeekDay.MONDAY)
    tx = BashOperator(task_id='print_date_in_bash', bash_command='date')
    tx.trigger_rule = TriggerRule.NONE_FAILED
    [t0, t0a, t1, t1a, t2, t2a, t3, t4] >> tx
    t5 >> t6 >> tx
    t7 >> tx
    [t8, t9] >> tx
    t10 >> tx