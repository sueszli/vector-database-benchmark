"""Example DAG demonstrating the usage of the sensor decorator."""
from __future__ import annotations
import pendulum
from airflow.decorators import dag, task
from airflow.sensors.base import PokeReturnValue

@dag(schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example'])
def example_sensor_decorator():
    if False:
        for i in range(10):
            print('nop')

    @task.sensor(poke_interval=60, timeout=3600, mode='reschedule')
    def wait_for_upstream() -> PokeReturnValue:
        if False:
            print('Hello World!')
        return PokeReturnValue(is_done=True, xcom_value='xcom_value')

    @task
    def dummy_operator() -> None:
        if False:
            while True:
                i = 10
        pass
    wait_for_upstream() >> dummy_operator()
tutorial_etl_dag = example_sensor_decorator()