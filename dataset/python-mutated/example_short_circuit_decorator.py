"""Example DAG demonstrating the usage of the `@task.short_circuit()` TaskFlow decorator."""
from __future__ import annotations
import pendulum
from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

@dag(start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example'])
def example_short_circuit_decorator():
    if False:
        i = 10
        return i + 15

    @task.short_circuit()
    def check_condition(condition):
        if False:
            for i in range(10):
                print('nop')
        return condition
    ds_true = [EmptyOperator(task_id=f'true_{i}') for i in [1, 2]]
    ds_false = [EmptyOperator(task_id=f'false_{i}') for i in [1, 2]]
    condition_is_true = check_condition.override(task_id='condition_is_true')(condition=True)
    condition_is_false = check_condition.override(task_id='condition_is_false')(condition=False)
    chain(condition_is_true, *ds_true)
    chain(condition_is_false, *ds_false)
    [task_1, task_2, task_3, task_4, task_5, task_6] = [EmptyOperator(task_id=f'task_{i}') for i in range(1, 7)]
    task_7 = EmptyOperator(task_id='task_7', trigger_rule=TriggerRule.ALL_DONE)
    short_circuit = check_condition.override(task_id='short_circuit', ignore_downstream_trigger_rules=False)(condition=False)
    chain(task_1, [task_2, short_circuit], [task_3, task_4], [task_5, task_6], task_7)
example_dag = example_short_circuit_decorator()