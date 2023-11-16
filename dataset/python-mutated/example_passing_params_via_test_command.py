"""Example DAG demonstrating the usage of the params arguments in templated arguments."""
from __future__ import annotations
import datetime
import os
import textwrap
import pendulum
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

@task(task_id='run_this')
def my_py_command(params, test_mode=None, task=None):
    if False:
        print('Hello World!')
    '\n    Print out the "foo" param passed in via\n    `airflow tasks test example_passing_params_via_test_command run_this <date>\n    -t \'{"foo":"bar"}\'`\n    '
    if test_mode:
        print(f" 'foo' was passed in via test={test_mode} command : kwargs[params][foo] = {task.params['foo']}")
    print(f" 'miff' was passed in via task params = {params['miff']}")
    return 1

@task(task_id='env_var_test_task')
def print_env_vars(test_mode=None):
    if False:
        i = 10
        return i + 15
    '\n    Print out the "foo" param passed in via\n    `airflow tasks test example_passing_params_via_test_command env_var_test_task <date>\n    --env-vars \'{"foo":"bar"}\'`\n    '
    if test_mode:
        print(f"foo={os.environ.get('foo')}")
        print(f"AIRFLOW_TEST_MODE={os.environ.get('AIRFLOW_TEST_MODE')}")
with DAG('example_passing_params_via_test_command', schedule='*/1 * * * *', start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, dagrun_timeout=datetime.timedelta(minutes=4), tags=['example']) as dag:
    run_this = my_py_command(params={'miff': 'agg'})
    my_command = textwrap.dedent('\n        echo "\'foo\' was passed in via Airflow CLI Test command with value \'$FOO\'"\n        echo "\'miff\' was passed in via BashOperator with value \'$MIFF\'"\n        ')
    also_run_this = BashOperator(task_id='also_run_this', bash_command=my_command, params={'miff': 'agg'}, env={'FOO': '{{ params.foo }}', 'MIFF': '{{ params.miff }}'})
    env_var_test_task = print_env_vars()
    run_this >> also_run_this