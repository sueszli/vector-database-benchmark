"""Example DAG demonstrating the usage of the Classic branching Python operators.

It is showcasing the basic BranchPythonOperator and its sisters BranchExternalPythonOperator
and BranchPythonVirtualenvOperator."""
from __future__ import annotations
import random
import sys
import tempfile
from pathlib import Path
import pendulum
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchExternalPythonOperator, BranchPythonOperator, BranchPythonVirtualenvOperator, ExternalPythonOperator, PythonOperator, PythonVirtualenvOperator
from airflow.utils.edgemodifier import Label
from airflow.utils.trigger_rule import TriggerRule
PATH_TO_PYTHON_BINARY = sys.executable
with DAG(dag_id='example_branch_operator', start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, schedule='@daily', tags=['example', 'example2'], orientation='TB') as dag:
    run_this_first = EmptyOperator(task_id='run_this_first')
    options = ['a', 'b', 'c', 'd']
    branching = BranchPythonOperator(task_id='branching', python_callable=lambda : f'branch_{random.choice(options)}')
    run_this_first >> branching
    join = EmptyOperator(task_id='join', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    for option in options:
        t = PythonOperator(task_id=f'branch_{option}', python_callable=lambda : print('Hello World'))
        empty_follow = EmptyOperator(task_id='follow_' + option)
        branching >> Label(option) >> t >> empty_follow >> join

    def branch_with_external_python(choices):
        if False:
            for i in range(10):
                print('nop')
        import random
        return f'ext_py_{random.choice(choices)}'
    branching_ext_py = BranchExternalPythonOperator(task_id='branching_ext_python', python=PATH_TO_PYTHON_BINARY, python_callable=branch_with_external_python, op_args=[options])
    join >> branching_ext_py
    join_ext_py = EmptyOperator(task_id='join_ext_python', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    def hello_world_with_external_python():
        if False:
            print('Hello World!')
        print('Hello World from external Python')
    for option in options:
        t = ExternalPythonOperator(task_id=f'ext_py_{option}', python=PATH_TO_PYTHON_BINARY, python_callable=hello_world_with_external_python)
        branching_ext_py >> Label(option) >> t >> join_ext_py
    VENV_CACHE_PATH = Path(tempfile.gettempdir())

    def branch_with_venv(choices):
        if False:
            print('Hello World!')
        import random
        import numpy as np
        print(f'Some numpy stuff: {np.arange(6)}')
        return f'venv_{random.choice(choices)}'
    branching_venv = BranchPythonVirtualenvOperator(task_id='branching_venv', requirements=['numpy~=1.24.4'], venv_cache_path=VENV_CACHE_PATH, python_callable=branch_with_venv, op_args=[options])
    join_ext_py >> branching_venv
    join_venv = EmptyOperator(task_id='join_venv', trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    def hello_world_with_venv():
        if False:
            for i in range(10):
                print('nop')
        import numpy as np
        print(f'Hello World with some numpy stuff: {np.arange(6)}')
    for option in options:
        t = PythonVirtualenvOperator(task_id=f'venv_{option}', requirements=['numpy~=1.24.4'], venv_cache_path=VENV_CACHE_PATH, python_callable=hello_world_with_venv)
        branching_venv >> Label(option) >> t >> join_venv