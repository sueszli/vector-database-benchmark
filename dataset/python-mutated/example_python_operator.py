"""
Example DAG demonstrating the usage of the TaskFlow API to execute Python functions natively and within a
virtual environment.
"""
from __future__ import annotations
import logging
import sys
import time
from pprint import pprint
import pendulum
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.python import ExternalPythonOperator, PythonVirtualenvOperator, is_venv_installed
log = logging.getLogger(__name__)
PATH_TO_PYTHON_BINARY = sys.executable

def x():
    if False:
        i = 10
        return i + 15
    pass
with DAG(dag_id='example_python_operator', schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example']) as dag:

    @task(task_id='print_the_context')
    def print_context(ds=None, **kwargs):
        if False:
            print('Hello World!')
        'Print the Airflow context and ds variable from the context.'
        pprint(kwargs)
        print(ds)
        return 'Whatever you return gets printed in the logs'
    run_this = print_context()

    @task(task_id='log_sql_query', templates_dict={'query': 'sql/sample.sql'}, templates_exts=['.sql'])
    def log_sql(**kwargs):
        if False:
            while True:
                i = 10
        logging.info('Python task decorator query: %s', str(kwargs['templates_dict']['query']))
    log_the_sql = log_sql()
    for i in range(5):

        @task(task_id=f'sleep_for_{i}')
        def my_sleeping_function(random_base):
            if False:
                while True:
                    i = 10
            'This is a function that will run within the DAG execution'
            time.sleep(random_base)
        sleeping_task = my_sleeping_function(random_base=i / 10)
        run_this >> log_the_sql >> sleeping_task
    if not is_venv_installed():
        log.warning('The virtalenv_python example task requires virtualenv, please install it.')
    else:

        @task.virtualenv(task_id='virtualenv_python', requirements=['colorama==0.4.0'], system_site_packages=False)
        def callable_virtualenv():
            if False:
                for i in range(10):
                    print('nop')
            '\n            Example function that will be performed in a virtual environment.\n\n            Importing at the module level ensures that it will not attempt to import the\n            library before it is installed.\n            '
            from time import sleep
            from colorama import Back, Fore, Style
            print(Fore.RED + 'some red text')
            print(Back.GREEN + 'and with a green background')
            print(Style.DIM + 'and in dim text')
            print(Style.RESET_ALL)
            for _ in range(4):
                print(Style.DIM + 'Please wait...', flush=True)
                sleep(1)
            print('Finished')
        virtualenv_task = callable_virtualenv()
        sleeping_task >> virtualenv_task

        @task.external_python(task_id='external_python', python=PATH_TO_PYTHON_BINARY)
        def callable_external_python():
            if False:
                for i in range(10):
                    print('nop')
            '\n            Example function that will be performed in a virtual environment.\n\n            Importing at the module level ensures that it will not attempt to import the\n            library before it is installed.\n            '
            import sys
            from time import sleep
            print(f'Running task via {sys.executable}')
            print('Sleeping')
            for _ in range(4):
                print('Please wait...', flush=True)
                sleep(1)
            print('Finished')
        external_python_task = callable_external_python()
        external_classic = ExternalPythonOperator(task_id='external_python_classic', python=PATH_TO_PYTHON_BINARY, python_callable=x)
        virtual_classic = PythonVirtualenvOperator(task_id='virtualenv_classic', requirements='colorama==0.4.0', python_callable=x)
        run_this >> external_classic >> external_python_task >> virtual_classic