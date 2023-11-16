from __future__ import annotations
import logging
from datetime import datetime
from airflow.decorators import dag, task
from airflow.operators.python import is_venv_installed
log = logging.getLogger(__name__)
if not is_venv_installed():
    log.warning('The tutorial_taskflow_api_virtualenv example DAG requires virtualenv, please install it.')
else:

    @dag(schedule=None, start_date=datetime(2021, 1, 1), catchup=False, tags=['example'])
    def tutorial_taskflow_api_virtualenv():
        if False:
            for i in range(10):
                print('nop')
        '\n        ### TaskFlow API example using virtualenv\n        This is a simple data pipeline example which demonstrates the use of\n        the TaskFlow API using three simple tasks for Extract, Transform, and Load.\n        '

        @task.virtualenv(use_dill=True, system_site_packages=False, requirements=['funcsigs'])
        def extract():
            if False:
                i = 10
                return i + 15
            '\n            #### Extract task\n            A simple Extract task to get data ready for the rest of the data\n            pipeline. In this case, getting data is simulated by reading from a\n            hardcoded JSON string.\n            '
            import json
            data_string = '{"1001": 301.27, "1002": 433.21, "1003": 502.22}'
            order_data_dict = json.loads(data_string)
            return order_data_dict

        @task(multiple_outputs=True)
        def transform(order_data_dict: dict):
            if False:
                for i in range(10):
                    print('nop')
            '\n            #### Transform task\n            A simple Transform task which takes in the collection of order data and\n            computes the total order value.\n            '
            total_order_value = 0
            for value in order_data_dict.values():
                total_order_value += value
            return {'total_order_value': total_order_value}

        @task()
        def load(total_order_value: float):
            if False:
                print('Hello World!')
            '\n            #### Load task\n            A simple Load task which takes in the result of the Transform task and\n            instead of saving it to end user review, just prints it out.\n            '
            print(f'Total order value is: {total_order_value:.2f}')
        order_data = extract()
        order_summary = transform(order_data)
        load(order_summary['total_order_value'])
    tutorial_dag = tutorial_taskflow_api_virtualenv()