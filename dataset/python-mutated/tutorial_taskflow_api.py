from __future__ import annotations
import json
import pendulum
from airflow.decorators import dag, task

@dag(schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example'])
def tutorial_taskflow_api():
    if False:
        print('Hello World!')
    '\n    ### TaskFlow API Tutorial Documentation\n    This is a simple data pipeline example which demonstrates the use of\n    the TaskFlow API using three simple tasks for Extract, Transform, and Load.\n    Documentation that goes along with the Airflow TaskFlow API tutorial is\n    located\n    [here](https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html)\n    '

    @task()
    def extract():
        if False:
            print('Hello World!')
        '\n        #### Extract task\n        A simple Extract task to get data ready for the rest of the data\n        pipeline. In this case, getting data is simulated by reading from a\n        hardcoded JSON string.\n        '
        data_string = '{"1001": 301.27, "1002": 433.21, "1003": 502.22}'
        order_data_dict = json.loads(data_string)
        return order_data_dict

    @task(multiple_outputs=True)
    def transform(order_data_dict: dict):
        if False:
            return 10
        '\n        #### Transform task\n        A simple Transform task which takes in the collection of order data and\n        computes the total order value.\n        '
        total_order_value = 0
        for value in order_data_dict.values():
            total_order_value += value
        return {'total_order_value': total_order_value}

    @task()
    def load(total_order_value: float):
        if False:
            for i in range(10):
                print('nop')
        '\n        #### Load task\n        A simple Load task which takes in the result of the Transform task and\n        instead of saving it to end user review, just prints it out.\n        '
        print(f'Total order value is: {total_order_value:.2f}')
    order_data = extract()
    order_summary = transform(order_data)
    load(order_summary['total_order_value'])
tutorial_taskflow_api()