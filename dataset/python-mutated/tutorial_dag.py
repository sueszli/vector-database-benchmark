"""
### DAG Tutorial Documentation
This DAG is demonstrating an Extract -> Transform -> Load pipeline
"""
from __future__ import annotations
import json
import textwrap
import pendulum
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
with DAG('tutorial_dag', default_args={'retries': 2}, description='DAG tutorial', schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example']) as dag:
    dag.doc_md = __doc__

    def extract(**kwargs):
        if False:
            return 10
        ti = kwargs['ti']
        data_string = '{"1001": 301.27, "1002": 433.21, "1003": 502.22}'
        ti.xcom_push('order_data', data_string)

    def transform(**kwargs):
        if False:
            i = 10
            return i + 15
        ti = kwargs['ti']
        extract_data_string = ti.xcom_pull(task_ids='extract', key='order_data')
        order_data = json.loads(extract_data_string)
        total_order_value = 0
        for value in order_data.values():
            total_order_value += value
        total_value = {'total_order_value': total_order_value}
        total_value_json_string = json.dumps(total_value)
        ti.xcom_push('total_order_value', total_value_json_string)

    def load(**kwargs):
        if False:
            for i in range(10):
                print('nop')
        ti = kwargs['ti']
        total_value_string = ti.xcom_pull(task_ids='transform', key='total_order_value')
        total_order_value = json.loads(total_value_string)
        print(total_order_value)
    extract_task = PythonOperator(task_id='extract', python_callable=extract)
    extract_task.doc_md = textwrap.dedent('    #### Extract task\n    A simple Extract task to get data ready for the rest of the data pipeline.\n    In this case, getting data is simulated by reading from a hardcoded JSON string.\n    This data is then put into xcom, so that it can be processed by the next task.\n    ')
    transform_task = PythonOperator(task_id='transform', python_callable=transform)
    transform_task.doc_md = textwrap.dedent('    #### Transform task\n    A simple Transform task which takes in the collection of order data from xcom\n    and computes the total order value.\n    This computed value is then put into xcom, so that it can be processed by the next task.\n    ')
    load_task = PythonOperator(task_id='load', python_callable=load)
    load_task.doc_md = textwrap.dedent('    #### Load task\n    A simple Load task which takes in the result of the Transform task, by reading it\n    from xcom and instead of saving it to end user review, just prints it out.\n    ')
    extract_task >> transform_task >> load_task