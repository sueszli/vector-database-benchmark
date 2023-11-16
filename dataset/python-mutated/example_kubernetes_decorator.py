from __future__ import annotations
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
with DAG(dag_id='example_kubernetes_decorator', schedule=None, start_date=datetime(2021, 1, 1), tags=['example', 'cncf', 'kubernetes'], catchup=False) as dag:

    @task.kubernetes(image='python:3.8-slim-buster', name='k8s_test', namespace='default', in_cluster=False, config_file='/path/to/.kube/config')
    def execute_in_k8s_pod():
        if False:
            for i in range(10):
                print('nop')
        import time
        print('Hello from k8s pod')
        time.sleep(2)

    @task.kubernetes(image='python:3.8-slim-buster', namespace='default', in_cluster=False)
    def print_pattern():
        if False:
            print('Hello World!')
        n = 5
        for i in range(n):
            for j in range(i + 1):
                print('* ', end='')
            print('\r')
    execute_in_k8s_pod_instance = execute_in_k8s_pod()
    print_pattern_instance = print_pattern()
    execute_in_k8s_pod_instance >> print_pattern_instance
from tests.system.utils import get_test_run
test_run = get_test_run(dag)