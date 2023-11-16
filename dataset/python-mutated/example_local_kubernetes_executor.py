"""
This is an example dag for using a Local Kubernetes Executor Configuration.
"""
from __future__ import annotations
import logging
from datetime import datetime
from airflow.configuration import conf
from airflow.decorators import task
from airflow.example_dags.libs.helper import print_stuff
from airflow.models.dag import DAG
log = logging.getLogger(__name__)
worker_container_repository = conf.get('kubernetes_executor', 'worker_container_repository')
worker_container_tag = conf.get('kubernetes_executor', 'worker_container_tag')
try:
    from kubernetes.client import models as k8s
except ImportError:
    log.warning('Could not import DAGs in example_local_kubernetes_executor.py', exc_info=True)
    log.warning('Install Kubernetes dependencies with: pip install apache-airflow[cncf.kubernetes]')
    k8s = None
if k8s:
    with DAG(dag_id='example_local_kubernetes_executor', schedule=None, start_date=datetime(2021, 1, 1), catchup=False, tags=['example3']) as dag:
        start_task_executor_config = {'pod_override': k8s.V1Pod(metadata=k8s.V1ObjectMeta(annotations={'test': 'annotation'}))}

        @task(executor_config=start_task_executor_config, queue='kubernetes', task_id='task_with_kubernetes_executor')
        def task_with_template():
            if False:
                return 10
            print_stuff()

        @task(task_id='task_with_local_executor')
        def task_with_local(ds=None, **kwargs):
            if False:
                print('Hello World!')
            'Print the Airflow context and ds variable from the context.'
            print(kwargs)
            print(ds)
            return 'Whatever you return gets printed in the logs'
        task_with_local() >> task_with_template()