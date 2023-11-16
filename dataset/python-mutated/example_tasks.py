"""
Example Airflow DAG that creates and deletes Queues and creates, gets, lists,
runs and deletes Tasks in the Google Cloud Tasks service in the Google Cloud.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
from google.api_core.retry import Retry
from google.cloud.tasks_v2.types import Queue
from google.protobuf import timestamp_pb2
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.tasks import CloudTasksQueueCreateOperator, CloudTasksQueueDeleteOperator, CloudTasksTaskCreateOperator, CloudTasksTaskDeleteOperator, CloudTasksTaskGetOperator, CloudTasksTaskRunOperator, CloudTasksTasksListOperator
from airflow.utils.trigger_rule import TriggerRule
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'cloud_tasks_tasks'
timestamp = timestamp_pb2.Timestamp()
timestamp.FromDatetime(datetime.now() + timedelta(hours=12))
LOCATION = 'us-central1'
QUEUE_ID = f"queue-{ENV_ID}-{DAG_ID.replace('_', '-')}"
TASK_NAME = 'task-to-run'
TASK = {'http_request': {'http_method': 'POST', 'url': 'http://www.example.com/example', 'body': b''}, 'schedule_time': timestamp}
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['example', 'tasks']) as dag:

    @task(task_id='random_string')
    def generate_random_string():
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate random string for queue and task names.\n        Queue name cannot be repeated in preceding 7 days and\n        task name in the last 1 hour.\n        '
        import random
        import string
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    random_string = generate_random_string()
    create_queue = CloudTasksQueueCreateOperator(location=LOCATION, task_queue=Queue(stackdriver_logging_config=dict(sampling_ratio=0.5)), queue_name=QUEUE_ID + "{{ task_instance.xcom_pull(task_ids='random_string') }}", retry=Retry(maximum=10.0), timeout=5, task_id='create_queue')
    delete_queue = CloudTasksQueueDeleteOperator(location=LOCATION, queue_name=QUEUE_ID + "{{ task_instance.xcom_pull(task_ids='random_string') }}", task_id='delete_queue')
    delete_queue.trigger_rule = TriggerRule.ALL_DONE
    create_task = CloudTasksTaskCreateOperator(location=LOCATION, queue_name=QUEUE_ID + "{{ task_instance.xcom_pull(task_ids='random_string') }}", task=TASK, task_name=TASK_NAME + "{{ task_instance.xcom_pull(task_ids='random_string') }}", retry=Retry(maximum=10.0), timeout=5, task_id='create_task_to_run')
    tasks_get = CloudTasksTaskGetOperator(location=LOCATION, queue_name=QUEUE_ID + "{{ task_instance.xcom_pull(task_ids='random_string') }}", task_name=TASK_NAME + "{{ task_instance.xcom_pull(task_ids='random_string') }}", task_id='tasks_get')
    run_task = CloudTasksTaskRunOperator(location=LOCATION, queue_name=QUEUE_ID + "{{ task_instance.xcom_pull(task_ids='random_string') }}", task_name=TASK_NAME + "{{ task_instance.xcom_pull(task_ids='random_string') }}", retry=Retry(maximum=10.0), task_id='run_task')
    list_tasks = CloudTasksTasksListOperator(location=LOCATION, queue_name=QUEUE_ID + "{{ task_instance.xcom_pull(task_ids='random_string') }}", task_id='list_tasks')
    delete_task = CloudTasksTaskDeleteOperator(location=LOCATION, queue_name=QUEUE_ID + "{{ task_instance.xcom_pull(task_ids='random_string') }}", task_name=TASK_NAME + "{{ task_instance.xcom_pull(task_ids='random_string') }}", task_id='delete_task')
    chain(random_string, create_queue, create_task, tasks_get, list_tasks, run_task, delete_task, delete_queue)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)