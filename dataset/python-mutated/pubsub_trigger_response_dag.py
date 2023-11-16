""" Two Airflow DAGs that demonstrate the mechanism of triggering DAGs with Pub/Sub messages

    Usage: Replace <PROJECT_ID> with the project ID of your project
"""
from __future__ import annotations
from datetime import datetime
import time
from airflow import DAG
from airflow import XComArg
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.google.cloud.operators.pubsub import PubSubCreateSubscriptionOperator, PubSubPullOperator
PROJECT_ID = '<PROJECT_ID>'
TOPIC_ID = 'dag-topic-trigger'
SUBSCRIPTION = 'trigger_dag_subscription'

def handle_messages(pulled_messages, context):
    if False:
        print('Hello World!')
    dag_ids = list()
    for (idx, m) in enumerate(pulled_messages):
        data = m.message.data.decode('utf-8')
        print(f'message {idx} data is {data}')
        dag_ids.append(data)
    return dag_ids
with DAG('trigger_dag', start_date=datetime(2021, 1, 1), schedule_interval='* * * * *', max_active_runs=1, catchup=False) as trigger_dag:
    subscribe_task = PubSubCreateSubscriptionOperator(task_id='subscribe_task', project_id=PROJECT_ID, topic=TOPIC_ID, subscription=SUBSCRIPTION)
    subscription = subscribe_task.output
    pull_messages_operator = PubSubPullOperator(task_id='pull_messages_operator', project_id=PROJECT_ID, ack_messages=True, messages_callback=handle_messages, subscription=subscription, max_messages=50)
    trigger_target_dag = TriggerDagRunOperator.partial(task_id='trigger_target').expand(trigger_dag_id=XComArg(pull_messages_operator))
    subscribe_task >> pull_messages_operator >> trigger_target_dag

def _some_heavy_task():
    if False:
        return 10
    print('Do some operation...')
    time.sleep(1)
    print('Done!')
with DAG('target_dag', start_date=datetime(2022, 1, 1), schedule_interval=None, catchup=False) as target_dag:
    some_heavy_task = PythonOperator(task_id='some_heavy_task', python_callable=_some_heavy_task)
    some_heavy_task