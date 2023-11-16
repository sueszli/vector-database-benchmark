from __future__ import annotations
from datetime import datetime
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.sqs import SqsHook
from airflow.providers.amazon.aws.operators.sqs import SqsPublishOperator
from airflow.providers.amazon.aws.sensors.sqs import SqsSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import SystemTestContextBuilder
sys_test_context_task = SystemTestContextBuilder().build()
DAG_ID = 'example_sqs'

@task
def create_queue(queue_name) -> str:
    if False:
        for i in range(10):
            print('nop')
    return SqsHook().create_queue(queue_name=queue_name)['QueueUrl']

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_queue(queue_url):
    if False:
        i = 10
        return i + 15
    SqsHook().conn.delete_queue(QueueUrl=queue_url)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context['ENV_ID']
    sns_queue_name = f'{env_id}-example-queue'
    sqs_queue = create_queue(sns_queue_name)
    publish_to_queue_1 = SqsPublishOperator(task_id='publish_to_queue_1', sqs_queue=sqs_queue, message_content='{{ task_instance }}-{{ logical_date }}')
    publish_to_queue_2 = SqsPublishOperator(task_id='publish_to_queue_2', sqs_queue=sqs_queue, message_content='{{ task_instance }}-{{ logical_date }}')
    read_from_queue = SqsSensor(task_id='read_from_queue', sqs_queue=sqs_queue)
    read_from_queue_in_batch = SqsSensor(task_id='read_from_queue_in_batch', sqs_queue=sqs_queue, max_messages=10, num_batches=3)
    chain(test_context, sqs_queue, publish_to_queue_1, read_from_queue, publish_to_queue_2, read_from_queue_in_batch, delete_queue(sqs_queue))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)