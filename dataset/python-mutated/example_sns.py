from __future__ import annotations
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.sns import SnsPublishOperator
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import SystemTestContextBuilder
sys_test_context_task = SystemTestContextBuilder().build()
DAG_ID = 'example_sns'

@task
def create_topic(topic_name) -> str:
    if False:
        print('Hello World!')
    return boto3.client('sns').create_topic(Name=topic_name)['TopicArn']

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_topic(topic_arn) -> None:
    if False:
        i = 10
        return i + 15
    boto3.client('sns').delete_topic(TopicArn=topic_arn)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context['ENV_ID']
    sns_topic_name = f'{env_id}-test-topic'
    create_sns_topic = create_topic(sns_topic_name)
    publish_message = SnsPublishOperator(task_id='publish_message', target_arn=create_sns_topic, message='This is a sample message sent to SNS via an Apache Airflow DAG task.')
    chain(test_context, create_sns_topic, publish_message, delete_topic(create_sns_topic))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)