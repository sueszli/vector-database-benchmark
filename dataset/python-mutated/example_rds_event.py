from __future__ import annotations
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.rds import RdsCreateDbInstanceOperator, RdsCreateEventSubscriptionOperator, RdsDeleteDbInstanceOperator, RdsDeleteEventSubscriptionOperator
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_rds_event'
sys_test_context_task = SystemTestContextBuilder().build()

@task
def create_sns_topic(env_id) -> str:
    if False:
        print('Hello World!')
    return boto3.client('sns').create_topic(Name=f'{env_id}-topic')['TopicArn']

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_sns_topic(topic_arn) -> None:
    if False:
        while True:
            i = 10
    boto3.client('sns').delete_topic(TopicArn=topic_arn)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    rds_db_name = f'{test_context[ENV_ID_KEY]}_db'
    rds_instance_name = f'{test_context[ENV_ID_KEY]}-instance'
    rds_subscription_name = f'{test_context[ENV_ID_KEY]}-subscription'
    sns_topic = create_sns_topic(test_context[ENV_ID_KEY])
    create_db_instance = RdsCreateDbInstanceOperator(task_id='create_db_instance', db_instance_identifier=rds_instance_name, db_instance_class='db.t4g.micro', engine='postgres', rds_kwargs={'MasterUsername': 'rds_username', 'MasterUserPassword': 'rds_password', 'AllocatedStorage': 20, 'DBName': rds_db_name, 'PubliclyAccessible': False})
    create_subscription = RdsCreateEventSubscriptionOperator(task_id='create_subscription', subscription_name=rds_subscription_name, sns_topic_arn=sns_topic, source_type='db-instance', source_ids=[rds_instance_name], event_categories=['availability'])
    delete_subscription = RdsDeleteEventSubscriptionOperator(task_id='delete_subscription', subscription_name=rds_subscription_name)
    delete_subscription.trigger_rule = TriggerRule.ALL_DONE
    delete_db_instance = RdsDeleteDbInstanceOperator(task_id='delete_db_instance', db_instance_identifier=rds_instance_name, rds_kwargs={'SkipFinalSnapshot': True}, trigger_rule=TriggerRule.ALL_DONE)
    chain(test_context, sns_topic, create_db_instance, create_subscription, delete_subscription, delete_db_instance, delete_sns_topic(sns_topic))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)