from __future__ import annotations
import json
from datetime import datetime
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.hooks.step_function import StepFunctionHook
from airflow.providers.amazon.aws.operators.step_function import StepFunctionGetExecutionOutputOperator, StepFunctionStartExecutionOperator
from airflow.providers.amazon.aws.sensors.step_function import StepFunctionExecutionSensor
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_step_functions'
ROLE_ARN_KEY = 'ROLE_ARN'
sys_test_context_task = SystemTestContextBuilder().add_variable(ROLE_ARN_KEY).build()
STATE_MACHINE_DEFINITION = {'StartAt': 'Wait', 'States': {'Wait': {'Type': 'Wait', 'Seconds': 7, 'Next': 'Success'}, 'Success': {'Type': 'Succeed'}}}

@task
def create_state_machine(env_id, role_arn):
    if False:
        print('Hello World!')
    return StepFunctionHook().get_conn().create_state_machine(name=f'{DAG_ID}_{env_id}', definition=json.dumps(STATE_MACHINE_DEFINITION), roleArn=role_arn)['stateMachineArn']

@task
def delete_state_machine(state_machine_arn):
    if False:
        i = 10
        return i + 15
    StepFunctionHook().get_conn().delete_state_machine(stateMachineArn=state_machine_arn)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    role_arn = test_context[ROLE_ARN_KEY]
    state_machine_arn = create_state_machine(env_id, role_arn)
    start_execution = StepFunctionStartExecutionOperator(task_id='start_execution', state_machine_arn=state_machine_arn)
    execution_arn = start_execution.output
    wait_for_execution = StepFunctionExecutionSensor(task_id='wait_for_execution', execution_arn=execution_arn)
    wait_for_execution.poke_interval = 1
    get_execution_output = StepFunctionGetExecutionOutputOperator(task_id='get_execution_output', execution_arn=execution_arn)
    chain(test_context, state_machine_arn, start_execution, wait_for_execution, get_execution_output, delete_state_machine(state_machine_arn))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)