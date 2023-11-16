from __future__ import annotations
import json
import zipfile
from datetime import datetime
from io import BytesIO
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.lambda_function import LambdaCreateFunctionOperator, LambdaInvokeFunctionOperator
from airflow.providers.amazon.aws.sensors.lambda_function import LambdaFunctionStateSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder, prune_logs
DAG_ID = 'example_lambda'
ROLE_ARN_KEY = 'ROLE_ARN'
sys_test_context_task = SystemTestContextBuilder().add_variable(ROLE_ARN_KEY).build()
CODE_CONTENT = "\ndef test(*args):\n    print('Hello')\n"

def create_zip(content: str):
    if False:
        return 10
    with BytesIO() as zip_output:
        with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            info = zipfile.ZipInfo('lambda_function.py')
            info.external_attr = 511 << 16
            zip_file.writestr(info, content)
        zip_output.seek(0)
        return zip_output.read()

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_lambda(function_name: str):
    if False:
        print('Hello World!')
    client = boto3.client('lambda')
    client.delete_function(FunctionName=function_name)
with DAG(DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    lambda_function_name: str = f'{test_context[ENV_ID_KEY]}-function'
    role_arn = test_context[ROLE_ARN_KEY]
    create_lambda_function = LambdaCreateFunctionOperator(task_id='create_lambda_function', function_name=lambda_function_name, runtime='python3.9', role=role_arn, handler='lambda_function.test', code={'ZipFile': create_zip(CODE_CONTENT)})
    wait_lambda_function_state = LambdaFunctionStateSensor(task_id='wait_lambda_function_state', function_name=lambda_function_name)
    wait_lambda_function_state.poke_interval = 1
    invoke_lambda_function = LambdaInvokeFunctionOperator(task_id='invoke_lambda_function', function_name=lambda_function_name, payload=json.dumps({'SampleEvent': {'SampleData': {'Name': 'XYZ', 'DoB': '1993-01-01'}}}))
    log_cleanup = prune_logs([(f'/aws/lambda/{lambda_function_name}', None)], force_delete=True, retry=True)
    chain(test_context, create_lambda_function, wait_lambda_function_state, invoke_lambda_function, delete_lambda(lambda_function_name), log_cleanup)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)