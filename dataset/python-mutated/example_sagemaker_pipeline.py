from __future__ import annotations
from datetime import datetime
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerStartPipelineOperator, SageMakerStopPipelineOperator
from airflow.providers.amazon.aws.sensors.sagemaker import SageMakerPipelineSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder
DAG_ID = 'example_sagemaker_pipeline'
ROLE_ARN_KEY = 'ROLE_ARN'
sys_test_context_task = SystemTestContextBuilder().add_variable(ROLE_ARN_KEY).build()

@task
def create_pipeline(name: str, role_arn: str):
    if False:
        while True:
            i = 10
    pipeline_json_definition = '{"Version": "2020-12-01", "Metadata": {}, "Parameters": [], "PipelineExperimentConfig": {"ExperimentName": {"Get": "Execution.PipelineName"}, "TrialName": {"Get": "Execution.PipelineExecutionId"}}, "Steps": [{"Name": "DummyCond29", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond28", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond27", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond26", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond25", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond24", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond23", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond22", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond21", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond20", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond19", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond18", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond17", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond16", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond15", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond14", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond13", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond12", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond11", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond10", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond9", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond8", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond7", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond6", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond5", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond4", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond3", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond2", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond1", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond0", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [{"Name": "DummyCond", "Type": "Condition", "Arguments": {"Conditions": [{"Type": "LessThanOrEqualTo", "LeftValue": 3.0, "RightValue": 6.0}], "IfSteps": [], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}], "ElseSteps": []}}]}'
    sgmk_client = boto3.client('sagemaker')
    sgmk_client.create_pipeline(PipelineName=name, PipelineDefinition=pipeline_json_definition, RoleArn=role_arn)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_pipeline(name: str):
    if False:
        print('Hello World!')
    sgmk_client = boto3.client('sagemaker')
    sgmk_client.delete_pipeline(PipelineName=name)
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    pipeline_name = f'{env_id}-pipeline'
    create_pipeline = create_pipeline(pipeline_name, test_context[ROLE_ARN_KEY])
    start_pipeline1 = SageMakerStartPipelineOperator(task_id='start_pipeline1', pipeline_name=pipeline_name)
    stop_pipeline1 = SageMakerStopPipelineOperator(task_id='stop_pipeline1', pipeline_exec_arn=start_pipeline1.output)
    start_pipeline2 = SageMakerStartPipelineOperator(task_id='start_pipeline2', pipeline_name=pipeline_name)
    await_pipeline2 = SageMakerPipelineSensor(task_id='await_pipeline2', pipeline_exec_arn=start_pipeline2.output)
    await_pipeline2.poke_interval = 10
    chain(test_context, create_pipeline, start_pipeline1, start_pipeline2, stop_pipeline1, await_pipeline2, delete_pipeline(pipeline_name))
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)