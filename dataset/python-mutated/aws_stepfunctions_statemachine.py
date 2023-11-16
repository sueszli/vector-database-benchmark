from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import LOG, OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.strings import to_str

class StepFunctionsStateMachineProperties(TypedDict):
    RoleArn: Optional[str]
    Arn: Optional[str]
    Definition: Optional[dict]
    DefinitionS3Location: Optional[S3Location]
    DefinitionString: Optional[str]
    DefinitionSubstitutions: Optional[dict]
    LoggingConfiguration: Optional[LoggingConfiguration]
    Name: Optional[str]
    StateMachineName: Optional[str]
    StateMachineRevisionId: Optional[str]
    StateMachineType: Optional[str]
    Tags: Optional[list[TagsEntry]]
    TracingConfiguration: Optional[TracingConfiguration]

class CloudWatchLogsLogGroup(TypedDict):
    LogGroupArn: Optional[str]

class LogDestination(TypedDict):
    CloudWatchLogsLogGroup: Optional[CloudWatchLogsLogGroup]

class LoggingConfiguration(TypedDict):
    Destinations: Optional[list[LogDestination]]
    IncludeExecutionData: Optional[bool]
    Level: Optional[str]

class TracingConfiguration(TypedDict):
    Enabled: Optional[bool]

class S3Location(TypedDict):
    Bucket: Optional[str]
    Key: Optional[str]
    Version: Optional[str]

class TagsEntry(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class StepFunctionsStateMachineProvider(ResourceProvider[StepFunctionsStateMachineProperties]):
    TYPE = 'AWS::StepFunctions::StateMachine'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[StepFunctionsStateMachineProperties]) -> ProgressEvent[StepFunctionsStateMachineProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Arn\n\n        Required properties:\n          - RoleArn\n\n        Create-only properties:\n          - /properties/StateMachineName\n          - /properties/StateMachineType\n\n        Read-only properties:\n          - /properties/Arn\n          - /properties/Name\n          - /properties/StateMachineRevisionId\n\n        IAM permissions required:\n          - states:CreateStateMachine\n          - iam:PassRole\n          - s3:GetObject\n\n        '
        model = request.desired_state
        step_function = request.aws_client_factory.stepfunctions
        if not model.get('StateMachineName'):
            model['StateMachineName'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
        params = {'name': model.get('StateMachineName'), 'roleArn': model.get('RoleArn'), 'type': model.get('StateMachineType', 'STANDARD')}
        s3_client = request.aws_client_factory.s3
        definition_str = self._get_definition(model, s3_client)
        params['definition'] = definition_str
        response = step_function.create_state_machine(**params)
        model['Arn'] = response['stateMachineArn']
        model['Name'] = model['StateMachineName']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def _get_definition(self, model, s3_client):
        if False:
            while True:
                i = 10
        definition_str = model.get('DefinitionString')
        s3_location = model.get('DefinitionS3Location')
        if not definition_str and s3_location:
            LOG.debug('Fetching state machine definition from S3: %s', s3_location)
            result = s3_client.get_object(Bucket=s3_location['Bucket'], Key=s3_location['Key'])
            definition_str = to_str(result['Body'].read())
        substitutions = model.get('DefinitionSubstitutions')
        if substitutions is not None:
            definition_str = _apply_substitutions(definition_str, substitutions)
        return definition_str

    def read(self, request: ResourceRequest[StepFunctionsStateMachineProperties]) -> ProgressEvent[StepFunctionsStateMachineProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - states:DescribeStateMachine\n          - states:ListTagsForResource\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[StepFunctionsStateMachineProperties]) -> ProgressEvent[StepFunctionsStateMachineProperties]:
        if False:
            print('Hello World!')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - states:DeleteStateMachine\n          - states:DescribeStateMachine\n        '
        model = request.desired_state
        step_function = request.aws_client_factory.stepfunctions
        step_function.delete_state_machine(stateMachineArn=model['Arn'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[StepFunctionsStateMachineProperties]) -> ProgressEvent[StepFunctionsStateMachineProperties]:
        if False:
            while True:
                i = 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - states:UpdateStateMachine\n          - states:TagResource\n          - states:UntagResource\n          - states:ListTagsForResource\n          - iam:PassRole\n        '
        model = request.desired_state
        step_function = request.aws_client_factory.stepfunctions
        if not model.get('Arn'):
            model['Arn'] = request.previous_state['Arn']
        params = {'stateMachineArn': model['Arn'], 'definition': model['DefinitionString']}
        step_function.update_state_machine(**params)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

def _apply_substitutions(definition: str, substitutions: dict[str, str]) -> str:
    if False:
        for i in range(10):
            print('nop')
    substitution_regex = re.compile('\\${[a-zA-Z0-9_]+}')
    tokens = substitution_regex.findall(definition)
    result = definition
    for token in tokens:
        raw_token = token[2:-1]
        if raw_token not in substitutions.keys():
            raise
        result = result.replace(token, substitutions[raw_token])
    return result