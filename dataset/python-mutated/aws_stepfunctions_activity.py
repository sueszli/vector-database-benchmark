from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class StepFunctionsActivityProperties(TypedDict):
    Name: Optional[str]
    Arn: Optional[str]
    Tags: Optional[list[TagsEntry]]

class TagsEntry(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class StepFunctionsActivityProvider(ResourceProvider[StepFunctionsActivityProperties]):
    TYPE = 'AWS::StepFunctions::Activity'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[StepFunctionsActivityProperties]) -> ProgressEvent[StepFunctionsActivityProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Arn\n\n        Required properties:\n          - Name\n\n        Create-only properties:\n          - /properties/Name\n\n        Read-only properties:\n          - /properties/Arn\n\n        IAM permissions required:\n          - states:CreateActivity\n\n        '
        model = request.desired_state
        step_functions = request.aws_client_factory.stepfunctions
        if not model.get('Tags'):
            response = step_functions.create_activity(name=model['Name'])
        else:
            response = step_functions.create_activity(name=model['Name'], tags=model['Tags'])
        model['Arn'] = response['activityArn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[StepFunctionsActivityProperties]) -> ProgressEvent[StepFunctionsActivityProperties]:
        if False:
            return 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - states:DescribeActivity\n          - states:ListTagsForResource\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[StepFunctionsActivityProperties]) -> ProgressEvent[StepFunctionsActivityProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - states:DeleteActivity\n        '
        model = request.desired_state
        step_functions = request.aws_client_factory.stepfunctions
        step_functions.delete_activity(activityArn=model['Arn'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[StepFunctionsActivityProperties]) -> ProgressEvent[StepFunctionsActivityProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n\n        IAM permissions required:\n          - states:ListTagsForResource\n          - states:TagResource\n          - states:UntagResource\n        '
        raise NotImplementedError