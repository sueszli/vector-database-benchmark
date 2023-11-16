from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class IAMAccessKeyProperties(TypedDict):
    UserName: Optional[str]
    Id: Optional[str]
    SecretAccessKey: Optional[str]
    Serial: Optional[int]
    Status: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class IAMAccessKeyProvider(ResourceProvider[IAMAccessKeyProperties]):
    TYPE = 'AWS::IAM::AccessKey'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[IAMAccessKeyProperties]) -> ProgressEvent[IAMAccessKeyProperties]:
        if False:
            return 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - UserName\n\n        Create-only properties:\n          - /properties/UserName\n          - /properties/Serial\n\n        Read-only properties:\n          - /properties/SecretAccessKey\n          - /properties/Id\n\n        '
        model = request.desired_state
        iam_client = request.aws_client_factory.iam
        access_key = iam_client.create_access_key(UserName=model['UserName'])
        model['SecretAccessKey'] = access_key['AccessKey']['SecretAccessKey']
        model['Id'] = access_key['AccessKey']['AccessKeyId']
        if model.get('Status') == 'Inactive':
            iam_client.update_access_key(AccessKeyId=model['Id'], UserName=model['UserName'], Status=model['Status'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[IAMAccessKeyProperties]) -> ProgressEvent[IAMAccessKeyProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[IAMAccessKeyProperties]) -> ProgressEvent[IAMAccessKeyProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n        '
        iam_client = request.aws_client_factory.iam
        model = request.previous_state
        iam_client.delete_access_key(AccessKeyId=model['Id'], UserName=model['UserName'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[IAMAccessKeyProperties]) -> ProgressEvent[IAMAccessKeyProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n        '
        iam_client = request.aws_client_factory.iam
        user_name_changed = request.desired_state['UserName'] != request.previous_state['UserName']
        serial_changed = request.desired_state['Serial'] != request.previous_state['Serial']
        if user_name_changed or serial_changed:
            self.delete(request)
            create_event = self.create(request)
            return create_event
        iam_client.update_access_key(AccessKeyId=request.previous_state['Id'], UserName=request.previous_state['UserName'], Status=request.desired_state['Status'])
        old_model = request.previous_state
        old_model['Status'] = request.desired_state['Status']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=old_model)