from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class IAMServiceLinkedRoleProperties(TypedDict):
    AWSServiceName: Optional[str]
    CustomSuffix: Optional[str]
    Description: Optional[str]
    Id: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class IAMServiceLinkedRoleProvider(ResourceProvider[IAMServiceLinkedRoleProperties]):
    TYPE = 'AWS::IAM::ServiceLinkedRole'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[IAMServiceLinkedRoleProperties]) -> ProgressEvent[IAMServiceLinkedRoleProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - AWSServiceName\n\n        Create-only properties:\n          - /properties/CustomSuffix\n          - /properties/AWSServiceName\n\n        Read-only properties:\n          - /properties/Id\n\n        '
        model = request.desired_state
        response = request.aws_client_factory.iam.create_service_linked_role(**model)
        model['Id'] = response['Role']['RoleName']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[IAMServiceLinkedRoleProperties]) -> ProgressEvent[IAMServiceLinkedRoleProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[IAMServiceLinkedRoleProperties]) -> ProgressEvent[IAMServiceLinkedRoleProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n        '
        request.aws_client_factory.iam.delete_service_linked_role(RoleName=request.previous_state['Id'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[IAMServiceLinkedRoleProperties]) -> ProgressEvent[IAMServiceLinkedRoleProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError