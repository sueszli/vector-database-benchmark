from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class SSMMaintenanceWindowTargetProperties(TypedDict):
    ResourceType: Optional[str]
    Targets: Optional[list[Targets]]
    WindowId: Optional[str]
    Description: Optional[str]
    Id: Optional[str]
    Name: Optional[str]
    OwnerInformation: Optional[str]

class Targets(TypedDict):
    Key: Optional[str]
    Values: Optional[list[str]]
REPEATED_INVOCATION = 'repeated_invocation'

class SSMMaintenanceWindowTargetProvider(ResourceProvider[SSMMaintenanceWindowTargetProperties]):
    TYPE = 'AWS::SSM::MaintenanceWindowTarget'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[SSMMaintenanceWindowTargetProperties]) -> ProgressEvent[SSMMaintenanceWindowTargetProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - WindowId\n          - ResourceType\n          - Targets\n\n        Create-only properties:\n          - /properties/WindowId\n\n        Read-only properties:\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        ssm = request.aws_client_factory.ssm
        params = util.select_attributes(model=model, params=['Description', 'Name', 'OwnerInformation', 'ResourceType', 'Targets', 'WindowId'])
        response = ssm.register_target_with_maintenance_window(**params)
        model['Id'] = response['WindowTargetId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[SSMMaintenanceWindowTargetProperties]) -> ProgressEvent[SSMMaintenanceWindowTargetProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[SSMMaintenanceWindowTargetProperties]) -> ProgressEvent[SSMMaintenanceWindowTargetProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        ssm = request.aws_client_factory.ssm
        ssm.deregister_target_from_maintenance_window(WindowId=model['WindowId'], WindowTargetId=model['Id'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[SSMMaintenanceWindowTargetProperties]) -> ProgressEvent[SSMMaintenanceWindowTargetProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError