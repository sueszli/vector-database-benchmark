from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class SSMMaintenanceWindowProperties(TypedDict):
    AllowUnassociatedTargets: Optional[bool]
    Cutoff: Optional[int]
    Duration: Optional[int]
    Name: Optional[str]
    Schedule: Optional[str]
    Description: Optional[str]
    EndDate: Optional[str]
    Id: Optional[str]
    ScheduleOffset: Optional[int]
    ScheduleTimezone: Optional[str]
    StartDate: Optional[str]
    Tags: Optional[list[Tag]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class SSMMaintenanceWindowProvider(ResourceProvider[SSMMaintenanceWindowProperties]):
    TYPE = 'AWS::SSM::MaintenanceWindow'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[SSMMaintenanceWindowProperties]) -> ProgressEvent[SSMMaintenanceWindowProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - AllowUnassociatedTargets\n          - Cutoff\n          - Schedule\n          - Duration\n          - Name\n\n\n\n        Read-only properties:\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        ssm_client = request.aws_client_factory.ssm
        params = util.select_attributes(model, ['AllowUnassociatedTargets', 'Cutoff', 'Duration', 'Name', 'Schedule', 'ScheduleOffset', 'ScheduleTimezone', 'StartDate', 'EndDate', 'Description', 'Tags'])
        response = ssm_client.create_maintenance_window(**params)
        model['Id'] = response['WindowId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[SSMMaintenanceWindowProperties]) -> ProgressEvent[SSMMaintenanceWindowProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[SSMMaintenanceWindowProperties]) -> ProgressEvent[SSMMaintenanceWindowProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        ssm_client = request.aws_client_factory.ssm
        ssm_client.delete_maintenance_window(WindowId=model['Id'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[SSMMaintenanceWindowProperties]) -> ProgressEvent[SSMMaintenanceWindowProperties]:
        if False:
            while True:
                i = 10
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError