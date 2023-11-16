from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class SSMMaintenanceWindowTaskProperties(TypedDict):
    Priority: Optional[int]
    TaskArn: Optional[str]
    TaskType: Optional[str]
    WindowId: Optional[str]
    CutoffBehavior: Optional[str]
    Description: Optional[str]
    Id: Optional[str]
    LoggingInfo: Optional[LoggingInfo]
    MaxConcurrency: Optional[str]
    MaxErrors: Optional[str]
    Name: Optional[str]
    ServiceRoleArn: Optional[str]
    Targets: Optional[list[Target]]
    TaskInvocationParameters: Optional[TaskInvocationParameters]
    TaskParameters: Optional[dict]

class Target(TypedDict):
    Key: Optional[str]
    Values: Optional[list[str]]

class MaintenanceWindowStepFunctionsParameters(TypedDict):
    Input: Optional[str]
    Name: Optional[str]

class CloudWatchOutputConfig(TypedDict):
    CloudWatchLogGroupName: Optional[str]
    CloudWatchOutputEnabled: Optional[bool]

class NotificationConfig(TypedDict):
    NotificationArn: Optional[str]
    NotificationEvents: Optional[list[str]]
    NotificationType: Optional[str]

class MaintenanceWindowRunCommandParameters(TypedDict):
    CloudWatchOutputConfig: Optional[CloudWatchOutputConfig]
    Comment: Optional[str]
    DocumentHash: Optional[str]
    DocumentHashType: Optional[str]
    DocumentVersion: Optional[str]
    NotificationConfig: Optional[NotificationConfig]
    OutputS3BucketName: Optional[str]
    OutputS3KeyPrefix: Optional[str]
    Parameters: Optional[dict]
    ServiceRoleArn: Optional[str]
    TimeoutSeconds: Optional[int]

class MaintenanceWindowLambdaParameters(TypedDict):
    ClientContext: Optional[str]
    Payload: Optional[str]
    Qualifier: Optional[str]

class MaintenanceWindowAutomationParameters(TypedDict):
    DocumentVersion: Optional[str]
    Parameters: Optional[dict]

class TaskInvocationParameters(TypedDict):
    MaintenanceWindowAutomationParameters: Optional[MaintenanceWindowAutomationParameters]
    MaintenanceWindowLambdaParameters: Optional[MaintenanceWindowLambdaParameters]
    MaintenanceWindowRunCommandParameters: Optional[MaintenanceWindowRunCommandParameters]
    MaintenanceWindowStepFunctionsParameters: Optional[MaintenanceWindowStepFunctionsParameters]

class LoggingInfo(TypedDict):
    Region: Optional[str]
    S3Bucket: Optional[str]
    S3Prefix: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class SSMMaintenanceWindowTaskProvider(ResourceProvider[SSMMaintenanceWindowTaskProperties]):
    TYPE = 'AWS::SSM::MaintenanceWindowTask'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[SSMMaintenanceWindowTaskProperties]) -> ProgressEvent[SSMMaintenanceWindowTaskProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - WindowId\n          - Priority\n          - TaskType\n          - TaskArn\n\n        Create-only properties:\n          - /properties/WindowId\n          - /properties/TaskType\n\n        Read-only properties:\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        ssm = request.aws_client_factory.ssm
        params = util.select_attributes(model=model, params=['Description', 'Name', 'OwnerInformation', 'Priority', 'ServiceRoleArn', 'Targets', 'TaskArn', 'TaskParameters', 'TaskType', 'WindowId'])
        if (invocation_params := model.get('TaskInvocationParameters')):
            task_type_map = {'MaintenanceWindowAutomationParameters': 'Automation', 'MaintenanceWindowLambdaParameters': 'Lambda', 'MaintenanceWindowRunCommandParameters': 'RunCommand', 'MaintenanceWindowStepFunctionsParameters': 'StepFunctions'}
            params['TaskInvocationParameters'] = {task_type_map[k]: v for (k, v) in invocation_params.items()}
        response = ssm.register_task_with_maintenance_window(**params)
        model['Id'] = response['WindowTaskId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[SSMMaintenanceWindowTaskProperties]) -> ProgressEvent[SSMMaintenanceWindowTaskProperties]:
        if False:
            return 10
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[SSMMaintenanceWindowTaskProperties]) -> ProgressEvent[SSMMaintenanceWindowTaskProperties]:
        if False:
            while True:
                i = 10
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        ssm = request.aws_client_factory.ssm
        ssm.deregister_task_from_maintenance_window(WindowId=model['WindowId'], WindowTaskId=model['Id'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[SSMMaintenanceWindowTaskProperties]) -> ProgressEvent[SSMMaintenanceWindowTaskProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError