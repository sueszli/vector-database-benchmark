from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class Route53HealthCheckProperties(TypedDict):
    HealthCheckConfig: Optional[dict]
    HealthCheckId: Optional[str]
    HealthCheckTags: Optional[list[HealthCheckTag]]

class HealthCheckTag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class Route53HealthCheckProvider(ResourceProvider[Route53HealthCheckProperties]):
    TYPE = 'AWS::Route53::HealthCheck'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[Route53HealthCheckProperties]) -> ProgressEvent[Route53HealthCheckProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/HealthCheckId\n\n        Required properties:\n          - HealthCheckConfig\n\n        Create-only properties:\n          - /properties/HealthCheckConfig/Type\n          - /properties/HealthCheckConfig/MeasureLatency\n          - /properties/HealthCheckConfig/RequestInterval\n\n        Read-only properties:\n          - /properties/HealthCheckId\n\n        IAM permissions required:\n          - route53:CreateHealthCheck\n          - route53:ChangeTagsForResource\n          - cloudwatch:DescribeAlarms\n          - route53-recovery-control-config:DescribeRoutingControl\n\n        '
        model = request.desired_state
        create_params = util.select_attributes(model, ['HealthCheckConfig', 'CallerReference'])
        if not create_params.get('CallerReference'):
            create_params['CallerReference'] = util.generate_default_name_without_stack(request.logical_resource_id)
        result = request.aws_client_factory.route53.create_health_check(**create_params)
        model['HealthCheckId'] = result['HealthCheck']['Id']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[Route53HealthCheckProperties]) -> ProgressEvent[Route53HealthCheckProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - route53:GetHealthCheck\n          - route53:ListTagsForResource\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[Route53HealthCheckProperties]) -> ProgressEvent[Route53HealthCheckProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n\n        IAM permissions required:\n          - route53:DeleteHealthCheck\n        '
        model = request.desired_state
        request.aws_client_factory.route53.delete_health_check(HealthCheckId=model['HealthCheckId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[Route53HealthCheckProperties]) -> ProgressEvent[Route53HealthCheckProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Update a resource\n\n        IAM permissions required:\n          - route53:UpdateHealthCheck\n          - route53:ChangeTagsForResource\n          - route53:ListTagsForResource\n          - cloudwatch:DescribeAlarms\n        '
        raise NotImplementedError