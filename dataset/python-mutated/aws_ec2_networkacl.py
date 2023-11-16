from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EC2NetworkAclProperties(TypedDict):
    VpcId: Optional[str]
    Id: Optional[str]
    Tags: Optional[list[Tag]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2NetworkAclProvider(ResourceProvider[EC2NetworkAclProperties]):
    TYPE = 'AWS::EC2::NetworkAcl'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2NetworkAclProperties]) -> ProgressEvent[EC2NetworkAclProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - VpcId\n\n        Create-only properties:\n          - /properties/VpcId\n\n        Read-only properties:\n          - /properties/Id\n\n        IAM permissions required:\n          - ec2:CreateNetworkAcl\n          - ec2:DescribeNetworkAcls\n\n        '
        model = request.desired_state
        create_params = {'VpcId': model['VpcId']}
        if model.get('Tags'):
            create_params['TagSpecifications'] = [{'ResourceType': 'network-acl', 'Tags': [{'Key': tag['Key'], 'Value': tag['Value']} for tag in model['Tags']]}]
        response = request.aws_client_factory.ec2.create_network_acl(**create_params)
        model['Id'] = response['NetworkAcl']['NetworkAclId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[EC2NetworkAclProperties]) -> ProgressEvent[EC2NetworkAclProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - ec2:DescribeNetworkAcls\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2NetworkAclProperties]) -> ProgressEvent[EC2NetworkAclProperties]:
        if False:
            return 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - ec2:DeleteNetworkAcl\n          - ec2:DescribeNetworkAcls\n        '
        model = request.desired_state
        request.aws_client_factory.ec2.delete_network_acl(NetworkAclId=model['Id'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[EC2NetworkAclProperties]) -> ProgressEvent[EC2NetworkAclProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - ec2:DescribeNetworkAcls\n          - ec2:DeleteTags\n          - ec2:CreateTags\n        '
        raise NotImplementedError