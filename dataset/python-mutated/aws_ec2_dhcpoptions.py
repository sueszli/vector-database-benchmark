from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EC2DHCPOptionsProperties(TypedDict):
    DhcpOptionsId: Optional[str]
    DomainName: Optional[str]
    DomainNameServers: Optional[list[str]]
    NetbiosNameServers: Optional[list[str]]
    NetbiosNodeType: Optional[int]
    NtpServers: Optional[list[str]]
    Tags: Optional[list[Tag]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class EC2DHCPOptionsProvider(ResourceProvider[EC2DHCPOptionsProperties]):
    TYPE = 'AWS::EC2::DHCPOptions'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EC2DHCPOptionsProperties]) -> ProgressEvent[EC2DHCPOptionsProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/DhcpOptionsId\n\n\n\n        Create-only properties:\n          - /properties/NetbiosNameServers\n          - /properties/NetbiosNodeType\n          - /properties/NtpServers\n          - /properties/DomainName\n          - /properties/DomainNameServers\n\n        Read-only properties:\n          - /properties/DhcpOptionsId\n\n        IAM permissions required:\n          - ec2:CreateDhcpOptions\n          - ec2:DescribeDhcpOptions\n          - ec2:CreateTags\n\n        '
        model = request.desired_state
        dhcp_configurations = []
        if model.get('DomainName'):
            dhcp_configurations.append({'Key': 'domain-name', 'Values': [model['DomainName']]})
        if model.get('DomainNameServers'):
            dhcp_configurations.append({'Key': 'domain-name-servers', 'Values': model['DomainNameServers']})
        if model.get('NetbiosNameServers'):
            dhcp_configurations.append({'Key': 'netbios-name-servers', 'Values': model['NetbiosNameServers']})
        if model.get('NetbiosNodeType'):
            dhcp_configurations.append({'Key': 'netbios-node-type', 'Values': [str(model['NetbiosNodeType'])]})
        if model.get('NtpServers'):
            dhcp_configurations.append({'Key': 'ntp-servers', 'Values': model['NtpServers']})
        create_params = {'DhcpConfigurations': dhcp_configurations}
        if model.get('Tags'):
            tags = [{'Key': str(tag['Key']), 'Value': str(tag['Value'])} for tag in model['Tags']]
        else:
            tags = []
        default_tags = [{'Key': 'aws:cloudformation:logical-id', 'Value': request.logical_resource_id}, {'Key': 'aws:cloudformation:stack-id', 'Value': request.stack_id}, {'Key': 'aws:cloudformation:stack-name', 'Value': request.stack_name}]
        create_params['TagSpecifications'] = [{'ResourceType': 'dhcp-options', 'Tags': tags + default_tags}]
        result = request.aws_client_factory.ec2.create_dhcp_options(**create_params)
        model['DhcpOptionsId'] = result['DhcpOptions']['DhcpOptionsId']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def read(self, request: ResourceRequest[EC2DHCPOptionsProperties]) -> ProgressEvent[EC2DHCPOptionsProperties]:
        if False:
            print('Hello World!')
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - ec2:DescribeDhcpOptions\n          - ec2:DescribeTags\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EC2DHCPOptionsProperties]) -> ProgressEvent[EC2DHCPOptionsProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n\n        IAM permissions required:\n          - ec2:DeleteDhcpOptions\n          - ec2:DeleteTags\n        '
        model = request.desired_state
        request.aws_client_factory.ec2.delete_dhcp_options(DhcpOptionsId=model['DhcpOptionsId'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[EC2DHCPOptionsProperties]) -> ProgressEvent[EC2DHCPOptionsProperties]:
        if False:
            print('Hello World!')
        '\n        Update a resource\n\n        IAM permissions required:\n          - ec2:CreateTags\n          - ec2:DescribeDhcpOptions\n          - ec2:DeleteTags\n        '
        raise NotImplementedError