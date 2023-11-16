from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypedDict
if TYPE_CHECKING:
    from mypy_boto3_route53 import Route53Client
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class Route53RecordSetProperties(TypedDict):
    Name: Optional[str]
    Type: Optional[str]
    AliasTarget: Optional[AliasTarget]
    CidrRoutingConfig: Optional[CidrRoutingConfig]
    Comment: Optional[str]
    Failover: Optional[str]
    GeoLocation: Optional[GeoLocation]
    HealthCheckId: Optional[str]
    HostedZoneId: Optional[str]
    HostedZoneName: Optional[str]
    Id: Optional[str]
    MultiValueAnswer: Optional[bool]
    Region: Optional[str]
    ResourceRecords: Optional[list[str]]
    SetIdentifier: Optional[str]
    TTL: Optional[str]
    Weight: Optional[int]

class AliasTarget(TypedDict):
    DNSName: Optional[str]
    HostedZoneId: Optional[str]
    EvaluateTargetHealth: Optional[bool]

class CidrRoutingConfig(TypedDict):
    CollectionId: Optional[str]
    LocationName: Optional[str]

class GeoLocation(TypedDict):
    ContinentCode: Optional[str]
    CountryCode: Optional[str]
    SubdivisionCode: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class Route53RecordSetProvider(ResourceProvider[Route53RecordSetProperties]):
    TYPE = 'AWS::Route53::RecordSet'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[Route53RecordSetProperties]) -> ProgressEvent[Route53RecordSetProperties]:
        if False:
            while True:
                i = 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - Type\n          - Name\n\n        Create-only properties:\n          - /properties/HostedZoneName\n          - /properties/Name\n          - /properties/HostedZoneId\n\n        Read-only properties:\n          - /properties/Id\n        '
        model = request.desired_state
        route53 = request.aws_client_factory.route53
        if not model.get('HostedZoneId'):
            hosted_zone_name = model.get('HostedZoneName')
            hosted_zone_id = self.get_hosted_zone_id_from_name(hosted_zone_name, route53)
            model['HostedZoneId'] = hosted_zone_id
        attr_names = ['Name', 'Type', 'SetIdentifier', 'Weight', 'Region', 'GeoLocation', 'Failover', 'MultiValueAnswer', 'TTL', 'ResourceRecords', 'AliasTarget', 'HealthCheckId']
        attrs = util.select_attributes(model, attr_names)
        attrs['ResourceRecords'] = [{'Value': record} for record in attrs['ResourceRecords']]
        if 'TTL' in attrs:
            if isinstance(attrs['TTL'], str):
                attrs['TTL'] = int(attrs['TTL'])
        if 'AliasTarget' in attrs and 'EvaluateTargetHealth' not in attrs['AliasTarget']:
            attrs['AliasTarget']['EvaluateTargetHealth'] = False
        route53.change_resource_record_sets(HostedZoneId=model['HostedZoneId'], ChangeBatch={'Changes': [{'Action': 'UPSERT', 'ResourceRecordSet': attrs}]})
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def get_hosted_zone_id_from_name(self, hosted_zone_name: str, client: 'Route53Client'):
        if False:
            while True:
                i = 10
        if not hosted_zone_name:
            raise Exception('Either HostedZoneId or HostedZoneName must be present.')
        zones = client.list_hosted_zones_by_name(DNSName=hosted_zone_name)['HostedZones']
        if len(zones) != 1:
            raise Exception(f'Ambiguous HostedZoneName {hosted_zone_name} provided.')
        hosted_zone_id = zones[0]['Id']
        return hosted_zone_id

    def read(self, request: ResourceRequest[Route53RecordSetProperties]) -> ProgressEvent[Route53RecordSetProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[Route53RecordSetProperties]) -> ProgressEvent[Route53RecordSetProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        route53 = request.aws_client_factory.route53
        resource_records = [{'Value': record} for record in model['ResourceRecords']]
        route53.change_resource_record_sets(HostedZoneId=model['HostedZoneId'], ChangeBatch={'Changes': [{'Action': 'DELETE', 'ResourceRecordSet': {'Name': model['Name'], 'Type': model['Type'], 'TTL': int(model['TTL']), 'ResourceRecords': resource_records}}]})
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)

    def update(self, request: ResourceRequest[Route53RecordSetProperties]) -> ProgressEvent[Route53RecordSetProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError