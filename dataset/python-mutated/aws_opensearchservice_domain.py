from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class OpenSearchServiceDomainProperties(TypedDict):
    AccessPolicies: Optional[dict]
    AdvancedOptions: Optional[dict]
    AdvancedSecurityOptions: Optional[AdvancedSecurityOptionsInput]
    Arn: Optional[str]
    ClusterConfig: Optional[ClusterConfig]
    CognitoOptions: Optional[CognitoOptions]
    DomainArn: Optional[str]
    DomainEndpoint: Optional[str]
    DomainEndpointOptions: Optional[DomainEndpointOptions]
    DomainEndpoints: Optional[dict]
    DomainName: Optional[str]
    EBSOptions: Optional[EBSOptions]
    EncryptionAtRestOptions: Optional[EncryptionAtRestOptions]
    EngineVersion: Optional[str]
    Id: Optional[str]
    LogPublishingOptions: Optional[dict]
    NodeToNodeEncryptionOptions: Optional[NodeToNodeEncryptionOptions]
    OffPeakWindowOptions: Optional[OffPeakWindowOptions]
    ServiceSoftwareOptions: Optional[ServiceSoftwareOptions]
    SnapshotOptions: Optional[SnapshotOptions]
    SoftwareUpdateOptions: Optional[SoftwareUpdateOptions]
    Tags: Optional[list[Tag]]
    VPCOptions: Optional[VPCOptions]

class ZoneAwarenessConfig(TypedDict):
    AvailabilityZoneCount: Optional[int]

class ClusterConfig(TypedDict):
    DedicatedMasterCount: Optional[int]
    DedicatedMasterEnabled: Optional[bool]
    DedicatedMasterType: Optional[str]
    InstanceCount: Optional[int]
    InstanceType: Optional[str]
    WarmCount: Optional[int]
    WarmEnabled: Optional[bool]
    WarmType: Optional[str]
    ZoneAwarenessConfig: Optional[ZoneAwarenessConfig]
    ZoneAwarenessEnabled: Optional[bool]

class SnapshotOptions(TypedDict):
    AutomatedSnapshotStartHour: Optional[int]

class VPCOptions(TypedDict):
    SecurityGroupIds: Optional[list[str]]
    SubnetIds: Optional[list[str]]

class NodeToNodeEncryptionOptions(TypedDict):
    Enabled: Optional[bool]

class DomainEndpointOptions(TypedDict):
    CustomEndpoint: Optional[str]
    CustomEndpointCertificateArn: Optional[str]
    CustomEndpointEnabled: Optional[bool]
    EnforceHTTPS: Optional[bool]
    TLSSecurityPolicy: Optional[str]

class CognitoOptions(TypedDict):
    Enabled: Optional[bool]
    IdentityPoolId: Optional[str]
    RoleArn: Optional[str]
    UserPoolId: Optional[str]

class MasterUserOptions(TypedDict):
    MasterUserARN: Optional[str]
    MasterUserName: Optional[str]
    MasterUserPassword: Optional[str]

class Idp(TypedDict):
    EntityId: Optional[str]
    MetadataContent: Optional[str]

class SAMLOptions(TypedDict):
    Enabled: Optional[bool]
    Idp: Optional[Idp]
    MasterBackendRole: Optional[str]
    MasterUserName: Optional[str]
    RolesKey: Optional[str]
    SessionTimeoutMinutes: Optional[int]
    SubjectKey: Optional[str]

class AdvancedSecurityOptionsInput(TypedDict):
    AnonymousAuthDisableDate: Optional[str]
    AnonymousAuthEnabled: Optional[bool]
    Enabled: Optional[bool]
    InternalUserDatabaseEnabled: Optional[bool]
    MasterUserOptions: Optional[MasterUserOptions]
    SAMLOptions: Optional[SAMLOptions]

class EBSOptions(TypedDict):
    EBSEnabled: Optional[bool]
    Iops: Optional[int]
    Throughput: Optional[int]
    VolumeSize: Optional[int]
    VolumeType: Optional[str]

class EncryptionAtRestOptions(TypedDict):
    Enabled: Optional[bool]
    KmsKeyId: Optional[str]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]

class ServiceSoftwareOptions(TypedDict):
    AutomatedUpdateDate: Optional[str]
    Cancellable: Optional[bool]
    CurrentVersion: Optional[str]
    Description: Optional[str]
    NewVersion: Optional[str]
    OptionalDeployment: Optional[bool]
    UpdateAvailable: Optional[bool]
    UpdateStatus: Optional[str]

class WindowStartTime(TypedDict):
    Hours: Optional[int]
    Minutes: Optional[int]

class OffPeakWindow(TypedDict):
    WindowStartTime: Optional[WindowStartTime]

class OffPeakWindowOptions(TypedDict):
    Enabled: Optional[bool]
    OffPeakWindow: Optional[OffPeakWindow]

class SoftwareUpdateOptions(TypedDict):
    AutoSoftwareUpdateEnabled: Optional[bool]
REPEATED_INVOCATION = 'repeated_invocation'

class OpenSearchServiceDomainProvider(ResourceProvider[OpenSearchServiceDomainProperties]):
    TYPE = 'AWS::OpenSearchService::Domain'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[OpenSearchServiceDomainProperties]) -> ProgressEvent[OpenSearchServiceDomainProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/DomainName\n\n\n\n        Create-only properties:\n          - /properties/DomainName\n\n        Read-only properties:\n          - /properties/Id\n          - /properties/Arn\n          - /properties/DomainArn\n          - /properties/DomainEndpoint\n          - /properties/DomainEndpoints\n          - /properties/ServiceSoftwareOptions\n          - /properties/AdvancedSecurityOptions/AnonymousAuthDisableDate\n\n        IAM permissions required:\n          - es:CreateDomain\n          - es:DescribeDomain\n          - es:AddTags\n          - es:ListTags\n\n        '
        model = request.desired_state
        opensearch_client = request.aws_client_factory.opensearch
        if not request.custom_context.get(REPEATED_INVOCATION):
            request.custom_context[REPEATED_INVOCATION] = True
            domain_name = model.get('DomainName')
            if not domain_name:
                domain_name = util.generate_default_name(request.stack_name, request.logical_resource_id).lower()[0:28]
                model['DomainName'] = domain_name
            properties = util.remove_none_values(model)
            cluster_config = properties.get('ClusterConfig')
            if isinstance(cluster_config, dict):
                cluster_config.setdefault('DedicatedMasterType', 'm3.medium.search')
                cluster_config.setdefault('WarmType', 'ultrawarm1.medium.search')
                for key in ['DedicatedMasterCount', 'InstanceCount', 'WarmCount']:
                    if key in cluster_config and isinstance(cluster_config[key], str):
                        cluster_config[key] = int(cluster_config[key])
            if properties.get('AccessPolicies'):
                properties['AccessPolicies'] = json.dumps(properties['AccessPolicies'])
            if (ebs_options := properties.get('EBSOptions')):
                for key in ['Iops', 'Throughput', 'VolumeSize']:
                    if key in ebs_options and isinstance(ebs_options[key], str):
                        ebs_options[key] = int(ebs_options[key])
            create_kwargs = {**util.deselect_attributes(properties, ['Tags'])}
            if (tags := properties.get('Tags')):
                create_kwargs['TagList'] = tags
            opensearch_client.create_domain(**create_kwargs)
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model, custom_context=request.custom_context)
        opensearch_domain = opensearch_client.describe_domain(DomainName=model['DomainName'])
        if opensearch_domain['DomainStatus']['Processing'] is False:
            model['Arn'] = opensearch_domain['DomainStatus']['ARN']
            model['Id'] = opensearch_domain['DomainStatus']['DomainId']
            model['DomainArn'] = opensearch_domain['DomainStatus']['ARN']
            model['DomainEndpoint'] = opensearch_domain['DomainStatus'].get('Endpoint')
            model['DomainEndpoints'] = opensearch_domain['DomainStatus'].get('Endpoints')
            model['ServiceSoftwareOptions'] = opensearch_domain['DomainStatus'].get('ServiceSoftwareOptions')
            model.setdefault('AdvancedSecurityOptions', {})['AnonymousAuthDisableDate'] = opensearch_domain['DomainStatus'].get('AdvancedSecurityOptions', {}).get('AnonymousAuthDisableDate')
            return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model)
        else:
            return ProgressEvent(status=OperationStatus.IN_PROGRESS, resource_model=model)

    def read(self, request: ResourceRequest[OpenSearchServiceDomainProperties]) -> ProgressEvent[OpenSearchServiceDomainProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - es:DescribeDomain\n          - es:ListTags\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[OpenSearchServiceDomainProperties]) -> ProgressEvent[OpenSearchServiceDomainProperties]:
        if False:
            return 10
        '\n        Delete a resource\n\n        IAM permissions required:\n          - es:DeleteDomain\n          - es:DescribeDomain\n        '
        opensearch_client = request.aws_client_factory.opensearch
        opensearch_client.delete_domain(DomainName=request.previous_state['DomainName'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model={})

    def update(self, request: ResourceRequest[OpenSearchServiceDomainProperties]) -> ProgressEvent[OpenSearchServiceDomainProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - es:UpdateDomain\n          - es:UpgradeDomain\n          - es:DescribeDomain\n          - es:AddTags\n          - es:RemoveTags\n          - es:ListTags\n        '
        raise NotImplementedError