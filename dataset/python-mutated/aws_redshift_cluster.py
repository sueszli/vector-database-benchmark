from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class RedshiftClusterProperties(TypedDict):
    ClusterType: Optional[str]
    DBName: Optional[str]
    MasterUserPassword: Optional[str]
    MasterUsername: Optional[str]
    NodeType: Optional[str]
    AllowVersionUpgrade: Optional[bool]
    AquaConfigurationStatus: Optional[str]
    AutomatedSnapshotRetentionPeriod: Optional[int]
    AvailabilityZone: Optional[str]
    AvailabilityZoneRelocation: Optional[bool]
    AvailabilityZoneRelocationStatus: Optional[str]
    Classic: Optional[bool]
    ClusterIdentifier: Optional[str]
    ClusterParameterGroupName: Optional[str]
    ClusterSecurityGroups: Optional[list[str]]
    ClusterSubnetGroupName: Optional[str]
    ClusterVersion: Optional[str]
    DeferMaintenance: Optional[bool]
    DeferMaintenanceDuration: Optional[int]
    DeferMaintenanceEndTime: Optional[str]
    DeferMaintenanceIdentifier: Optional[str]
    DeferMaintenanceStartTime: Optional[str]
    DestinationRegion: Optional[str]
    ElasticIp: Optional[str]
    Encrypted: Optional[bool]
    Endpoint: Optional[Endpoint]
    EnhancedVpcRouting: Optional[bool]
    HsmClientCertificateIdentifier: Optional[str]
    HsmConfigurationIdentifier: Optional[str]
    IamRoles: Optional[list[str]]
    Id: Optional[str]
    KmsKeyId: Optional[str]
    LoggingProperties: Optional[LoggingProperties]
    MaintenanceTrackName: Optional[str]
    ManualSnapshotRetentionPeriod: Optional[int]
    NumberOfNodes: Optional[int]
    OwnerAccount: Optional[str]
    Port: Optional[int]
    PreferredMaintenanceWindow: Optional[str]
    PubliclyAccessible: Optional[bool]
    ResourceAction: Optional[str]
    RevisionTarget: Optional[str]
    RotateEncryptionKey: Optional[bool]
    SnapshotClusterIdentifier: Optional[str]
    SnapshotCopyGrantName: Optional[str]
    SnapshotCopyManual: Optional[bool]
    SnapshotCopyRetentionPeriod: Optional[int]
    SnapshotIdentifier: Optional[str]
    Tags: Optional[list[Tag]]
    VpcSecurityGroupIds: Optional[list[str]]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]

class LoggingProperties(TypedDict):
    BucketName: Optional[str]
    S3KeyPrefix: Optional[str]

class Endpoint(TypedDict):
    Address: Optional[str]
    Port: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class RedshiftClusterProvider(ResourceProvider[RedshiftClusterProperties]):
    TYPE = 'AWS::Redshift::Cluster'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[RedshiftClusterProperties]) -> ProgressEvent[RedshiftClusterProperties]:
        if False:
            return 10
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/ClusterIdentifier\n\n        Required properties:\n          - MasterUserPassword\n          - NodeType\n          - MasterUsername\n          - DBName\n          - ClusterType\n\n        Create-only properties:\n          - /properties/ClusterIdentifier\n          - /properties/OwnerAccount\n          - /properties/SnapshotIdentifier\n          - /properties/DBName\n          - /properties/SnapshotClusterIdentifier\n          - /properties/ClusterSubnetGroupName\n          - /properties/MasterUsername\n\n        Read-only properties:\n          - /properties/Id\n          - /properties/DeferMaintenanceIdentifier\n          - /properties/Endpoint/Port\n          - /properties/Endpoint/Address\n\n        IAM permissions required:\n          - redshift:DescribeClusters\n          - redshift:CreateCluster\n          - redshift:RestoreFromClusterSnapshot\n          - redshift:EnableLogging\n\n        '
        model = request.desired_state
        redshift = request.aws_client_factory.redshift
        if not model.get('ClusterIdentifier'):
            model['ClusterIdentifier'] = util.generate_default_name(stack_name=request.stack_name, logical_resource_id=request.logical_resource_id)
        redshift.create_cluster(**model)
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[RedshiftClusterProperties]) -> ProgressEvent[RedshiftClusterProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - redshift:DescribeClusters\n          - redshift:DescribeLoggingStatus\n          - redshift:DescribeSnapshotCopyGrant\n          - redshift:DescribeClusterDbRevisions\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[RedshiftClusterProperties]) -> ProgressEvent[RedshiftClusterProperties]:
        if False:
            print('Hello World!')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - redshift:DescribeClusters\n          - redshift:DeleteCluster\n        '
        model = request.desired_state
        redshift = request.aws_client_factory.redshift
        redshift.delete_cluster(ClusterIdentifier=model['ClusterIdentifier'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[RedshiftClusterProperties]) -> ProgressEvent[RedshiftClusterProperties]:
        if False:
            return 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - redshift:DescribeClusters\n          - redshift:ModifyCluster\n          - redshift:ModifyClusterIamRoles\n          - redshift:EnableLogging\n          - redshift:CreateTags\n          - redshift:DeleteTags\n          - redshift:DisableLogging\n          - redshift:RebootCluster\n          - redshift:EnableSnapshotCopy\n          - redshift:DisableSnapshotCopy\n          - redshift:ModifySnapshotCopyRetentionPeriod\n          - redshift:ModifyAquaConfiguration\n          - redshift:ResizeCluster\n          - redshift:ModifyClusterMaintenance\n          - redshift:DescribeClusterDbRevisions\n          - redshift:ModifyClusterDbRevisions\n          - redshift:PauseCluster\n          - redshift:ResumeCluster\n          - redshift:RotateEncryptionKey\n        '
        raise NotImplementedError