from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.constants import AWS_REGION_US_EAST_1, DEFAULT_AWS_ACCOUNT_ID
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.aws import arns
LOG = logging.getLogger(__name__)
default_repos_per_stack = {}

class ECRRepositoryProperties(TypedDict):
    Arn: Optional[str]
    EncryptionConfiguration: Optional[EncryptionConfiguration]
    ImageScanningConfiguration: Optional[ImageScanningConfiguration]
    ImageTagMutability: Optional[str]
    LifecyclePolicy: Optional[LifecyclePolicy]
    RepositoryName: Optional[str]
    RepositoryPolicyText: Optional[dict | str]
    RepositoryUri: Optional[str]
    Tags: Optional[list[Tag]]

class LifecyclePolicy(TypedDict):
    LifecyclePolicyText: Optional[str]
    RegistryId: Optional[str]

class Tag(TypedDict):
    Key: Optional[str]
    Value: Optional[str]

class ImageScanningConfiguration(TypedDict):
    ScanOnPush: Optional[bool]

class EncryptionConfiguration(TypedDict):
    EncryptionType: Optional[str]
    KmsKey: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class ECRRepositoryProvider(ResourceProvider[ECRRepositoryProperties]):
    TYPE = 'AWS::ECR::Repository'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[ECRRepositoryProperties]) -> ProgressEvent[ECRRepositoryProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/RepositoryName\n\n        Create-only properties:\n          - /properties/RepositoryName\n          - /properties/EncryptionConfiguration\n          - /properties/EncryptionConfiguration/EncryptionType\n          - /properties/EncryptionConfiguration/KmsKey\n\n        Read-only properties:\n          - /properties/Arn\n          - /properties/RepositoryUri\n\n        IAM permissions required:\n          - ecr:CreateRepository\n          - ecr:PutLifecyclePolicy\n          - ecr:SetRepositoryPolicy\n          - ecr:TagResource\n          - kms:DescribeKey\n          - kms:CreateGrant\n          - kms:RetireGrant\n\n        '
        model = request.desired_state
        default_repos_per_stack[request.stack_name] = model['RepositoryName']
        LOG.warning('Creating a Mock ECR Repository for CloudFormation. This is only intended to be used for allowing a successful CDK bootstrap and does not provision any underlying ECR repository.')
        model.update({'Arn': arns.get_ecr_repository_arn(model['RepositoryName'], DEFAULT_AWS_ACCOUNT_ID, AWS_REGION_US_EAST_1), 'RepositoryUri': 'http://localhost:4566', 'ImageTagMutability': 'MUTABLE', 'ImageScanningConfiguration': {'scanOnPush': True}})
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[ECRRepositoryProperties]) -> ProgressEvent[ECRRepositoryProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - ecr:DescribeRepositories\n          - ecr:GetLifecyclePolicy\n          - ecr:GetRepositoryPolicy\n          - ecr:ListTagsForResource\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[ECRRepositoryProperties]) -> ProgressEvent[ECRRepositoryProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Delete a resource\n\n        IAM permissions required:\n          - ecr:DeleteRepository\n          - kms:RetireGrant\n        '
        if default_repos_per_stack.get(request.stack_name):
            del default_repos_per_stack[request.stack_name]
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=request.desired_state, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[ECRRepositoryProperties]) -> ProgressEvent[ECRRepositoryProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n        IAM permissions required:\n          - ecr:PutLifecyclePolicy\n          - ecr:SetRepositoryPolicy\n          - ecr:TagResource\n          - ecr:UntagResource\n          - ecr:DeleteLifecyclePolicy\n          - ecr:DeleteRepositoryPolicy\n          - ecr:PutImageScanningConfiguration\n          - ecr:PutImageTagMutability\n          - kms:DescribeKey\n          - kms:CreateGrant\n          - kms:RetireGrant\n        '
        raise NotImplementedError