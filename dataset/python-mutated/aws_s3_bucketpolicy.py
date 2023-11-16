from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest
from localstack.utils.json import canonical_json
from localstack.utils.strings import md5

class S3BucketPolicyProperties(TypedDict):
    Bucket: Optional[str]
    PolicyDocument: Optional[dict]
    Id: Optional[str]
REPEATED_INVOCATION = 'repeated_invocation'

class S3BucketPolicyProvider(ResourceProvider[S3BucketPolicyProperties]):
    TYPE = 'AWS::S3::BucketPolicy'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[S3BucketPolicyProperties]) -> ProgressEvent[S3BucketPolicyProperties]:
        if False:
            print('Hello World!')
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Id\n\n        Required properties:\n          - Bucket\n          - PolicyDocument\n\n        Create-only properties:\n          - /properties/Bucket\n\n        Read-only properties:\n          - /properties/Id\n\n\n\n        '
        model = request.desired_state
        s3 = request.aws_client_factory.s3
        s3.put_bucket_policy(Bucket=model['Bucket'], Policy=json.dumps(model['PolicyDocument']))
        model['Id'] = md5(canonical_json(model['PolicyDocument']))
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[S3BucketPolicyProperties]) -> ProgressEvent[S3BucketPolicyProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch resource information\n\n\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[S3BucketPolicyProperties]) -> ProgressEvent[S3BucketPolicyProperties]:
        if False:
            return 10
        '\n        Delete a resource\n\n\n        '
        model = request.desired_state
        s3 = request.aws_client_factory.s3
        s3.delete_bucket_policy(Bucket=model['Bucket'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[S3BucketPolicyProperties]) -> ProgressEvent[S3BucketPolicyProperties]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a resource\n\n\n        '
        raise NotImplementedError