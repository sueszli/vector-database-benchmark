from __future__ import annotations
from pathlib import Path
from typing import Optional, TypedDict
import localstack.services.cloudformation.provider_utils as util
from localstack.services.cloudformation.resource_provider import OperationStatus, ProgressEvent, ResourceProvider, ResourceRequest

class EventsConnectionProperties(TypedDict):
    AuthParameters: Optional[AuthParameters]
    AuthorizationType: Optional[str]
    Arn: Optional[str]
    Description: Optional[str]
    Name: Optional[str]
    SecretArn: Optional[str]

class ApiKeyAuthParameters(TypedDict):
    ApiKeyName: Optional[str]
    ApiKeyValue: Optional[str]

class BasicAuthParameters(TypedDict):
    Password: Optional[str]
    Username: Optional[str]

class ClientParameters(TypedDict):
    ClientID: Optional[str]
    ClientSecret: Optional[str]

class Parameter(TypedDict):
    Key: Optional[str]
    Value: Optional[str]
    IsValueSecret: Optional[bool]

class ConnectionHttpParameters(TypedDict):
    BodyParameters: Optional[list[Parameter]]
    HeaderParameters: Optional[list[Parameter]]
    QueryStringParameters: Optional[list[Parameter]]

class OAuthParameters(TypedDict):
    AuthorizationEndpoint: Optional[str]
    ClientParameters: Optional[ClientParameters]
    HttpMethod: Optional[str]
    OAuthHttpParameters: Optional[ConnectionHttpParameters]

class AuthParameters(TypedDict):
    ApiKeyAuthParameters: Optional[ApiKeyAuthParameters]
    BasicAuthParameters: Optional[BasicAuthParameters]
    InvocationHttpParameters: Optional[ConnectionHttpParameters]
    OAuthParameters: Optional[OAuthParameters]
REPEATED_INVOCATION = 'repeated_invocation'

class EventsConnectionProvider(ResourceProvider[EventsConnectionProperties]):
    TYPE = 'AWS::Events::Connection'
    SCHEMA = util.get_schema_path(Path(__file__))

    def create(self, request: ResourceRequest[EventsConnectionProperties]) -> ProgressEvent[EventsConnectionProperties]:
        if False:
            i = 10
            return i + 15
        '\n        Create a new resource.\n\n        Primary identifier fields:\n          - /properties/Name\n\n        Required properties:\n          - AuthorizationType\n          - AuthParameters\n\n        Create-only properties:\n          - /properties/Name\n\n        Read-only properties:\n          - /properties/Arn\n          - /properties/SecretArn\n\n        IAM permissions required:\n          - events:CreateConnection\n          - secretsmanager:CreateSecret\n          - secretsmanager:GetSecretValue\n          - secretsmanager:PutSecretValue\n          - iam:CreateServiceLinkedRole\n\n        '
        model = request.desired_state
        events = request.aws_client_factory.events
        response = events.create_connection(**model)
        model['Arn'] = response['ConnectionArn']
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def read(self, request: ResourceRequest[EventsConnectionProperties]) -> ProgressEvent[EventsConnectionProperties]:
        if False:
            while True:
                i = 10
        '\n        Fetch resource information\n\n        IAM permissions required:\n          - events:DescribeConnection\n        '
        raise NotImplementedError

    def delete(self, request: ResourceRequest[EventsConnectionProperties]) -> ProgressEvent[EventsConnectionProperties]:
        if False:
            print('Hello World!')
        '\n        Delete a resource\n\n        IAM permissions required:\n          - events:DeleteConnection\n        '
        model = request.desired_state
        events = request.aws_client_factory.events
        events.delete_connection(Name=model['Name'])
        return ProgressEvent(status=OperationStatus.SUCCESS, resource_model=model, custom_context=request.custom_context)

    def update(self, request: ResourceRequest[EventsConnectionProperties]) -> ProgressEvent[EventsConnectionProperties]:
        if False:
            while True:
                i = 10
        '\n        Update a resource\n\n        IAM permissions required:\n          - events:UpdateConnection\n          - events:DescribeConnection\n          - secretsmanager:CreateSecret\n          - secretsmanager:UpdateSecret\n          - secretsmanager:GetSecretValue\n          - secretsmanager:PutSecretValue\n        '
        raise NotImplementedError