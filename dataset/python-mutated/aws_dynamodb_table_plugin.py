from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class DynamoDBTableProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::DynamoDB::Table'

    def __init__(self):
        if False:
            print('Hello World!')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            while True:
                i = 10
        from localstack.services.dynamodb.resource_providers.aws_dynamodb_table import DynamoDBTableProvider
        self.factory = DynamoDBTableProvider