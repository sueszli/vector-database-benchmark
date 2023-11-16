from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class RedshiftClusterProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::Redshift::Cluster'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            while True:
                i = 10
        from localstack.services.redshift.resource_providers.aws_redshift_cluster import RedshiftClusterProvider
        self.factory = RedshiftClusterProvider