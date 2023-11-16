from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class SSMMaintenanceWindowProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::SSM::MaintenanceWindow'

    def __init__(self):
        if False:
            while True:
                i = 10
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            i = 10
            return i + 15
        from localstack.services.ssm.resource_providers.aws_ssm_maintenancewindow import SSMMaintenanceWindowProvider
        self.factory = SSMMaintenanceWindowProvider