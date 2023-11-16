from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class SSMPatchBaselineProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::SSM::PatchBaseline'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            while True:
                i = 10
        from localstack.services.ssm.resource_providers.aws_ssm_patchbaseline import SSMPatchBaselineProvider
        self.factory = SSMPatchBaselineProvider