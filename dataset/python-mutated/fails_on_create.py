from typing import Optional
import pulumi

class FailsOnCreate(pulumi.CustomResource):

    def __init__(self, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            return 10
        super().__init__('testprovider:index:FailsOnCreate', resource_name, {}, opts)