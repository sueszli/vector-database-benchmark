from typing import Optional
import pulumi

class FailsOnDelete(pulumi.CustomResource):

    def __init__(self, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            while True:
                i = 10
        super().__init__('testprovider:index:FailsOnDelete', resource_name, {}, opts)