import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
from . import outputs
__all__ = ['GetCustomDbRolesResult', 'AwaitableGetCustomDbRolesResult', 'get_custom_db_roles', 'get_custom_db_roles_output']

@pulumi.output_type
class GetCustomDbRolesResult:

    def __init__(__self__, result=None):
        if False:
            for i in range(10):
                print('nop')
        if result and (not isinstance(result, dict)):
            raise TypeError("Expected argument 'result' to be a dict")
        pulumi.set(__self__, 'result', result)

    @property
    @pulumi.getter
    def result(self) -> Optional['outputs.GetCustomDbRolesResult']:
        if False:
            i = 10
            return i + 15
        return pulumi.get(self, 'result')

class AwaitableGetCustomDbRolesResult(GetCustomDbRolesResult):

    def __await__(self):
        if False:
            i = 10
            return i + 15
        if False:
            yield self
        return GetCustomDbRolesResult(result=self.result)

def get_custom_db_roles(opts: Optional[pulumi.InvokeOptions]=None) -> AwaitableGetCustomDbRolesResult:
    if False:
        return 10
    '\n    Use this data source to access information about an existing resource.\n    '
    __args__ = dict()
    opts = pulumi.InvokeOptions.merge(_utilities.get_invoke_opts_defaults(), opts)
    __ret__ = pulumi.runtime.invoke('mongodbatlas::getCustomDbRoles', __args__, opts=opts, typ=GetCustomDbRolesResult).value
    return AwaitableGetCustomDbRolesResult(result=pulumi.get(__ret__, 'result'))

@_utilities.lift_output_func(get_custom_db_roles)
def get_custom_db_roles_output(opts: Optional[pulumi.InvokeOptions]=None) -> pulumi.Output[GetCustomDbRolesResult]:
    if False:
        print('Hello World!')
    '\n    Use this data source to access information about an existing resource.\n    '
    ...