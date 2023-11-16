import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
from .resource import Resource
__all__ = ['ArgFunctionResult', 'AwaitableArgFunctionResult', 'arg_function', 'arg_function_output']

@pulumi.output_type
class ArgFunctionResult:

    def __init__(__self__, result=None):
        if False:
            return 10
        if result and (not isinstance(result, Resource)):
            raise TypeError("Expected argument 'result' to be a Resource")
        pulumi.set(__self__, 'result', result)

    @property
    @pulumi.getter
    def result(self) -> Optional['Resource']:
        if False:
            print('Hello World!')
        return pulumi.get(self, 'result')

class AwaitableArgFunctionResult(ArgFunctionResult):

    def __await__(self):
        if False:
            return 10
        if False:
            yield self
        return ArgFunctionResult(result=self.result)

def arg_function(arg1: Optional['Resource']=None, opts: Optional[pulumi.InvokeOptions]=None) -> AwaitableArgFunctionResult:
    if False:
        i = 10
        return i + 15
    '\n    Use this data source to access information about an existing resource.\n    '
    __args__ = dict()
    __args__['arg1'] = arg1
    opts = pulumi.InvokeOptions.merge(_utilities.get_invoke_opts_defaults(), opts)
    __ret__ = pulumi.runtime.invoke('example::argFunction', __args__, opts=opts, typ=ArgFunctionResult).value
    return AwaitableArgFunctionResult(result=pulumi.get(__ret__, 'result'))

@_utilities.lift_output_func(arg_function)
def arg_function_output(arg1: Optional[pulumi.Input[Optional['Resource']]]=None, opts: Optional[pulumi.InvokeOptions]=None) -> pulumi.Output[ArgFunctionResult]:
    if False:
        while True:
            i = 10
    '\n    Use this data source to access information about an existing resource.\n    '
    ...