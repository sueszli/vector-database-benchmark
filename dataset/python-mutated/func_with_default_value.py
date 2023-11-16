import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
__all__ = ['FuncWithDefaultValueResult', 'AwaitableFuncWithDefaultValueResult', 'func_with_default_value', 'func_with_default_value_output']

@pulumi.output_type
class FuncWithDefaultValueResult:

    def __init__(__self__, r=None):
        if False:
            return 10
        if r and (not isinstance(r, str)):
            raise TypeError("Expected argument 'r' to be a str")
        pulumi.set(__self__, 'r', r)

    @property
    @pulumi.getter
    def r(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return pulumi.get(self, 'r')

class AwaitableFuncWithDefaultValueResult(FuncWithDefaultValueResult):

    def __await__(self):
        if False:
            i = 10
            return i + 15
        if False:
            yield self
        return FuncWithDefaultValueResult(r=self.r)

def func_with_default_value(a: Optional[str]=None, b: Optional[str]=None, opts: Optional[pulumi.InvokeOptions]=None) -> AwaitableFuncWithDefaultValueResult:
    if False:
        for i in range(10):
            print('nop')
    '\n    Check codegen of functions with default values.\n    '
    __args__ = dict()
    __args__['a'] = a
    __args__['b'] = b
    opts = pulumi.InvokeOptions.merge(_utilities.get_invoke_opts_defaults(), opts)
    __ret__ = pulumi.runtime.invoke('mypkg::funcWithDefaultValue', __args__, opts=opts, typ=FuncWithDefaultValueResult).value
    return AwaitableFuncWithDefaultValueResult(r=pulumi.get(__ret__, 'r'))

@_utilities.lift_output_func(func_with_default_value)
def func_with_default_value_output(a: Optional[pulumi.Input[str]]=None, b: Optional[pulumi.Input[Optional[str]]]=None, opts: Optional[pulumi.InvokeOptions]=None) -> pulumi.Output[FuncWithDefaultValueResult]:
    if False:
        i = 10
        return i + 15
    '\n    Check codegen of functions with default values.\n    '
    ...