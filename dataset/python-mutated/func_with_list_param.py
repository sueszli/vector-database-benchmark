import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
__all__ = ['FuncWithListParamResult', 'AwaitableFuncWithListParamResult', 'func_with_list_param', 'func_with_list_param_output']

@pulumi.output_type
class FuncWithListParamResult:

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

class AwaitableFuncWithListParamResult(FuncWithListParamResult):

    def __await__(self):
        if False:
            while True:
                i = 10
        if False:
            yield self
        return FuncWithListParamResult(r=self.r)

def func_with_list_param(a: Optional[Sequence[str]]=None, b: Optional[str]=None, opts: Optional[pulumi.InvokeOptions]=None) -> AwaitableFuncWithListParamResult:
    if False:
        print('Hello World!')
    '\n    Check codegen of functions with a List parameter.\n    '
    __args__ = dict()
    __args__['a'] = a
    __args__['b'] = b
    opts = pulumi.InvokeOptions.merge(_utilities.get_invoke_opts_defaults(), opts)
    __ret__ = pulumi.runtime.invoke('mypkg::funcWithListParam', __args__, opts=opts, typ=FuncWithListParamResult).value
    return AwaitableFuncWithListParamResult(r=pulumi.get(__ret__, 'r'))

@_utilities.lift_output_func(func_with_list_param)
def func_with_list_param_output(a: Optional[pulumi.Input[Optional[Sequence[str]]]]=None, b: Optional[pulumi.Input[Optional[str]]]=None, opts: Optional[pulumi.InvokeOptions]=None) -> pulumi.Output[FuncWithListParamResult]:
    if False:
        print('Hello World!')
    '\n    Check codegen of functions with a List parameter.\n    '
    ...