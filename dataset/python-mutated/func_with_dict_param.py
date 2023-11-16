import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
__all__ = ['FuncWithDictParamResult', 'AwaitableFuncWithDictParamResult', 'func_with_dict_param', 'func_with_dict_param_output']

@pulumi.output_type
class FuncWithDictParamResult:

    def __init__(__self__, r=None):
        if False:
            while True:
                i = 10
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

class AwaitableFuncWithDictParamResult(FuncWithDictParamResult):

    def __await__(self):
        if False:
            return 10
        if False:
            yield self
        return FuncWithDictParamResult(r=self.r)

def func_with_dict_param(a: Optional[Mapping[str, str]]=None, b: Optional[str]=None, opts: Optional[pulumi.InvokeOptions]=None) -> AwaitableFuncWithDictParamResult:
    if False:
        i = 10
        return i + 15
    '\n    Check codegen of functions with a Dict<str,str> parameter.\n    '
    __args__ = dict()
    __args__['a'] = a
    __args__['b'] = b
    opts = pulumi.InvokeOptions.merge(_utilities.get_invoke_opts_defaults(), opts)
    __ret__ = pulumi.runtime.invoke('mypkg::funcWithDictParam', __args__, opts=opts, typ=FuncWithDictParamResult).value
    return AwaitableFuncWithDictParamResult(r=pulumi.get(__ret__, 'r'))

@_utilities.lift_output_func(func_with_dict_param)
def func_with_dict_param_output(a: Optional[pulumi.Input[Optional[Mapping[str, str]]]]=None, b: Optional[pulumi.Input[Optional[str]]]=None, opts: Optional[pulumi.InvokeOptions]=None) -> pulumi.Output[FuncWithDictParamResult]:
    if False:
        i = 10
        return i + 15
    '\n    Check codegen of functions with a Dict<str,str> parameter.\n    '
    ...