import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
__all__ = ['FuncWithAllOptionalInputsResult', 'AwaitableFuncWithAllOptionalInputsResult', 'func_with_all_optional_inputs', 'func_with_all_optional_inputs_output']

@pulumi.output_type
class FuncWithAllOptionalInputsResult:

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
            return 10
        return pulumi.get(self, 'r')

class AwaitableFuncWithAllOptionalInputsResult(FuncWithAllOptionalInputsResult):

    def __await__(self):
        if False:
            i = 10
            return i + 15
        if False:
            yield self
        return FuncWithAllOptionalInputsResult(r=self.r)

def func_with_all_optional_inputs(a: Optional[str]=None, b: Optional[str]=None, opts: Optional[pulumi.InvokeOptions]=None) -> AwaitableFuncWithAllOptionalInputsResult:
    if False:
        i = 10
        return i + 15
    '\n    Check codegen of functions with all optional inputs.\n\n\n    :param str a: Property A\n    :param str b: Property B\n    '
    __args__ = dict()
    __args__['a'] = a
    __args__['b'] = b
    opts = pulumi.InvokeOptions.merge(_utilities.get_invoke_opts_defaults(), opts)
    __ret__ = pulumi.runtime.invoke('mypkg::funcWithAllOptionalInputs', __args__, opts=opts, typ=FuncWithAllOptionalInputsResult).value
    return AwaitableFuncWithAllOptionalInputsResult(r=pulumi.get(__ret__, 'r'))

@_utilities.lift_output_func(func_with_all_optional_inputs)
def func_with_all_optional_inputs_output(a: Optional[pulumi.Input[Optional[str]]]=None, b: Optional[pulumi.Input[Optional[str]]]=None, opts: Optional[pulumi.InvokeOptions]=None) -> pulumi.Output[FuncWithAllOptionalInputsResult]:
    if False:
        i = 10
        return i + 15
    '\n    Check codegen of functions with all optional inputs.\n\n\n    :param str a: Property A\n    :param str b: Property B\n    '
    ...