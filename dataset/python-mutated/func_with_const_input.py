import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload, Awaitable
from . import _utilities
__all__ = ['func_with_const_input']

def func_with_const_input(plain_input: Optional[str]=None, opts: Optional[pulumi.InvokeOptions]=None) -> Awaitable[None]:
    if False:
        i = 10
        return i + 15
    '\n    Codegen demo with const inputs\n    '
    __args__ = dict()
    __args__['plainInput'] = plain_input
    opts = pulumi.InvokeOptions.merge(_utilities.get_invoke_opts_defaults(), opts)
    __ret__ = pulumi.runtime.invoke('mypkg::funcWithConstInput', __args__, opts=opts).value