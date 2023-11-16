import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload, Awaitable
from . import _utilities
from ._enums import *
__all__ = ['example_func']

def example_func(enums: Optional[Sequence[Union[str, 'MyEnum']]]=None, opts: Optional[pulumi.InvokeOptions]=None) -> Awaitable[None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Use this data source to access information about an existing resource.\n    '
    __args__ = dict()
    __args__['enums'] = enums
    opts = pulumi.InvokeOptions.merge(_utilities.get_invoke_opts_defaults(), opts)
    __ret__ = pulumi.runtime.invoke('my8110::exampleFunc', __args__, opts=opts).value