import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload, Awaitable
from . import _utilities
__all__ = ['func_with_empty_outputs']

def func_with_empty_outputs(name: Optional[str]=None, opts: Optional[pulumi.InvokeOptions]=None) -> Awaitable[None]:
    if False:
        return 10
    '\n    n/a\n\n\n    :param str name: The Name of the FeatureGroup.\n    '
    __args__ = dict()
    __args__['name'] = name
    opts = pulumi.InvokeOptions.merge(_utilities.get_invoke_opts_defaults(), opts)
    __ret__ = pulumi.runtime.invoke('mypkg::funcWithEmptyOutputs', __args__, opts=opts).value