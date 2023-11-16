import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from ... import _utilities
__all__ = ['ENIConfigSpecArgs']

@pulumi.input_type
class ENIConfigSpecArgs:

    def __init__(__self__, *, security_groups: Optional[pulumi.Input[Sequence[pulumi.Input[str]]]]=None, subnet: Optional[pulumi.Input[str]]=None):
        if False:
            print('Hello World!')
        if security_groups is not None:
            pulumi.set(__self__, 'security_groups', security_groups)
        if subnet is not None:
            pulumi.set(__self__, 'subnet', subnet)

    @property
    @pulumi.getter(name='securityGroups')
    def security_groups(self) -> Optional[pulumi.Input[Sequence[pulumi.Input[str]]]]:
        if False:
            print('Hello World!')
        return pulumi.get(self, 'security_groups')

    @security_groups.setter
    def security_groups(self, value: Optional[pulumi.Input[Sequence[pulumi.Input[str]]]]):
        if False:
            print('Hello World!')
        pulumi.set(self, 'security_groups', value)

    @property
    @pulumi.getter
    def subnet(self) -> Optional[pulumi.Input[str]]:
        if False:
            return 10
        return pulumi.get(self, 'subnet')

    @subnet.setter
    def subnet(self, value: Optional[pulumi.Input[str]]):
        if False:
            return 10
        pulumi.set(self, 'subnet', value)