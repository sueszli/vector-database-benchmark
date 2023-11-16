import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
import pulumi_azure_native
__all__ = ['RegistryGeoReplicationArgs', 'RegistryGeoReplication']

@pulumi.input_type
class RegistryGeoReplicationArgs:

    def __init__(__self__, *, resource_group: pulumi.Input['pulumi_azure_native.resources.ResourceGroup']):
        if False:
            for i in range(10):
                print('nop')
        "\n        The set of arguments for constructing a RegistryGeoReplication resource.\n        :param pulumi.Input['pulumi_azure_native.resources.ResourceGroup'] resource_group: The resource group that hosts the component resource\n        "
        pulumi.set(__self__, 'resource_group', resource_group)

    @property
    @pulumi.getter(name='resourceGroup')
    def resource_group(self) -> pulumi.Input['pulumi_azure_native.resources.ResourceGroup']:
        if False:
            i = 10
            return i + 15
        '\n        The resource group that hosts the component resource\n        '
        return pulumi.get(self, 'resource_group')

    @resource_group.setter
    def resource_group(self, value: pulumi.Input['pulumi_azure_native.resources.ResourceGroup']):
        if False:
            for i in range(10):
                print('nop')
        pulumi.set(self, 'resource_group', value)

class RegistryGeoReplication(pulumi.ComponentResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, resource_group: Optional[pulumi.Input['pulumi_azure_native.resources.ResourceGroup']]=None, __props__=None):
        if False:
            print('Hello World!')
        "\n        Create a RegistryGeoReplication resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        :param pulumi.Input['pulumi_azure_native.resources.ResourceGroup'] resource_group: The resource group that hosts the component resource\n        "
        ...

    @overload
    def __init__(__self__, resource_name: str, args: RegistryGeoReplicationArgs, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            return 10
        "\n        Create a RegistryGeoReplication resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param RegistryGeoReplicationArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            print('Hello World!')
        (resource_args, opts) = _utilities.get_resource_args_opts(RegistryGeoReplicationArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, resource_group: Optional[pulumi.Input['pulumi_azure_native.resources.ResourceGroup']]=None, __props__=None):
        if False:
            return 10
        opts = pulumi.ResourceOptions.merge(_utilities.get_resource_opts_defaults(), opts)
        if not isinstance(opts, pulumi.ResourceOptions):
            raise TypeError('Expected resource options to be a ResourceOptions instance')
        if opts.id is not None:
            raise ValueError('ComponentResource classes do not support opts.id')
        else:
            if __props__ is not None:
                raise TypeError('__props__ is only valid when passed in combination with a valid opts.id to get an existing resource')
            __props__ = RegistryGeoReplicationArgs.__new__(RegistryGeoReplicationArgs)
            if resource_group is None and (not opts.urn):
                raise TypeError("Missing required property 'resource_group'")
            __props__.__dict__['resource_group'] = resource_group
            __props__.__dict__['acr_login_server_out'] = None
            __props__.__dict__['registry'] = None
            __props__.__dict__['replication'] = None
        super(RegistryGeoReplication, __self__).__init__('registrygeoreplication:index:RegistryGeoReplication', resource_name, __props__, opts, remote=True)

    @property
    @pulumi.getter(name='acrLoginServerOut')
    def acr_login_server_out(self) -> pulumi.Output[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The login server url\n        '
        return pulumi.get(self, 'acr_login_server_out')

    @property
    @pulumi.getter
    def registry(self) -> pulumi.Output['pulumi_azure_native.containerregistry.Registry']:
        if False:
            while True:
                i = 10
        '\n        The Registry\n        '
        return pulumi.get(self, 'registry')

    @property
    @pulumi.getter
    def replication(self) -> pulumi.Output['pulumi_azure_native.containerregistry.Replication']:
        if False:
            return 10
        '\n        The replication policy\n        '
        return pulumi.get(self, 'replication')