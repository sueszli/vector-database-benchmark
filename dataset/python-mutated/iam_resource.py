import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from .. import _utilities
import pulumi_google_native
__all__ = ['IamResourceArgs', 'IamResource']

@pulumi.input_type
class IamResourceArgs:

    def __init__(__self__, *, config: Optional[pulumi.Input['pulumi_google_native.iam.v1.AuditConfigArgs']]=None):
        if False:
            i = 10
            return i + 15
        '\n        The set of arguments for constructing a IamResource resource.\n        '
        if config is not None:
            pulumi.set(__self__, 'config', config)

    @property
    @pulumi.getter
    def config(self) -> Optional[pulumi.Input['pulumi_google_native.iam.v1.AuditConfigArgs']]:
        if False:
            i = 10
            return i + 15
        return pulumi.get(self, 'config')

    @config.setter
    def config(self, value: Optional[pulumi.Input['pulumi_google_native.iam.v1.AuditConfigArgs']]):
        if False:
            for i in range(10):
                print('nop')
        pulumi.set(self, 'config', value)

class IamResource(pulumi.ComponentResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, config: Optional[pulumi.Input[pulumi.InputType['pulumi_google_native.iam.v1.AuditConfigArgs']]]=None, __props__=None):
        if False:
            return 10
        '\n        Create a IamResource resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        '
        ...

    @overload
    def __init__(__self__, resource_name: str, args: Optional[IamResourceArgs]=None, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a IamResource resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param IamResourceArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            print('Hello World!')
        (resource_args, opts) = _utilities.get_resource_args_opts(IamResourceArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, config: Optional[pulumi.Input[pulumi.InputType['pulumi_google_native.iam.v1.AuditConfigArgs']]]=None, __props__=None):
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
            __props__ = IamResourceArgs.__new__(IamResourceArgs)
            __props__.__dict__['config'] = config
        super(IamResource, __self__).__init__('example:myModule:IamResource', resource_name, __props__, opts, remote=True)