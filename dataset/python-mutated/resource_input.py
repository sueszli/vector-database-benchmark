import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
__all__ = ['ResourceInputArgs', 'ResourceInput']

@pulumi.input_type
class ResourceInputArgs:

    def __init__(__self__):
        if False:
            i = 10
            return i + 15
        '\n        The set of arguments for constructing a ResourceInput resource.\n        '
        pass

class ResourceInput(pulumi.CustomResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, __props__=None):
        if False:
            while True:
                i = 10
        '\n        Create a ResourceInput resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        '
        ...

    @overload
    def __init__(__self__, resource_name: str, args: Optional[ResourceInputArgs]=None, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            while True:
                i = 10
        "\n        Create a ResourceInput resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param ResourceInputArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            return 10
        (resource_args, opts) = _utilities.get_resource_args_opts(ResourceInputArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, __props__=None):
        if False:
            for i in range(10):
                print('nop')
        opts = pulumi.ResourceOptions.merge(_utilities.get_resource_opts_defaults(), opts)
        if not isinstance(opts, pulumi.ResourceOptions):
            raise TypeError('Expected resource options to be a ResourceOptions instance')
        if opts.id is None:
            if __props__ is not None:
                raise TypeError('__props__ is only valid when passed in combination with a valid opts.id to get an existing resource')
            __props__ = ResourceInputArgs.__new__(ResourceInputArgs)
            __props__.__dict__['bar'] = None
        super(ResourceInput, __self__).__init__('example::ResourceInput', resource_name, __props__, opts)

    @staticmethod
    def get(resource_name: str, id: pulumi.Input[str], opts: Optional[pulumi.ResourceOptions]=None) -> 'ResourceInput':
        if False:
            while True:
                i = 10
        "\n        Get an existing ResourceInput resource's state with the given name, id, and optional extra\n        properties used to qualify the lookup.\n\n        :param str resource_name: The unique name of the resulting resource.\n        :param pulumi.Input[str] id: The unique provider ID of the resource to lookup.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        opts = pulumi.ResourceOptions.merge(opts, pulumi.ResourceOptions(id=id))
        __props__ = ResourceInputArgs.__new__(ResourceInputArgs)
        __props__.__dict__['bar'] = None
        return ResourceInput(resource_name, opts=opts, __props__=__props__)

    @property
    @pulumi.getter
    def bar(self) -> pulumi.Output[Optional[str]]:
        if False:
            for i in range(10):
                print('nop')
        return pulumi.get(self, 'bar')