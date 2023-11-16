import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
__all__ = ['MainComponentArgs', 'MainComponent']

@pulumi.input_type
class MainComponentArgs:

    def __init__(__self__):
        if False:
            print('Hello World!')
        '\n        The set of arguments for constructing a MainComponent resource.\n        '
        pass

class MainComponent(pulumi.CustomResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, __props__=None):
        if False:
            while True:
                i = 10
        '\n        Create a MainComponent resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        '
        ...

    @overload
    def __init__(__self__, resource_name: str, args: Optional[MainComponentArgs]=None, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            print('Hello World!')
        "\n        Create a MainComponent resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param MainComponentArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            return 10
        (resource_args, opts) = _utilities.get_resource_args_opts(MainComponentArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, __props__=None):
        if False:
            i = 10
            return i + 15
        opts = pulumi.ResourceOptions.merge(_utilities.get_resource_opts_defaults(), opts)
        if not isinstance(opts, pulumi.ResourceOptions):
            raise TypeError('Expected resource options to be a ResourceOptions instance')
        if opts.id is None:
            if __props__ is not None:
                raise TypeError('__props__ is only valid when passed in combination with a valid opts.id to get an existing resource')
            __props__ = MainComponentArgs.__new__(MainComponentArgs)
        super(MainComponent, __self__).__init__('example::MainComponent', resource_name, __props__, opts)

    @staticmethod
    def get(resource_name: str, id: pulumi.Input[str], opts: Optional[pulumi.ResourceOptions]=None) -> 'MainComponent':
        if False:
            print('Hello World!')
        "\n        Get an existing MainComponent resource's state with the given name, id, and optional extra\n        properties used to qualify the lookup.\n\n        :param str resource_name: The unique name of the resulting resource.\n        :param pulumi.Input[str] id: The unique provider ID of the resource to lookup.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        opts = pulumi.ResourceOptions.merge(opts, pulumi.ResourceOptions(id=id))
        __props__ = MainComponentArgs.__new__(MainComponentArgs)
        return MainComponent(resource_name, opts=opts, __props__=__props__)