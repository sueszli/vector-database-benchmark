import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
from . import outputs
from .cat import Cat
from .dog import Dog
__all__ = ['ToyStoreArgs', 'ToyStore']

@pulumi.input_type
class ToyStoreArgs:

    def __init__(__self__):
        if False:
            print('Hello World!')
        '\n        The set of arguments for constructing a ToyStore resource.\n        '
        pass

class ToyStore(pulumi.CustomResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, __props__=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a ToyStore resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        '
        ...

    @overload
    def __init__(__self__, resource_name: str, args: Optional[ToyStoreArgs]=None, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            print('Hello World!')
        "\n        Create a ToyStore resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param ToyStoreArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            print('Hello World!')
        (resource_args, opts) = _utilities.get_resource_args_opts(ToyStoreArgs, pulumi.ResourceOptions, *args, **kwargs)
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
            __props__ = ToyStoreArgs.__new__(ToyStoreArgs)
            __props__.__dict__['chew'] = None
            __props__.__dict__['laser'] = None
            __props__.__dict__['stuff'] = None
            __props__.__dict__['wanted'] = None
        replace_on_changes = pulumi.ResourceOptions(replace_on_changes=['chew.owner', 'laser.batteries', 'stuff[*].associated.color', 'stuff[*].color', 'wanted[*]'])
        opts = pulumi.ResourceOptions.merge(opts, replace_on_changes)
        super(ToyStore, __self__).__init__('example::ToyStore', resource_name, __props__, opts)

    @staticmethod
    def get(resource_name: str, id: pulumi.Input[str], opts: Optional[pulumi.ResourceOptions]=None) -> 'ToyStore':
        if False:
            i = 10
            return i + 15
        "\n        Get an existing ToyStore resource's state with the given name, id, and optional extra\n        properties used to qualify the lookup.\n\n        :param str resource_name: The unique name of the resulting resource.\n        :param pulumi.Input[str] id: The unique provider ID of the resource to lookup.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        opts = pulumi.ResourceOptions.merge(opts, pulumi.ResourceOptions(id=id))
        __props__ = ToyStoreArgs.__new__(ToyStoreArgs)
        __props__.__dict__['chew'] = None
        __props__.__dict__['laser'] = None
        __props__.__dict__['stuff'] = None
        __props__.__dict__['wanted'] = None
        return ToyStore(resource_name, opts=opts, __props__=__props__)

    @property
    @pulumi.getter
    def chew(self) -> pulumi.Output[Optional['outputs.Chew']]:
        if False:
            for i in range(10):
                print('nop')
        return pulumi.get(self, 'chew')

    @property
    @pulumi.getter
    def laser(self) -> pulumi.Output[Optional['outputs.Laser']]:
        if False:
            while True:
                i = 10
        return pulumi.get(self, 'laser')

    @property
    @pulumi.getter
    def stuff(self) -> pulumi.Output[Optional[Sequence['outputs.Toy']]]:
        if False:
            while True:
                i = 10
        return pulumi.get(self, 'stuff')

    @property
    @pulumi.getter
    def wanted(self) -> pulumi.Output[Optional[Sequence['outputs.Toy']]]:
        if False:
            for i in range(10):
                print('nop')
        return pulumi.get(self, 'wanted')