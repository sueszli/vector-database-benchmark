import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
from .resource import Resource
__all__ = ['OtherResourceArgs', 'OtherResource']

@pulumi.input_type
class OtherResourceArgs:

    def __init__(__self__, *, foo: Optional[pulumi.Input['Resource']]=None):
        if False:
            print('Hello World!')
        '\n        The set of arguments for constructing a OtherResource resource.\n        '
        if foo is not None:
            pulumi.set(__self__, 'foo', foo)

    @property
    @pulumi.getter
    def foo(self) -> Optional[pulumi.Input['Resource']]:
        if False:
            return 10
        return pulumi.get(self, 'foo')

    @foo.setter
    def foo(self, value: Optional[pulumi.Input['Resource']]):
        if False:
            return 10
        pulumi.set(self, 'foo', value)

class OtherResource(pulumi.ComponentResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, foo: Optional[pulumi.Input['Resource']]=None, __props__=None):
        if False:
            return 10
        '\n        Create a OtherResource resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        '
        ...

    @overload
    def __init__(__self__, resource_name: str, args: Optional[OtherResourceArgs]=None, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            return 10
        "\n        Create a OtherResource resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param OtherResourceArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            while True:
                i = 10
        (resource_args, opts) = _utilities.get_resource_args_opts(OtherResourceArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, foo: Optional[pulumi.Input['Resource']]=None, __props__=None):
        if False:
            while True:
                i = 10
        opts = pulumi.ResourceOptions.merge(_utilities.get_resource_opts_defaults(), opts)
        if not isinstance(opts, pulumi.ResourceOptions):
            raise TypeError('Expected resource options to be a ResourceOptions instance')
        if opts.id is not None:
            raise ValueError('ComponentResource classes do not support opts.id')
        else:
            if __props__ is not None:
                raise TypeError('__props__ is only valid when passed in combination with a valid opts.id to get an existing resource')
            __props__ = OtherResourceArgs.__new__(OtherResourceArgs)
            __props__.__dict__['foo'] = foo
        super(OtherResource, __self__).__init__('example::OtherResource', resource_name, __props__, opts, remote=True)

    @property
    @pulumi.getter
    def foo(self) -> pulumi.Output[Optional['Resource']]:
        if False:
            i = 10
            return i + 15
        return pulumi.get(self, 'foo')