import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from ... import _utilities
from ._enums import *
__all__ = ['NurseryArgs', 'Nursery']

@pulumi.input_type
class NurseryArgs:

    def __init__(__self__, *, varieties: pulumi.Input[Sequence[pulumi.Input['RubberTreeVariety']]], sizes: Optional[pulumi.Input[Mapping[str, pulumi.Input['TreeSize']]]]=None):
        if False:
            return 10
        "\n        The set of arguments for constructing a Nursery resource.\n        :param pulumi.Input[Sequence[pulumi.Input['RubberTreeVariety']]] varieties: The varieties available\n        :param pulumi.Input[Mapping[str, pulumi.Input['TreeSize']]] sizes: The sizes of trees available\n        "
        pulumi.set(__self__, 'varieties', varieties)
        if sizes is not None:
            pulumi.set(__self__, 'sizes', sizes)

    @property
    @pulumi.getter
    def varieties(self) -> pulumi.Input[Sequence[pulumi.Input['RubberTreeVariety']]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The varieties available\n        '
        return pulumi.get(self, 'varieties')

    @varieties.setter
    def varieties(self, value: pulumi.Input[Sequence[pulumi.Input['RubberTreeVariety']]]):
        if False:
            print('Hello World!')
        pulumi.set(self, 'varieties', value)

    @property
    @pulumi.getter
    def sizes(self) -> Optional[pulumi.Input[Mapping[str, pulumi.Input['TreeSize']]]]:
        if False:
            i = 10
            return i + 15
        '\n        The sizes of trees available\n        '
        return pulumi.get(self, 'sizes')

    @sizes.setter
    def sizes(self, value: Optional[pulumi.Input[Mapping[str, pulumi.Input['TreeSize']]]]):
        if False:
            while True:
                i = 10
        pulumi.set(self, 'sizes', value)

class Nursery(pulumi.CustomResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, sizes: Optional[pulumi.Input[Mapping[str, pulumi.Input['TreeSize']]]]=None, varieties: Optional[pulumi.Input[Sequence[pulumi.Input['RubberTreeVariety']]]]=None, __props__=None):
        if False:
            while True:
                i = 10
        "\n        Create a Nursery resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        :param pulumi.Input[Mapping[str, pulumi.Input['TreeSize']]] sizes: The sizes of trees available\n        :param pulumi.Input[Sequence[pulumi.Input['RubberTreeVariety']]] varieties: The varieties available\n        "
        ...

    @overload
    def __init__(__self__, resource_name: str, args: NurseryArgs, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            i = 10
            return i + 15
        "\n        Create a Nursery resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param NurseryArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            return 10
        (resource_args, opts) = _utilities.get_resource_args_opts(NurseryArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, sizes: Optional[pulumi.Input[Mapping[str, pulumi.Input['TreeSize']]]]=None, varieties: Optional[pulumi.Input[Sequence[pulumi.Input['RubberTreeVariety']]]]=None, __props__=None):
        if False:
            print('Hello World!')
        opts = pulumi.ResourceOptions.merge(_utilities.get_resource_opts_defaults(), opts)
        if not isinstance(opts, pulumi.ResourceOptions):
            raise TypeError('Expected resource options to be a ResourceOptions instance')
        if opts.id is None:
            if __props__ is not None:
                raise TypeError('__props__ is only valid when passed in combination with a valid opts.id to get an existing resource')
            __props__ = NurseryArgs.__new__(NurseryArgs)
            __props__.__dict__['sizes'] = sizes
            if varieties is None and (not opts.urn):
                raise TypeError("Missing required property 'varieties'")
            __props__.__dict__['varieties'] = varieties
        super(Nursery, __self__).__init__('plant:tree/v1:Nursery', resource_name, __props__, opts)

    @staticmethod
    def get(resource_name: str, id: pulumi.Input[str], opts: Optional[pulumi.ResourceOptions]=None) -> 'Nursery':
        if False:
            return 10
        "\n        Get an existing Nursery resource's state with the given name, id, and optional extra\n        properties used to qualify the lookup.\n\n        :param str resource_name: The unique name of the resulting resource.\n        :param pulumi.Input[str] id: The unique provider ID of the resource to lookup.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        opts = pulumi.ResourceOptions.merge(opts, pulumi.ResourceOptions(id=id))
        __props__ = NurseryArgs.__new__(NurseryArgs)
        return Nursery(resource_name, opts=opts, __props__=__props__)