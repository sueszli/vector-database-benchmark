import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
from . import outputs
from ._inputs import *
__all__ = ['PersonArgs', 'Person']

@pulumi.input_type
class PersonArgs:

    def __init__(__self__, *, name: Optional[pulumi.Input[str]]=None, pets: Optional[pulumi.Input[Sequence[pulumi.Input['PetArgs']]]]=None):
        if False:
            i = 10
            return i + 15
        '\n        The set of arguments for constructing a Person resource.\n        '
        if name is not None:
            pulumi.set(__self__, 'name', name)
        if pets is not None:
            pulumi.set(__self__, 'pets', pets)

    @property
    @pulumi.getter
    def name(self) -> Optional[pulumi.Input[str]]:
        if False:
            i = 10
            return i + 15
        return pulumi.get(self, 'name')

    @name.setter
    def name(self, value: Optional[pulumi.Input[str]]):
        if False:
            i = 10
            return i + 15
        pulumi.set(self, 'name', value)

    @property
    @pulumi.getter
    def pets(self) -> Optional[pulumi.Input[Sequence[pulumi.Input['PetArgs']]]]:
        if False:
            print('Hello World!')
        return pulumi.get(self, 'pets')

    @pets.setter
    def pets(self, value: Optional[pulumi.Input[Sequence[pulumi.Input['PetArgs']]]]):
        if False:
            i = 10
            return i + 15
        pulumi.set(self, 'pets', value)

class Person(pulumi.CustomResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, name: Optional[pulumi.Input[str]]=None, pets: Optional[pulumi.Input[Sequence[pulumi.Input[pulumi.InputType['PetArgs']]]]]=None, __props__=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a Person resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        '
        ...

    @overload
    def __init__(__self__, resource_name: str, args: Optional[PersonArgs]=None, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            print('Hello World!')
        "\n        Create a Person resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param PersonArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            print('Hello World!')
        (resource_args, opts) = _utilities.get_resource_args_opts(PersonArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, name: Optional[pulumi.Input[str]]=None, pets: Optional[pulumi.Input[Sequence[pulumi.Input[pulumi.InputType['PetArgs']]]]]=None, __props__=None):
        if False:
            return 10
        opts = pulumi.ResourceOptions.merge(_utilities.get_resource_opts_defaults(), opts)
        if not isinstance(opts, pulumi.ResourceOptions):
            raise TypeError('Expected resource options to be a ResourceOptions instance')
        if opts.id is None:
            if __props__ is not None:
                raise TypeError('__props__ is only valid when passed in combination with a valid opts.id to get an existing resource')
            __props__ = PersonArgs.__new__(PersonArgs)
            __props__.__dict__['name'] = name
            __props__.__dict__['pets'] = pets
        super(Person, __self__).__init__('example::Person', resource_name, __props__, opts)

    @staticmethod
    def get(resource_name: str, id: pulumi.Input[str], opts: Optional[pulumi.ResourceOptions]=None) -> 'Person':
        if False:
            for i in range(10):
                print('nop')
        "\n        Get an existing Person resource's state with the given name, id, and optional extra\n        properties used to qualify the lookup.\n\n        :param str resource_name: The unique name of the resulting resource.\n        :param pulumi.Input[str] id: The unique provider ID of the resource to lookup.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        opts = pulumi.ResourceOptions.merge(opts, pulumi.ResourceOptions(id=id))
        __props__ = PersonArgs.__new__(PersonArgs)
        __props__.__dict__['name'] = None
        __props__.__dict__['pets'] = None
        return Person(resource_name, opts=opts, __props__=__props__)

    @property
    @pulumi.getter
    def name(self) -> pulumi.Output[Optional[str]]:
        if False:
            for i in range(10):
                print('nop')
        return pulumi.get(self, 'name')

    @property
    @pulumi.getter
    def pets(self) -> pulumi.Output[Optional[Sequence['outputs.Pet']]]:
        if False:
            for i in range(10):
                print('nop')
        return pulumi.get(self, 'pets')