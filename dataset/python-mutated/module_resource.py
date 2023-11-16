import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
from ._enums import *
__all__ = ['ModuleResourceArgs', 'ModuleResource']

@pulumi.input_type
class ModuleResourceArgs:

    def __init__(__self__, *, plain_required_bool: Optional[bool]=None, plain_required_const: Optional[str]=None, plain_required_number: Optional[float]=None, plain_required_string: Optional[str]=None, required_bool: Optional[pulumi.Input[bool]]=None, required_enum: Optional[pulumi.Input['EnumThing']]=None, required_number: Optional[pulumi.Input[float]]=None, required_string: Optional[pulumi.Input[str]]=None, optional_bool: Optional[pulumi.Input[bool]]=None, optional_const: Optional[pulumi.Input[str]]=None, optional_enum: Optional[pulumi.Input['EnumThing']]=None, optional_number: Optional[pulumi.Input[float]]=None, optional_string: Optional[pulumi.Input[str]]=None, plain_optional_bool: Optional[bool]=None, plain_optional_const: Optional[str]=None, plain_optional_number: Optional[float]=None, plain_optional_string: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        '\n        The set of arguments for constructing a ModuleResource resource.\n        '
        if plain_required_bool is None:
            plain_required_bool = True
        pulumi.set(__self__, 'plain_required_bool', plain_required_bool)
        if plain_required_const is None:
            plain_required_const = 'another'
        pulumi.set(__self__, 'plain_required_const', 'val')
        if plain_required_number is None:
            plain_required_number = 42
        pulumi.set(__self__, 'plain_required_number', plain_required_number)
        if plain_required_string is None:
            plain_required_string = 'buzzer'
        pulumi.set(__self__, 'plain_required_string', plain_required_string)
        if required_bool is None:
            required_bool = True
        pulumi.set(__self__, 'required_bool', required_bool)
        if required_enum is None:
            required_enum = 4
        pulumi.set(__self__, 'required_enum', required_enum)
        if required_number is None:
            required_number = 42
        pulumi.set(__self__, 'required_number', required_number)
        if required_string is None:
            required_string = 'buzzer'
        pulumi.set(__self__, 'required_string', required_string)
        if optional_bool is None:
            optional_bool = True
        if optional_bool is not None:
            pulumi.set(__self__, 'optional_bool', optional_bool)
        if optional_const is None:
            optional_const = 'another'
        if optional_const is not None:
            pulumi.set(__self__, 'optional_const', 'val')
        if optional_enum is None:
            optional_enum = 8
        if optional_enum is not None:
            pulumi.set(__self__, 'optional_enum', optional_enum)
        if optional_number is None:
            optional_number = 42
        if optional_number is not None:
            pulumi.set(__self__, 'optional_number', optional_number)
        if optional_string is None:
            optional_string = 'buzzer'
        if optional_string is not None:
            pulumi.set(__self__, 'optional_string', optional_string)
        if plain_optional_bool is None:
            plain_optional_bool = True
        if plain_optional_bool is not None:
            pulumi.set(__self__, 'plain_optional_bool', plain_optional_bool)
        if plain_optional_const is None:
            plain_optional_const = 'another'
        if plain_optional_const is not None:
            pulumi.set(__self__, 'plain_optional_const', 'val')
        if plain_optional_number is None:
            plain_optional_number = 42
        if plain_optional_number is not None:
            pulumi.set(__self__, 'plain_optional_number', plain_optional_number)
        if plain_optional_string is None:
            plain_optional_string = 'buzzer'
        if plain_optional_string is not None:
            pulumi.set(__self__, 'plain_optional_string', plain_optional_string)

    @property
    @pulumi.getter
    def plain_required_bool(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return pulumi.get(self, 'plain_required_bool')

    @plain_required_bool.setter
    def plain_required_bool(self, value: bool):
        if False:
            for i in range(10):
                print('nop')
        pulumi.set(self, 'plain_required_bool', value)

    @property
    @pulumi.getter
    def plain_required_const(self) -> str:
        if False:
            return 10
        return pulumi.get(self, 'plain_required_const')

    @plain_required_const.setter
    def plain_required_const(self, value: str):
        if False:
            return 10
        pulumi.set(self, 'plain_required_const', value)

    @property
    @pulumi.getter
    def plain_required_number(self) -> float:
        if False:
            print('Hello World!')
        return pulumi.get(self, 'plain_required_number')

    @plain_required_number.setter
    def plain_required_number(self, value: float):
        if False:
            i = 10
            return i + 15
        pulumi.set(self, 'plain_required_number', value)

    @property
    @pulumi.getter
    def plain_required_string(self) -> str:
        if False:
            i = 10
            return i + 15
        return pulumi.get(self, 'plain_required_string')

    @plain_required_string.setter
    def plain_required_string(self, value: str):
        if False:
            i = 10
            return i + 15
        pulumi.set(self, 'plain_required_string', value)

    @property
    @pulumi.getter
    def required_bool(self) -> pulumi.Input[bool]:
        if False:
            i = 10
            return i + 15
        return pulumi.get(self, 'required_bool')

    @required_bool.setter
    def required_bool(self, value: pulumi.Input[bool]):
        if False:
            i = 10
            return i + 15
        pulumi.set(self, 'required_bool', value)

    @property
    @pulumi.getter
    def required_enum(self) -> pulumi.Input['EnumThing']:
        if False:
            while True:
                i = 10
        return pulumi.get(self, 'required_enum')

    @required_enum.setter
    def required_enum(self, value: pulumi.Input['EnumThing']):
        if False:
            print('Hello World!')
        pulumi.set(self, 'required_enum', value)

    @property
    @pulumi.getter
    def required_number(self) -> pulumi.Input[float]:
        if False:
            for i in range(10):
                print('nop')
        return pulumi.get(self, 'required_number')

    @required_number.setter
    def required_number(self, value: pulumi.Input[float]):
        if False:
            while True:
                i = 10
        pulumi.set(self, 'required_number', value)

    @property
    @pulumi.getter
    def required_string(self) -> pulumi.Input[str]:
        if False:
            return 10
        return pulumi.get(self, 'required_string')

    @required_string.setter
    def required_string(self, value: pulumi.Input[str]):
        if False:
            i = 10
            return i + 15
        pulumi.set(self, 'required_string', value)

    @property
    @pulumi.getter
    def optional_bool(self) -> Optional[pulumi.Input[bool]]:
        if False:
            print('Hello World!')
        return pulumi.get(self, 'optional_bool')

    @optional_bool.setter
    def optional_bool(self, value: Optional[pulumi.Input[bool]]):
        if False:
            return 10
        pulumi.set(self, 'optional_bool', value)

    @property
    @pulumi.getter
    def optional_const(self) -> Optional[pulumi.Input[str]]:
        if False:
            while True:
                i = 10
        return pulumi.get(self, 'optional_const')

    @optional_const.setter
    def optional_const(self, value: Optional[pulumi.Input[str]]):
        if False:
            i = 10
            return i + 15
        pulumi.set(self, 'optional_const', value)

    @property
    @pulumi.getter
    def optional_enum(self) -> Optional[pulumi.Input['EnumThing']]:
        if False:
            print('Hello World!')
        return pulumi.get(self, 'optional_enum')

    @optional_enum.setter
    def optional_enum(self, value: Optional[pulumi.Input['EnumThing']]):
        if False:
            i = 10
            return i + 15
        pulumi.set(self, 'optional_enum', value)

    @property
    @pulumi.getter
    def optional_number(self) -> Optional[pulumi.Input[float]]:
        if False:
            while True:
                i = 10
        return pulumi.get(self, 'optional_number')

    @optional_number.setter
    def optional_number(self, value: Optional[pulumi.Input[float]]):
        if False:
            for i in range(10):
                print('nop')
        pulumi.set(self, 'optional_number', value)

    @property
    @pulumi.getter
    def optional_string(self) -> Optional[pulumi.Input[str]]:
        if False:
            while True:
                i = 10
        return pulumi.get(self, 'optional_string')

    @optional_string.setter
    def optional_string(self, value: Optional[pulumi.Input[str]]):
        if False:
            i = 10
            return i + 15
        pulumi.set(self, 'optional_string', value)

    @property
    @pulumi.getter
    def plain_optional_bool(self) -> Optional[bool]:
        if False:
            return 10
        return pulumi.get(self, 'plain_optional_bool')

    @plain_optional_bool.setter
    def plain_optional_bool(self, value: Optional[bool]):
        if False:
            while True:
                i = 10
        pulumi.set(self, 'plain_optional_bool', value)

    @property
    @pulumi.getter
    def plain_optional_const(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return pulumi.get(self, 'plain_optional_const')

    @plain_optional_const.setter
    def plain_optional_const(self, value: Optional[str]):
        if False:
            return 10
        pulumi.set(self, 'plain_optional_const', value)

    @property
    @pulumi.getter
    def plain_optional_number(self) -> Optional[float]:
        if False:
            return 10
        return pulumi.get(self, 'plain_optional_number')

    @plain_optional_number.setter
    def plain_optional_number(self, value: Optional[float]):
        if False:
            while True:
                i = 10
        pulumi.set(self, 'plain_optional_number', value)

    @property
    @pulumi.getter
    def plain_optional_string(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return pulumi.get(self, 'plain_optional_string')

    @plain_optional_string.setter
    def plain_optional_string(self, value: Optional[str]):
        if False:
            for i in range(10):
                print('nop')
        pulumi.set(self, 'plain_optional_string', value)

class ModuleResource(pulumi.CustomResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, optional_bool: Optional[pulumi.Input[bool]]=None, optional_const: Optional[pulumi.Input[str]]=None, optional_enum: Optional[pulumi.Input['EnumThing']]=None, optional_number: Optional[pulumi.Input[float]]=None, optional_string: Optional[pulumi.Input[str]]=None, plain_optional_bool: Optional[bool]=None, plain_optional_const: Optional[str]=None, plain_optional_number: Optional[float]=None, plain_optional_string: Optional[str]=None, plain_required_bool: Optional[bool]=None, plain_required_const: Optional[str]=None, plain_required_number: Optional[float]=None, plain_required_string: Optional[str]=None, required_bool: Optional[pulumi.Input[bool]]=None, required_enum: Optional[pulumi.Input['EnumThing']]=None, required_number: Optional[pulumi.Input[float]]=None, required_string: Optional[pulumi.Input[str]]=None, __props__=None):
        if False:
            while True:
                i = 10
        '\n        Create a ModuleResource resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        '
        ...

    @overload
    def __init__(__self__, resource_name: str, args: ModuleResourceArgs, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            while True:
                i = 10
        "\n        Create a ModuleResource resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param ModuleResourceArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (resource_args, opts) = _utilities.get_resource_args_opts(ModuleResourceArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, optional_bool: Optional[pulumi.Input[bool]]=None, optional_const: Optional[pulumi.Input[str]]=None, optional_enum: Optional[pulumi.Input['EnumThing']]=None, optional_number: Optional[pulumi.Input[float]]=None, optional_string: Optional[pulumi.Input[str]]=None, plain_optional_bool: Optional[bool]=None, plain_optional_const: Optional[str]=None, plain_optional_number: Optional[float]=None, plain_optional_string: Optional[str]=None, plain_required_bool: Optional[bool]=None, plain_required_const: Optional[str]=None, plain_required_number: Optional[float]=None, plain_required_string: Optional[str]=None, required_bool: Optional[pulumi.Input[bool]]=None, required_enum: Optional[pulumi.Input['EnumThing']]=None, required_number: Optional[pulumi.Input[float]]=None, required_string: Optional[pulumi.Input[str]]=None, __props__=None):
        if False:
            for i in range(10):
                print('nop')
        opts = pulumi.ResourceOptions.merge(_utilities.get_resource_opts_defaults(), opts)
        if not isinstance(opts, pulumi.ResourceOptions):
            raise TypeError('Expected resource options to be a ResourceOptions instance')
        if opts.id is None:
            if __props__ is not None:
                raise TypeError('__props__ is only valid when passed in combination with a valid opts.id to get an existing resource')
            __props__ = ModuleResourceArgs.__new__(ModuleResourceArgs)
            if optional_bool is None:
                optional_bool = True
            __props__.__dict__['optional_bool'] = optional_bool
            if optional_const is None:
                optional_const = 'another'
            __props__.__dict__['optional_const'] = 'val'
            if optional_enum is None:
                optional_enum = 8
            __props__.__dict__['optional_enum'] = optional_enum
            if optional_number is None:
                optional_number = 42
            __props__.__dict__['optional_number'] = optional_number
            if optional_string is None:
                optional_string = 'buzzer'
            __props__.__dict__['optional_string'] = optional_string
            if plain_optional_bool is None:
                plain_optional_bool = True
            __props__.__dict__['plain_optional_bool'] = plain_optional_bool
            if plain_optional_const is None:
                plain_optional_const = 'another'
            __props__.__dict__['plain_optional_const'] = 'val'
            if plain_optional_number is None:
                plain_optional_number = 42
            __props__.__dict__['plain_optional_number'] = plain_optional_number
            if plain_optional_string is None:
                plain_optional_string = 'buzzer'
            __props__.__dict__['plain_optional_string'] = plain_optional_string
            if plain_required_bool is None:
                plain_required_bool = True
            if plain_required_bool is None and (not opts.urn):
                raise TypeError("Missing required property 'plain_required_bool'")
            __props__.__dict__['plain_required_bool'] = plain_required_bool
            if plain_required_const is None:
                plain_required_const = 'another'
            if plain_required_const is None and (not opts.urn):
                raise TypeError("Missing required property 'plain_required_const'")
            __props__.__dict__['plain_required_const'] = 'val'
            if plain_required_number is None:
                plain_required_number = 42
            if plain_required_number is None and (not opts.urn):
                raise TypeError("Missing required property 'plain_required_number'")
            __props__.__dict__['plain_required_number'] = plain_required_number
            if plain_required_string is None:
                plain_required_string = 'buzzer'
            if plain_required_string is None and (not opts.urn):
                raise TypeError("Missing required property 'plain_required_string'")
            __props__.__dict__['plain_required_string'] = plain_required_string
            if required_bool is None:
                required_bool = True
            if required_bool is None and (not opts.urn):
                raise TypeError("Missing required property 'required_bool'")
            __props__.__dict__['required_bool'] = required_bool
            if required_enum is None:
                required_enum = 4
            if required_enum is None and (not opts.urn):
                raise TypeError("Missing required property 'required_enum'")
            __props__.__dict__['required_enum'] = required_enum
            if required_number is None:
                required_number = 42
            if required_number is None and (not opts.urn):
                raise TypeError("Missing required property 'required_number'")
            __props__.__dict__['required_number'] = required_number
            if required_string is None:
                required_string = 'buzzer'
            if required_string is None and (not opts.urn):
                raise TypeError("Missing required property 'required_string'")
            __props__.__dict__['required_string'] = required_string
        super(ModuleResource, __self__).__init__('foobar::ModuleResource', resource_name, __props__, opts)

    @staticmethod
    def get(resource_name: str, id: pulumi.Input[str], opts: Optional[pulumi.ResourceOptions]=None) -> 'ModuleResource':
        if False:
            return 10
        "\n        Get an existing ModuleResource resource's state with the given name, id, and optional extra\n        properties used to qualify the lookup.\n\n        :param str resource_name: The unique name of the resulting resource.\n        :param pulumi.Input[str] id: The unique provider ID of the resource to lookup.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        opts = pulumi.ResourceOptions.merge(opts, pulumi.ResourceOptions(id=id))
        __props__ = ModuleResourceArgs.__new__(ModuleResourceArgs)
        __props__.__dict__['optional_bool'] = None
        return ModuleResource(resource_name, opts=opts, __props__=__props__)

    @property
    @pulumi.getter
    def optional_bool(self) -> pulumi.Output[Optional[bool]]:
        if False:
            return 10
        return pulumi.get(self, 'optional_bool')