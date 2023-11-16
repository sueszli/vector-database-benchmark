"""Domain objects relating to parameters."""
from __future__ import annotations
import re
from core import feconf
from core import utils
from core.domain import value_generators_domain
from typing import Dict, List, TypedDict, Union

class CustomizationArgsDict(TypedDict):
    """Dictionary representing the customization_args argument."""
    parse_with_jinja: bool

class CustomizationArgsDictWithValue(CustomizationArgsDict):
    """Dictionary representing the customization_args argument
    containing value key.
    """
    value: str

class CustomizationArgsDictWithValueList(CustomizationArgsDict):
    """Dictionary representing the customization_args argument
    containing list_of_values key.
    """
    list_of_values: List[str]
AllowedCustomizationArgsDict = Union[CustomizationArgsDictWithValue, CustomizationArgsDictWithValueList]

class ParamSpecDict(TypedDict):
    """Dictionary representing the ParamSpec object."""
    obj_type: str

class ParamSpec:
    """Value object for an exploration parameter specification."""

    def __init__(self, obj_type: str) -> None:
        if False:
            i = 10
            return i + 15
        'Initializes a ParamSpec object with the specified object type.\n\n        Args:\n            obj_type: unicode. The object type with which the parameter is\n                initialized.\n        '
        self.obj_type = obj_type

    def to_dict(self) -> ParamSpecDict:
        if False:
            return 10
        'Returns a dict representation of this ParamSpec.\n\n        Returns:\n            dict. A dict with a single key, whose value is the type\n            of the parameter represented by this ParamSpec.\n        '
        return {'obj_type': self.obj_type}

    @classmethod
    def from_dict(cls, param_spec_dict: ParamSpecDict) -> ParamSpec:
        if False:
            return 10
        'Creates a ParamSpec object from its dict representation.\n\n        Args:\n            param_spec_dict: dict. The dictionary containing the specification\n                of the parameter. It contains the following key (object_type).\n                `object_type` determines the data type of the parameter.\n\n        Returns:\n            ParamSpec. A ParamSpec object created from the specified\n            object type.\n        '
        return cls(param_spec_dict['obj_type'])

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Validate the existence of the object class.'
        if self.obj_type not in feconf.SUPPORTED_OBJ_TYPES:
            raise utils.ValidationError('%s is not among the supported object types for parameters: {%s}.' % (self.obj_type, ', '.join(sorted(feconf.SUPPORTED_OBJ_TYPES))))

class ParamChangeDict(TypedDict):
    """Dictionary representing the ParamChange object."""
    name: str
    generator_id: str
    customization_args: AllowedCustomizationArgsDict

class ParamChange:
    """Value object for a parameter change."""

    def __init__(self, name: str, generator_id: str, customization_args: AllowedCustomizationArgsDict) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize a ParamChange object with the specified arguments.\n\n        Args:\n            name: unicode. The name of the parameter.\n            generator_id: unicode. The type of generator used to create the\n                parameter, e.g., "Copier".\n            customization_args: dict. A dict containing the following keys:\n                (value, parse_with_jinja). `value` specifies the value of the\n                parameter, and `parse_with_jinja` indicates whether parsing is\n                to be done with the Jinja template engine. If the parameter is\n                specified using one of several possible values, this dict\n                contains a list (`list_of_values`) of possible values (instead\n                of `value`).\n        '
        self._name = name
        self._generator_id = generator_id
        self._customization_args = customization_args

    @property
    def name(self) -> str:
        if False:
            print('Hello World!')
        'The name of the changing parameter.\n\n        Returns:\n            unicode. The name of the parameter.\n        '
        return self._name

    @property
    def generator(self) -> value_generators_domain.BaseValueGenerator:
        if False:
            print('Hello World!')
        'The value generator used to define the new value of the\n        changing parameter.\n\n        Returns:\n            subclass of BaseValueGenerator. The generator object for the\n            parameter.\n        '
        return value_generators_domain.Registry.get_generator_class_by_id(self._generator_id)()

    @property
    def customization_args(self) -> AllowedCustomizationArgsDict:
        if False:
            print('Hello World!')
        'A dict containing several arguments that determine the changing value\n        of the parameter.\n\n        Returns:\n            dict: A dict specifying the following customization arguments for\n            the parameter. In case of a parameter change to a single value,\n            this dict contains the value of the parameter and a key-value\n            pair specifying whether parsing is done using the Jinja template\n            engine. If the parameter is changed to one amongst several values,\n            this dict contains a list of possible values.\n         '
        return self._customization_args

    def to_dict(self) -> ParamChangeDict:
        if False:
            while True:
                i = 10
        'Returns a dict representing this ParamChange domain object.\n\n        Returns:\n            dict. A dict representation of the ParamChange instance.\n        '
        return {'name': self.name, 'generator_id': self.generator.id, 'customization_args': self.customization_args}

    @classmethod
    def from_dict(cls, param_change_dict: ParamChangeDict) -> ParamChange:
        if False:
            i = 10
            return i + 15
        'Create a ParamChange object with the specified arguments.\n\n        Args:\n            param_change_dict: dict. A dict containing data about the\n                following keys: (customization_args(dict), name, generator_id).\n                `customization_args` is a dict with the following keys:\n                (value, parse_with_jinja). `value` specifies the value of the\n                parameter and `parse_with_jinja` indicates whether parsing\n                change be performed using the Jinja template engine. If the\n                parameter changed to one amongst several values, this dict\n                contains a list of possible values.\n                `name` is the name of the parameter.\n                `generator_id` is the type of value generator used to\n                generate the new value for the parameter.\n\n        Returns:\n            ParamChange. The ParamChange object created from the\n            `param_change_dict` dict, which specifies the name,\n            customization arguments and the generator used.\n        '
        return cls(param_change_dict['name'], param_change_dict['generator_id'], param_change_dict['customization_args'])

    def get_value(self, context_params: Dict[str, str]) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Generates a single value for a parameter change.'
        value: str = self.generator.generate_value(context_params, **self.customization_args)
        return value

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Checks that the properties of this ParamChange object are valid.'
        if not isinstance(self.name, str):
            raise utils.ValidationError('Expected param_change name to be a string, received %s' % self.name)
        if not re.match(feconf.ALPHANUMERIC_REGEX, self.name):
            raise utils.ValidationError('Only parameter names with characters in [a-zA-Z0-9] are accepted.')
        if not isinstance(self._generator_id, str):
            raise utils.ValidationError('Expected generator ID to be a string, received %s ' % self._generator_id)
        try:
            hasattr(self, 'generator')
        except KeyError as e:
            raise utils.ValidationError('Invalid generator ID %s' % self._generator_id) from e
        if not isinstance(self.customization_args, dict):
            raise utils.ValidationError('Expected a dict of customization_args, received %s' % self.customization_args)
        for arg_name in self.customization_args:
            if not isinstance(arg_name, str):
                raise Exception('Invalid parameter change customization_arg name: %s' % arg_name)