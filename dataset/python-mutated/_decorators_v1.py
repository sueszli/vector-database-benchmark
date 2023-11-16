"""Logic for V1 validators, e.g. `@validator` and `@root_validator`."""
from __future__ import annotations as _annotations
from inspect import Parameter, signature
from typing import Any, Dict, Tuple, Union, cast
from pydantic_core import core_schema
from typing_extensions import Protocol
from ..errors import PydanticUserError
from ._decorators import can_be_positional

class V1OnlyValueValidator(Protocol):
    """A simple validator, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        ...

class V1ValidatorWithValues(Protocol):
    """A validator with `values` argument, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any, values: dict[str, Any]) -> Any:
        if False:
            for i in range(10):
                print('nop')
        ...

class V1ValidatorWithValuesKwOnly(Protocol):
    """A validator with keyword only `values` argument, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any, *, values: dict[str, Any]) -> Any:
        if False:
            i = 10
            return i + 15
        ...

class V1ValidatorWithKwargs(Protocol):
    """A validator with `kwargs` argument, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any, **kwargs: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        ...

class V1ValidatorWithValuesAndKwargs(Protocol):
    """A validator with `values` and `kwargs` arguments, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any, values: dict[str, Any], **kwargs: Any) -> Any:
        if False:
            print('Hello World!')
        ...
V1Validator = Union[V1ValidatorWithValues, V1ValidatorWithValuesKwOnly, V1ValidatorWithKwargs, V1ValidatorWithValuesAndKwargs]

def can_be_keyword(param: Parameter) -> bool:
    if False:
        while True:
            i = 10
    return param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)

def make_generic_v1_field_validator(validator: V1Validator) -> core_schema.WithInfoValidatorFunction:
    if False:
        for i in range(10):
            print('nop')
    'Wrap a V1 style field validator for V2 compatibility.\n\n    Args:\n        validator: The V1 style field validator.\n\n    Returns:\n        A wrapped V2 style field validator.\n\n    Raises:\n        PydanticUserError: If the signature is not supported or the parameters are\n            not available in Pydantic V2.\n    '
    sig = signature(validator)
    needs_values_kw = False
    for (param_num, (param_name, parameter)) in enumerate(sig.parameters.items()):
        if can_be_keyword(parameter) and param_name in ('field', 'config'):
            raise PydanticUserError('The `field` and `config` parameters are not available in Pydantic V2, please use the `info` parameter instead.', code='validator-field-config-info')
        if parameter.kind is Parameter.VAR_KEYWORD:
            needs_values_kw = True
        elif can_be_keyword(parameter) and param_name == 'values':
            needs_values_kw = True
        elif can_be_positional(parameter) and param_num == 0:
            continue
        elif parameter.default is Parameter.empty:
            raise PydanticUserError(f'Unsupported signature for V1 style validator {validator}: {sig} is not supported.', code='validator-v1-signature')
    if needs_values_kw:
        val1 = cast(V1ValidatorWithValues, validator)

        def wrapper1(value: Any, info: core_schema.ValidationInfo) -> Any:
            if False:
                print('Hello World!')
            return val1(value, values=info.data)
        return wrapper1
    else:
        val2 = cast(V1OnlyValueValidator, validator)

        def wrapper2(value: Any, _: core_schema.ValidationInfo) -> Any:
            if False:
                return 10
            return val2(value)
        return wrapper2
RootValidatorValues = Dict[str, Any]
RootValidatorFieldsTuple = Tuple[Any, ...]

class V1RootValidatorFunction(Protocol):
    """A simple root validator, supported for V1 validators and V2 validators."""

    def __call__(self, __values: RootValidatorValues) -> RootValidatorValues:
        if False:
            while True:
                i = 10
        ...

class V2CoreBeforeRootValidator(Protocol):
    """V2 validator with mode='before'."""

    def __call__(self, __values: RootValidatorValues, __info: core_schema.ValidationInfo) -> RootValidatorValues:
        if False:
            for i in range(10):
                print('nop')
        ...

class V2CoreAfterRootValidator(Protocol):
    """V2 validator with mode='after'."""

    def __call__(self, __fields_tuple: RootValidatorFieldsTuple, __info: core_schema.ValidationInfo) -> RootValidatorFieldsTuple:
        if False:
            return 10
        ...

def make_v1_generic_root_validator(validator: V1RootValidatorFunction, pre: bool) -> V2CoreBeforeRootValidator | V2CoreAfterRootValidator:
    if False:
        i = 10
        return i + 15
    'Wrap a V1 style root validator for V2 compatibility.\n\n    Args:\n        validator: The V1 style field validator.\n        pre: Whether the validator is a pre validator.\n\n    Returns:\n        A wrapped V2 style validator.\n    '
    if pre is True:

        def _wrapper1(values: RootValidatorValues, _: core_schema.ValidationInfo) -> RootValidatorValues:
            if False:
                return 10
            return validator(values)
        return _wrapper1

    def _wrapper2(fields_tuple: RootValidatorFieldsTuple, _: core_schema.ValidationInfo) -> RootValidatorFieldsTuple:
        if False:
            while True:
                i = 10
        if len(fields_tuple) == 2:
            (values, init_vars) = fields_tuple
            values = validator(values)
            return (values, init_vars)
        else:
            (model_dict, model_extra, fields_set) = fields_tuple
            if model_extra:
                fields = set(model_dict.keys())
                model_dict.update(model_extra)
                model_dict_new = validator(model_dict)
                for k in list(model_dict_new.keys()):
                    if k not in fields:
                        model_extra[k] = model_dict_new.pop(k)
            else:
                model_dict_new = validator(model_dict)
            return (model_dict_new, model_extra, fields_set)
    return _wrapper2