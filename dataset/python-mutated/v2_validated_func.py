"""
This module contains an implementation of pydantic v1's ValidateFunction 
modified to validate function arguments and return a pydantic v2 model.

Specifically it allows for us to validate v2 models used as flow/task 
arguments.
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, create_model
from pydantic.v1 import validator
from pydantic.v1.decorator import ValidatedFunction
from pydantic.v1.errors import ConfigError
from pydantic.v1.utils import to_camel
if TYPE_CHECKING:
    ConfigType = Union[None, Type[Any], Dict[str, Any]]
V_POSITIONAL_ONLY_NAME = 'v__positional_only'
V_DUPLICATE_KWARGS = 'v__duplicate_kwargs'

class V2ValidatedFunction(ValidatedFunction):

    def create_model(self, fields: Dict[str, Any], takes_args: bool, takes_kwargs: bool, config: 'ConfigType') -> None:
        if False:
            i = 10
            return i + 15
        pos_args = len(self.arg_mapping)

        class CustomConfig:
            pass
        if not TYPE_CHECKING:
            if isinstance(config, dict):
                CustomConfig = type('Config', (), config)
            elif config is not None:
                CustomConfig = config
        if hasattr(CustomConfig, 'fields') or hasattr(CustomConfig, 'alias_generator'):
            raise ConfigError('Setting the "fields" and "alias_generator" property on custom Config for @validate_arguments is not yet supported, please remove.')

        class DecoratorBaseModel(BaseModel):

            @validator(self.v_args_name, check_fields=False, allow_reuse=True)
            def check_args(cls, v: Optional[List[Any]]) -> Optional[List[Any]]:
                if False:
                    for i in range(10):
                        print('nop')
                if takes_args or v is None:
                    return v
                raise TypeError(f'{pos_args} positional arguments expected but {pos_args + len(v)} given')

            @validator(self.v_kwargs_name, check_fields=False, allow_reuse=True)
            def check_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
                if False:
                    return 10
                if takes_kwargs or v is None:
                    return v
                plural = '' if len(v) == 1 else 's'
                keys = ', '.join(map(repr, v.keys()))
                raise TypeError(f'unexpected keyword argument{plural}: {keys}')

            @validator(V_POSITIONAL_ONLY_NAME, check_fields=False, allow_reuse=True)
            def check_positional_only(cls, v: Optional[List[str]]) -> None:
                if False:
                    return 10
                if v is None:
                    return
                plural = '' if len(v) == 1 else 's'
                keys = ', '.join(map(repr, v))
                raise TypeError(f'positional-only argument{plural} passed as keyword argument{plural}: {keys}')

            @validator(V_DUPLICATE_KWARGS, check_fields=False, allow_reuse=True)
            def check_duplicate_kwargs(cls, v: Optional[List[str]]) -> None:
                if False:
                    while True:
                        i = 10
                if v is None:
                    return
                plural = '' if len(v) == 1 else 's'
                keys = ', '.join(map(repr, v))
                raise TypeError(f'multiple values for argument{plural}: {keys}')

            class Config(CustomConfig):
                extra = getattr(CustomConfig, 'extra', 'forbid')
        self.model = create_model(to_camel(self.raw_function.__name__), __base__=DecoratorBaseModel, **fields)