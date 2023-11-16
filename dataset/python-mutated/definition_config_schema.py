from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Union
import dagster._check as check
from dagster._config import ConfigAnyInstance, ConfigType, EvaluateValueResult, Field, UserConfigSchema, convert_potential_field, process_config
from dagster._core.errors import DagsterConfigMappingFunctionError, user_code_error_boundary
if TYPE_CHECKING:
    from dagster._core.definitions.configurable import ConfigurableDefinition
CoercableToConfigSchema = Union[None, UserConfigSchema, 'IDefinitionConfigSchema']

def convert_user_facing_definition_config_schema(potential_schema: CoercableToConfigSchema) -> 'IDefinitionConfigSchema':
    if False:
        for i in range(10):
            print('nop')
    if potential_schema is None:
        return DefinitionConfigSchema(Field(ConfigAnyInstance, is_required=False))
    elif isinstance(potential_schema, IDefinitionConfigSchema):
        return potential_schema
    else:
        return DefinitionConfigSchema(convert_potential_field(potential_schema))

class IDefinitionConfigSchema(ABC):

    @abstractmethod
    def as_field(self) -> Field:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @property
    def config_type(self) -> Optional[ConfigType]:
        if False:
            for i in range(10):
                print('nop')
        field = self.as_field()
        return field.config_type if field else None

    @property
    def is_required(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        field = self.as_field()
        return field.is_required if field else False

    @property
    def default_provided(self) -> bool:
        if False:
            print('Hello World!')
        field = self.as_field()
        return field.default_provided if field else False

    @property
    def default_value(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        field = self.as_field()
        check.invariant(self.default_provided, 'Asking for default value when none was provided')
        return field.default_value if field else None

    @property
    def default_value_as_json_str(self) -> str:
        if False:
            while True:
                i = 10
        field = self.as_field()
        check.invariant(self.default_provided, 'Asking for default value when none was provided')
        return field.default_value_as_json_str

    @property
    def description(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        field = self.as_field()
        return field.description if field else None

class DefinitionConfigSchema(IDefinitionConfigSchema):

    def __init__(self, config_field: Field):
        if False:
            for i in range(10):
                print('nop')
        self._config_field = check.inst_param(config_field, 'config_field', Field)

    def as_field(self) -> Field:
        if False:
            return 10
        return self._config_field

def _get_user_code_error_str_lambda(configured_definition: 'ConfigurableDefinition') -> Callable[[], str]:
    if False:
        while True:
            i = 10
    return lambda : 'The config mapping function on a `configured` {} has thrown an unexpected error during its execution.'.format(configured_definition.__class__.__name__)

class ConfiguredDefinitionConfigSchema(IDefinitionConfigSchema):
    parent_def: 'ConfigurableDefinition'
    _current_field: Optional[Field]
    _config_fn: Callable[..., object]

    def __init__(self, parent_definition: 'ConfigurableDefinition', config_schema: Optional[IDefinitionConfigSchema], config_or_config_fn: object):
        if False:
            while True:
                i = 10
        from .configurable import ConfigurableDefinition
        self.parent_def = check.inst_param(parent_definition, 'parent_definition', ConfigurableDefinition)
        check.opt_inst_param(config_schema, 'config_schema', IDefinitionConfigSchema)
        self._current_field = config_schema.as_field() if config_schema else None
        if not callable(config_or_config_fn):
            self._config_fn = lambda _: config_or_config_fn
        else:
            self._config_fn = config_or_config_fn

    def as_field(self) -> Field:
        if False:
            i = 10
            return i + 15
        return check.not_none(self._current_field)

    def _invoke_user_config_fn(self, processed_config: Mapping[str, Any]) -> Mapping[str, object]:
        if False:
            i = 10
            return i + 15
        with user_code_error_boundary(DagsterConfigMappingFunctionError, _get_user_code_error_str_lambda(self.parent_def)):
            return {'config': self._config_fn(processed_config.get('config', {}))}

    def resolve_config(self, processed_config: Mapping[str, object]) -> EvaluateValueResult:
        if False:
            print('Hello World!')
        check.mapping_param(processed_config, 'processed_config')
        config_evr = process_config({'config': self.parent_def.config_field or {}}, self._invoke_user_config_fn(processed_config))
        if config_evr.success:
            return self.parent_def.apply_config_mapping(config_evr.value)
        else:
            return config_evr