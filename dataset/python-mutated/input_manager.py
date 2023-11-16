from abc import ABC, abstractmethod
from functools import update_wrapper
from typing import TYPE_CHECKING, AbstractSet, Callable, Optional, Union, cast, overload
from typing_extensions import TypeAlias, TypeGuard
import dagster._check as check
from dagster._core.decorator_utils import has_at_least_one_parameter
from dagster._core.definitions.config import is_callable_valid_config_arg
from dagster._core.definitions.definition_config_schema import CoercableToConfigSchema, IDefinitionConfigSchema, convert_user_facing_definition_config_schema
from dagster._core.definitions.resource_definition import ResourceDefinition, ResourceFunction
if TYPE_CHECKING:
    from dagster._core.execution.context.input import InputContext
InputLoadFn: TypeAlias = Union[Callable[['InputContext'], object], Callable[[], object]]

class InputManager(ABC):
    """Base interface for classes that are responsible for loading solid inputs."""

    @abstractmethod
    def load_input(self, context: 'InputContext') -> object:
        if False:
            i = 10
            return i + 15
        'The user-defined read method that loads an input to a solid.\n\n        Args:\n            context (InputContext): The input context.\n\n        Returns:\n            Any: The data object.\n        '

class IInputManagerDefinition:

    @property
    @abstractmethod
    def input_config_schema(self) -> IDefinitionConfigSchema:
        if False:
            print('Hello World!')
        'The schema for per-input configuration for inputs that are managed by this\n        input manager.\n        '

class InputManagerDefinition(ResourceDefinition, IInputManagerDefinition):
    """Definition of an input manager resource.

    Input managers load op inputs.

    An InputManagerDefinition is a :py:class:`ResourceDefinition` whose resource_fn returns an
    :py:class:`InputManager`.

    The easiest way to create an InputManagerDefinition is with the
    :py:func:`@input_manager <input_manager>` decorator.
    """

    def __init__(self, resource_fn: ResourceFunction, config_schema: Optional[CoercableToConfigSchema]=None, description: Optional[str]=None, input_config_schema: Optional[CoercableToConfigSchema]=None, required_resource_keys: Optional[AbstractSet[str]]=None, version: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        self._input_config_schema = convert_user_facing_definition_config_schema(input_config_schema)
        super(InputManagerDefinition, self).__init__(resource_fn=resource_fn, config_schema=config_schema, description=description, required_resource_keys=required_resource_keys, version=version)

    @property
    def input_config_schema(self) -> IDefinitionConfigSchema:
        if False:
            for i in range(10):
                print('nop')
        return self._input_config_schema

    def copy_for_configured(self, description: Optional[str], config_schema: CoercableToConfigSchema) -> 'InputManagerDefinition':
        if False:
            print('Hello World!')
        return InputManagerDefinition(config_schema=config_schema, description=description or self.description, resource_fn=self.resource_fn, required_resource_keys=self.required_resource_keys, input_config_schema=self.input_config_schema)

@overload
def input_manager(config_schema: InputLoadFn) -> InputManagerDefinition:
    if False:
        print('Hello World!')
    ...

@overload
def input_manager(config_schema: Optional[CoercableToConfigSchema]=None, description: Optional[str]=None, input_config_schema: Optional[CoercableToConfigSchema]=None, required_resource_keys: Optional[AbstractSet[str]]=None, version: Optional[str]=None) -> Callable[[InputLoadFn], InputManagerDefinition]:
    if False:
        for i in range(10):
            print('nop')
    ...

def input_manager(config_schema: Union[InputLoadFn, Optional[CoercableToConfigSchema]]=None, description: Optional[str]=None, input_config_schema: Optional[CoercableToConfigSchema]=None, required_resource_keys: Optional[AbstractSet[str]]=None, version: Optional[str]=None) -> Union[InputManagerDefinition, Callable[[InputLoadFn], InputManagerDefinition]]:
    if False:
        print('Hello World!')
    'Define an input manager.\n\n    Input managers load op inputs, either from upstream outputs or by providing default values.\n\n    The decorated function should accept a :py:class:`InputContext` and resource config, and return\n    a loaded object that will be passed into one of the inputs of an op.\n\n    The decorator produces an :py:class:`InputManagerDefinition`.\n\n    Args:\n        config_schema (Optional[ConfigSchema]): The schema for the resource-level config. If not\n            set, Dagster will accept any config provided.\n        description (Optional[str]): A human-readable description of the resource.\n        input_config_schema (Optional[ConfigSchema]): A schema for the input-level config. Each\n            input that uses this input manager can be configured separately using this config.\n            If not set, Dagster will accept any config provided.\n        required_resource_keys (Optional[Set[str]]): Keys for the resources required by the input\n            manager.\n        version (Optional[str]): (Experimental) the version of the input manager definition.\n\n    **Examples:**\n\n    .. code-block:: python\n\n        from dagster import input_manager, op, job, In\n\n        @input_manager\n        def csv_loader(_):\n            return read_csv("some/path")\n\n        @op(ins={"input1": In(input_manager_key="csv_loader_key")})\n        def my_op(_, input1):\n            do_stuff(input1)\n\n        @job(resource_defs={"csv_loader_key": csv_loader})\n        def my_job():\n            my_op()\n\n        @input_manager(config_schema={"base_dir": str})\n        def csv_loader(context):\n            return read_csv(context.resource_config["base_dir"] + "/some/path")\n\n        @input_manager(input_config_schema={"path": str})\n        def csv_loader(context):\n            return read_csv(context.config["path"])\n    '
    if _is_input_load_fn(config_schema):
        return _InputManagerDecoratorCallable()(config_schema)

    def _wrap(load_fn: InputLoadFn) -> InputManagerDefinition:
        if False:
            i = 10
            return i + 15
        return _InputManagerDecoratorCallable(config_schema=cast(CoercableToConfigSchema, config_schema), description=description, version=version, input_config_schema=input_config_schema, required_resource_keys=required_resource_keys)(load_fn)
    return _wrap

def _is_input_load_fn(obj: Union[InputLoadFn, CoercableToConfigSchema]) -> TypeGuard[InputLoadFn]:
    if False:
        i = 10
        return i + 15
    return callable(obj) and (not is_callable_valid_config_arg(obj))

class InputManagerWrapper(InputManager):

    def __init__(self, load_fn: InputLoadFn):
        if False:
            for i in range(10):
                print('nop')
        self._load_fn = load_fn

    def load_input(self, context: 'InputContext') -> object:
        if False:
            for i in range(10):
                print('nop')
        intermediate = self._load_fn(context) if has_at_least_one_parameter(self._load_fn) else self._load_fn()
        if isinstance(intermediate, InputManager):
            return intermediate.load_input(context)
        return intermediate

class _InputManagerDecoratorCallable:

    def __init__(self, config_schema: CoercableToConfigSchema=None, description: Optional[str]=None, version: Optional[str]=None, input_config_schema: CoercableToConfigSchema=None, required_resource_keys: Optional[AbstractSet[str]]=None):
        if False:
            print('Hello World!')
        self.config_schema = config_schema
        self.description = check.opt_str_param(description, 'description')
        self.version = check.opt_str_param(version, 'version')
        self.input_config_schema = input_config_schema
        self.required_resource_keys = required_resource_keys

    def __call__(self, load_fn: InputLoadFn) -> InputManagerDefinition:
        if False:
            print('Hello World!')
        check.callable_param(load_fn, 'load_fn')

        def _resource_fn(_):
            if False:
                i = 10
                return i + 15
            return InputManagerWrapper(load_fn)
        input_manager_def = InputManagerDefinition(resource_fn=_resource_fn, config_schema=self.config_schema, description=self.description, version=self.version, input_config_schema=self.input_config_schema, required_resource_keys=self.required_resource_keys)
        update_wrapper(input_manager_def, wrapped=load_fn)
        return input_manager_def