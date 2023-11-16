from functools import update_wrapper
from typing import TYPE_CHECKING, AbstractSet, Any, Callable, Dict, Iterator, Mapping, Optional, Union, cast, overload
from typing_extensions import TypeAlias
import dagster._check as check
from dagster._annotations import experimental_param, public
from dagster._core.decorator_utils import format_docstring_for_description
from dagster._core.definitions.config import is_callable_valid_config_arg
from dagster._core.definitions.configurable import AnonymousConfigurableDefinition
from dagster._core.errors import DagsterInvalidDefinitionError, DagsterInvalidInvocationError
from dagster._utils import IHasInternalInit
from ..decorator_utils import get_function_params, has_at_least_one_parameter, is_required_param, positional_arg_name_list, validate_expected_params
from .definition_config_schema import CoercableToConfigSchema, IDefinitionConfigSchema, convert_user_facing_definition_config_schema
from .resource_invocation import resource_invocation_result
from .resource_requirement import RequiresResources, ResourceDependencyRequirement, ResourceRequirement
from .scoped_resources_builder import IContainsGenerator as IContainsGenerator, Resources as Resources, ScopedResourcesBuilder as ScopedResourcesBuilder
if TYPE_CHECKING:
    from dagster._core.execution.resources_init import InitResourceContext
ResourceFunctionWithContext: TypeAlias = Callable[['InitResourceContext'], Any]
ResourceFunctionWithoutContext: TypeAlias = Callable[[], Any]
ResourceFunction: TypeAlias = Union[ResourceFunctionWithContext, ResourceFunctionWithoutContext]

@experimental_param(param='version')
class ResourceDefinition(AnonymousConfigurableDefinition, RequiresResources, IHasInternalInit):
    """Core class for defining resources.

    Resources are scoped ways to make external resources (like database connections) available to
    ops and assets during job execution and to clean up after execution resolves.

    If resource_fn yields once rather than returning (in the manner of functions decorable with
    :py:func:`@contextlib.contextmanager <python:contextlib.contextmanager>`) then the body of the
    function after the yield will be run after execution resolves, allowing users to write their
    own teardown/cleanup logic.

    Depending on your executor, resources may be instantiated and cleaned up more than once in a
    job execution.

    Args:
        resource_fn (Callable[[InitResourceContext], Any]): User-provided function to instantiate
            the resource, which will be made available to executions keyed on the
            ``context.resources`` object.
        config_schema (Optional[ConfigSchema): The schema for the config. If set, Dagster will check
            that config provided for the resource matches this schema and fail if it does not. If
            not set, Dagster will accept any config provided for the resource.
        description (Optional[str]): A human-readable description of the resource.
        required_resource_keys: (Optional[Set[str]]) Keys for the resources required by this
            resource. A DagsterInvariantViolationError will be raised during initialization if
            dependencies are cyclic.
        version (Optional[str]): (Experimental) The version of the resource's definition fn. Two
            wrapped resource functions should only have the same version if they produce the same
            resource definition when provided with the same inputs.
    """

    def __init__(self, resource_fn: ResourceFunction, config_schema: CoercableToConfigSchema=None, description: Optional[str]=None, required_resource_keys: Optional[AbstractSet[str]]=None, version: Optional[str]=None):
        if False:
            while True:
                i = 10
        self._resource_fn = check.callable_param(resource_fn, 'resource_fn')
        self._config_schema = convert_user_facing_definition_config_schema(config_schema)
        self._description = check.opt_str_param(description, 'description')
        self._required_resource_keys = check.opt_set_param(required_resource_keys, 'required_resource_keys')
        self._version = check.opt_str_param(version, 'version')
        self._dagster_maintained = False
        self._hardcoded_resource_type = None

    @staticmethod
    def dagster_internal_init(*, resource_fn: ResourceFunction, config_schema: CoercableToConfigSchema, description: Optional[str], required_resource_keys: Optional[AbstractSet[str]], version: Optional[str]) -> 'ResourceDefinition':
        if False:
            for i in range(10):
                print('nop')
        return ResourceDefinition(resource_fn=resource_fn, config_schema=config_schema, description=description, required_resource_keys=required_resource_keys, version=version)

    @property
    def resource_fn(self) -> ResourceFunction:
        if False:
            while True:
                i = 10
        return self._resource_fn

    @property
    def config_schema(self) -> IDefinitionConfigSchema:
        if False:
            while True:
                i = 10
        return self._config_schema

    @public
    @property
    def description(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        'A human-readable description of the resource.'
        return self._description

    @public
    @property
    def version(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        'A string which can be used to identify a particular code version of a resource definition.'
        return self._version

    @public
    @property
    def required_resource_keys(self) -> AbstractSet[str]:
        if False:
            for i in range(10):
                print('nop')
        "A set of the resource keys that this resource depends on. These keys will be made available\n        to the resource's init context during execution, and the resource will not be instantiated\n        until all required resources are available.\n        "
        return self._required_resource_keys

    def _is_dagster_maintained(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._dagster_maintained

    @public
    @staticmethod
    def none_resource(description: Optional[str]=None) -> 'ResourceDefinition':
        if False:
            return 10
        'A helper function that returns a none resource.\n\n        Args:\n            description ([Optional[str]]): The description of the resource. Defaults to None.\n\n        Returns:\n            [ResourceDefinition]: A resource that does nothing.\n        '
        return ResourceDefinition.hardcoded_resource(value=None, description=description)

    @public
    @staticmethod
    def hardcoded_resource(value: Any, description: Optional[str]=None) -> 'ResourceDefinition':
        if False:
            print('Hello World!')
        'A helper function that creates a ``ResourceDefinition`` with a hardcoded object.\n\n        Args:\n            value (Any): The value that will be accessible via context.resources.resource_name.\n            description ([Optional[str]]): The description of the resource. Defaults to None.\n\n        Returns:\n            [ResourceDefinition]: A hardcoded resource.\n        '
        resource_def = ResourceDefinition(resource_fn=lambda _init_context: value, description=description)
        if hasattr(value, '_is_dagster_maintained'):
            resource_def._dagster_maintained = value._is_dagster_maintained()
            resource_def._hardcoded_resource_type = type(value)
        return resource_def

    @public
    @staticmethod
    def mock_resource(description: Optional[str]=None) -> 'ResourceDefinition':
        if False:
            for i in range(10):
                print('nop')
        'A helper function that creates a ``ResourceDefinition`` which wraps a ``mock.MagicMock``.\n\n        Args:\n            description ([Optional[str]]): The description of the resource. Defaults to None.\n\n        Returns:\n            [ResourceDefinition]: A resource that creates the magic methods automatically and helps\n                you mock existing resources.\n        '
        from unittest import mock
        return ResourceDefinition(resource_fn=lambda _init_context: mock.MagicMock(), description=description)

    @public
    @staticmethod
    def string_resource(description: Optional[str]=None) -> 'ResourceDefinition':
        if False:
            while True:
                i = 10
        'Creates a ``ResourceDefinition`` which takes in a single string as configuration\n        and returns this configured string to any ops or assets which depend on it.\n\n        Args:\n            description ([Optional[str]]): The description of the string resource. Defaults to None.\n\n        Returns:\n            [ResourceDefinition]: A resource that takes in a single string as configuration and\n                returns that string.\n        '
        return ResourceDefinition(resource_fn=lambda init_context: init_context.resource_config, config_schema=str, description=description)

    def copy_for_configured(self, description: Optional[str], config_schema: CoercableToConfigSchema) -> 'ResourceDefinition':
        if False:
            i = 10
            return i + 15
        resource_def = ResourceDefinition.dagster_internal_init(config_schema=config_schema, description=description or self.description, resource_fn=self.resource_fn, required_resource_keys=self.required_resource_keys, version=self.version)
        resource_def._dagster_maintained = self._is_dagster_maintained()
        return resource_def

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        from dagster._core.execution.context.init import UnboundInitResourceContext
        if has_at_least_one_parameter(self.resource_fn):
            if len(args) + len(kwargs) == 0:
                raise DagsterInvalidInvocationError('Resource initialization function has context argument, but no context was provided when invoking.')
            if len(args) + len(kwargs) > 1:
                raise DagsterInvalidInvocationError('Initialization of resource received multiple arguments. Only a first positional context parameter should be provided when invoking.')
            context_param_name = get_function_params(self.resource_fn)[0].name
            if args:
                check.opt_inst_param(args[0], context_param_name, UnboundInitResourceContext)
                return resource_invocation_result(self, cast(Optional[UnboundInitResourceContext], args[0]))
            else:
                if context_param_name not in kwargs:
                    raise DagsterInvalidInvocationError(f"Resource initialization expected argument '{context_param_name}'.")
                check.opt_inst_param(kwargs[context_param_name], context_param_name, UnboundInitResourceContext)
                return resource_invocation_result(self, cast(Optional[UnboundInitResourceContext], kwargs[context_param_name]))
        elif len(args) + len(kwargs) > 0:
            raise DagsterInvalidInvocationError('Attempted to invoke resource with argument, but underlying function has no context argument. Either specify a context argument on the resource function, or remove the passed-in argument.')
        else:
            return resource_invocation_result(self, None)

    def get_resource_requirements(self, outer_context: Optional[object]=None) -> Iterator[ResourceRequirement]:
        if False:
            while True:
                i = 10
        source_key = cast(str, outer_context)
        for resource_key in sorted(list(self.required_resource_keys)):
            yield ResourceDependencyRequirement(key=resource_key, source_key=source_key)

def dagster_maintained_resource(resource_def: ResourceDefinition) -> ResourceDefinition:
    if False:
        i = 10
        return i + 15
    resource_def._dagster_maintained = True
    return resource_def

class _ResourceDecoratorCallable:

    def __init__(self, config_schema: Optional[Mapping[str, Any]]=None, description: Optional[str]=None, required_resource_keys: Optional[AbstractSet[str]]=None, version: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        self.config_schema = config_schema
        self.description = check.opt_str_param(description, 'description')
        self.version = check.opt_str_param(version, 'version')
        self.required_resource_keys = check.opt_set_param(required_resource_keys, 'required_resource_keys')

    def __call__(self, resource_fn: ResourceFunction) -> ResourceDefinition:
        if False:
            return 10
        check.callable_param(resource_fn, 'resource_fn')
        any_name = ['*'] if has_at_least_one_parameter(resource_fn) else []
        params = get_function_params(resource_fn)
        missing_positional = validate_expected_params(params, any_name)
        if missing_positional:
            raise DagsterInvalidDefinitionError(f"@resource decorated function '{resource_fn.__name__}' expects a single positional argument.")
        extras = params[len(any_name):]
        required_extras = list(filter(is_required_param, extras))
        if required_extras:
            raise DagsterInvalidDefinitionError(f"@resource decorated function '{resource_fn.__name__}' expects only a single positional required argument. Got required extra params {', '.join(positional_arg_name_list(required_extras))}")
        resource_def = ResourceDefinition.dagster_internal_init(resource_fn=resource_fn, config_schema=self.config_schema, description=self.description or format_docstring_for_description(resource_fn), version=self.version, required_resource_keys=self.required_resource_keys)
        update_wrapper(resource_def, wrapped=resource_fn)
        return resource_def

@overload
def resource(config_schema: ResourceFunction) -> ResourceDefinition:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def resource(config_schema: CoercableToConfigSchema=..., description: Optional[str]=..., required_resource_keys: Optional[AbstractSet[str]]=..., version: Optional[str]=...) -> Callable[[ResourceFunction], 'ResourceDefinition']:
    if False:
        return 10
    ...

def resource(config_schema: Union[ResourceFunction, CoercableToConfigSchema]=None, description: Optional[str]=None, required_resource_keys: Optional[AbstractSet[str]]=None, version: Optional[str]=None) -> Union[Callable[[ResourceFunction], 'ResourceDefinition'], 'ResourceDefinition']:
    if False:
        for i in range(10):
            print('nop')
    'Define a resource.\n\n    The decorated function should accept an :py:class:`InitResourceContext` and return an instance of\n    the resource. This function will become the ``resource_fn`` of an underlying\n    :py:class:`ResourceDefinition`.\n\n    If the decorated function yields once rather than returning (in the manner of functions\n    decorable with :py:func:`@contextlib.contextmanager <python:contextlib.contextmanager>`) then\n    the body of the function after the yield will be run after execution resolves, allowing users\n    to write their own teardown/cleanup logic.\n\n    Args:\n        config_schema (Optional[ConfigSchema]): The schema for the config. Configuration data available in\n            `init_context.resource_config`. If not set, Dagster will accept any config provided.\n        description(Optional[str]): A human-readable description of the resource.\n        version (Optional[str]): (Experimental) The version of a resource function. Two wrapped\n            resource functions should only have the same version if they produce the same resource\n            definition when provided with the same inputs.\n        required_resource_keys (Optional[Set[str]]): Keys for the resources required by this resource.\n    '
    if callable(config_schema) and (not is_callable_valid_config_arg(config_schema)):
        return _ResourceDecoratorCallable()(config_schema)

    def _wrap(resource_fn: ResourceFunction) -> 'ResourceDefinition':
        if False:
            return 10
        return _ResourceDecoratorCallable(config_schema=cast(Optional[Dict[str, Any]], config_schema), description=description, required_resource_keys=required_resource_keys, version=version)(resource_fn)
    return _wrap

def make_values_resource(**kwargs: Any) -> ResourceDefinition:
    if False:
        return 10
    'A helper function that creates a ``ResourceDefinition`` to take in user-defined values.\n\n        This is useful for sharing values between ops.\n\n    Args:\n        **kwargs: Arbitrary keyword arguments that will be passed to the config schema of the\n            returned resource definition. If not set, Dagster will accept any config provided for\n            the resource.\n\n    For example:\n\n    .. code-block:: python\n\n        @op(required_resource_keys={"globals"})\n        def my_op(context):\n            print(context.resources.globals["my_str_var"])\n\n        @job(resource_defs={"globals": make_values_resource(my_str_var=str, my_int_var=int)})\n        def my_job():\n            my_op()\n\n    Returns:\n        ResourceDefinition: A resource that passes in user-defined values.\n    '
    return ResourceDefinition(resource_fn=lambda init_context: init_context.resource_config, config_schema=kwargs or Any)