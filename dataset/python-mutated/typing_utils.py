from typing import TYPE_CHECKING, Any, Generic, Optional, Type, TypeVar, Union, cast
from pydantic import Field
from typing_extensions import Annotated, dataclass_transform, get_origin
from dagster._core.errors import DagsterInvalidDagsterTypeInPythonicConfigDefinitionError
from .type_check_utils import safe_is_subclass
try:
    from pydantic.main import ModelMetaclass
except ImportError:
    from pydantic._internal._model_construction import ModelMetaclass
if TYPE_CHECKING:
    from dagster._config.pythonic_config import PartialResource

class LateBoundTypesForResourceTypeChecking:
    _TResValue = TypeVar('_TResValue')

    class _Temp(Generic[_TResValue]):
        pass
    _ResourceDep: Type = _Temp
    _Resource: Type = _Temp
    _PartialResource: Type = _Temp

    @staticmethod
    def get_resource_rep_type() -> Type:
        if False:
            while True:
                i = 10
        return LateBoundTypesForResourceTypeChecking._ResourceDep

    @staticmethod
    def get_resource_type() -> Type:
        if False:
            while True:
                i = 10
        return LateBoundTypesForResourceTypeChecking._Resource

    @staticmethod
    def get_partial_resource_type(base: Type) -> Type:
        if False:
            return 10
        return LateBoundTypesForResourceTypeChecking._PartialResource

    @staticmethod
    def set_actual_types_for_type_checking(resource_dep_type: Type, resource_type: Type, partial_resource_type: Type) -> None:
        if False:
            i = 10
            return i + 15
        LateBoundTypesForResourceTypeChecking._ResourceDep = resource_dep_type
        LateBoundTypesForResourceTypeChecking._Resource = resource_type
        LateBoundTypesForResourceTypeChecking._PartialResource = partial_resource_type

@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class BaseConfigMeta(ModelMetaclass):

    def __new__(cls, name, bases, namespaces, **kwargs) -> Any:
        if False:
            print('Hello World!')
        annotations = namespaces.get('__annotations__', {})
        try:
            from dagster._core.types.dagster_type import DagsterType
            for field in annotations:
                if isinstance(annotations[field], DagsterType):
                    raise DagsterInvalidDagsterTypeInPythonicConfigDefinitionError(name, field)
        except ImportError:
            pass
        return super().__new__(cls, name, bases, namespaces, **kwargs)

@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class BaseResourceMeta(BaseConfigMeta):
    """Custom metaclass for Resource and PartialResource. This metaclass is responsible for
    transforming the type annotations on the class so that Pydantic constructor-time validation
    does not error when users provide partially configured resources to resource params.

    For example, the following code would ordinarily fail Pydantic validation:

    .. code-block:: python

        class FooResource(ConfigurableResource):
            bar: BarResource

        # Types as PartialResource[BarResource]
        partial_bar = BarResource.configure_at_runtime()

        # Pydantic validation fails because bar is not a BarResource
        foo = FooResource(bar=partial_bar)

    This metaclass transforms the type annotations on the class so that Pydantic validation
    accepts either a PartialResource or a Resource as a value for the resource dependency.
    """

    def __new__(cls, name, bases, namespaces, **kwargs) -> Any:
        if False:
            return 10
        annotations = namespaces.get('__annotations__', {})
        for field in annotations:
            if not field.startswith('__'):
                if get_origin(annotations[field]) == LateBoundTypesForResourceTypeChecking.get_resource_rep_type():
                    annotations[field] = Annotated[Any, 'resource_dependency']
                elif safe_is_subclass(annotations[field], LateBoundTypesForResourceTypeChecking.get_resource_type()):
                    base = annotations[field]
                    annotations[field] = Annotated[Union[LateBoundTypesForResourceTypeChecking.get_partial_resource_type(base), base], 'resource_dependency']
        namespaces['__annotations__'] = annotations
        return super().__new__(cls, name, bases, namespaces, **kwargs)
Self = TypeVar('Self', bound='TypecheckAllowPartialResourceInitParams')

class TypecheckAllowPartialResourceInitParams:
    """Implementation of the Python descriptor protocol (https://docs.python.org/3/howto/descriptor.html)
    to adjust the types of resource inputs and outputs, e.g. resource dependencies can be passed in
    as PartialResources or Resources, but will always be returned as Resources.

    For example, given a resource with the following signature:

    .. code-block:: python

        class FooResource(Resource):
            bar: BarResource

    The following code will work:

    .. code-block:: python

        # Types as PartialResource[BarResource]
        partial_bar = BarResource.configure_at_runtime()

        # bar parameter takes BarResource | PartialResource[BarResource]
        foo = FooResource(bar=partial_bar)

        # initialization of FooResource succeeds,
        # populating the bar attribute with a full BarResource

        # bar attribute is typed as BarResource, since
        # it is fully initialized when a user accesses it
        print(foo.bar)

    Very similar to https://github.com/pydantic/pydantic/discussions/4262.
    """

    def __set_name__(self, _owner, name):
        if False:
            print('Hello World!')
        self._assigned_name = name

    def __get__(self: 'Self', obj: Any, __owner: Any) -> 'Self':
        if False:
            i = 10
            return i + 15
        return cast(Self, getattr(obj, self._assigned_name))

    def __set__(self: 'Self', obj: Optional[object], value: Union['Self', 'PartialResource[Self]']) -> None:
        if False:
            i = 10
            return i + 15
        setattr(obj, self._assigned_name, value)