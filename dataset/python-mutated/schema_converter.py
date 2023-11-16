from __future__ import annotations
import dataclasses
import sys
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union, cast
from typing_extensions import Protocol
from graphql import GraphQLAbstractType, GraphQLArgument, GraphQLDirective, GraphQLEnumType, GraphQLEnumValue, GraphQLField, GraphQLInputField, GraphQLInputObjectType, GraphQLInterfaceType, GraphQLList, GraphQLNonNull, GraphQLObjectType, GraphQLUnionType, Undefined, ValueNode, default_type_resolver
from graphql.language.directive_locations import DirectiveLocation
from strawberry.annotation import StrawberryAnnotation
from strawberry.arguments import StrawberryArgument, convert_arguments
from strawberry.custom_scalar import ScalarWrapper
from strawberry.enum import EnumDefinition
from strawberry.exceptions import DuplicatedTypeName, InvalidTypeInputForUnion, InvalidUnionTypeError, MissingTypesForGenericError, ScalarAlreadyRegisteredError, UnresolvedFieldTypeError
from strawberry.field import UNRESOLVED
from strawberry.lazy_type import LazyType
from strawberry.private import is_private
from strawberry.schema.types.scalar import _make_scalar_type
from strawberry.type import StrawberryList, StrawberryOptional, StrawberryType, get_object_definition, has_object_definition
from strawberry.types.info import Info
from strawberry.types.types import StrawberryObjectDefinition
from strawberry.union import StrawberryUnion
from strawberry.unset import UNSET
from strawberry.utils.await_maybe import await_maybe
from ..extensions.field_extension import build_field_extension_resolvers
from . import compat
from .types.concrete_type import ConcreteType
if TYPE_CHECKING:
    from graphql import GraphQLInputType, GraphQLNullableType, GraphQLOutputType, GraphQLResolveInfo, GraphQLScalarType
    from strawberry.custom_scalar import ScalarDefinition
    from strawberry.directive import StrawberryDirective
    from strawberry.enum import EnumValue
    from strawberry.field import StrawberryField
    from strawberry.schema.config import StrawberryConfig
    from strawberry.schema_directive import StrawberrySchemaDirective
FieldType = TypeVar('FieldType', bound=Union[GraphQLField, GraphQLInputField], covariant=True)

class FieldConverterProtocol(Generic[FieldType], Protocol):

    def __call__(self, field: StrawberryField, *, type_definition: Optional[StrawberryObjectDefinition]=None) -> FieldType:
        if False:
            i = 10
            return i + 15
        ...

def _get_thunk_mapping(type_definition: StrawberryObjectDefinition, name_converter: Callable[[StrawberryField], str], field_converter: FieldConverterProtocol[FieldType]) -> Dict[str, FieldType]:
    if False:
        for i in range(10):
            print('nop')
    'Create a GraphQL core `ThunkMapping` mapping of field names to field types.\n\n    This method filters out remaining `strawberry.Private` annotated fields that\n    could not be filtered during the initialization of a `TypeDefinition` due to\n    postponed type-hint evaluation (PEP-563). Performing this filtering now (at\n    schema conversion time) ensures that all types to be included in the schema\n    should have already been resolved.\n\n    Raises:\n        TypeError: If the type of a field in ``fields`` is `UNRESOLVED`\n    '
    thunk_mapping: Dict[str, FieldType] = {}
    for field in type_definition.fields:
        field_type = field.type
        if field_type is UNRESOLVED:
            raise UnresolvedFieldTypeError(type_definition, field)
        if not is_private(field_type):
            thunk_mapping[name_converter(field)] = field_converter(field, type_definition=type_definition)
    return thunk_mapping

class CustomGraphQLEnumType(GraphQLEnumType):

    def __init__(self, enum: EnumDefinition, *args: Any, **kwargs: Any):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.wrapped_cls = enum.wrapped_cls

    def serialize(self, output_value: Any) -> str:
        if False:
            i = 10
            return i + 15
        if isinstance(output_value, self.wrapped_cls):
            for (name, value) in self.values.items():
                if output_value.value == value.value:
                    return name
            raise ValueError(f'Invalid value for enum {self.name}: {output_value}')
        return super().serialize(output_value)

    def parse_value(self, input_value: str) -> Any:
        if False:
            i = 10
            return i + 15
        return self.wrapped_cls(super().parse_value(input_value))

    def parse_literal(self, value_node: ValueNode, _variables: Optional[Dict[str, Any]]=None) -> Any:
        if False:
            while True:
                i = 10
        return self.wrapped_cls(super().parse_literal(value_node, _variables))

class GraphQLCoreConverter:
    DEFINITION_BACKREF = 'strawberry-definition'

    def __init__(self, config: StrawberryConfig, scalar_registry: Dict[object, Union[ScalarWrapper, ScalarDefinition]]):
        if False:
            return 10
        self.type_map: Dict[str, ConcreteType] = {}
        self.config = config
        self.scalar_registry = scalar_registry

    def from_argument(self, argument: StrawberryArgument) -> GraphQLArgument:
        if False:
            i = 10
            return i + 15
        argument_type = cast('GraphQLInputType', self.from_maybe_optional(argument.type))
        default_value = Undefined if argument.default is UNSET else argument.default
        return GraphQLArgument(type_=argument_type, default_value=default_value, description=argument.description, deprecation_reason=argument.deprecation_reason, extensions={GraphQLCoreConverter.DEFINITION_BACKREF: argument})

    def from_enum(self, enum: EnumDefinition) -> CustomGraphQLEnumType:
        if False:
            for i in range(10):
                print('nop')
        enum_name = self.config.name_converter.from_type(enum)
        assert enum_name is not None
        cached_type = self.type_map.get(enum_name, None)
        if cached_type:
            self.validate_same_type_definition(enum_name, enum, cached_type)
            graphql_enum = cached_type.implementation
            assert isinstance(graphql_enum, CustomGraphQLEnumType)
            return graphql_enum
        graphql_enum = CustomGraphQLEnumType(enum=enum, name=enum_name, values={self.config.name_converter.from_enum_value(enum, item): self.from_enum_value(item) for item in enum.values}, description=enum.description, extensions={GraphQLCoreConverter.DEFINITION_BACKREF: enum})
        self.type_map[enum_name] = ConcreteType(definition=enum, implementation=graphql_enum)
        return graphql_enum

    def from_enum_value(self, enum_value: EnumValue) -> GraphQLEnumValue:
        if False:
            for i in range(10):
                print('nop')
        return GraphQLEnumValue(enum_value.value, deprecation_reason=enum_value.deprecation_reason, description=enum_value.description, extensions={GraphQLCoreConverter.DEFINITION_BACKREF: enum_value})

    def from_directive(self, directive: StrawberryDirective) -> GraphQLDirective:
        if False:
            return 10
        graphql_arguments = {}
        for argument in directive.arguments:
            argument_name = self.config.name_converter.from_argument(argument)
            graphql_arguments[argument_name] = self.from_argument(argument)
        directive_name = self.config.name_converter.from_type(directive)
        return GraphQLDirective(name=directive_name, locations=directive.locations, args=graphql_arguments, description=directive.description, extensions={GraphQLCoreConverter.DEFINITION_BACKREF: directive})

    def from_schema_directive(self, cls: Type) -> GraphQLDirective:
        if False:
            i = 10
            return i + 15
        strawberry_directive = cast('StrawberrySchemaDirective', cls.__strawberry_directive__)
        module = sys.modules[cls.__module__]
        args: Dict[str, GraphQLArgument] = {}
        for field in strawberry_directive.fields:
            default = field.default
            if default == dataclasses.MISSING:
                default = UNSET
            name = self.config.name_converter.get_graphql_name(field)
            args[name] = self.from_argument(StrawberryArgument(python_name=field.python_name or field.name, graphql_name=None, type_annotation=StrawberryAnnotation(annotation=field.type, namespace=module.__dict__), default=default))
        return GraphQLDirective(name=self.config.name_converter.from_directive(strawberry_directive), locations=[DirectiveLocation(loc.value) for loc in strawberry_directive.locations], args=args, is_repeatable=strawberry_directive.repeatable, description=strawberry_directive.description, extensions={GraphQLCoreConverter.DEFINITION_BACKREF: strawberry_directive})

    def from_field(self, field: StrawberryField, *, type_definition: Optional[StrawberryObjectDefinition]=None) -> GraphQLField:
        if False:
            return 10
        resolver = self.from_resolver(field)
        field_type = cast('GraphQLOutputType', self.from_maybe_optional(field.resolve_type(type_definition=type_definition)))
        subscribe = None
        if field.is_subscription:
            subscribe = resolver
            resolver = lambda event, *_, **__: event
        graphql_arguments = {}
        for argument in field.arguments:
            argument_name = self.config.name_converter.from_argument(argument)
            graphql_arguments[argument_name] = self.from_argument(argument)
        return GraphQLField(type_=field_type, args=graphql_arguments, resolve=resolver, subscribe=subscribe, description=field.description, deprecation_reason=field.deprecation_reason, extensions={GraphQLCoreConverter.DEFINITION_BACKREF: field})

    def from_input_field(self, field: StrawberryField, *, type_definition: Optional[StrawberryObjectDefinition]=None) -> GraphQLInputField:
        if False:
            while True:
                i = 10
        field_type = cast('GraphQLInputType', self.from_maybe_optional(field.resolve_type(type_definition=type_definition)))
        default_value: object
        if field.default_value is UNSET or field.default_value is dataclasses.MISSING:
            default_value = Undefined
        else:
            default_value = field.default_value
        return GraphQLInputField(type_=field_type, default_value=default_value, description=field.description, deprecation_reason=field.deprecation_reason, extensions={GraphQLCoreConverter.DEFINITION_BACKREF: field})

    def get_graphql_fields(self, type_definition: StrawberryObjectDefinition) -> Dict[str, GraphQLField]:
        if False:
            print('Hello World!')
        return _get_thunk_mapping(type_definition=type_definition, name_converter=self.config.name_converter.from_field, field_converter=self.from_field)

    def get_graphql_input_fields(self, type_definition: StrawberryObjectDefinition) -> Dict[str, GraphQLInputField]:
        if False:
            print('Hello World!')
        return _get_thunk_mapping(type_definition=type_definition, name_converter=self.config.name_converter.from_field, field_converter=self.from_input_field)

    def from_input_object(self, object_type: type) -> GraphQLInputObjectType:
        if False:
            i = 10
            return i + 15
        type_definition = object_type.__strawberry_definition__
        type_name = self.config.name_converter.from_type(type_definition)
        cached_type = self.type_map.get(type_name, None)
        if cached_type:
            self.validate_same_type_definition(type_name, type_definition, cached_type)
            graphql_object_type = self.type_map[type_name].implementation
            assert isinstance(graphql_object_type, GraphQLInputObjectType)
            return graphql_object_type
        graphql_object_type = GraphQLInputObjectType(name=type_name, fields=lambda : self.get_graphql_input_fields(type_definition), description=type_definition.description, extensions={GraphQLCoreConverter.DEFINITION_BACKREF: type_definition})
        self.type_map[type_name] = ConcreteType(definition=type_definition, implementation=graphql_object_type)
        return graphql_object_type

    def from_interface(self, interface: StrawberryObjectDefinition) -> GraphQLInterfaceType:
        if False:
            i = 10
            return i + 15
        interface_name = self.config.name_converter.from_type(interface)
        cached_type = self.type_map.get(interface_name, None)
        if cached_type:
            self.validate_same_type_definition(interface_name, interface, cached_type)
            graphql_interface = cached_type.implementation
            assert isinstance(graphql_interface, GraphQLInterfaceType)
            return graphql_interface

        def _get_resolve_type():
            if False:
                print('Hello World!')
            if interface.resolve_type:
                return interface.resolve_type

            def resolve_type(obj: Any, info: GraphQLResolveInfo, abstract_type: GraphQLAbstractType) -> Union[Awaitable[Optional[str]], str, None]:
                if False:
                    i = 10
                    return i + 15
                if isinstance(obj, interface.origin):
                    type_definition = get_object_definition(obj, strict=True)
                    if not type_definition.is_graphql_generic:
                        return obj.__strawberry_definition__.name
                return default_type_resolver(obj, info, abstract_type)
            return resolve_type
        graphql_interface = GraphQLInterfaceType(name=interface_name, fields=lambda : self.get_graphql_fields(interface), interfaces=list(map(self.from_interface, interface.interfaces)), description=interface.description, extensions={GraphQLCoreConverter.DEFINITION_BACKREF: interface}, resolve_type=_get_resolve_type())
        self.type_map[interface_name] = ConcreteType(definition=interface, implementation=graphql_interface)
        return graphql_interface

    def from_list(self, type_: StrawberryList) -> GraphQLList:
        if False:
            for i in range(10):
                print('nop')
        of_type = self.from_maybe_optional(type_.of_type)
        return GraphQLList(of_type)

    def from_object(self, object_type: StrawberryObjectDefinition) -> GraphQLObjectType:
        if False:
            for i in range(10):
                print('nop')
        object_type_name = self.config.name_converter.from_type(object_type)
        cached_type = self.type_map.get(object_type_name, None)
        if cached_type:
            self.validate_same_type_definition(object_type_name, object_type, cached_type)
            graphql_object_type = cached_type.implementation
            assert isinstance(graphql_object_type, GraphQLObjectType)
            return graphql_object_type

        def _get_is_type_of() -> Optional[Callable[[Any, GraphQLResolveInfo], bool]]:
            if False:
                i = 10
                return i + 15
            if object_type.is_type_of:
                return object_type.is_type_of
            if not object_type.interfaces:
                return None
            possible_types = (*tuple((interface.origin for interface in object_type.interfaces)), object_type.origin)

            def is_type_of(obj: Any, _info: GraphQLResolveInfo) -> bool:
                if False:
                    for i in range(10):
                        print('nop')
                if object_type.concrete_of and (has_object_definition(obj) and obj.__strawberry_definition__.origin is object_type.concrete_of.origin):
                    return True
                return isinstance(obj, possible_types)
            return is_type_of
        graphql_object_type = GraphQLObjectType(name=object_type_name, fields=lambda : self.get_graphql_fields(object_type), interfaces=list(map(self.from_interface, object_type.interfaces)), description=object_type.description, is_type_of=_get_is_type_of(), extensions={GraphQLCoreConverter.DEFINITION_BACKREF: object_type})
        self.type_map[object_type_name] = ConcreteType(definition=object_type, implementation=graphql_object_type)
        return graphql_object_type

    def from_resolver(self, field: StrawberryField) -> Callable:
        if False:
            for i in range(10):
                print('nop')
        field.default_resolver = self.config.default_resolver
        if field.is_basic_field:

            def _get_basic_result(_source: Any, *args: str, **kwargs: Any):
                if False:
                    print('Hello World!')
                return field.get_result(_source, info=None, args=[], kwargs={})
            _get_basic_result._is_default = True
            return _get_basic_result

        def _get_arguments(source: Any, info: Info, kwargs: Any) -> Tuple[List[Any], Dict[str, Any]]:
            if False:
                for i in range(10):
                    print('nop')
            field_arguments = field.arguments[:]
            if field.base_resolver:
                existing = {arg.python_name for arg in field_arguments}
                field_arguments.extend([arg for arg in field.base_resolver.arguments if arg.python_name not in existing])
            kwargs = convert_arguments(kwargs, field_arguments, scalar_registry=self.scalar_registry, config=self.config)
            args = []
            if field.base_resolver:
                if field.base_resolver.self_parameter:
                    args.append(source)
                if (parent_parameter := field.base_resolver.parent_parameter):
                    kwargs[parent_parameter.name] = source
                if (root_parameter := field.base_resolver.root_parameter):
                    kwargs[root_parameter.name] = source
                if (info_parameter := field.base_resolver.info_parameter):
                    kwargs[info_parameter.name] = info
            return (args, kwargs)

        def _check_permissions(source: Any, info: Info, kwargs: Any):
            if False:
                return 10
            '\n            Checks if the permission should be accepted and\n            raises an exception if not\n            '
            for permission_class in field.permission_classes:
                permission = permission_class()
                if not permission.has_permission(source, info, **kwargs):
                    message = getattr(permission, 'message', None)
                    raise PermissionError(message)

        async def _check_permissions_async(source: Any, info: Info, kwargs: Any):
            for permission_class in field.permission_classes:
                permission = permission_class()
                has_permission: bool
                has_permission = await await_maybe(permission.has_permission(source, info, **kwargs))
                if not has_permission:
                    message = getattr(permission, 'message', None)
                    raise PermissionError(message)

        def _strawberry_info_from_graphql(info: GraphQLResolveInfo) -> Info:
            if False:
                print('Hello World!')
            return Info(_raw_info=info, _field=field)

        def _get_result(_source: Any, info: Info, field_args: List[Any], field_kwargs: Dict[str, Any]):
            if False:
                for i in range(10):
                    print('nop')
            return field.get_result(_source, info=info, args=field_args, kwargs=field_kwargs)

        def wrap_field_extensions() -> Callable[..., Any]:
            if False:
                i = 10
                return i + 15
            'Wrap the provided field resolver with the middleware.'
            for extension in field.extensions:
                extension.apply(field)
            extension_functions = build_field_extension_resolvers(field)

            def extension_resolver(_source: Any, info: Info, **kwargs: Any):
                if False:
                    print('Hello World!')
                (field_args, field_kwargs) = _get_arguments(source=_source, info=info, kwargs=kwargs)
                resolver_requested_info = False
                if 'info' in field_kwargs:
                    resolver_requested_info = True
                    field_kwargs.pop('info')

                def wrapped_get_result(_source: Any, info: Info, **kwargs: Any):
                    if False:
                        print('Hello World!')
                    if resolver_requested_info:
                        kwargs['info'] = info
                    return _get_result(_source, info, field_args=field_args, field_kwargs=kwargs)
                return reduce(lambda chained_fn, next_fn: partial(next_fn, chained_fn), extension_functions, wrapped_get_result)(_source, info, **field_kwargs)
            return extension_resolver
        _get_result_with_extensions = wrap_field_extensions()

        def _resolver(_source: Any, info: GraphQLResolveInfo, **kwargs: Any):
            if False:
                while True:
                    i = 10
            strawberry_info = _strawberry_info_from_graphql(info)
            _check_permissions(_source, strawberry_info, kwargs)
            return _get_result_with_extensions(_source, strawberry_info, **kwargs)

        async def _async_resolver(_source: Any, info: GraphQLResolveInfo, **kwargs: Any):
            strawberry_info = _strawberry_info_from_graphql(info)
            await _check_permissions_async(_source, strawberry_info, kwargs)
            return await await_maybe(_get_result_with_extensions(_source, strawberry_info, **kwargs))
        if field.is_async:
            _async_resolver._is_default = not field.base_resolver
            return _async_resolver
        else:
            _resolver._is_default = not field.base_resolver
            return _resolver

    def from_scalar(self, scalar: Type) -> GraphQLScalarType:
        if False:
            i = 10
            return i + 15
        scalar_definition: ScalarDefinition
        if scalar in self.scalar_registry:
            _scalar_definition = self.scalar_registry[scalar]
            if isinstance(_scalar_definition, ScalarWrapper):
                scalar_definition = _scalar_definition._scalar_definition
            else:
                scalar_definition = _scalar_definition
        else:
            scalar_definition = scalar._scalar_definition
        scalar_name = self.config.name_converter.from_type(scalar_definition)
        if scalar_name not in self.type_map:
            implementation = scalar_definition.implementation if scalar_definition.implementation is not None else _make_scalar_type(scalar_definition)
            self.type_map[scalar_name] = ConcreteType(definition=scalar_definition, implementation=implementation)
        else:
            other_definition = self.type_map[scalar_name].definition
            if other_definition != scalar_definition:
                other_definition = cast('ScalarDefinition', other_definition)
                raise ScalarAlreadyRegisteredError(scalar_definition, other_definition)
            implementation = cast('GraphQLScalarType', self.type_map[scalar_name].implementation)
        return implementation

    def from_maybe_optional(self, type_: Union[StrawberryType, type]) -> Union[GraphQLNullableType, GraphQLNonNull]:
        if False:
            return 10
        NoneType = type(None)
        if type_ is None or type_ is NoneType:
            return self.from_type(type_)
        elif isinstance(type_, StrawberryOptional):
            return self.from_type(type_.of_type)
        else:
            return GraphQLNonNull(self.from_type(type_))

    def from_type(self, type_: Union[StrawberryType, type]) -> GraphQLNullableType:
        if False:
            print('Hello World!')
        if compat.is_graphql_generic(type_):
            raise MissingTypesForGenericError(type_)
        if isinstance(type_, EnumDefinition):
            return self.from_enum(type_)
        elif compat.is_input_type(type_):
            return self.from_input_object(type_)
        elif isinstance(type_, StrawberryList):
            return self.from_list(type_)
        elif compat.is_interface_type(type_):
            type_definition: StrawberryObjectDefinition = type_.__strawberry_definition__
            return self.from_interface(type_definition)
        elif has_object_definition(type_):
            return self.from_object(type_.__strawberry_definition__)
        elif compat.is_enum(type_):
            enum_definition: EnumDefinition = type_._enum_definition
            return self.from_enum(enum_definition)
        elif isinstance(type_, StrawberryObjectDefinition):
            return self.from_object(type_)
        elif isinstance(type_, StrawberryUnion):
            return self.from_union(type_)
        elif isinstance(type_, LazyType):
            return self.from_type(type_.resolve_type())
        elif compat.is_scalar(type_, self.scalar_registry):
            return self.from_scalar(type_)
        raise TypeError(f"Unexpected type '{type_}'")

    def from_union(self, union: StrawberryUnion) -> GraphQLUnionType:
        if False:
            for i in range(10):
                print('nop')
        union_name = self.config.name_converter.from_type(union)
        for type_ in union.types:
            if not StrawberryUnion.is_valid_union_type(type_):
                raise InvalidUnionTypeError(union_name, type_, union_definition=union)
        if union_name in self.type_map:
            graphql_union = self.type_map[union_name].implementation
            assert isinstance(graphql_union, GraphQLUnionType)
            return graphql_union
        graphql_types: List[GraphQLObjectType] = []
        for type_ in union.types:
            graphql_type = self.from_type(type_)
            if isinstance(graphql_type, GraphQLInputObjectType):
                raise InvalidTypeInputForUnion(graphql_type)
            assert isinstance(graphql_type, GraphQLObjectType)
            graphql_types.append(graphql_type)
        graphql_union = GraphQLUnionType(name=union_name, types=graphql_types, description=union.description, resolve_type=union.get_type_resolver(self.type_map), extensions={GraphQLCoreConverter.DEFINITION_BACKREF: union})
        self.type_map[union_name] = ConcreteType(definition=union, implementation=graphql_union)
        return graphql_union

    def _get_is_type_of(self, object_type: StrawberryObjectDefinition) -> Optional[Callable[[Any, GraphQLResolveInfo], bool]]:
        if False:
            i = 10
            return i + 15
        if object_type.is_type_of:
            return object_type.is_type_of
        if object_type.interfaces:

            def is_type_of(obj: Any, _info: GraphQLResolveInfo) -> bool:
                if False:
                    print('Hello World!')
                if object_type.concrete_of and (has_object_definition(obj) and obj.__strawberry_definition__.origin is object_type.concrete_of.origin):
                    return True
                return isinstance(obj, object_type.origin)
            return is_type_of
        return None

    def validate_same_type_definition(self, name: str, type_definition: StrawberryType, cached_type: ConcreteType) -> None:
        if False:
            for i in range(10):
                print('nop')
        if cached_type.definition == type_definition:
            return
        first_type_definition = cached_type.definition
        second_type_definition = type_definition
        if isinstance(first_type_definition, StrawberryObjectDefinition) and isinstance(second_type_definition, StrawberryObjectDefinition) and (first_type_definition.concrete_of is not None) and (first_type_definition.concrete_of == second_type_definition.concrete_of) and (first_type_definition.type_var_map.keys() == second_type_definition.type_var_map.keys()):
            equal = True
            for (type_var, type1) in first_type_definition.type_var_map.items():
                type2 = second_type_definition.type_var_map[type_var]
                if isinstance(type1, LazyType):
                    type1 = type1.resolve_type()
                elif isinstance(type1, StrawberryOptional) and isinstance(type1.of_type, LazyType):
                    type1.of_type = type1.of_type.resolve_type()
                if isinstance(type2, LazyType):
                    type2 = type2.resolve_type()
                elif isinstance(type2, StrawberryOptional) and isinstance(type2.of_type, LazyType):
                    type2.of_type = type2.of_type.resolve_type()
                if type1 != type2:
                    equal = False
                    break
            if equal:
                return
        if isinstance(second_type_definition, StrawberryObjectDefinition):
            first_origin = second_type_definition.origin
        elif isinstance(second_type_definition, EnumDefinition):
            first_origin = second_type_definition.wrapped_cls
        else:
            first_origin = None
        if isinstance(first_type_definition, StrawberryObjectDefinition):
            second_origin = first_type_definition.origin
        elif isinstance(first_type_definition, EnumDefinition):
            second_origin = first_type_definition.wrapped_cls
        else:
            second_origin = None
        raise DuplicatedTypeName(first_origin, second_origin, name)