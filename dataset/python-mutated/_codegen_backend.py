"""DTO backends do the heavy lifting of decoding and validating raw bytes into domain models, and
back again, to bytes.
"""
from __future__ import annotations
import re
import textwrap
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Generator, Mapping, Protocol, cast
from msgspec import UNSET
from litestar.dto._backend import DTOBackend
from litestar.dto._types import CollectionType, CompositeType, SimpleType, TransferDTOFieldDefinition, TransferType, UnionType
from litestar.utils.helpers import unique_name_for_scope
if TYPE_CHECKING:
    from litestar.connection import ASGIConnection
    from litestar.dto import AbstractDTO
    from litestar.types.serialization import LitestarEncodableType
    from litestar.typing import FieldDefinition
__all__ = ('DTOCodegenBackend',)

class DTOCodegenBackend(DTOBackend):
    __slots__ = ('_transfer_to_dict', '_transfer_to_model_type', '_transfer_data_from_builtins', '_transfer_data_from_builtins_with_overrides', '_encode_data')

    def __init__(self, dto_factory: type[AbstractDTO], field_definition: FieldDefinition, handler_id: str, is_data_field: bool, model_type: type[Any], wrapper_attribute_name: str | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create dto backend instance.\n\n        Args:\n            dto_factory: The DTO factory class calling this backend.\n            field_definition: Parsed type.\n            handler_id: The name of the handler that this backend is for.\n            is_data_field: Whether the field is a subclass of DTOData.\n            model_type: Model type.\n            wrapper_attribute_name: If the data that DTO should operate upon is wrapped in a generic datastructure,\n              this is the name of the attribute that the data is stored in.\n        '
        super().__init__(dto_factory=dto_factory, field_definition=field_definition, handler_id=handler_id, is_data_field=is_data_field, model_type=model_type, wrapper_attribute_name=wrapper_attribute_name)
        self._transfer_to_dict = self._create_transfer_data_fn(destination_type=dict, field_definition=self.field_definition)
        self._transfer_to_model_type = self._create_transfer_data_fn(destination_type=self.model_type, field_definition=self.field_definition)
        self._transfer_data_from_builtins = self._create_transfer_data_fn(destination_type=self.model_type, field_definition=self.field_definition, override_serialization_name=False)
        self._transfer_data_from_builtins_with_overrides = self._create_transfer_data_fn(destination_type=self.model_type, field_definition=self.field_definition, override_serialization_name=True)
        self._encode_data = self._create_transfer_data_fn(destination_type=self.transfer_model_type, field_definition=self.field_definition)

    def populate_data_from_builtins(self, builtins: Any, asgi_connection: ASGIConnection) -> Any:
        if False:
            while True:
                i = 10
        'Populate model instance from builtin types.\n\n        Args:\n            builtins: Builtin type.\n            asgi_connection: The current ASGI Connection\n\n        Returns:\n            Instance or collection of ``model_type`` instances.\n        '
        if self.dto_data_type:
            return self.dto_data_type(backend=self, data_as_builtins=self._transfer_to_dict(self.parse_builtins(builtins, asgi_connection)))
        return self.transfer_data_from_builtins(self.parse_builtins(builtins, asgi_connection))

    def transfer_data_from_builtins(self, builtins: Any, override_serialization_name: bool=False) -> Any:
        if False:
            i = 10
            return i + 15
        'Populate model instance from builtin types.\n\n        Args:\n            builtins: Builtin type.\n            override_serialization_name: Use the original field names, used when creating\n                                         an instance using `DTOData.create_instance`\n\n        Returns:\n            Instance or collection of ``model_type`` instances.\n        '
        if override_serialization_name:
            return self._transfer_data_from_builtins_with_overrides(builtins)
        return self._transfer_data_from_builtins(builtins)

    def populate_data_from_raw(self, raw: bytes, asgi_connection: ASGIConnection) -> Any:
        if False:
            while True:
                i = 10
        'Parse raw bytes into instance of `model_type`.\n\n        Args:\n            raw: bytes\n            asgi_connection: The current ASGI Connection\n\n        Returns:\n            Instance or collection of ``model_type`` instances.\n        '
        if self.dto_data_type:
            return self.dto_data_type(backend=self, data_as_builtins=self._transfer_to_dict(self.parse_raw(raw, asgi_connection)))
        return self._transfer_to_model_type(self.parse_raw(raw, asgi_connection))

    def encode_data(self, data: Any) -> LitestarEncodableType:
        if False:
            i = 10
            return i + 15
        'Encode data into a ``LitestarEncodableType``.\n\n        Args:\n            data: Data to encode.\n\n        Returns:\n            Encoded data.\n        '
        if self.wrapper_attribute_name:
            wrapped_transfer = self._encode_data(getattr(data, self.wrapper_attribute_name))
            setattr(data, self.wrapper_attribute_name, wrapped_transfer)
            return cast('LitestarEncodableType', data)
        return cast('LitestarEncodableType', self._encode_data(data))

    def _create_transfer_data_fn(self, destination_type: type[Any], field_definition: FieldDefinition, override_serialization_name: bool | None=None) -> Any:
        if False:
            while True:
                i = 10
        'Create instance or iterable of instances of ``destination_type``.\n\n        Args:\n            destination_type: the model type received by the DTO on type narrowing.\n            field_definition: the parsed type that represents the handler annotation for which the DTO is being applied.\n            override_serialization_name: Override serialization name\n\n        Returns:\n            Data parsed into ``destination_type``.\n        '
        return TransferFunctionFactory.create_transfer_data(destination_type=destination_type, field_definitions=self.parsed_field_definitions, is_data_field=self.is_data_field, override_serialization_name=override_serialization_name if override_serialization_name is not None else self.override_serialization_name, field_definition=field_definition)

class FieldAccessManager(Protocol):

    def __call__(self, source_name: str, field_name: str, expect_optional: bool) -> ContextManager[str]:
        if False:
            while True:
                i = 10
        ...

class TransferFunctionFactory:

    def __init__(self, is_data_field: bool, override_serialization_name: bool, nested_as_dict: bool) -> None:
        if False:
            while True:
                i = 10
        self.is_data_field = is_data_field
        self.override_serialization_name = override_serialization_name
        self._fn_locals: dict[str, Any] = {'Mapping': Mapping, 'UNSET': UNSET}
        self._indentation = 1
        self._body = ''
        self.names: set[str] = set()
        self.nested_as_dict = nested_as_dict
        self._re_index_access = re.compile('\\[[\'\\"](\\w+?)[\'\\"]]')

    def _add_to_fn_globals(self, name: str, value: Any) -> str:
        if False:
            print('Hello World!')
        unique_name = unique_name_for_scope(name, self._fn_locals)
        self._fn_locals[unique_name] = value
        return unique_name

    def _create_local_name(self, name: str) -> str:
        if False:
            while True:
                i = 10
        unique_name = unique_name_for_scope(name, self.names)
        self.names.add(unique_name)
        return unique_name

    def _make_function(self, source_value_name: str, return_value_name: str, fn_name: str='func') -> Callable[[Any], Any]:
        if False:
            i = 10
            return i + 15
        'Wrap the current body contents in a function definition and turn it into a callable object'
        source = f'def {fn_name}({source_value_name}):\n{self._body} return {return_value_name}'
        ctx: dict[str, Any] = {**self._fn_locals}
        exec(source, ctx)
        return ctx['func']

    def _add_stmt(self, stmt: str) -> None:
        if False:
            print('Hello World!')
        self._body += textwrap.indent(stmt + '\n', ' ' * self._indentation)

    @contextmanager
    def _start_block(self, expr: str | None=None) -> Generator[None, None, None]:
        if False:
            i = 10
            return i + 15
        'Start an indented block. If `expr` is given, use it as the "opening line"\n        of the block.\n        '
        if expr is not None:
            self._add_stmt(expr)
        self._indentation += 1
        yield
        self._indentation -= 1

    @contextmanager
    def _try_except_pass(self, exception: str) -> Generator[None, None, None]:
        if False:
            while True:
                i = 10
        'Enter a `try / except / pass` block. Content written while inside this context\n        will go into the `try` block.\n        '
        with self._start_block('try:'):
            yield
        with self._start_block(expr=f'except {exception}:'):
            self._add_stmt('pass')

    @contextmanager
    def _access_mapping_item(self, source_name: str, field_name: str, expect_optional: bool) -> Generator[str, None, None]:
        if False:
            print('Hello World!')
        'Enter a context within which an item of a mapping can be accessed safely,\n        i.e. only if it is contained within that mapping.\n        Yields an expression that accesses the mapping item. Content written while\n        within this context can use this expression to access the desired value.\n        '
        value_expr = f"{source_name}['{field_name}']"
        if expect_optional:
            with self._start_block(f"if '{field_name}' in {source_name}:"):
                yield value_expr
        else:
            with self._try_except_pass('KeyError'):
                yield value_expr

    @contextmanager
    def _access_attribute(self, source_name: str, field_name: str, expect_optional: bool) -> Generator[str, None, None]:
        if False:
            for i in range(10):
                print('nop')
        'Enter a context within which an attribute of an object can be accessed\n        safely, i.e. only if the object actually has the attribute.\n        Yields an expression that retrieves the object attribute. Content written while\n        within this context can use this expression to access the desired value.\n        '
        value_expr = f'{source_name}.{field_name}'
        if expect_optional:
            with self._start_block(f"if hasattr({source_name}, '{field_name}'):"):
                yield value_expr
        else:
            with self._try_except_pass('AttributeError'):
                yield value_expr

    @classmethod
    def create_transfer_instance_data(cls, field_definitions: tuple[TransferDTOFieldDefinition, ...], destination_type: type[Any], is_data_field: bool, override_serialization_name: bool) -> Callable[[Any], Any]:
        if False:
            while True:
                i = 10
        factory = cls(is_data_field=is_data_field, override_serialization_name=override_serialization_name, nested_as_dict=destination_type is dict)
        tmp_return_type_name = factory._create_local_name('tmp_return_type')
        source_instance_name = factory._create_local_name('source_instance')
        destination_type_name = factory._add_to_fn_globals('destination_type', destination_type)
        factory._create_transfer_instance_data(tmp_return_type_name=tmp_return_type_name, source_instance_name=source_instance_name, destination_type_name=destination_type_name, field_definitions=field_definitions, destination_type_is_dict=destination_type is dict)
        return factory._make_function(source_value_name=source_instance_name, return_value_name=tmp_return_type_name)

    @classmethod
    def create_transfer_type_data(cls, transfer_type: TransferType, is_data_field: bool, override_serialization_name: bool) -> Callable[[Any], Any]:
        if False:
            i = 10
            return i + 15
        factory = cls(is_data_field=is_data_field, override_serialization_name=override_serialization_name, nested_as_dict=False)
        tmp_return_type_name = factory._create_local_name('tmp_return_type')
        source_value_name = factory._create_local_name('source_value')
        factory._create_transfer_type_data_body(transfer_type=transfer_type, nested_as_dict=False, assignment_target=tmp_return_type_name, source_value_name=source_value_name)
        return factory._make_function(source_value_name=source_value_name, return_value_name=tmp_return_type_name)

    @classmethod
    def create_transfer_data(cls, destination_type: type[Any], field_definitions: tuple[TransferDTOFieldDefinition, ...], is_data_field: bool, override_serialization_name: bool, field_definition: FieldDefinition | None=None) -> Callable[[Any], Any]:
        if False:
            return 10
        if field_definition and field_definition.is_non_string_collection and (not field_definition.is_mapping):
            factory = cls(is_data_field=is_data_field, override_serialization_name=override_serialization_name, nested_as_dict=False)
            source_value_name = factory._create_local_name('source_value')
            return_value_name = factory._create_local_name('tmp_return_value')
            factory._create_transfer_data_body_nested(field_definitions=field_definitions, field_definition=field_definition, destination_type=destination_type, source_data_name=source_value_name, assignment_target=return_value_name)
            return factory._make_function(source_value_name=source_value_name, return_value_name=return_value_name)
        return cls.create_transfer_instance_data(destination_type=destination_type, field_definitions=field_definitions, is_data_field=is_data_field, override_serialization_name=override_serialization_name)

    def _create_transfer_data_body_nested(self, field_definition: FieldDefinition, field_definitions: tuple[TransferDTOFieldDefinition, ...], destination_type: type[Any], source_data_name: str, assignment_target: str) -> None:
        if False:
            i = 10
            return i + 15
        origin_name = self._add_to_fn_globals('origin', field_definition.instantiable_origin)
        transfer_func = TransferFunctionFactory.create_transfer_data(is_data_field=self.is_data_field, destination_type=destination_type, field_definition=field_definition.inner_types[0], field_definitions=field_definitions, override_serialization_name=self.override_serialization_name)
        transfer_func_name = self._add_to_fn_globals('transfer_data', transfer_func)
        self._add_stmt(f'{assignment_target} = {origin_name}({transfer_func_name}(item) for item in {source_data_name})')

    def _create_transfer_instance_data(self, tmp_return_type_name: str, source_instance_name: str, destination_type_name: str, field_definitions: tuple[TransferDTOFieldDefinition, ...], destination_type_is_dict: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        local_dict_name = self._create_local_name('unstructured_data')
        self._add_stmt(f'{local_dict_name} = {{}}')
        if (field_definitions := tuple((f for f in field_definitions if self.is_data_field or not f.is_excluded))):
            if len(field_definitions) > 1 and ('.' in source_instance_name or '[' in source_instance_name):
                if '.' in source_instance_name:
                    (level_1, level_2) = source_instance_name.split('.', 1)
                else:
                    (level_1, level_2, *_) = self._re_index_access.split(source_instance_name, maxsplit=1)
                new_source_instance_name = self._create_local_name(f'{level_1}_{level_2}')
                self._add_stmt(f'{new_source_instance_name} = {source_instance_name}')
                source_instance_name = new_source_instance_name
            for source_type in ('mapping', 'object'):
                if source_type == 'mapping':
                    block_expr = f'if isinstance({source_instance_name}, Mapping):'
                    access_item = self._access_mapping_item
                else:
                    block_expr = 'else:'
                    access_item = self._access_attribute
                with self._start_block(expr=block_expr):
                    self._create_transfer_instance_data_inner(local_dict_name=local_dict_name, field_definitions=field_definitions, access_field_safe=access_item, source_instance_name=source_instance_name)
        if not destination_type_is_dict:
            self._add_stmt(f'{tmp_return_type_name} = {destination_type_name}(**{local_dict_name})')
        else:
            self._add_stmt(f'{tmp_return_type_name} = {local_dict_name}')

    def _create_transfer_instance_data_inner(self, *, local_dict_name: str, field_definitions: tuple[TransferDTOFieldDefinition, ...], access_field_safe: FieldAccessManager, source_instance_name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        should_use_serialization_name = not self.override_serialization_name and self.is_data_field
        for field_definition in field_definitions:
            source_field_name = field_definition.serialization_name if should_use_serialization_name else field_definition.name
            destination_name = field_definition.name if self.is_data_field else field_definition.serialization_name
            with access_field_safe(source_name=source_instance_name, field_name=source_field_name, expect_optional=field_definition.is_partial or field_definition.is_optional) as source_value_expr:
                if self.is_data_field and field_definition.is_partial:
                    source_value_name = self._create_local_name('source_value')
                    self._add_stmt(f'{source_value_name} = {source_value_expr}')
                    ctx = self._start_block(f'if {source_value_name} is not UNSET:')
                else:
                    source_value_name = source_value_expr
                    ctx = nullcontext()
                with ctx:
                    self._create_transfer_type_data_body(transfer_type=field_definition.transfer_type, nested_as_dict=self.nested_as_dict, source_value_name=source_value_name, assignment_target=f"{local_dict_name}['{destination_name}']")

    def _create_transfer_type_data_body(self, transfer_type: TransferType, nested_as_dict: bool, source_value_name: str, assignment_target: str) -> None:
        if False:
            return 10
        if isinstance(transfer_type, SimpleType) and transfer_type.nested_field_info:
            if nested_as_dict:
                destination_type: Any = dict
            elif self.is_data_field:
                destination_type = transfer_type.field_definition.annotation
            else:
                destination_type = transfer_type.nested_field_info.model
            self._create_transfer_instance_data(field_definitions=transfer_type.nested_field_info.field_definitions, tmp_return_type_name=assignment_target, source_instance_name=source_value_name, destination_type_name=self._add_to_fn_globals('destination_type', destination_type), destination_type_is_dict=destination_type is dict)
            return
        if isinstance(transfer_type, UnionType) and transfer_type.has_nested:
            self._create_transfer_nested_union_type_data(transfer_type=transfer_type, source_value_name=source_value_name, assignment_target=assignment_target)
            return
        if isinstance(transfer_type, CollectionType):
            origin_name = self._add_to_fn_globals('origin', transfer_type.field_definition.instantiable_origin)
            if transfer_type.has_nested:
                transfer_type_data_fn = TransferFunctionFactory.create_transfer_type_data(is_data_field=self.is_data_field, override_serialization_name=self.override_serialization_name, transfer_type=transfer_type.inner_type)
                transfer_type_data_name = self._add_to_fn_globals('transfer_type_data', transfer_type_data_fn)
                self._add_stmt(f'{assignment_target} = {origin_name}({transfer_type_data_name}(item) for item in {source_value_name})')
                return
            self._add_stmt(f'{assignment_target} = {origin_name}({source_value_name})')
            return
        self._add_stmt(f'{assignment_target} = {source_value_name}')

    def _create_transfer_nested_union_type_data(self, transfer_type: UnionType, source_value_name: str, assignment_target: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        for inner_type in transfer_type.inner_types:
            if isinstance(inner_type, CompositeType):
                continue
            if inner_type.nested_field_info:
                if self.is_data_field:
                    constraint_type = inner_type.nested_field_info.model
                    destination_type = inner_type.field_definition.annotation
                else:
                    constraint_type = inner_type.field_definition.annotation
                    destination_type = inner_type.nested_field_info.model
                constraint_type_name = self._add_to_fn_globals('constraint_type', constraint_type)
                destination_type_name = self._add_to_fn_globals('destination_type', destination_type)
                with self._start_block(f'if isinstance({source_value_name}, {constraint_type_name}):'):
                    self._create_transfer_instance_data(destination_type_name=destination_type_name, destination_type_is_dict=destination_type is dict, field_definitions=inner_type.nested_field_info.field_definitions, source_instance_name=source_value_name, tmp_return_type_name=assignment_target)
                    return
        self._add_stmt(f'{assignment_target} = {source_value_name}')