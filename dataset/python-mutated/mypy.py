"""This module includes classes and functions designed specifically for use with the mypy plugin."""
from __future__ import annotations
import sys
from configparser import ConfigParser
from typing import Any, Callable, Iterator
from mypy.errorcodes import ErrorCode
from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR2, MDEF, Argument, AssignmentStmt, Block, CallExpr, ClassDef, Context, Decorator, DictExpr, EllipsisExpr, Expression, FuncDef, IfStmt, JsonDict, MemberExpr, NameExpr, PassStmt, PlaceholderNode, RefExpr, Statement, StrExpr, SymbolTableNode, TempNode, TypeAlias, TypeInfo, Var
from mypy.options import Options
from mypy.plugin import CheckerPluginInterface, ClassDefContext, FunctionContext, MethodContext, Plugin, ReportConfigContext, SemanticAnalyzerPluginInterface
from mypy.plugins import dataclasses
from mypy.plugins.common import deserialize_and_fixup_type
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.state import state
from mypy.typeops import map_type_from_supertype
from mypy.types import AnyType, CallableType, Instance, NoneType, Overloaded, Type, TypeOfAny, TypeType, TypeVarType, UnionType, get_proper_type
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic._internal import _fields
from pydantic.version import parse_mypy_version
try:
    from mypy.types import TypeVarDef
except ImportError:
    from mypy.types import TypeVarType as TypeVarDef
CONFIGFILE_KEY = 'pydantic-mypy'
METADATA_KEY = 'pydantic-mypy-metadata'
BASEMODEL_FULLNAME = 'pydantic.main.BaseModel'
BASESETTINGS_FULLNAME = 'pydantic_settings.main.BaseSettings'
ROOT_MODEL_FULLNAME = 'pydantic.root_model.RootModel'
MODEL_METACLASS_FULLNAME = 'pydantic._internal._model_construction.ModelMetaclass'
FIELD_FULLNAME = 'pydantic.fields.Field'
DATACLASS_FULLNAME = 'pydantic.dataclasses.dataclass'
MODEL_VALIDATOR_FULLNAME = 'pydantic.functional_validators.model_validator'
DECORATOR_FULLNAMES = {'pydantic.functional_validators.field_validator', 'pydantic.functional_validators.model_validator', 'pydantic.functional_serializers.serializer', 'pydantic.functional_serializers.model_serializer', 'pydantic.deprecated.class_validators.validator', 'pydantic.deprecated.class_validators.root_validator'}
MYPY_VERSION_TUPLE = parse_mypy_version(mypy_version)
BUILTINS_NAME = 'builtins' if MYPY_VERSION_TUPLE >= (0, 930) else '__builtins__'
__version__ = 2

def plugin(version: str) -> type[Plugin]:
    if False:
        print('Hello World!')
    '`version` is the mypy version string.\n\n    We might want to use this to print a warning if the mypy version being used is\n    newer, or especially older, than we expect (or need).\n\n    Args:\n        version: The mypy version string.\n\n    Return:\n        The Pydantic mypy plugin type.\n    '
    return PydanticPlugin

class _DeferAnalysis(Exception):
    pass

class PydanticPlugin(Plugin):
    """The Pydantic mypy plugin."""

    def __init__(self, options: Options) -> None:
        if False:
            print('Hello World!')
        self.plugin_config = PydanticPluginConfig(options)
        self._plugin_data = self.plugin_config.to_data()
        super().__init__(options)

    def get_base_class_hook(self, fullname: str) -> Callable[[ClassDefContext], bool] | None:
        if False:
            while True:
                i = 10
        'Update Pydantic model class.'
        sym = self.lookup_fully_qualified(fullname)
        if sym and isinstance(sym.node, TypeInfo):
            if any((base.fullname == BASEMODEL_FULLNAME for base in sym.node.mro)):
                return self._pydantic_model_class_maker_callback
        return None

    def get_metaclass_hook(self, fullname: str) -> Callable[[ClassDefContext], None] | None:
        if False:
            print('Hello World!')
        'Update Pydantic `ModelMetaclass` definition.'
        if fullname == MODEL_METACLASS_FULLNAME:
            return self._pydantic_model_metaclass_marker_callback
        return None

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        if False:
            print('Hello World!')
        'Adjust the return type of the `Field` function.'
        sym = self.lookup_fully_qualified(fullname)
        if sym and sym.fullname == FIELD_FULLNAME:
            return self._pydantic_field_callback
        return None

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        if False:
            for i in range(10):
                print('nop')
        'Adjust return type of `from_orm` method call.'
        if fullname.endswith('.from_orm'):
            return from_attributes_callback
        return None

    def get_class_decorator_hook(self, fullname: str) -> Callable[[ClassDefContext], None] | None:
        if False:
            return 10
        'Mark pydantic.dataclasses as dataclass.\n\n        Mypy version 1.1.1 added support for `@dataclass_transform` decorator.\n        '
        if fullname == DATACLASS_FULLNAME and MYPY_VERSION_TUPLE < (1, 1):
            return dataclasses.dataclass_class_maker_callback
        return None

    def report_config_data(self, ctx: ReportConfigContext) -> dict[str, Any]:
        if False:
            return 10
        'Return all plugin config data.\n\n        Used by mypy to determine if cache needs to be discarded.\n        '
        return self._plugin_data

    def _pydantic_model_class_maker_callback(self, ctx: ClassDefContext) -> bool:
        if False:
            print('Hello World!')
        transformer = PydanticModelTransformer(ctx.cls, ctx.reason, ctx.api, self.plugin_config)
        return transformer.transform()

    def _pydantic_model_metaclass_marker_callback(self, ctx: ClassDefContext) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Reset dataclass_transform_spec attribute of ModelMetaclass.\n\n        Let the plugin handle it. This behavior can be disabled\n        if 'debug_dataclass_transform' is set to True', for testing purposes.\n        "
        if self.plugin_config.debug_dataclass_transform:
            return
        info_metaclass = ctx.cls.info.declared_metaclass
        assert info_metaclass, "callback not passed from 'get_metaclass_hook'"
        if getattr(info_metaclass.type, 'dataclass_transform_spec', None):
            info_metaclass.type.dataclass_transform_spec = None

    def _pydantic_field_callback(self, ctx: FunctionContext) -> Type:
        if False:
            return 10
        'Extract the type of the `default` argument from the Field function, and use it as the return type.\n\n        In particular:\n        * Check whether the default and default_factory argument is specified.\n        * Output an error if both are specified.\n        * Retrieve the type of the argument which is specified, and use it as return type for the function.\n        '
        default_any_type = ctx.default_return_type
        assert ctx.callee_arg_names[0] == 'default', '"default" is no longer first argument in Field()'
        assert ctx.callee_arg_names[1] == 'default_factory', '"default_factory" is no longer second argument in Field()'
        default_args = ctx.args[0]
        default_factory_args = ctx.args[1]
        if default_args and default_factory_args:
            error_default_and_default_factory_specified(ctx.api, ctx.context)
            return default_any_type
        if default_args:
            default_type = ctx.arg_types[0][0]
            default_arg = default_args[0]
            if not isinstance(default_arg, EllipsisExpr):
                return default_type
        elif default_factory_args:
            default_factory_type = ctx.arg_types[1][0]
            if isinstance(default_factory_type, Overloaded):
                default_factory_type = default_factory_type.items[0]
            if isinstance(default_factory_type, CallableType):
                ret_type = default_factory_type.ret_type
                args = getattr(ret_type, 'args', None)
                if args:
                    if all((isinstance(arg, TypeVarType) for arg in args)):
                        ret_type.args = tuple((default_any_type for _ in args))
                return ret_type
        return default_any_type

class PydanticPluginConfig:
    """A Pydantic mypy plugin config holder.

    Attributes:
        init_forbid_extra: Whether to add a `**kwargs` at the end of the generated `__init__` signature.
        init_typed: Whether to annotate fields in the generated `__init__`.
        warn_required_dynamic_aliases: Whether to raise required dynamic aliases error.
        debug_dataclass_transform: Whether to not reset `dataclass_transform_spec` attribute
            of `ModelMetaclass` for testing purposes.
    """
    __slots__ = ('init_forbid_extra', 'init_typed', 'warn_required_dynamic_aliases', 'debug_dataclass_transform')
    init_forbid_extra: bool
    init_typed: bool
    warn_required_dynamic_aliases: bool
    debug_dataclass_transform: bool

    def __init__(self, options: Options) -> None:
        if False:
            i = 10
            return i + 15
        if options.config_file is None:
            return
        toml_config = parse_toml(options.config_file)
        if toml_config is not None:
            config = toml_config.get('tool', {}).get('pydantic-mypy', {})
            for key in self.__slots__:
                setting = config.get(key, False)
                if not isinstance(setting, bool):
                    raise ValueError(f'Configuration value must be a boolean for key: {key}')
                setattr(self, key, setting)
        else:
            plugin_config = ConfigParser()
            plugin_config.read(options.config_file)
            for key in self.__slots__:
                setting = plugin_config.getboolean(CONFIGFILE_KEY, key, fallback=False)
                setattr(self, key, setting)

    def to_data(self) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict of config names to their values.'
        return {key: getattr(self, key) for key in self.__slots__}

def from_attributes_callback(ctx: MethodContext) -> Type:
    if False:
        i = 10
        return i + 15
    'Raise an error if from_attributes is not enabled.'
    model_type: Instance
    ctx_type = ctx.type
    if isinstance(ctx_type, TypeType):
        ctx_type = ctx_type.item
    if isinstance(ctx_type, CallableType) and isinstance(ctx_type.ret_type, Instance):
        model_type = ctx_type.ret_type
    elif isinstance(ctx_type, Instance):
        model_type = ctx_type
    else:
        detail = f'ctx.type: {ctx_type} (of type {ctx_type.__class__.__name__})'
        error_unexpected_behavior(detail, ctx.api, ctx.context)
        return ctx.default_return_type
    pydantic_metadata = model_type.type.metadata.get(METADATA_KEY)
    if pydantic_metadata is None:
        return ctx.default_return_type
    from_attributes = pydantic_metadata.get('config', {}).get('from_attributes')
    if from_attributes is not True:
        error_from_attributes(model_type.type.name, ctx.api, ctx.context)
    return ctx.default_return_type

class PydanticModelField:
    """Based on mypy.plugins.dataclasses.DataclassAttribute."""

    def __init__(self, name: str, alias: str | None, has_dynamic_alias: bool, has_default: bool, line: int, column: int, type: Type | None, info: TypeInfo):
        if False:
            return 10
        self.name = name
        self.alias = alias
        self.has_dynamic_alias = has_dynamic_alias
        self.has_default = has_default
        self.line = line
        self.column = column
        self.type = type
        self.info = info

    def to_argument(self, current_info: TypeInfo, typed: bool, force_optional: bool, use_alias: bool) -> Argument:
        if False:
            return 10
        'Based on mypy.plugins.dataclasses.DataclassAttribute.to_argument.'
        return Argument(variable=self.to_var(current_info, use_alias), type_annotation=self.expand_type(current_info) if typed else AnyType(TypeOfAny.explicit), initializer=None, kind=ARG_NAMED_OPT if force_optional or self.has_default else ARG_NAMED)

    def expand_type(self, current_info: TypeInfo) -> Type | None:
        if False:
            print('Hello World!')
        'Based on mypy.plugins.dataclasses.DataclassAttribute.expand_type.'
        if self.type is not None and getattr(self.info, 'self_type', None) is not None:
            expanded_type = expand_type(self.type, {self.info.self_type.id: fill_typevars(current_info)})
            if isinstance(self.type, UnionType) and (not isinstance(expanded_type, UnionType)):
                raise _DeferAnalysis()
            return expanded_type
        return self.type

    def to_var(self, current_info: TypeInfo, use_alias: bool) -> Var:
        if False:
            print('Hello World!')
        'Based on mypy.plugins.dataclasses.DataclassAttribute.to_var.'
        if use_alias and self.alias is not None:
            name = self.alias
        else:
            name = self.name
        return Var(name, self.expand_type(current_info))

    def serialize(self) -> JsonDict:
        if False:
            i = 10
            return i + 15
        'Based on mypy.plugins.dataclasses.DataclassAttribute.serialize.'
        assert self.type
        return {'name': self.name, 'alias': self.alias, 'has_dynamic_alias': self.has_dynamic_alias, 'has_default': self.has_default, 'line': self.line, 'column': self.column, 'type': self.type.serialize()}

    @classmethod
    def deserialize(cls, info: TypeInfo, data: JsonDict, api: SemanticAnalyzerPluginInterface) -> PydanticModelField:
        if False:
            for i in range(10):
                print('nop')
        'Based on mypy.plugins.dataclasses.DataclassAttribute.deserialize.'
        data = data.copy()
        typ = deserialize_and_fixup_type(data.pop('type'), api)
        return cls(type=typ, info=info, **data)

    def expand_typevar_from_subtype(self, sub_type: TypeInfo) -> None:
        if False:
            i = 10
            return i + 15
        'Expands type vars in the context of a subtype when an attribute is inherited\n        from a generic super type.\n        '
        if self.type is not None:
            self.type = map_type_from_supertype(self.type, sub_type, self.info)

class PydanticModelClassVar:
    """Class vars are stored to be ignored by subclasses.

    Attributes:
        name: the class var name
    """

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.name = name

    @classmethod
    def deserialize(cls, data: JsonDict) -> PydanticModelClassVar:
        if False:
            i = 10
            return i + 15
        'Based on mypy.plugins.dataclasses.DataclassAttribute.deserialize.'
        data = data.copy()
        return cls(**data)

    def serialize(self) -> JsonDict:
        if False:
            while True:
                i = 10
        'Based on mypy.plugins.dataclasses.DataclassAttribute.serialize.'
        return {'name': self.name}

class PydanticModelTransformer:
    """Transform the BaseModel subclass according to the plugin settings.

    Attributes:
        tracked_config_fields: A set of field configs that the plugin has to track their value.
    """
    tracked_config_fields: set[str] = {'extra', 'frozen', 'from_attributes', 'populate_by_name', 'alias_generator'}

    def __init__(self, cls: ClassDef, reason: Expression | Statement, api: SemanticAnalyzerPluginInterface, plugin_config: PydanticPluginConfig) -> None:
        if False:
            print('Hello World!')
        self._cls = cls
        self._reason = reason
        self._api = api
        self.plugin_config = plugin_config

    def transform(self) -> bool:
        if False:
            return 10
        'Configures the BaseModel subclass according to the plugin settings.\n\n        In particular:\n\n        * determines the model config and fields,\n        * adds a fields-aware signature for the initializer and construct methods\n        * freezes the class if frozen = True\n        * stores the fields, config, and if the class is settings in the mypy metadata for access by subclasses\n        '
        info = self._cls.info
        is_root_model = any((ROOT_MODEL_FULLNAME in base.fullname for base in info.mro[:-1]))
        config = self.collect_config()
        (fields, classvars) = self.collect_fields_and_classvar(config, is_root_model)
        if fields is None or classvars is None:
            return False
        for field in fields:
            if field.type is None:
                return False
        is_settings = any((base.fullname == BASESETTINGS_FULLNAME for base in info.mro[:-1]))
        self.add_initializer(fields, config, is_settings, is_root_model)
        self.add_model_construct_method(fields, config, is_settings)
        try:
            self.set_frozen(fields, frozen=config.frozen is True)
        except _DeferAnalysis:
            if not self._api.final_iteration:
                self._api.defer()
        self.adjust_decorator_signatures()
        info.metadata[METADATA_KEY] = {'fields': {field.name: field.serialize() for field in fields}, 'classvars': {classvar.name: classvar.serialize() for classvar in classvars}, 'config': config.get_values_dict()}
        return True

    def adjust_decorator_signatures(self) -> None:
        if False:
            print('Hello World!')
        'When we decorate a function `f` with `pydantic.validator(...)`, `pydantic.field_validator`\n        or `pydantic.serializer(...)`, mypy sees `f` as a regular method taking a `self` instance,\n        even though pydantic internally wraps `f` with `classmethod` if necessary.\n\n        Teach mypy this by marking any function whose outermost decorator is a `validator()`,\n        `field_validator()` or `serializer()` call as a `classmethod`.\n        '
        for (name, sym) in self._cls.info.names.items():
            if isinstance(sym.node, Decorator):
                first_dec = sym.node.original_decorators[0]
                if isinstance(first_dec, CallExpr) and isinstance(first_dec.callee, NameExpr) and (first_dec.callee.fullname in DECORATOR_FULLNAMES) and (not (first_dec.callee.fullname == MODEL_VALIDATOR_FULLNAME and any((first_dec.arg_names[i] == 'mode' and isinstance(arg, StrExpr) and (arg.value == 'after') for (i, arg) in enumerate(first_dec.args))))):
                    sym.node.func.is_class = True

    def collect_config(self) -> ModelConfigData:
        if False:
            i = 10
            return i + 15
        'Collects the values of the config attributes that are used by the plugin, accounting for parent classes.'
        cls = self._cls
        config = ModelConfigData()
        has_config_kwargs = False
        has_config_from_namespace = False
        for (name, expr) in cls.keywords.items():
            config_data = self.get_config_update(name, expr)
            if config_data:
                has_config_kwargs = True
                config.update(config_data)
        stmt: Statement | None = None
        for stmt in cls.defs.body:
            if not isinstance(stmt, (AssignmentStmt, ClassDef)):
                continue
            if isinstance(stmt, AssignmentStmt):
                lhs = stmt.lvalues[0]
                if not isinstance(lhs, NameExpr) or lhs.name != 'model_config':
                    continue
                if isinstance(stmt.rvalue, CallExpr):
                    for (arg_name, arg) in zip(stmt.rvalue.arg_names, stmt.rvalue.args):
                        if arg_name is None:
                            continue
                        config.update(self.get_config_update(arg_name, arg))
                elif isinstance(stmt.rvalue, DictExpr):
                    for (key_expr, value_expr) in stmt.rvalue.items:
                        if not isinstance(key_expr, StrExpr):
                            continue
                        config.update(self.get_config_update(key_expr.value, value_expr))
            elif isinstance(stmt, ClassDef):
                if stmt.name != 'Config':
                    continue
                for substmt in stmt.defs.body:
                    if not isinstance(substmt, AssignmentStmt):
                        continue
                    lhs = substmt.lvalues[0]
                    if not isinstance(lhs, NameExpr):
                        continue
                    config.update(self.get_config_update(lhs.name, substmt.rvalue))
            if has_config_kwargs:
                self._api.fail('Specifying config in two places is ambiguous, use either Config attribute or class kwargs', cls)
                break
            has_config_from_namespace = True
        if has_config_kwargs or has_config_from_namespace:
            if stmt and config.has_alias_generator and (not config.populate_by_name) and self.plugin_config.warn_required_dynamic_aliases:
                error_required_dynamic_aliases(self._api, stmt)
        for info in cls.info.mro[1:]:
            if METADATA_KEY not in info.metadata:
                continue
            self._api.add_plugin_dependency(make_wildcard_trigger(info.fullname))
            for (name, value) in info.metadata[METADATA_KEY]['config'].items():
                config.setdefault(name, value)
        return config

    def collect_fields_and_classvar(self, model_config: ModelConfigData, is_root_model: bool) -> tuple[list[PydanticModelField] | None, list[PydanticModelClassVar] | None]:
        if False:
            print('Hello World!')
        'Collects the fields for the model, accounting for parent classes.'
        cls = self._cls
        found_fields: dict[str, PydanticModelField] = {}
        found_classvars: dict[str, PydanticModelClassVar] = {}
        for info in reversed(cls.info.mro[1:-1]):
            if METADATA_KEY not in info.metadata:
                continue
            self._api.add_plugin_dependency(make_wildcard_trigger(info.fullname))
            for (name, data) in info.metadata[METADATA_KEY]['fields'].items():
                field = PydanticModelField.deserialize(info, data, self._api)
                with state.strict_optional_set(self._api.options.strict_optional):
                    field.expand_typevar_from_subtype(cls.info)
                found_fields[name] = field
                sym_node = cls.info.names.get(name)
                if sym_node and sym_node.node and (not isinstance(sym_node.node, Var)):
                    self._api.fail('BaseModel field may only be overridden by another field', sym_node.node)
            for (name, data) in info.metadata[METADATA_KEY]['classvars'].items():
                found_classvars[name] = PydanticModelClassVar.deserialize(data)
        current_field_names: set[str] = set()
        current_classvars_names: set[str] = set()
        for stmt in self._get_assignment_statements_from_block(cls.defs):
            maybe_field = self.collect_field_and_classvars_from_stmt(stmt, model_config, found_classvars)
            if isinstance(maybe_field, PydanticModelField):
                lhs = stmt.lvalues[0]
                if is_root_model and lhs.name != 'root':
                    error_extra_fields_on_root_model(self._api, stmt)
                else:
                    current_field_names.add(lhs.name)
                    found_fields[lhs.name] = maybe_field
            elif isinstance(maybe_field, PydanticModelClassVar):
                lhs = stmt.lvalues[0]
                current_classvars_names.add(lhs.name)
                found_classvars[lhs.name] = maybe_field
        return (list(found_fields.values()), list(found_classvars.values()))

    def _get_assignment_statements_from_if_statement(self, stmt: IfStmt) -> Iterator[AssignmentStmt]:
        if False:
            for i in range(10):
                print('nop')
        for body in stmt.body:
            if not body.is_unreachable:
                yield from self._get_assignment_statements_from_block(body)
        if stmt.else_body is not None and (not stmt.else_body.is_unreachable):
            yield from self._get_assignment_statements_from_block(stmt.else_body)

    def _get_assignment_statements_from_block(self, block: Block) -> Iterator[AssignmentStmt]:
        if False:
            return 10
        for stmt in block.body:
            if isinstance(stmt, AssignmentStmt):
                yield stmt
            elif isinstance(stmt, IfStmt):
                yield from self._get_assignment_statements_from_if_statement(stmt)

    def collect_field_and_classvars_from_stmt(self, stmt: AssignmentStmt, model_config: ModelConfigData, classvars: dict[str, PydanticModelClassVar]) -> PydanticModelField | PydanticModelClassVar | None:
        if False:
            i = 10
            return i + 15
        'Get pydantic model field from statement.\n\n        Args:\n            stmt: The statement.\n            model_config: Configuration settings for the model.\n\n        Returns:\n            A pydantic model field if it could find the field in statement. Otherwise, `None`.\n        '
        cls = self._cls
        lhs = stmt.lvalues[0]
        if not isinstance(lhs, NameExpr) or not _fields.is_valid_field_name(lhs.name) or lhs.name == 'model_config':
            return None
        if not stmt.new_syntax:
            if isinstance(stmt.rvalue, CallExpr) and isinstance(stmt.rvalue.callee, CallExpr) and isinstance(stmt.rvalue.callee.callee, NameExpr) and (stmt.rvalue.callee.callee.fullname in DECORATOR_FULLNAMES):
                return None
            if lhs.name in classvars:
                return None
            error_untyped_fields(self._api, stmt)
            return None
        lhs = stmt.lvalues[0]
        if not isinstance(lhs, NameExpr):
            return None
        if not _fields.is_valid_field_name(lhs.name) or lhs.name == 'model_config':
            return None
        sym = cls.info.names.get(lhs.name)
        if sym is None:
            return None
        node = sym.node
        if isinstance(node, PlaceholderNode):
            return None
        if isinstance(node, TypeAlias):
            self._api.fail('Type aliases inside BaseModel definitions are not supported at runtime', node)
            return None
        if not isinstance(node, Var):
            return None
        if node.is_classvar:
            return PydanticModelClassVar(lhs.name)
        node_type = get_proper_type(node.type)
        if isinstance(node_type, Instance) and node_type.type.fullname == 'dataclasses.InitVar':
            self._api.fail('InitVar is not supported in BaseModel', node)
        has_default = self.get_has_default(stmt)
        if sym.type is None and node.is_final and node.is_inferred:
            typ = self._api.analyze_simple_literal_type(stmt.rvalue, is_final=True)
            if typ:
                node.type = typ
            else:
                self._api.fail('Need type argument for Final[...] with non-literal default in BaseModel', stmt)
                node.type = AnyType(TypeOfAny.from_error)
        (alias, has_dynamic_alias) = self.get_alias_info(stmt)
        if has_dynamic_alias and (not model_config.populate_by_name) and self.plugin_config.warn_required_dynamic_aliases:
            error_required_dynamic_aliases(self._api, stmt)
        init_type = self._infer_dataclass_attr_init_type(sym, lhs.name, stmt)
        return PydanticModelField(name=lhs.name, has_dynamic_alias=has_dynamic_alias, has_default=has_default, alias=alias, line=stmt.line, column=stmt.column, type=init_type, info=cls.info)

    def _infer_dataclass_attr_init_type(self, sym: SymbolTableNode, name: str, context: Context) -> Type | None:
        if False:
            for i in range(10):
                print('nop')
        'Infer __init__ argument type for an attribute.\n\n        In particular, possibly use the signature of __set__.\n        '
        default = sym.type
        if sym.implicit:
            return default
        t = get_proper_type(sym.type)
        if not isinstance(t, Instance):
            return default
        setter = t.type.get('__set__')
        if setter:
            if isinstance(setter.node, FuncDef):
                super_info = t.type.get_containing_type_info('__set__')
                assert super_info
                if setter.type:
                    setter_type = get_proper_type(map_type_from_supertype(setter.type, t.type, super_info))
                else:
                    return AnyType(TypeOfAny.unannotated)
                if isinstance(setter_type, CallableType) and setter_type.arg_kinds == [ARG_POS, ARG_POS, ARG_POS]:
                    return expand_type_by_instance(setter_type.arg_types[2], t)
                else:
                    self._api.fail(f'Unsupported signature for "__set__" in "{t.type.name}"', context)
            else:
                self._api.fail(f'Unsupported "__set__" in "{t.type.name}"', context)
        return default

    def add_initializer(self, fields: list[PydanticModelField], config: ModelConfigData, is_settings: bool, is_root_model: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Adds a fields-aware `__init__` method to the class.\n\n        The added `__init__` will be annotated with types vs. all `Any` depending on the plugin settings.\n        '
        if '__init__' in self._cls.info.names and (not self._cls.info.names['__init__'].plugin_generated):
            return
        typed = self.plugin_config.init_typed
        use_alias = config.populate_by_name is not True
        requires_dynamic_aliases = bool(config.has_alias_generator and (not config.populate_by_name))
        with state.strict_optional_set(self._api.options.strict_optional):
            args = self.get_field_arguments(fields, typed=typed, requires_dynamic_aliases=requires_dynamic_aliases, use_alias=use_alias, is_settings=is_settings)
            if is_root_model:
                args[0].kind = ARG_POS if args[0].kind == ARG_NAMED else ARG_OPT
            if is_settings:
                base_settings_node = self._api.lookup_fully_qualified(BASESETTINGS_FULLNAME).node
                if '__init__' in base_settings_node.names:
                    base_settings_init_node = base_settings_node.names['__init__'].node
                    if base_settings_init_node is not None and base_settings_init_node.type is not None:
                        func_type = base_settings_init_node.type
                        for (arg_idx, arg_name) in enumerate(func_type.arg_names):
                            if arg_name.startswith('__') or not arg_name.startswith('_'):
                                continue
                            analyzed_variable_type = self._api.anal_type(func_type.arg_types[arg_idx])
                            variable = Var(arg_name, analyzed_variable_type)
                            args.append(Argument(variable, analyzed_variable_type, None, ARG_OPT))
        if not self.should_init_forbid_extra(fields, config):
            var = Var('kwargs')
            args.append(Argument(var, AnyType(TypeOfAny.explicit), None, ARG_STAR2))
        add_method(self._api, self._cls, '__init__', args=args, return_type=NoneType())

    def add_model_construct_method(self, fields: list[PydanticModelField], config: ModelConfigData, is_settings: bool) -> None:
        if False:
            return 10
        'Adds a fully typed `model_construct` classmethod to the class.\n\n        Similar to the fields-aware __init__ method, but always uses the field names (not aliases),\n        and does not treat settings fields as optional.\n        '
        set_str = self._api.named_type(f'{BUILTINS_NAME}.set', [self._api.named_type(f'{BUILTINS_NAME}.str')])
        optional_set_str = UnionType([set_str, NoneType()])
        fields_set_argument = Argument(Var('_fields_set', optional_set_str), optional_set_str, None, ARG_OPT)
        with state.strict_optional_set(self._api.options.strict_optional):
            args = self.get_field_arguments(fields, typed=True, requires_dynamic_aliases=False, use_alias=False, is_settings=is_settings)
        if not self.should_init_forbid_extra(fields, config):
            var = Var('kwargs')
            args.append(Argument(var, AnyType(TypeOfAny.explicit), None, ARG_STAR2))
        args = [fields_set_argument] + args
        add_method(self._api, self._cls, 'model_construct', args=args, return_type=fill_typevars(self._cls.info), is_classmethod=True)

    def set_frozen(self, fields: list[PydanticModelField], frozen: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Marks all fields as properties so that attempts to set them trigger mypy errors.\n\n        This is the same approach used by the attrs and dataclasses plugins.\n        '
        info = self._cls.info
        for field in fields:
            sym_node = info.names.get(field.name)
            if sym_node is not None:
                var = sym_node.node
                if isinstance(var, Var):
                    var.is_property = frozen
                elif isinstance(var, PlaceholderNode) and (not self._api.final_iteration):
                    self._api.defer()
                else:
                    try:
                        var_str = str(var)
                    except TypeError:
                        var_str = repr(var)
                    detail = f'sym_node.node: {var_str} (of type {var.__class__})'
                    error_unexpected_behavior(detail, self._api, self._cls)
            else:
                var = field.to_var(info, use_alias=False)
                var.info = info
                var.is_property = frozen
                var._fullname = info.fullname + '.' + var.name
                info.names[var.name] = SymbolTableNode(MDEF, var)

    def get_config_update(self, name: str, arg: Expression) -> ModelConfigData | None:
        if False:
            return 10
        "Determines the config update due to a single kwarg in the ConfigDict definition.\n\n        Warns if a tracked config attribute is set to a value the plugin doesn't know how to interpret (e.g., an int)\n        "
        if name not in self.tracked_config_fields:
            return None
        if name == 'extra':
            if isinstance(arg, StrExpr):
                forbid_extra = arg.value == 'forbid'
            elif isinstance(arg, MemberExpr):
                forbid_extra = arg.name == 'forbid'
            else:
                error_invalid_config_value(name, self._api, arg)
                return None
            return ModelConfigData(forbid_extra=forbid_extra)
        if name == 'alias_generator':
            has_alias_generator = True
            if isinstance(arg, NameExpr) and arg.fullname == 'builtins.None':
                has_alias_generator = False
            return ModelConfigData(has_alias_generator=has_alias_generator)
        if isinstance(arg, NameExpr) and arg.fullname in ('builtins.True', 'builtins.False'):
            return ModelConfigData(**{name: arg.fullname == 'builtins.True'})
        error_invalid_config_value(name, self._api, arg)
        return None

    @staticmethod
    def get_has_default(stmt: AssignmentStmt) -> bool:
        if False:
            return 10
        'Returns a boolean indicating whether the field defined in `stmt` is a required field.'
        expr = stmt.rvalue
        if isinstance(expr, TempNode):
            return False
        if isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr) and (expr.callee.fullname == FIELD_FULLNAME):
            for (arg, name) in zip(expr.args, expr.arg_names):
                if name is None or name == 'default':
                    return arg.__class__ is not EllipsisExpr
                if name == 'default_factory':
                    return not (isinstance(arg, NameExpr) and arg.fullname == 'builtins.None')
            return False
        return not isinstance(expr, EllipsisExpr)

    @staticmethod
    def get_alias_info(stmt: AssignmentStmt) -> tuple[str | None, bool]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a pair (alias, has_dynamic_alias), extracted from the declaration of the field defined in `stmt`.\n\n        `has_dynamic_alias` is True if and only if an alias is provided, but not as a string literal.\n        If `has_dynamic_alias` is True, `alias` will be None.\n        '
        expr = stmt.rvalue
        if isinstance(expr, TempNode):
            return (None, False)
        if not (isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr) and (expr.callee.fullname == FIELD_FULLNAME)):
            return (None, False)
        for (i, arg_name) in enumerate(expr.arg_names):
            if arg_name != 'alias':
                continue
            arg = expr.args[i]
            if isinstance(arg, StrExpr):
                return (arg.value, False)
            else:
                return (None, True)
        return (None, False)

    def get_field_arguments(self, fields: list[PydanticModelField], typed: bool, use_alias: bool, requires_dynamic_aliases: bool, is_settings: bool) -> list[Argument]:
        if False:
            return 10
        'Helper function used during the construction of the `__init__` and `model_construct` method signatures.\n\n        Returns a list of mypy Argument instances for use in the generated signatures.\n        '
        info = self._cls.info
        arguments = [field.to_argument(info, typed=typed, force_optional=requires_dynamic_aliases or is_settings, use_alias=use_alias) for field in fields if not (use_alias and field.has_dynamic_alias)]
        return arguments

    def should_init_forbid_extra(self, fields: list[PydanticModelField], config: ModelConfigData) -> bool:
        if False:
            while True:
                i = 10
        'Indicates whether the generated `__init__` should get a `**kwargs` at the end of its signature.\n\n        We disallow arbitrary kwargs if the extra config setting is "forbid", or if the plugin config says to,\n        *unless* a required dynamic alias is present (since then we can\'t determine a valid signature).\n        '
        if not config.populate_by_name:
            if self.is_dynamic_alias_present(fields, bool(config.has_alias_generator)):
                return False
        if config.forbid_extra:
            return True
        return self.plugin_config.init_forbid_extra

    @staticmethod
    def is_dynamic_alias_present(fields: list[PydanticModelField], has_alias_generator: bool) -> bool:
        if False:
            return 10
        'Returns whether any fields on the model have a "dynamic alias", i.e., an alias that cannot be\n        determined during static analysis.\n        '
        for field in fields:
            if field.has_dynamic_alias:
                return True
        if has_alias_generator:
            for field in fields:
                if field.alias is None:
                    return True
        return False

class ModelConfigData:
    """Pydantic mypy plugin model config class."""

    def __init__(self, forbid_extra: bool | None=None, frozen: bool | None=None, from_attributes: bool | None=None, populate_by_name: bool | None=None, has_alias_generator: bool | None=None):
        if False:
            return 10
        self.forbid_extra = forbid_extra
        self.frozen = frozen
        self.from_attributes = from_attributes
        self.populate_by_name = populate_by_name
        self.has_alias_generator = has_alias_generator

    def get_values_dict(self) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict of Pydantic model config names to their values.\n\n        It includes the config if config value is not `None`.\n        '
        return {k: v for (k, v) in self.__dict__.items() if v is not None}

    def update(self, config: ModelConfigData | None) -> None:
        if False:
            i = 10
            return i + 15
        'Update Pydantic model config values.'
        if config is None:
            return
        for (k, v) in config.get_values_dict().items():
            setattr(self, k, v)

    def setdefault(self, key: str, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set default value for Pydantic model config if config value is `None`.'
        if getattr(self, key) is None:
            setattr(self, key, value)
ERROR_ORM = ErrorCode('pydantic-orm', 'Invalid from_attributes call', 'Pydantic')
ERROR_CONFIG = ErrorCode('pydantic-config', 'Invalid config value', 'Pydantic')
ERROR_ALIAS = ErrorCode('pydantic-alias', 'Dynamic alias disallowed', 'Pydantic')
ERROR_UNEXPECTED = ErrorCode('pydantic-unexpected', 'Unexpected behavior', 'Pydantic')
ERROR_UNTYPED = ErrorCode('pydantic-field', 'Untyped field disallowed', 'Pydantic')
ERROR_FIELD_DEFAULTS = ErrorCode('pydantic-field', 'Invalid Field defaults', 'Pydantic')
ERROR_EXTRA_FIELD_ROOT_MODEL = ErrorCode('pydantic-field', 'Extra field on RootModel subclass', 'Pydantic')

def error_from_attributes(model_name: str, api: CheckerPluginInterface, context: Context) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Emits an error when the model does not have `from_attributes=True`.'
    api.fail(f'"{model_name}" does not have from_attributes=True', context, code=ERROR_ORM)

def error_invalid_config_value(name: str, api: SemanticAnalyzerPluginInterface, context: Context) -> None:
    if False:
        return 10
    'Emits an error when the config value is invalid.'
    api.fail(f'Invalid value for "Config.{name}"', context, code=ERROR_CONFIG)

def error_required_dynamic_aliases(api: SemanticAnalyzerPluginInterface, context: Context) -> None:
    if False:
        print('Hello World!')
    'Emits required dynamic aliases error.\n\n    This will be called when `warn_required_dynamic_aliases=True`.\n    '
    api.fail('Required dynamic aliases disallowed', context, code=ERROR_ALIAS)

def error_unexpected_behavior(detail: str, api: CheckerPluginInterface | SemanticAnalyzerPluginInterface, context: Context) -> None:
    if False:
        while True:
            i = 10
    'Emits unexpected behavior error.'
    link = 'https://github.com/pydantic/pydantic/issues/new/choose'
    full_message = f'The pydantic mypy plugin ran into unexpected behavior: {detail}\n'
    full_message += f'Please consider reporting this bug at {link} so we can try to fix it!'
    api.fail(full_message, context, code=ERROR_UNEXPECTED)

def error_untyped_fields(api: SemanticAnalyzerPluginInterface, context: Context) -> None:
    if False:
        print('Hello World!')
    'Emits an error when there is an untyped field in the model.'
    api.fail('Untyped fields disallowed', context, code=ERROR_UNTYPED)

def error_extra_fields_on_root_model(api: CheckerPluginInterface, context: Context) -> None:
    if False:
        i = 10
        return i + 15
    'Emits an error when there is more than just a root field defined for a subclass of RootModel.'
    api.fail('Only `root` is allowed as a field of a `RootModel`', context, code=ERROR_EXTRA_FIELD_ROOT_MODEL)

def error_default_and_default_factory_specified(api: CheckerPluginInterface, context: Context) -> None:
    if False:
        return 10
    'Emits an error when `Field` has both `default` and `default_factory` together.'
    api.fail('Field default and default_factory cannot be specified together', context, code=ERROR_FIELD_DEFAULTS)

def add_method(api: SemanticAnalyzerPluginInterface | CheckerPluginInterface, cls: ClassDef, name: str, args: list[Argument], return_type: Type, self_type: Type | None=None, tvar_def: TypeVarDef | None=None, is_classmethod: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Very closely related to `mypy.plugins.common.add_method_to_class`, with a few pydantic-specific changes.'
    info = cls.info
    if name in info.names:
        sym = info.names[name]
        if sym.plugin_generated and isinstance(sym.node, FuncDef):
            cls.defs.body.remove(sym.node)
    if isinstance(api, SemanticAnalyzerPluginInterface):
        function_type = api.named_type('builtins.function')
    else:
        function_type = api.named_generic_type('builtins.function', [])
    if is_classmethod:
        self_type = self_type or TypeType(fill_typevars(info))
        first = [Argument(Var('_cls'), self_type, None, ARG_POS, True)]
    else:
        self_type = self_type or fill_typevars(info)
        first = [Argument(Var('__pydantic_self__'), self_type, None, ARG_POS)]
    args = first + args
    (arg_types, arg_names, arg_kinds) = ([], [], [])
    for arg in args:
        assert arg.type_annotation, 'All arguments must be fully typed.'
        arg_types.append(arg.type_annotation)
        arg_names.append(arg.variable.name)
        arg_kinds.append(arg.kind)
    signature = CallableType(arg_types, arg_kinds, arg_names, return_type, function_type)
    if tvar_def:
        signature.variables = [tvar_def]
    func = FuncDef(name, args, Block([PassStmt()]))
    func.info = info
    func.type = set_callable_name(signature, func)
    func.is_class = is_classmethod
    func._fullname = info.fullname + '.' + name
    func.line = info.line
    if name in info.names:
        r_name = get_unique_redefinition_name(name, info.names)
        info.names[r_name] = info.names[name]
    if is_classmethod:
        func.is_decorated = True
        v = Var(name, func.type)
        v.info = info
        v._fullname = func._fullname
        v.is_classmethod = True
        dec = Decorator(func, [NameExpr('classmethod')], v)
        dec.line = info.line
        sym = SymbolTableNode(MDEF, dec)
    else:
        sym = SymbolTableNode(MDEF, func)
    sym.plugin_generated = True
    info.names[name] = sym
    info.defn.defs.body.append(func)

def parse_toml(config_file: str) -> dict[str, Any] | None:
    if False:
        while True:
            i = 10
    'Returns a dict of config keys to values.\n\n    It reads configs from toml file and returns `None` if the file is not a toml file.\n    '
    if not config_file.endswith('.toml'):
        return None
    if sys.version_info >= (3, 11):
        import tomllib as toml_
    else:
        try:
            import tomli as toml_
        except ImportError:
            import warnings
            warnings.warn('No TOML parser installed, cannot read configuration from `pyproject.toml`.')
            return None
    with open(config_file, 'rb') as rf:
        return toml_.load(rf)