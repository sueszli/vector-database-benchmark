from __future__ import annotations
import re
import typing
import warnings
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Set, Union, cast
from mypy.nodes import ARG_OPT, ARG_STAR2, GDEF, MDEF, Argument, AssignmentStmt, Block, CallExpr, CastExpr, FuncDef, IndexExpr, MemberExpr, NameExpr, PassStmt, SymbolTableNode, TupleExpr, TypeAlias, Var
from mypy.plugin import Plugin, SemanticAnalyzerPluginInterface
from mypy.plugins.common import _get_argument, add_method
from mypy.semanal_shared import set_callable_name
from mypy.types import AnyType, CallableType, Instance, NoneType, TypeOfAny, TypeVarType, UnionType
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
try:
    from mypy.types import TypeVarDef
except ImportError:
    TypeVarDef = TypeVarType
try:
    from pydantic.mypy import METADATA_KEY as PYDANTIC_METADATA_KEY
    from pydantic.mypy import PydanticModelField
    from strawberry.experimental.pydantic._compat import IS_PYDANTIC_V1
except ImportError:
    PYDANTIC_METADATA_KEY = ''
    IS_PYDANTIC_V1 = False
if TYPE_CHECKING:
    from mypy.nodes import ClassDef, Expression
    from mypy.plugins import AnalyzeTypeContext, CheckerPluginInterface, ClassDefContext, DynamicClassDefContext
    from mypy.types import Type
VERSION_RE = re.compile('(^0|^(?:[1-9][0-9]*))\\.(0|(?:[1-9][0-9]*))')
FALLBACK_VERSION = Decimal('0.800')

class MypyVersion:
    """Stores the mypy version to be used by the plugin"""
    VERSION: Decimal

class InvalidNodeTypeException(Exception):

    def __init__(self, node: Any) -> None:
        if False:
            print('Hello World!')
        self.message = f'Invalid node type: {node!s}'
        super().__init__()

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.message

def lazy_type_analyze_callback(ctx: AnalyzeTypeContext) -> Type:
    if False:
        for i in range(10):
            print('nop')
    if len(ctx.type.args) == 0:
        return AnyType(TypeOfAny.special_form)
    type_name = ctx.type.args[0]
    type_ = ctx.api.analyze_type(type_name)
    return type_

def _get_named_type(name: str, api: SemanticAnalyzerPluginInterface):
    if False:
        return 10
    if '.' in name:
        return api.named_type_or_none(name)
    return api.named_type(name)

def _get_type_for_expr(expr: Expression, api: SemanticAnalyzerPluginInterface) -> Type:
    if False:
        i = 10
        return i + 15
    if isinstance(expr, NameExpr):
        if expr.fullname:
            sym = api.lookup_fully_qualified_or_none(expr.fullname)
            if sym and isinstance(sym.node, Var):
                raise InvalidNodeTypeException(sym.node)
        return _get_named_type(expr.fullname or expr.name, api)
    if isinstance(expr, IndexExpr):
        type_ = _get_type_for_expr(expr.base, api)
        type_.args = (_get_type_for_expr(expr.index, api),)
        return type_
    if isinstance(expr, MemberExpr):
        if expr.fullname:
            return _get_named_type(expr.fullname, api)
        else:
            raise InvalidNodeTypeException(expr)
    if isinstance(expr, CallExpr):
        if expr.analyzed:
            return _get_type_for_expr(expr.analyzed, api)
        else:
            raise InvalidNodeTypeException(expr)
    if isinstance(expr, CastExpr):
        return expr.type
    raise ValueError(f'Unsupported expression {type(expr)}')

def create_type_hook(ctx: DynamicClassDefContext) -> None:
    if False:
        for i in range(10):
            print('nop')
    type_alias = TypeAlias(AnyType(TypeOfAny.from_error), fullname=ctx.api.qualified_name(ctx.name), line=ctx.call.line, column=ctx.call.column)
    ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, type_alias, plugin_generated=True))
    return

def union_hook(ctx: DynamicClassDefContext) -> None:
    if False:
        i = 10
        return i + 15
    try:
        types = ctx.call.args[ctx.call.arg_names.index('types')]
    except ValueError:
        types = ctx.call.args[1]
    if isinstance(types, TupleExpr):
        try:
            type_ = UnionType(tuple((_get_type_for_expr(x, ctx.api) for x in types.items)))
        except InvalidNodeTypeException:
            type_alias = TypeAlias(AnyType(TypeOfAny.from_error), fullname=ctx.api.qualified_name(ctx.name), line=ctx.call.line, column=ctx.call.column)
            ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, type_alias, plugin_generated=False))
            return
        type_alias = TypeAlias(type_, fullname=ctx.api.qualified_name(ctx.name), line=ctx.call.line, column=ctx.call.column)
        ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, type_alias, plugin_generated=False))

def enum_hook(ctx: DynamicClassDefContext) -> None:
    if False:
        for i in range(10):
            print('nop')
    first_argument = ctx.call.args[0]
    if isinstance(first_argument, NameExpr):
        if not first_argument.node:
            ctx.api.defer()
            return
        if isinstance(first_argument.node, Var):
            var_type = first_argument.node.type or AnyType(TypeOfAny.implementation_artifact)
            type_alias = TypeAlias(var_type, fullname=ctx.api.qualified_name(ctx.name), line=ctx.call.line, column=ctx.call.column)
            ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, type_alias, plugin_generated=False))
            return
    enum_type: Optional[Type]
    try:
        enum_type = _get_type_for_expr(first_argument, ctx.api)
    except InvalidNodeTypeException:
        enum_type = None
    if not enum_type:
        enum_type = AnyType(TypeOfAny.from_error)
    type_alias = TypeAlias(enum_type, fullname=ctx.api.qualified_name(ctx.name), line=ctx.call.line, column=ctx.call.column)
    ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, type_alias, plugin_generated=False))

def scalar_hook(ctx: DynamicClassDefContext) -> None:
    if False:
        return 10
    first_argument = ctx.call.args[0]
    if isinstance(first_argument, NameExpr):
        if not first_argument.node:
            ctx.api.defer()
            return
        if isinstance(first_argument.node, Var):
            var_type = first_argument.node.type or AnyType(TypeOfAny.implementation_artifact)
            type_alias = TypeAlias(var_type, fullname=ctx.api.qualified_name(ctx.name), line=ctx.call.line, column=ctx.call.column)
            ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, type_alias, plugin_generated=False))
            return
    scalar_type: Optional[Type]
    try:
        scalar_type = _get_type_for_expr(first_argument, ctx.api)
    except InvalidNodeTypeException:
        scalar_type = None
    if not scalar_type:
        scalar_type = AnyType(TypeOfAny.from_error)
    type_alias = TypeAlias(scalar_type, fullname=ctx.api.qualified_name(ctx.name), line=ctx.call.line, column=ctx.call.column)
    ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, type_alias, plugin_generated=False))

def add_static_method_to_class(api: Union[SemanticAnalyzerPluginInterface, CheckerPluginInterface], cls: ClassDef, name: str, args: List[Argument], return_type: Type, tvar_def: Optional[TypeVarType]=None) -> None:
    if False:
        print('Hello World!')
    'Adds a static method\n    Edited add_method_to_class to incorporate static method logic\n    https://github.com/python/mypy/blob/9c05d3d19/mypy/plugins/common.py\n    '
    info = cls.info
    if name in info.names:
        sym = info.names[name]
        if sym.plugin_generated and isinstance(sym.node, FuncDef):
            cls.defs.body.remove(sym.node)
    if MypyVersion.VERSION < Decimal('0.93'):
        function_type = api.named_type('__builtins__.function')
    elif isinstance(api, SemanticAnalyzerPluginInterface):
        function_type = api.named_type('builtins.function')
    else:
        function_type = api.named_generic_type('builtins.function', [])
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
    func.is_static = True
    func.info = info
    func.type = set_callable_name(signature, func)
    func._fullname = f'{info.fullname}.{name}'
    func.line = info.line
    if name in info.names:
        r_name = get_unique_redefinition_name(name, info.names)
        info.names[r_name] = info.names[name]
    info.names[name] = SymbolTableNode(MDEF, func, plugin_generated=True)
    info.defn.defs.body.append(func)

def strawberry_pydantic_class_callback(ctx: ClassDefContext) -> None:
    if False:
        while True:
            i = 10
    model_expression = _get_argument(call=ctx.reason, name='model')
    if model_expression is None:
        ctx.api.fail('model argument in decorator failed to be parsed', ctx.reason)
    else:
        init_args = [Argument(Var('kwargs'), AnyType(TypeOfAny.explicit), None, ARG_STAR2)]
        add_method(ctx, '__init__', init_args, NoneType())
        model_type = cast(Instance, _get_type_for_expr(model_expression, ctx.api))
        new_strawberry_fields: Set[str] = set()
        for stmt in ctx.cls.defs.body:
            if isinstance(stmt, AssignmentStmt):
                lhs = cast(NameExpr, stmt.lvalues[0])
                new_strawberry_fields.add(lhs.name)
        pydantic_fields: Set[PydanticModelField] = set()
        try:
            fields = model_type.type.metadata[PYDANTIC_METADATA_KEY]['fields']
            for data in fields.items():
                if IS_PYDANTIC_V1:
                    field = PydanticModelField.deserialize(ctx.cls.info, data[1])
                else:
                    field = PydanticModelField.deserialize(info=ctx.cls.info, data=data[1], api=ctx.api)
                pydantic_fields.add(field)
        except KeyError:
            ctx.api.fail('Pydantic plugin not installed, please add pydantic.mypy your mypy.ini plugins', ctx.reason)
        potentially_missing_fields: Set[PydanticModelField] = {f for f in pydantic_fields if f.name not in new_strawberry_fields}
        '\n        Need to check if all_fields=True from the pydantic decorator\n        There is no way to real check that Literal[True] was used\n        We just check if the strawberry type is missing all the fields\n        This means that the user is using all_fields=True\n        '
        is_all_fields: bool = len(potentially_missing_fields) == len(pydantic_fields)
        missing_pydantic_fields: Set[PydanticModelField] = potentially_missing_fields if not is_all_fields else set()
        if 'to_pydantic' not in ctx.cls.info.names:
            if IS_PYDANTIC_V1:
                add_method(ctx, 'to_pydantic', args=[f.to_argument(info=model_type.type, typed=True, force_optional=False, use_alias=True) for f in missing_pydantic_fields], return_type=model_type)
            else:
                add_method(ctx, 'to_pydantic', args=[f.to_argument(current_info=model_type.type, typed=True, force_optional=False, use_alias=True) for f in missing_pydantic_fields], return_type=model_type)
        model_argument = Argument(variable=Var(name='instance', type=model_type), type_annotation=model_type, initializer=None, kind=ARG_OPT)
        add_static_method_to_class(ctx.api, ctx.cls, name='from_pydantic', args=[model_argument], return_type=fill_typevars(ctx.cls.info))

class StrawberryPlugin(Plugin):

    def get_dynamic_class_hook(self, fullname: str) -> Optional[Callable[[DynamicClassDefContext], None]]:
        if False:
            for i in range(10):
                print('nop')
        if self._is_strawberry_union(fullname):
            return union_hook
        if self._is_strawberry_enum(fullname):
            return enum_hook
        if self._is_strawberry_scalar(fullname):
            return scalar_hook
        if self._is_strawberry_create_type(fullname):
            return create_type_hook
        return None

    def get_type_analyze_hook(self, fullname: str) -> Union[Callable[..., Type], None]:
        if False:
            return 10
        if self._is_strawberry_lazy_type(fullname):
            return lazy_type_analyze_callback
        return None

    def get_class_decorator_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        if False:
            i = 10
            return i + 15
        if self._is_strawberry_pydantic_decorator(fullname):
            return strawberry_pydantic_class_callback
        return None

    def _is_strawberry_union(self, fullname: str) -> bool:
        if False:
            print('Hello World!')
        return fullname == 'strawberry.union.union' or fullname.endswith('strawberry.union')

    def _is_strawberry_enum(self, fullname: str) -> bool:
        if False:
            i = 10
            return i + 15
        return fullname == 'strawberry.enum.enum' or fullname.endswith('strawberry.enum')

    def _is_strawberry_scalar(self, fullname: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return fullname == 'strawberry.custom_scalar.scalar' or fullname.endswith('strawberry.scalar')

    def _is_strawberry_lazy_type(self, fullname: str) -> bool:
        if False:
            return 10
        return fullname == 'strawberry.lazy_type.LazyType'

    def _is_strawberry_create_type(self, fullname: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return fullname == 'strawberry.tools.create_type.create_type' or fullname.endswith('.create_type')

    def _is_strawberry_pydantic_decorator(self, fullname: str) -> bool:
        if False:
            while True:
                i = 10
        if any((strawberry_decorator in fullname for strawberry_decorator in ('strawberry.experimental.pydantic.object_type.type', 'strawberry.experimental.pydantic.object_type.input', 'strawberry.experimental.pydantic.object_type.interface', 'strawberry.experimental.pydantic.error_type'))):
            return True
        return any((fullname.endswith(decorator) for decorator in ('strawberry.experimental.pydantic.type', 'strawberry.experimental.pydantic.input', 'strawberry.experimental.pydantic.error_type')))

def plugin(version: str) -> typing.Type[StrawberryPlugin]:
    if False:
        return 10
    match = VERSION_RE.match(version)
    if match:
        MypyVersion.VERSION = Decimal('.'.join(match.groups()))
    else:
        MypyVersion.VERSION = FALLBACK_VERSION
        warnings.warn(f'Mypy version {version} could not be parsed. Reverting to v0.800', stacklevel=1)
    return StrawberryPlugin