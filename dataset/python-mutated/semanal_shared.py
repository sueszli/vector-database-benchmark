"""Shared definitions used by different parts of semantic analysis."""
from __future__ import annotations
from abc import abstractmethod
from typing import Callable, Final, overload
from typing_extensions import Literal, Protocol
from mypy_extensions import trait
from mypy import join
from mypy.errorcodes import LITERAL_REQ, ErrorCode
from mypy.nodes import CallExpr, ClassDef, Context, DataclassTransformSpec, Decorator, Expression, FuncDef, NameExpr, Node, OverloadedFuncDef, RefExpr, SymbolNode, SymbolTable, SymbolTableNode, TypeInfo
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.tvar_scope import TypeVarLikeScope
from mypy.type_visitor import ANY_STRATEGY, BoolTypeQuery
from mypy.types import TPDICT_FB_NAMES, AnyType, FunctionLike, Instance, Parameters, ParamSpecFlavor, ParamSpecType, PlaceholderType, ProperType, TupleType, Type, TypeOfAny, TypeVarId, TypeVarLikeType, TypeVarTupleType, UnpackType, get_proper_type
ALLOW_INCOMPATIBLE_OVERRIDE: Final = ('__slots__', '__deletable__', '__match_args__')
PRIORITY_FALLBACKS: Final = 1

@trait
class SemanticAnalyzerCoreInterface:
    """A core abstract interface to generic semantic analyzer functionality.

    This is implemented by both semantic analyzer passes 2 and 3.
    """

    @abstractmethod
    def lookup_qualified(self, name: str, ctx: Context, suppress_errors: bool=False) -> SymbolTableNode | None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abstractmethod
    def lookup_fully_qualified(self, name: str) -> SymbolTableNode:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def lookup_fully_qualified_or_none(self, name: str) -> SymbolTableNode | None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @abstractmethod
    def fail(self, msg: str, ctx: Context, serious: bool=False, *, blocker: bool=False, code: ErrorCode | None=None) -> None:
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def note(self, msg: str, ctx: Context, *, code: ErrorCode | None=None) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def incomplete_feature_enabled(self, feature: str, ctx: Context) -> bool:
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def record_incomplete_ref(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abstractmethod
    def defer(self, debug_context: Context | None=None, force_progress: bool=False) -> None:
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def is_incomplete_namespace(self, fullname: str) -> bool:
        if False:
            print('Hello World!')
        'Is a module or class namespace potentially missing some definitions?'
        raise NotImplementedError

    @property
    @abstractmethod
    def final_iteration(self) -> bool:
        if False:
            return 10
        'Is this the final iteration of semantic analysis?'
        raise NotImplementedError

    @abstractmethod
    def is_future_flag_set(self, flag: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the specific __future__ feature imported'
        raise NotImplementedError

    @property
    @abstractmethod
    def is_stub_file(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abstractmethod
    def is_func_scope(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @property
    @abstractmethod
    def type(self) -> TypeInfo | None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

@trait
class SemanticAnalyzerInterface(SemanticAnalyzerCoreInterface):
    """A limited abstract interface to some generic semantic analyzer pass 2 functionality.

    We use this interface for various reasons:

    * Looser coupling
    * Cleaner import graph
    * Less need to pass around callback functions
    """
    tvar_scope: TypeVarLikeScope

    @abstractmethod
    def lookup(self, name: str, ctx: Context, suppress_errors: bool=False) -> SymbolTableNode | None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abstractmethod
    def named_type(self, fullname: str, args: list[Type] | None=None) -> Instance:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abstractmethod
    def named_type_or_none(self, fullname: str, args: list[Type] | None=None) -> Instance | None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def accept(self, node: Node) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def anal_type(self, t: Type, *, tvar_scope: TypeVarLikeScope | None=None, allow_tuple_literal: bool=False, allow_unbound_tvars: bool=False, allow_required: bool=False, allow_placeholder: bool=False, report_invalid_types: bool=True, prohibit_self_type: str | None=None) -> Type | None:
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def get_and_bind_all_tvars(self, type_exprs: list[Expression]) -> list[TypeVarLikeType]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def basic_new_typeinfo(self, name: str, basetype_or_fallback: Instance, line: int) -> TypeInfo:
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def schedule_patch(self, priority: int, fn: Callable[[], None]) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def add_symbol_table_node(self, name: str, stnode: SymbolTableNode) -> bool:
        if False:
            while True:
                i = 10
        'Add node to the current symbol table.'
        raise NotImplementedError

    @abstractmethod
    def current_symbol_table(self) -> SymbolTable:
        if False:
            while True:
                i = 10
        'Get currently active symbol table.\n\n        May be module, class, or local namespace.\n        '
        raise NotImplementedError

    @abstractmethod
    def add_symbol(self, name: str, node: SymbolNode, context: Context, module_public: bool=True, module_hidden: bool=False, can_defer: bool=True) -> bool:
        if False:
            return 10
        'Add symbol to the current symbol table.'
        raise NotImplementedError

    @abstractmethod
    def add_symbol_skip_local(self, name: str, node: SymbolNode) -> None:
        if False:
            print('Hello World!')
        'Add symbol to the current symbol table, skipping locals.\n\n        This is used to store symbol nodes in a symbol table that\n        is going to be serialized (local namespaces are not serialized).\n        See implementation docstring for more details.\n        '
        raise NotImplementedError

    @abstractmethod
    def parse_bool(self, expr: Expression) -> bool | None:
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def qualified_name(self, n: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @property
    @abstractmethod
    def is_typeshed_stub_file(self) -> bool:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def process_placeholder(self, name: str | None, kind: str, ctx: Context, force_progress: bool=False) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

def set_callable_name(sig: Type, fdef: FuncDef) -> ProperType:
    if False:
        for i in range(10):
            print('nop')
    sig = get_proper_type(sig)
    if isinstance(sig, FunctionLike):
        if fdef.info:
            if fdef.info.fullname in TPDICT_FB_NAMES:
                class_name = 'TypedDict'
            else:
                class_name = fdef.info.name
            return sig.with_name(f'{fdef.name} of {class_name}')
        else:
            return sig.with_name(fdef.name)
    else:
        return sig

def calculate_tuple_fallback(typ: TupleType) -> None:
    if False:
        return 10
    "Calculate a precise item type for the fallback of a tuple type.\n\n    This must be called only after the main semantic analysis pass, since joins\n    aren't available before that.\n\n    Note that there is an apparent chicken and egg problem with respect\n    to verifying type arguments against bounds. Verifying bounds might\n    require fallbacks, but we might use the bounds to calculate the\n    fallbacks. In practice this is not a problem, since the worst that\n    can happen is that we have invalid type argument values, and these\n    can happen in later stages as well (they will generate errors, but\n    we don't prevent their existence).\n    "
    fallback = typ.partial_fallback
    assert fallback.type.fullname == 'builtins.tuple'
    items = []
    for item in typ.items:
        if isinstance(item, UnpackType):
            unpacked_type = get_proper_type(item.type)
            if isinstance(unpacked_type, TypeVarTupleType):
                unpacked_type = get_proper_type(unpacked_type.upper_bound)
            if isinstance(unpacked_type, Instance) and unpacked_type.type.fullname == 'builtins.tuple':
                items.append(unpacked_type.args[0])
            else:
                raise NotImplementedError
        else:
            items.append(item)
    fallback.args = (join.join_type_list(items),)

class _NamedTypeCallback(Protocol):

    def __call__(self, fully_qualified_name: str, args: list[Type] | None=None) -> Instance:
        if False:
            while True:
                i = 10
        ...

def paramspec_args(name: str, fullname: str, id: TypeVarId | int, *, named_type_func: _NamedTypeCallback, line: int=-1, column: int=-1, prefix: Parameters | None=None) -> ParamSpecType:
    if False:
        for i in range(10):
            print('nop')
    return ParamSpecType(name, fullname, id, flavor=ParamSpecFlavor.ARGS, upper_bound=named_type_func('builtins.tuple', [named_type_func('builtins.object')]), default=AnyType(TypeOfAny.from_omitted_generics), line=line, column=column, prefix=prefix)

def paramspec_kwargs(name: str, fullname: str, id: TypeVarId | int, *, named_type_func: _NamedTypeCallback, line: int=-1, column: int=-1, prefix: Parameters | None=None) -> ParamSpecType:
    if False:
        print('Hello World!')
    return ParamSpecType(name, fullname, id, flavor=ParamSpecFlavor.KWARGS, upper_bound=named_type_func('builtins.dict', [named_type_func('builtins.str'), named_type_func('builtins.object')]), default=AnyType(TypeOfAny.from_omitted_generics), line=line, column=column, prefix=prefix)

class HasPlaceholders(BoolTypeQuery):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(ANY_STRATEGY)

    def visit_placeholder_type(self, t: PlaceholderType) -> bool:
        if False:
            i = 10
            return i + 15
        return True

def has_placeholder(typ: Type) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Check if a type contains any placeholder types (recursively).'
    return typ.accept(HasPlaceholders())

def find_dataclass_transform_spec(node: Node | None) -> DataclassTransformSpec | None:
    if False:
        i = 10
        return i + 15
    '\n    Find the dataclass transform spec for the given node, if any exists.\n\n    Per PEP 681 (https://peps.python.org/pep-0681/#the-dataclass-transform-decorator), dataclass\n    transforms can be specified in multiple ways, including decorator functions and\n    metaclasses/base classes. This function resolves the spec from any of these variants.\n    '
    if isinstance(node, CallExpr):
        node = node.callee
    if isinstance(node, RefExpr):
        node = node.node
    if isinstance(node, Decorator):
        node = node.func
    if isinstance(node, OverloadedFuncDef):
        for candidate in node.items:
            spec = find_dataclass_transform_spec(candidate)
            if spec is not None:
                return spec
        return find_dataclass_transform_spec(node.impl)
    if isinstance(node, FuncDef):
        return node.dataclass_transform_spec
    if isinstance(node, ClassDef):
        node = node.info
    if isinstance(node, TypeInfo):
        for base in node.mro[1:]:
            if base.dataclass_transform_spec is not None:
                return base.dataclass_transform_spec
        metaclass_type = node.metaclass_type
        if metaclass_type is not None and metaclass_type.type.dataclass_transform_spec is not None:
            return metaclass_type.type.dataclass_transform_spec
    return None

@overload
def require_bool_literal_argument(api: SemanticAnalyzerInterface | SemanticAnalyzerPluginInterface, expression: Expression, name: str, default: Literal[True] | Literal[False]) -> bool:
    if False:
        print('Hello World!')
    ...

@overload
def require_bool_literal_argument(api: SemanticAnalyzerInterface | SemanticAnalyzerPluginInterface, expression: Expression, name: str, default: None=None) -> bool | None:
    if False:
        for i in range(10):
            print('nop')
    ...

def require_bool_literal_argument(api: SemanticAnalyzerInterface | SemanticAnalyzerPluginInterface, expression: Expression, name: str, default: bool | None=None) -> bool | None:
    if False:
        for i in range(10):
            print('nop')
    "Attempt to interpret an expression as a boolean literal, and fail analysis if we can't."
    value = parse_bool(expression)
    if value is None:
        api.fail(f'"{name}" argument must be a True or False literal', expression, code=LITERAL_REQ)
        return default
    return value

def parse_bool(expr: Expression) -> bool | None:
    if False:
        while True:
            i = 10
    if isinstance(expr, NameExpr):
        if expr.fullname == 'builtins.True':
            return True
        if expr.fullname == 'builtins.False':
            return False
    return None