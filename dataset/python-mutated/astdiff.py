"""Utilities for comparing two versions of a module symbol table.

The goal is to find which AST nodes have externally visible changes, so
that we can fire triggers and re-process other parts of the program
that are stale because of the changes.

Only look at detail at definitions at the current module -- don't
recurse into other modules.

A summary of the module contents:

* snapshot_symbol_table(...) creates an opaque snapshot description of a
  module/class symbol table (recursing into nested class symbol tables).

* compare_symbol_table_snapshots(...) compares two snapshots for the same
  module id and returns fully qualified names of differences (which act as
  triggers).

To compare two versions of a module symbol table, take snapshots of both
versions and compare the snapshots. The use of snapshots makes it easy to
compare two versions of the *same* symbol table that is being mutated.

Summary of how this works for certain kinds of differences:

* If a symbol table node is deleted or added (only present in old/new version
  of the symbol table), it is considered different, of course.

* If a symbol table node refers to a different sort of thing in the new version,
  it is considered different (for example, if a class is replaced with a
  function).

* If the signature of a function has changed, it is considered different.

* If the type of a variable changes, it is considered different.

* If the MRO of a class changes, or a non-generic class is turned into a
  generic class, the class is considered different (there are other such "big"
  differences that cause a class to be considered changed). However, just changes
  to attributes or methods don't generally constitute a difference at the
  class level -- these are handled at attribute level (say, 'mod.Cls.method'
  is different rather than 'mod.Cls' being different).

* If an imported name targets a different name (say, 'from x import y' is
  replaced with 'from z import y'), the name in the module is considered
  different. If the target of an import continues to have the same name,
  but it's specifics change, this doesn't mean that the imported name is
  treated as changed. Say, there is 'from x import y' in 'm', and the
  type of 'x.y' has changed. This doesn't mean that that 'm.y' is considered
  changed. Instead, processing the difference in 'm' will be handled through
  fine-grained dependencies.
"""
from __future__ import annotations
from typing import Sequence, Tuple, Union
from typing_extensions import TypeAlias as _TypeAlias
from mypy.expandtype import expand_type
from mypy.nodes import UNBOUND_IMPORTED, Decorator, FuncBase, FuncDef, FuncItem, MypyFile, OverloadedFuncDef, ParamSpecExpr, SymbolNode, SymbolTable, TypeAlias, TypeInfo, TypeVarExpr, TypeVarTupleExpr, Var
from mypy.semanal_shared import find_dataclass_transform_spec
from mypy.state import state
from mypy.types import AnyType, CallableType, DeletedType, ErasedType, Instance, LiteralType, NoneType, Overloaded, Parameters, ParamSpecType, PartialType, TupleType, Type, TypeAliasType, TypedDictType, TypeType, TypeVarId, TypeVarLikeType, TypeVarTupleType, TypeVarType, TypeVisitor, UnboundType, UninhabitedType, UnionType, UnpackType
from mypy.util import get_prefix
Primitive: _TypeAlias = Union[str, float, int, bool]
SnapshotItem: _TypeAlias = Tuple[Union[Primitive, 'SnapshotItem'], ...]
SymbolSnapshot: _TypeAlias = Tuple[object, ...]

def compare_symbol_table_snapshots(name_prefix: str, snapshot1: dict[str, SymbolSnapshot], snapshot2: dict[str, SymbolSnapshot]) -> set[str]:
    if False:
        for i in range(10):
            print('nop')
    "Return names that are different in two snapshots of a symbol table.\n\n    Only shallow (intra-module) differences are considered. References to things defined\n    outside the module are compared based on the name of the target only.\n\n    Recurse into class symbol tables (if the class is defined in the target module).\n\n    Return a set of fully-qualified names (e.g., 'mod.func' or 'mod.Class.method').\n    "
    names1 = {f'{name_prefix}.{name}' for name in snapshot1}
    names2 = {f'{name_prefix}.{name}' for name in snapshot2}
    triggers = names1 ^ names2
    for name in set(snapshot1.keys()) & set(snapshot2.keys()):
        item1 = snapshot1[name]
        item2 = snapshot2[name]
        kind1 = item1[0]
        kind2 = item2[0]
        item_name = f'{name_prefix}.{name}'
        if kind1 != kind2:
            triggers.add(item_name)
        elif kind1 == 'TypeInfo':
            if item1[:-1] != item2[:-1]:
                triggers.add(item_name)
            assert isinstance(item1[-1], dict)
            assert isinstance(item2[-1], dict)
            triggers |= compare_symbol_table_snapshots(item_name, item1[-1], item2[-1])
        elif snapshot1[name] != snapshot2[name]:
            triggers.add(item_name)
    return triggers

def snapshot_symbol_table(name_prefix: str, table: SymbolTable) -> dict[str, SymbolSnapshot]:
    if False:
        for i in range(10):
            print('nop')
    'Create a snapshot description that represents the state of a symbol table.\n\n    The snapshot has a representation based on nested tuples and dicts\n    that makes it easy and fast to find differences.\n\n    Only "shallow" state is included in the snapshot -- references to\n    things defined in other modules are represented just by the names of\n    the targets.\n    '
    result: dict[str, SymbolSnapshot] = {}
    for (name, symbol) in table.items():
        node = symbol.node
        fullname = node.fullname if node else None
        common = (fullname, symbol.kind, symbol.module_public)
        if isinstance(node, MypyFile):
            result[name] = ('Moduleref', common)
        elif isinstance(node, TypeVarExpr):
            result[name] = ('TypeVar', node.variance, [snapshot_type(value) for value in node.values], snapshot_type(node.upper_bound), snapshot_type(node.default))
        elif isinstance(node, TypeAlias):
            result[name] = ('TypeAlias', snapshot_types(node.alias_tvars), node.normalized, node.no_args, snapshot_optional_type(node.target))
        elif isinstance(node, ParamSpecExpr):
            result[name] = ('ParamSpec', node.variance, snapshot_type(node.upper_bound), snapshot_type(node.default))
        elif isinstance(node, TypeVarTupleExpr):
            result[name] = ('TypeVarTuple', node.variance, snapshot_type(node.upper_bound), snapshot_type(node.default))
        else:
            assert symbol.kind != UNBOUND_IMPORTED
            if node and get_prefix(node.fullname) != name_prefix:
                result[name] = ('CrossRef', common)
            else:
                result[name] = snapshot_definition(node, common)
    return result

def snapshot_definition(node: SymbolNode | None, common: SymbolSnapshot) -> SymbolSnapshot:
    if False:
        return 10
    'Create a snapshot description of a symbol table node.\n\n    The representation is nested tuples and dicts. Only externally\n    visible attributes are included.\n    '
    if isinstance(node, FuncBase):
        if node.type:
            signature = snapshot_type(node.type)
        else:
            signature = snapshot_untyped_signature(node)
        impl: FuncDef | None = None
        if isinstance(node, FuncDef):
            impl = node
        elif isinstance(node, OverloadedFuncDef) and node.impl:
            impl = node.impl.func if isinstance(node.impl, Decorator) else node.impl
        is_trivial_body = impl.is_trivial_body if impl else False
        dataclass_transform_spec = find_dataclass_transform_spec(node)
        return ('Func', common, node.is_property, node.is_final, node.is_class, node.is_static, signature, is_trivial_body, dataclass_transform_spec.serialize() if dataclass_transform_spec is not None else None)
    elif isinstance(node, Var):
        return ('Var', common, snapshot_optional_type(node.type), node.is_final)
    elif isinstance(node, Decorator):
        return ('Decorator', node.is_overload, snapshot_optional_type(node.var.type), snapshot_definition(node.func, common))
    elif isinstance(node, TypeInfo):
        dataclass_transform_spec = node.dataclass_transform_spec
        if dataclass_transform_spec is None:
            dataclass_transform_spec = find_dataclass_transform_spec(node)
        attrs = (node.is_abstract, node.is_enum, node.is_protocol, node.fallback_to_any, node.meta_fallback_to_any, node.is_named_tuple, node.is_newtype, snapshot_optional_type(node.metaclass_type), snapshot_optional_type(node.tuple_type), snapshot_optional_type(node.typeddict_type), [base.fullname for base in node.mro], tuple((snapshot_type(tdef) for tdef in node.defn.type_vars)), [snapshot_type(base) for base in node.bases], [snapshot_type(p) for p in node._promote], dataclass_transform_spec.serialize() if dataclass_transform_spec is not None else None)
        prefix = node.fullname
        symbol_table = snapshot_symbol_table(prefix, node.names)
        symbol_table['(abstract)'] = ('Abstract', tuple(sorted(node.abstract_attributes)))
        return ('TypeInfo', common, attrs, symbol_table)
    else:
        assert False, type(node)

def snapshot_type(typ: Type) -> SnapshotItem:
    if False:
        print('Hello World!')
    'Create a snapshot representation of a type using nested tuples.'
    return typ.accept(SnapshotTypeVisitor())

def snapshot_optional_type(typ: Type | None) -> SnapshotItem:
    if False:
        while True:
            i = 10
    if typ:
        return snapshot_type(typ)
    else:
        return ('<not set>',)

def snapshot_types(types: Sequence[Type]) -> SnapshotItem:
    if False:
        return 10
    return tuple((snapshot_type(item) for item in types))

def snapshot_simple_type(typ: Type) -> SnapshotItem:
    if False:
        i = 10
        return i + 15
    return (type(typ).__name__,)

def encode_optional_str(s: str | None) -> str:
    if False:
        i = 10
        return i + 15
    if s is None:
        return '<None>'
    else:
        return s

class SnapshotTypeVisitor(TypeVisitor[SnapshotItem]):
    """Creates a read-only, self-contained snapshot of a type object.

    Properties of a snapshot:

    - Contains (nested) tuples and other immutable primitive objects only.
    - References to AST nodes are replaced with full names of targets.
    - Has no references to mutable or non-primitive objects.
    - Two snapshots represent the same object if and only if they are
      equal.
    - Results must be sortable. It's important that tuples have
      consistent types and can't arbitrarily mix str and None values,
      for example, since they can't be compared.
    """

    def visit_unbound_type(self, typ: UnboundType) -> SnapshotItem:
        if False:
            print('Hello World!')
        return ('UnboundType', typ.name, typ.optional, typ.empty_tuple_index, snapshot_types(typ.args))

    def visit_any(self, typ: AnyType) -> SnapshotItem:
        if False:
            return 10
        return snapshot_simple_type(typ)

    def visit_none_type(self, typ: NoneType) -> SnapshotItem:
        if False:
            print('Hello World!')
        return snapshot_simple_type(typ)

    def visit_uninhabited_type(self, typ: UninhabitedType) -> SnapshotItem:
        if False:
            return 10
        return snapshot_simple_type(typ)

    def visit_erased_type(self, typ: ErasedType) -> SnapshotItem:
        if False:
            return 10
        return snapshot_simple_type(typ)

    def visit_deleted_type(self, typ: DeletedType) -> SnapshotItem:
        if False:
            i = 10
            return i + 15
        return snapshot_simple_type(typ)

    def visit_instance(self, typ: Instance) -> SnapshotItem:
        if False:
            while True:
                i = 10
        return ('Instance', encode_optional_str(typ.type.fullname), snapshot_types(typ.args), ('None',) if typ.last_known_value is None else snapshot_type(typ.last_known_value))

    def visit_type_var(self, typ: TypeVarType) -> SnapshotItem:
        if False:
            while True:
                i = 10
        return ('TypeVar', typ.name, typ.fullname, typ.id.raw_id, typ.id.meta_level, snapshot_types(typ.values), snapshot_type(typ.upper_bound), snapshot_type(typ.default), typ.variance)

    def visit_param_spec(self, typ: ParamSpecType) -> SnapshotItem:
        if False:
            return 10
        return ('ParamSpec', typ.id.raw_id, typ.id.meta_level, typ.flavor, snapshot_type(typ.upper_bound), snapshot_type(typ.default))

    def visit_type_var_tuple(self, typ: TypeVarTupleType) -> SnapshotItem:
        if False:
            return 10
        return ('TypeVarTupleType', typ.id.raw_id, typ.id.meta_level, snapshot_type(typ.upper_bound), snapshot_type(typ.default))

    def visit_unpack_type(self, typ: UnpackType) -> SnapshotItem:
        if False:
            i = 10
            return i + 15
        return ('UnpackType', snapshot_type(typ.type))

    def visit_parameters(self, typ: Parameters) -> SnapshotItem:
        if False:
            i = 10
            return i + 15
        return ('Parameters', snapshot_types(typ.arg_types), tuple((encode_optional_str(name) for name in typ.arg_names)), tuple((k.value for k in typ.arg_kinds)))

    def visit_callable_type(self, typ: CallableType) -> SnapshotItem:
        if False:
            for i in range(10):
                print('nop')
        if typ.is_generic():
            typ = self.normalize_callable_variables(typ)
        return ('CallableType', snapshot_types(typ.arg_types), snapshot_type(typ.ret_type), tuple((encode_optional_str(name) for name in typ.arg_names)), tuple((k.value for k in typ.arg_kinds)), typ.is_type_obj(), typ.is_ellipsis_args, snapshot_types(typ.variables))

    def normalize_callable_variables(self, typ: CallableType) -> CallableType:
        if False:
            return 10
        'Normalize all type variable ids to run from -1 to -len(variables).'
        tvs = []
        tvmap: dict[TypeVarId, Type] = {}
        for (i, v) in enumerate(typ.variables):
            tid = TypeVarId(-1 - i)
            if isinstance(v, TypeVarType):
                tv: TypeVarLikeType = v.copy_modified(id=tid)
            elif isinstance(v, TypeVarTupleType):
                tv = v.copy_modified(id=tid)
            else:
                assert isinstance(v, ParamSpecType)
                tv = v.copy_modified(id=tid)
            tvs.append(tv)
            tvmap[v.id] = tv
        with state.strict_optional_set(True):
            return expand_type(typ, tvmap).copy_modified(variables=tvs)

    def visit_tuple_type(self, typ: TupleType) -> SnapshotItem:
        if False:
            print('Hello World!')
        return ('TupleType', snapshot_types(typ.items))

    def visit_typeddict_type(self, typ: TypedDictType) -> SnapshotItem:
        if False:
            print('Hello World!')
        items = tuple(((key, snapshot_type(item_type)) for (key, item_type) in typ.items.items()))
        required = tuple(sorted(typ.required_keys))
        return ('TypedDictType', items, required)

    def visit_literal_type(self, typ: LiteralType) -> SnapshotItem:
        if False:
            return 10
        return ('LiteralType', snapshot_type(typ.fallback), typ.value)

    def visit_union_type(self, typ: UnionType) -> SnapshotItem:
        if False:
            i = 10
            return i + 15
        items = {snapshot_type(item) for item in typ.items}
        normalized = tuple(sorted(items))
        return ('UnionType', normalized)

    def visit_overloaded(self, typ: Overloaded) -> SnapshotItem:
        if False:
            while True:
                i = 10
        return ('Overloaded', snapshot_types(typ.items))

    def visit_partial_type(self, typ: PartialType) -> SnapshotItem:
        if False:
            i = 10
            return i + 15
        raise RuntimeError

    def visit_type_type(self, typ: TypeType) -> SnapshotItem:
        if False:
            while True:
                i = 10
        return ('TypeType', snapshot_type(typ.item))

    def visit_type_alias_type(self, typ: TypeAliasType) -> SnapshotItem:
        if False:
            while True:
                i = 10
        assert typ.alias is not None
        return ('TypeAliasType', typ.alias.fullname, snapshot_types(typ.args))

def snapshot_untyped_signature(func: OverloadedFuncDef | FuncItem) -> SymbolSnapshot:
    if False:
        for i in range(10):
            print('nop')
    "Create a snapshot of the signature of a function that has no explicit signature.\n\n    If the arguments to a function without signature change, it must be\n    considered as different. We have this special casing since we don't store\n    the implicit signature anywhere, and we'd rather not construct new\n    Callable objects in this module (the idea is to only read properties of\n    the AST here).\n    "
    if isinstance(func, FuncItem):
        return (tuple(func.arg_names), tuple(func.arg_kinds))
    else:
        result: list[SymbolSnapshot] = []
        for item in func.items:
            if isinstance(item, Decorator):
                if item.var.type:
                    result.append(snapshot_type(item.var.type))
                else:
                    result.append(('DecoratorWithoutType',))
            else:
                result.append(snapshot_untyped_signature(item))
        return tuple(result)