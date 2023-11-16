"""Merge a new version of a module AST and symbol table to older versions of those.

When the source code of a module has a change in fine-grained incremental mode,
we build a new AST from the updated source. However, other parts of the program
may have direct references to parts of the old AST (namely, those nodes exposed
in the module symbol table). The merge operation changes the identities of new
AST nodes that have a correspondence in the old AST to the old ones so that
existing cross-references in other modules will continue to point to the correct
nodes. Also internal cross-references within the new AST are replaced. AST nodes
that aren't externally visible will get new, distinct object identities. This
applies to most expression and statement nodes, for example.

We perform this merge operation so that we don't have to update all
external references (which would be slow and fragile) or always perform
translation when looking up references (which would be hard to retrofit).

The AST merge operation is performed after semantic analysis. Semantic
analysis has to deal with potentially multiple aliases to certain AST
nodes (in particular, MypyFile nodes). Type checking assumes that we
don't have multiple variants of a single AST node visible to the type
checker.

Discussion of some notable special cases:

* If a node is replaced with a different kind of node (say, a function is
  replaced with a class), we don't perform the merge. Fine-grained dependencies
  will be used to rebind all references to the node.

* If a function is replaced with another function with an identical signature,
  call sites continue to point to the same object (by identity) and don't need
  to be reprocessed. Similarly, if a class is replaced with a class that is
  sufficiently similar (MRO preserved, etc.), class references don't need any
  processing. A typical incremental update to a file only changes a few
  externally visible things in a module, and this means that often only few
  external references need any processing, even if the modified module is large.

* A no-op update of a module should not require any processing outside the
  module, since all relevant object identities are preserved.

* The AST diff operation (mypy.server.astdiff) and the top-level fine-grained
  incremental logic (mypy.server.update) handle the cases where the new AST has
  differences from the old one that may need to be propagated to elsewhere in the
  program.

See the main entry point merge_asts for more details.
"""
from __future__ import annotations
from typing import TypeVar, cast
from mypy.nodes import MDEF, AssertTypeExpr, AssignmentStmt, Block, CallExpr, CastExpr, ClassDef, EnumCallExpr, FuncBase, FuncDef, LambdaExpr, MemberExpr, MypyFile, NamedTupleExpr, NameExpr, NewTypeExpr, OverloadedFuncDef, RefExpr, Statement, SuperExpr, SymbolNode, SymbolTable, TypeAlias, TypedDictExpr, TypeInfo, Var
from mypy.traverser import TraverserVisitor
from mypy.types import AnyType, CallableArgument, CallableType, DeletedType, EllipsisType, ErasedType, Instance, LiteralType, NoneType, Overloaded, Parameters, ParamSpecType, PartialType, PlaceholderType, RawExpressionType, SyntheticTypeVisitor, TupleType, Type, TypeAliasType, TypedDictType, TypeList, TypeType, TypeVarTupleType, TypeVarType, UnboundType, UninhabitedType, UnionType, UnpackType
from mypy.typestate import type_state
from mypy.util import get_prefix, replace_object_state

def merge_asts(old: MypyFile, old_symbols: SymbolTable, new: MypyFile, new_symbols: SymbolTable) -> None:
    if False:
        i = 10
        return i + 15
    "Merge a new version of a module AST to a previous version.\n\n    The main idea is to preserve the identities of externally visible\n    nodes in the old AST (that have a corresponding node in the new AST).\n    All old node state (outside identity) will come from the new AST.\n\n    When this returns, 'old' will refer to the merged AST, but 'new_symbols'\n    will be the new symbol table. 'new' and 'old_symbols' will no longer be\n    valid.\n    "
    assert new.fullname == old.fullname
    replacement_map = replacement_map_from_symbol_table(old_symbols, new_symbols, prefix=old.fullname)
    replacement_map[new] = old
    node = replace_nodes_in_ast(new, replacement_map)
    assert node is old
    replace_nodes_in_symbol_table(new_symbols, replacement_map)

def replacement_map_from_symbol_table(old: SymbolTable, new: SymbolTable, prefix: str) -> dict[SymbolNode, SymbolNode]:
    if False:
        return 10
    "Create a new-to-old object identity map by comparing two symbol table revisions.\n\n    Both symbol tables must refer to revisions of the same module id. The symbol tables\n    are compared recursively (recursing into nested class symbol tables), but only within\n    the given module prefix. Don't recurse into other modules accessible through the symbol\n    table.\n    "
    replacements: dict[SymbolNode, SymbolNode] = {}
    for (name, node) in old.items():
        if name in new and (node.kind == MDEF or (node.node and get_prefix(node.node.fullname) == prefix)):
            new_node = new[name]
            if type(new_node.node) == type(node.node) and new_node.node and node.node and (new_node.node.fullname == node.node.fullname) and (new_node.kind == node.kind):
                replacements[new_node.node] = node.node
                if isinstance(node.node, TypeInfo) and isinstance(new_node.node, TypeInfo):
                    type_repl = replacement_map_from_symbol_table(node.node.names, new_node.node.names, prefix)
                    replacements.update(type_repl)
                    if node.node.special_alias and new_node.node.special_alias:
                        replacements[new_node.node.special_alias] = node.node.special_alias
    return replacements

def replace_nodes_in_ast(node: SymbolNode, replacements: dict[SymbolNode, SymbolNode]) -> SymbolNode:
    if False:
        for i in range(10):
            print('nop')
    "Replace all references to replacement map keys within an AST node, recursively.\n\n    Also replace the *identity* of any nodes that have replacements. Return the\n    *replaced* version of the argument node (which may have a different identity, if\n    it's included in the replacement map).\n    "
    visitor = NodeReplaceVisitor(replacements)
    node.accept(visitor)
    return replacements.get(node, node)
SN = TypeVar('SN', bound=SymbolNode)

class NodeReplaceVisitor(TraverserVisitor):
    """Transform some nodes to new identities in an AST.

    Only nodes that live in the symbol table may be
    replaced, which simplifies the implementation some. Also
    replace all references to the old identities.
    """

    def __init__(self, replacements: dict[SymbolNode, SymbolNode]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.replacements = replacements

    def visit_mypy_file(self, node: MypyFile) -> None:
        if False:
            print('Hello World!')
        node = self.fixup(node)
        node.defs = self.replace_statements(node.defs)
        super().visit_mypy_file(node)

    def visit_block(self, node: Block) -> None:
        if False:
            for i in range(10):
                print('nop')
        node.body = self.replace_statements(node.body)
        super().visit_block(node)

    def visit_func_def(self, node: FuncDef) -> None:
        if False:
            i = 10
            return i + 15
        node = self.fixup(node)
        self.process_base_func(node)
        super().visit_func_def(node)

    def visit_overloaded_func_def(self, node: OverloadedFuncDef) -> None:
        if False:
            print('Hello World!')
        self.process_base_func(node)
        super().visit_overloaded_func_def(node)

    def visit_class_def(self, node: ClassDef) -> None:
        if False:
            print('Hello World!')
        node.info = self.fixup_and_reset_typeinfo(node.info)
        node.defs.body = self.replace_statements(node.defs.body)
        info = node.info
        for tv in node.type_vars:
            if isinstance(tv, TypeVarType):
                self.process_type_var_def(tv)
        if info:
            if info.is_named_tuple:
                self.process_synthetic_type_info(info)
            else:
                self.process_type_info(info)
        super().visit_class_def(node)

    def process_base_func(self, node: FuncBase) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.fixup_type(node.type)
        node.info = self.fixup(node.info)
        if node.unanalyzed_type:
            self.fixup_type(node.unanalyzed_type)

    def process_type_var_def(self, tv: TypeVarType) -> None:
        if False:
            i = 10
            return i + 15
        for value in tv.values:
            self.fixup_type(value)
        self.fixup_type(tv.upper_bound)
        self.fixup_type(tv.default)

    def process_param_spec_def(self, tv: ParamSpecType) -> None:
        if False:
            print('Hello World!')
        self.fixup_type(tv.upper_bound)
        self.fixup_type(tv.default)

    def process_type_var_tuple_def(self, tv: TypeVarTupleType) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.fixup_type(tv.upper_bound)
        self.fixup_type(tv.default)

    def visit_assignment_stmt(self, node: AssignmentStmt) -> None:
        if False:
            print('Hello World!')
        self.fixup_type(node.type)
        super().visit_assignment_stmt(node)

    def visit_name_expr(self, node: NameExpr) -> None:
        if False:
            return 10
        self.visit_ref_expr(node)

    def visit_member_expr(self, node: MemberExpr) -> None:
        if False:
            i = 10
            return i + 15
        if node.def_var:
            node.def_var = self.fixup(node.def_var)
        self.visit_ref_expr(node)
        super().visit_member_expr(node)

    def visit_ref_expr(self, node: RefExpr) -> None:
        if False:
            i = 10
            return i + 15
        if node.node is not None:
            node.node = self.fixup(node.node)
            if isinstance(node.node, Var):
                node.node.accept(self)

    def visit_namedtuple_expr(self, node: NamedTupleExpr) -> None:
        if False:
            print('Hello World!')
        super().visit_namedtuple_expr(node)
        node.info = self.fixup_and_reset_typeinfo(node.info)
        self.process_synthetic_type_info(node.info)

    def visit_cast_expr(self, node: CastExpr) -> None:
        if False:
            print('Hello World!')
        super().visit_cast_expr(node)
        self.fixup_type(node.type)

    def visit_assert_type_expr(self, node: AssertTypeExpr) -> None:
        if False:
            i = 10
            return i + 15
        super().visit_assert_type_expr(node)
        self.fixup_type(node.type)

    def visit_super_expr(self, node: SuperExpr) -> None:
        if False:
            i = 10
            return i + 15
        super().visit_super_expr(node)
        if node.info is not None:
            node.info = self.fixup(node.info)

    def visit_call_expr(self, node: CallExpr) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().visit_call_expr(node)
        if isinstance(node.analyzed, SymbolNode):
            node.analyzed = self.fixup(node.analyzed)

    def visit_newtype_expr(self, node: NewTypeExpr) -> None:
        if False:
            return 10
        if node.info:
            node.info = self.fixup_and_reset_typeinfo(node.info)
            self.process_synthetic_type_info(node.info)
        self.fixup_type(node.old_type)
        super().visit_newtype_expr(node)

    def visit_lambda_expr(self, node: LambdaExpr) -> None:
        if False:
            while True:
                i = 10
        node.info = self.fixup(node.info)
        super().visit_lambda_expr(node)

    def visit_typeddict_expr(self, node: TypedDictExpr) -> None:
        if False:
            return 10
        super().visit_typeddict_expr(node)
        node.info = self.fixup_and_reset_typeinfo(node.info)
        self.process_synthetic_type_info(node.info)

    def visit_enum_call_expr(self, node: EnumCallExpr) -> None:
        if False:
            return 10
        node.info = self.fixup_and_reset_typeinfo(node.info)
        self.process_synthetic_type_info(node.info)
        super().visit_enum_call_expr(node)

    def visit_var(self, node: Var) -> None:
        if False:
            i = 10
            return i + 15
        node.info = self.fixup(node.info)
        self.fixup_type(node.type)
        super().visit_var(node)

    def visit_type_alias(self, node: TypeAlias) -> None:
        if False:
            while True:
                i = 10
        self.fixup_type(node.target)
        for v in node.alias_tvars:
            self.fixup_type(v)
        super().visit_type_alias(node)

    def fixup(self, node: SN) -> SN:
        if False:
            i = 10
            return i + 15
        if node in self.replacements:
            new = self.replacements[node]
            skip_slots: tuple[str, ...] = ()
            if isinstance(node, TypeInfo) and isinstance(new, TypeInfo):
                skip_slots = ('special_alias',)
                replace_object_state(new.special_alias, node.special_alias)
            replace_object_state(new, node, skip_slots=skip_slots)
            return cast(SN, new)
        return node

    def fixup_and_reset_typeinfo(self, node: TypeInfo) -> TypeInfo:
        if False:
            return 10
        'Fix-up type info and reset subtype caches.\n\n        This needs to be called at least once per each merged TypeInfo, as otherwise we\n        may leak stale caches.\n        '
        if node in self.replacements:
            new = self.replacements[node]
            assert isinstance(new, TypeInfo)
            type_state.reset_all_subtype_caches_for(new)
        return self.fixup(node)

    def fixup_type(self, typ: Type | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if typ is not None:
            typ.accept(TypeReplaceVisitor(self.replacements))

    def process_type_info(self, info: TypeInfo | None) -> None:
        if False:
            i = 10
            return i + 15
        if info is None:
            return
        self.fixup_type(info.declared_metaclass)
        self.fixup_type(info.metaclass_type)
        for target in info._promote:
            self.fixup_type(target)
        self.fixup_type(info.tuple_type)
        self.fixup_type(info.typeddict_type)
        if info.special_alias:
            self.fixup_type(info.special_alias.target)
        info.defn.info = self.fixup(info)
        replace_nodes_in_symbol_table(info.names, self.replacements)
        for (i, item) in enumerate(info.mro):
            info.mro[i] = self.fixup(info.mro[i])
        for (i, base) in enumerate(info.bases):
            self.fixup_type(info.bases[i])

    def process_synthetic_type_info(self, info: TypeInfo) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.process_type_info(info)
        for (name, node) in info.names.items():
            if node.node:
                node.node.accept(self)

    def replace_statements(self, nodes: list[Statement]) -> list[Statement]:
        if False:
            for i in range(10):
                print('nop')
        result = []
        for node in nodes:
            if isinstance(node, SymbolNode):
                node = self.fixup(node)
            result.append(node)
        return result

class TypeReplaceVisitor(SyntheticTypeVisitor[None]):
    """Similar to NodeReplaceVisitor, but for type objects.

    Note: this visitor may sometimes visit unanalyzed types
    such as 'UnboundType' and 'RawExpressionType' For example, see
    NodeReplaceVisitor.process_base_func.
    """

    def __init__(self, replacements: dict[SymbolNode, SymbolNode]) -> None:
        if False:
            i = 10
            return i + 15
        self.replacements = replacements

    def visit_instance(self, typ: Instance) -> None:
        if False:
            print('Hello World!')
        typ.type = self.fixup(typ.type)
        for arg in typ.args:
            arg.accept(self)
        if typ.last_known_value:
            typ.last_known_value.accept(self)

    def visit_type_alias_type(self, typ: TypeAliasType) -> None:
        if False:
            return 10
        assert typ.alias is not None
        typ.alias = self.fixup(typ.alias)
        for arg in typ.args:
            arg.accept(self)

    def visit_any(self, typ: AnyType) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_none_type(self, typ: NoneType) -> None:
        if False:
            while True:
                i = 10
        pass

    def visit_callable_type(self, typ: CallableType) -> None:
        if False:
            print('Hello World!')
        for arg in typ.arg_types:
            arg.accept(self)
        typ.ret_type.accept(self)
        if typ.definition:
            typ.definition = self.replacements.get(typ.definition, typ.definition)
        if typ.fallback is not None:
            typ.fallback.accept(self)
        for tv in typ.variables:
            if isinstance(tv, TypeVarType):
                tv.upper_bound.accept(self)
                for value in tv.values:
                    value.accept(self)

    def visit_overloaded(self, t: Overloaded) -> None:
        if False:
            print('Hello World!')
        for item in t.items:
            item.accept(self)
        if t.fallback is not None:
            t.fallback.accept(self)

    def visit_erased_type(self, t: ErasedType) -> None:
        if False:
            while True:
                i = 10
        raise RuntimeError('Cannot handle erased type')

    def visit_deleted_type(self, typ: DeletedType) -> None:
        if False:
            print('Hello World!')
        pass

    def visit_partial_type(self, typ: PartialType) -> None:
        if False:
            return 10
        raise RuntimeError('Cannot handle partial type')

    def visit_tuple_type(self, typ: TupleType) -> None:
        if False:
            while True:
                i = 10
        for item in typ.items:
            item.accept(self)
        if typ.partial_fallback is not None:
            typ.partial_fallback.accept(self)

    def visit_type_type(self, typ: TypeType) -> None:
        if False:
            return 10
        typ.item.accept(self)

    def visit_type_var(self, typ: TypeVarType) -> None:
        if False:
            return 10
        typ.upper_bound.accept(self)
        typ.default.accept(self)
        for value in typ.values:
            value.accept(self)

    def visit_param_spec(self, typ: ParamSpecType) -> None:
        if False:
            i = 10
            return i + 15
        typ.upper_bound.accept(self)
        typ.default.accept(self)

    def visit_type_var_tuple(self, typ: TypeVarTupleType) -> None:
        if False:
            return 10
        typ.upper_bound.accept(self)
        typ.default.accept(self)

    def visit_unpack_type(self, typ: UnpackType) -> None:
        if False:
            i = 10
            return i + 15
        typ.type.accept(self)

    def visit_parameters(self, typ: Parameters) -> None:
        if False:
            print('Hello World!')
        for arg in typ.arg_types:
            arg.accept(self)

    def visit_typeddict_type(self, typ: TypedDictType) -> None:
        if False:
            print('Hello World!')
        for value_type in typ.items.values():
            value_type.accept(self)
        typ.fallback.accept(self)

    def visit_raw_expression_type(self, t: RawExpressionType) -> None:
        if False:
            while True:
                i = 10
        pass

    def visit_literal_type(self, typ: LiteralType) -> None:
        if False:
            return 10
        typ.fallback.accept(self)

    def visit_unbound_type(self, typ: UnboundType) -> None:
        if False:
            i = 10
            return i + 15
        for arg in typ.args:
            arg.accept(self)

    def visit_type_list(self, typ: TypeList) -> None:
        if False:
            return 10
        for item in typ.items:
            item.accept(self)

    def visit_callable_argument(self, typ: CallableArgument) -> None:
        if False:
            while True:
                i = 10
        typ.typ.accept(self)

    def visit_ellipsis_type(self, typ: EllipsisType) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def visit_uninhabited_type(self, typ: UninhabitedType) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def visit_union_type(self, typ: UnionType) -> None:
        if False:
            print('Hello World!')
        for item in typ.items:
            item.accept(self)

    def visit_placeholder_type(self, t: PlaceholderType) -> None:
        if False:
            print('Hello World!')
        for item in t.args:
            item.accept(self)

    def fixup(self, node: SN) -> SN:
        if False:
            i = 10
            return i + 15
        if node in self.replacements:
            new = self.replacements[node]
            return cast(SN, new)
        return node

def replace_nodes_in_symbol_table(symbols: SymbolTable, replacements: dict[SymbolNode, SymbolNode]) -> None:
    if False:
        print('Hello World!')
    for (name, node) in symbols.items():
        if node.node:
            if node.node in replacements:
                new = replacements[node.node]
                old = node.node
                replace_object_state(new, old, skip_slots=('special_alias',))
                node.node = new
            if isinstance(node.node, (Var, TypeAlias)):
                node.node.accept(NodeReplaceVisitor(replacements))