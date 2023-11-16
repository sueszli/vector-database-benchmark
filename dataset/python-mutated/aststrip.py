"""Strip/reset AST in-place to match state after semantic analyzer pre-analysis.

Fine-grained incremental mode reruns semantic analysis main pass
and type checking for *existing* AST nodes (targets) when changes are
propagated using fine-grained dependencies.  AST nodes attributes are
sometimes changed during semantic analysis main pass, and running
semantic analysis again on those nodes would produce incorrect
results, since this pass isn't idempotent. This pass resets AST
nodes to reflect the state after semantic pre-analysis, so that we
can rerun semantic analysis.
(The above is in contrast to behavior with modules that have source code
changes, for which we re-parse the entire module and reconstruct a fresh
AST. No stripping is required in this case. Both modes of operation should
have the same outcome.)
Notes:
* This is currently pretty fragile, as we must carefully undo whatever
  changes can be made in semantic analysis main pass, including changes
  to symbol tables.
* We reuse existing AST nodes because it makes it relatively straightforward
  to reprocess only a single target within a module efficiently. If there
  was a way to parse a single target within a file, in time proportional to
  the size of the target, we'd rather create fresh AST nodes than strip them.
  (This is possible only in Python 3.8+)
* Currently we don't actually reset all changes, but only those known to affect
  non-idempotent semantic analysis behavior.
  TODO: It would be more principled and less fragile to reset everything
      changed in semantic analysis main pass and later.
* Reprocessing may recreate AST nodes (such as Var nodes, and TypeInfo nodes
  created with assignment statements) that will get different identities from
  the original AST. Thus running an AST merge is necessary after stripping,
  even though some identities are preserved.
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from typing import Dict, Iterator, Tuple
from typing_extensions import TypeAlias as _TypeAlias
from mypy.nodes import CLASSDEF_NO_INFO, AssignmentStmt, Block, CallExpr, ClassDef, Decorator, ForStmt, FuncDef, ImportAll, ImportFrom, IndexExpr, ListExpr, MemberExpr, MypyFile, NameExpr, Node, OpExpr, OverloadedFuncDef, RefExpr, StarExpr, SuperExpr, SymbolTableNode, TupleExpr, TypeInfo, Var
from mypy.traverser import TraverserVisitor
from mypy.types import CallableType
from mypy.typestate import type_state
SavedAttributes: _TypeAlias = Dict[Tuple[ClassDef, str], SymbolTableNode]

def strip_target(node: MypyFile | FuncDef | OverloadedFuncDef, saved_attrs: SavedAttributes) -> None:
    if False:
        while True:
            i = 10
    'Reset a fine-grained incremental target to state before semantic analysis.\n\n    All TypeInfos are killed. Therefore we need to preserve the variables\n    defined as attributes on self. This is done by patches (callbacks)\n    returned from this function that re-add these variables when called.\n\n    Args:\n        node: node to strip\n        saved_attrs: collect attributes here that may need to be re-added to\n            classes afterwards if stripping a class body (this dict is mutated)\n    '
    visitor = NodeStripVisitor(saved_attrs)
    if isinstance(node, MypyFile):
        visitor.strip_file_top_level(node)
    else:
        node.accept(visitor)

class NodeStripVisitor(TraverserVisitor):

    def __init__(self, saved_class_attrs: SavedAttributes) -> None:
        if False:
            while True:
                i = 10
        self.type: TypeInfo | None = None
        self.is_class_body = False
        self.recurse_into_functions = True
        self.saved_class_attrs = saved_class_attrs

    def strip_file_top_level(self, file_node: MypyFile) -> None:
        if False:
            i = 10
            return i + 15
        "Strip a module top-level (don't recursive into functions)."
        self.recurse_into_functions = False
        file_node.plugin_deps.clear()
        file_node.accept(self)
        for name in file_node.names.copy():
            if '@' not in name:
                del file_node.names[name]

    def visit_block(self, b: Block) -> None:
        if False:
            return 10
        if b.is_unreachable:
            return
        super().visit_block(b)

    def visit_class_def(self, node: ClassDef) -> None:
        if False:
            print('Hello World!')
        "Strip class body and type info, but don't strip methods."
        if not self.recurse_into_functions:
            self.save_implicit_attributes(node)
        to_delete = {v.node for v in node.info.names.values() if v.plugin_generated}
        node.type_vars = []
        node.base_type_exprs.extend(node.removed_base_type_exprs)
        node.removed_base_type_exprs = []
        node.defs.body = [s for s in node.defs.body if s not in to_delete]
        with self.enter_class(node.info):
            super().visit_class_def(node)
        node.defs.body.extend(node.removed_statements)
        node.removed_statements = []
        type_state.reset_subtype_caches_for(node.info)
        node.info = CLASSDEF_NO_INFO
        node.analyzed = None

    def save_implicit_attributes(self, node: ClassDef) -> None:
        if False:
            print('Hello World!')
        'Produce callbacks that re-add attributes defined on self.'
        for (name, sym) in node.info.names.items():
            if isinstance(sym.node, Var) and sym.implicit:
                self.saved_class_attrs[node, name] = sym

    def visit_func_def(self, node: FuncDef) -> None:
        if False:
            while True:
                i = 10
        if not self.recurse_into_functions:
            return
        node.expanded = []
        node.type = node.unanalyzed_type
        if node.type:
            assert isinstance(node.type, CallableType)
            node.type.variables = []
        with self.enter_method(node.info) if node.info else nullcontext():
            super().visit_func_def(node)

    def visit_decorator(self, node: Decorator) -> None:
        if False:
            return 10
        node.var.type = None
        for expr in node.decorators:
            expr.accept(self)
        if self.recurse_into_functions:
            node.func.accept(self)
        else:
            node.var.is_final = False
            node.func.is_final = False

    def visit_overloaded_func_def(self, node: OverloadedFuncDef) -> None:
        if False:
            print('Hello World!')
        if not self.recurse_into_functions:
            return
        node.items = node.unanalyzed_items.copy()
        node.impl = None
        node.is_final = False
        super().visit_overloaded_func_def(node)

    def visit_assignment_stmt(self, node: AssignmentStmt) -> None:
        if False:
            while True:
                i = 10
        node.type = node.unanalyzed_type
        node.is_final_def = False
        node.is_alias_def = False
        if self.type and (not self.is_class_body):
            for lvalue in node.lvalues:
                self.process_lvalue_in_method(lvalue)
        super().visit_assignment_stmt(node)

    def visit_import_from(self, node: ImportFrom) -> None:
        if False:
            for i in range(10):
                print('nop')
        node.assignments = []

    def visit_import_all(self, node: ImportAll) -> None:
        if False:
            print('Hello World!')
        node.assignments = []

    def visit_for_stmt(self, node: ForStmt) -> None:
        if False:
            i = 10
            return i + 15
        node.index_type = node.unanalyzed_index_type
        node.inferred_item_type = None
        node.inferred_iterator_type = None
        super().visit_for_stmt(node)

    def visit_name_expr(self, node: NameExpr) -> None:
        if False:
            while True:
                i = 10
        self.strip_ref_expr(node)

    def visit_member_expr(self, node: MemberExpr) -> None:
        if False:
            i = 10
            return i + 15
        self.strip_ref_expr(node)
        super().visit_member_expr(node)

    def visit_index_expr(self, node: IndexExpr) -> None:
        if False:
            for i in range(10):
                print('nop')
        node.analyzed = None
        super().visit_index_expr(node)

    def visit_op_expr(self, node: OpExpr) -> None:
        if False:
            i = 10
            return i + 15
        node.analyzed = None
        super().visit_op_expr(node)

    def strip_ref_expr(self, node: RefExpr) -> None:
        if False:
            for i in range(10):
                print('nop')
        node.kind = None
        node.node = None
        node.fullname = ''
        node.is_new_def = False
        node.is_inferred_def = False

    def visit_call_expr(self, node: CallExpr) -> None:
        if False:
            for i in range(10):
                print('nop')
        node.analyzed = None
        super().visit_call_expr(node)

    def visit_super_expr(self, node: SuperExpr) -> None:
        if False:
            while True:
                i = 10
        node.info = None
        super().visit_super_expr(node)

    def process_lvalue_in_method(self, lvalue: Node) -> None:
        if False:
            return 10
        if isinstance(lvalue, MemberExpr):
            if lvalue.is_new_def:
                assert self.type is not None
                if lvalue.name in self.type.names:
                    del self.type.names[lvalue.name]
                key = (self.type.defn, lvalue.name)
                if key in self.saved_class_attrs:
                    del self.saved_class_attrs[key]
        elif isinstance(lvalue, (TupleExpr, ListExpr)):
            for item in lvalue.items:
                self.process_lvalue_in_method(item)
        elif isinstance(lvalue, StarExpr):
            self.process_lvalue_in_method(lvalue.expr)

    @contextmanager
    def enter_class(self, info: TypeInfo) -> Iterator[None]:
        if False:
            i = 10
            return i + 15
        old_type = self.type
        old_is_class_body = self.is_class_body
        self.type = info
        self.is_class_body = True
        yield
        self.type = old_type
        self.is_class_body = old_is_class_body

    @contextmanager
    def enter_method(self, info: TypeInfo) -> Iterator[None]:
        if False:
            for i in range(10):
                print('nop')
        old_type = self.type
        old_is_class_body = self.is_class_body
        self.type = info
        self.is_class_body = False
        yield
        self.type = old_type
        self.is_class_body = old_is_class_body