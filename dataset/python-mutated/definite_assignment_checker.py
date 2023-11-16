from __future__ import annotations
import ast
from ast import AnnAssign, Assign, AST, AsyncFunctionDef, AugAssign, ClassDef, comprehension, copy_location, DictComp, For, FunctionDef, GeneratorExp, If, Import, ImportFrom, Lambda, ListComp, Module, Name, NodeVisitor, Raise, SetComp, stmt, Try, While, With
from typing import Iterable, List, Optional, Set, Union
from ..consts import SC_CELL, SC_LOCAL
from ..symbols import Scope

class DefiniteAssignmentVisitor(NodeVisitor):

    def __init__(self, scope: Scope) -> None:
        if False:
            print('Hello World!')
        self.scope = scope
        self.assigned: Set[str] = set()
        self.unassigned: Set[Name] = set()

    def analyzeFunction(self, node: FunctionDef | AsyncFunctionDef) -> None:
        if False:
            print('Hello World!')
        for arg in node.args.args:
            self.assigned.add(arg.arg)
        for arg in node.args.kwonlyargs:
            self.assigned.add(arg.arg)
        for arg in node.args.posonlyargs:
            self.assigned.add(arg.arg)
        vararg = node.args.vararg
        if vararg:
            self.assigned.add(vararg.arg)
        kwarg = node.args.kwarg
        if kwarg:
            self.assigned.add(kwarg.arg)
        for stmt in node.body:
            self.visit(stmt)

    def set_assigned(self, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.is_local(name):
            self.assigned.add(name)

    def is_local(self, name: str) -> bool:
        if False:
            return 10
        scope = self.scope.check_name(name)
        return scope == SC_LOCAL or scope == SC_CELL

    def visit_Name(self, node: Name) -> None:
        if False:
            return 10
        if not self.is_local(node.id):
            return
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.assigned:
                self.unassigned.add(node)
        elif isinstance(node.ctx, ast.Del):
            if node.id not in self.assigned:
                self.unassigned.add(node)
            else:
                self.assigned.remove(node.id)
        else:
            self.assigned.add(node.id)

    def visit_Assign(self, node: Assign) -> None:
        if False:
            return 10
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AugAssign(self, node: AugAssign) -> None:
        if False:
            return 10
        target = node.target
        if isinstance(target, ast.Name):
            if target.id not in self.assigned:
                self.unassigned.add(target)
            self.generic_visit(node.value)
            return
        self.generic_visit(node)

    def visit_Try(self, node: Try) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not node.handlers:
            entry = set(self.assigned)
            self.walk_stmts(node.body)
            post_try = set(self.assigned)
            self.assigned = entry.intersection(post_try)
            self.walk_stmts(node.finalbody)
            for value in entry:
                if value not in self.assigned and value in post_try:
                    post_try.remove(value)
            post_try.update(self.assigned)
            self.assigned = post_try
            return
        entry = set(self.assigned)
        self.walk_stmts(node.body)
        elseentry = set(self.assigned)
        entry.intersection_update(self.assigned)
        finalentry = set(entry)
        for handler in node.handlers:
            self.assigned = set(entry)
            handler_name = handler.name
            if handler_name is not None:
                self.set_assigned(handler_name)
            self.walk_stmts(handler.body)
            finalentry.intersection_update(self.assigned)
        if node.orelse:
            self.assigned = elseentry
            self.walk_stmts(node.orelse)
            finalentry.intersection_update(self.assigned)
        self.assigned = finalentry
        if node.finalbody:
            self.walk_stmts(node.finalbody)

    def visit_ClassDef(self, node: ClassDef) -> None:
        if False:
            return 10
        for base in node.bases:
            self.visit(base)
        for kw in node.keywords:
            self.visit(kw)
        for dec in node.decorator_list:
            self.visit(dec)
        self.set_assigned(node.name)

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._visit_func_like(node)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        if False:
            i = 10
            return i + 15
        self._visit_func_like(node)

    def visit_Lambda(self, node: Lambda) -> None:
        if False:
            return 10
        self.visit(node.args)

    def _visit_func_like(self, node: Union[FunctionDef, AsyncFunctionDef]) -> None:
        if False:
            print('Hello World!')
        self.visit(node.args)
        returns = node.returns
        if returns:
            self.visit(returns)
        for dec in node.decorator_list:
            self.visit(dec)
        self.set_assigned(node.name)

    def visit_With(self, node: With) -> None:
        if False:
            return 10
        for item in node.items:
            self.visit(item)
        entry = set(self.assigned)
        self.walk_stmts(node.body)
        entry.intersection_update(self.assigned)
        self.assigned = entry

    def visit_Import(self, node: Import) -> None:
        if False:
            for i in range(10):
                print('nop')
        for name in node.names:
            self.set_assigned(name.asname or name.name.partition('.')[0])

    def visit_ImportFrom(self, node: ImportFrom) -> None:
        if False:
            print('Hello World!')
        for name in node.names:
            self.set_assigned(name.asname or name.name)

    def visit_AnnAssign(self, node: AnnAssign) -> None:
        if False:
            i = 10
            return i + 15
        if node.value:
            self.generic_visit(node)

    def visit_For(self, node: For) -> None:
        if False:
            i = 10
            return i + 15
        self.visit(node.iter)
        entry = set(self.assigned)
        self.visit(node.target)
        self.walk_stmts(node.body)
        entry.intersection_update(self.assigned)
        if node.orelse:
            self.assigned = set(entry)
            self.walk_stmts(node.orelse)
            entry.intersection_update(self.assigned)
        self.assigned = entry

    def visit_If(self, node: If) -> None:
        if False:
            while True:
                i = 10
        test = node.test
        self.visit(node.test)
        entry = set(self.assigned)
        self.walk_stmts(node.body)
        post_if = self.assigned
        if node.orelse:
            self.assigned = set(entry)
            self.walk_stmts(node.orelse)
            self.assigned = self.assigned.intersection(post_if)
        else:
            entry.intersection_update(post_if)
            self.assigned = entry

    def visit_While(self, node: While) -> None:
        if False:
            while True:
                i = 10
        self.visit(node.test)
        entry = set(self.assigned)
        self.walk_stmts(node.body)
        entry.intersection_update(self.assigned)
        if node.orelse:
            self.assigned = set(entry)
            self.walk_stmts(node.orelse)
            entry.intersection_update(self.assigned)
        self.assigned = entry

    def walk_stmts(self, nodes: Iterable[stmt]) -> None:
        if False:
            while True:
                i = 10
        for node in nodes:
            self.visit(node)
            if isinstance(node, Raise):
                return