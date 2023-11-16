from __future__ import annotations
import ast
from ast import AnnAssign, Assign, AST, AsyncFunctionDef, Attribute, ClassDef, Delete, ExceptHandler, FunctionDef, Global, Import, ImportFrom, Module, Name
from symtable import SymbolTable
from typing import final, List, MutableMapping, Optional, Set
from ..consts import CO_FUTURE_ANNOTATIONS
from ..pycodegen import find_futures
from .common import get_symbol_map, imported_name, ScopeStack, StrictModuleError, SymbolMap, SymbolScope
from .rewriter.rewriter import SymbolVisitor
CLASS_ATTR_CONFLICT_EXCEPTION = 'ClassAttributesConflictException'

class TransformerScope:

    def visit_Assign(self, node: Assign) -> None:
        if False:
            return 10
        pass

    def visit_AnnAssign(self, node: AnnAssign) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def loaded(self, name: str) -> None:
        if False:
            while True:
                i = 10
        pass

    def stored(self, name: str) -> None:
        if False:
            i = 10
            return i + 15
        pass

@final
class ClassScope(TransformerScope):

    def __init__(self) -> None:
        if False:
            return 10
        self.instance_fields: Set[str] = set()
        self.class_fields: Set[str] = set()

    def visit_AnnAssign(self, node: AnnAssign) -> None:
        if False:
            for i in range(10):
                print('nop')
        target = node.target
        if isinstance(target, Name):
            if node.value is None:
                self.instance_fields.add(target.id)
            else:
                self.class_fields.add(target.id)

    def stored(self, name: str) -> None:
        if False:
            print('Hello World!')
        self.class_fields.add(name)

@final
class FunctionScope(TransformerScope):

    def __init__(self, node: FunctionDef, parent: TransformerScope) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.node = node
        self.parent = parent

    def visit_AnnAssign(self, node: AnnAssign) -> None:
        if False:
            while True:
                i = 10
        parent = self.parent
        target = node.target
        if isinstance(self.node, FunctionDef) and self.node.name == '__init__' and self.node.args.args and isinstance(parent, ClassScope) and isinstance(target, Attribute):
            self.add_attr_name(target, parent)

    def add_attr_name(self, target: Attribute, parent: ClassScope) -> None:
        if False:
            return 10
        'records self.name = ... when salf matches the 1st parameter'
        value = target.value
        node = self.node
        if isinstance(node, FunctionDef) and isinstance(value, Name) and (value.id == node.args.args[0].arg):
            parent.instance_fields.add(target.attr)

    def visit_Assign(self, node: Assign) -> None:
        if False:
            print('Hello World!')
        parent = self.parent
        if isinstance(self.node, FunctionDef) and self.node.name == '__init__' and isinstance(parent, ClassScope) and self.node.args.args:
            for target in node.targets:
                if not isinstance(target, Attribute):
                    continue
                self.add_attr_name(target, parent)

@final
class ClassConflictChecker(SymbolVisitor[object, TransformerScope]):

    def __init__(self, symbols: SymbolTable, symbol_map: SymbolMap, filename: str, flags: int) -> None:
        if False:
            return 10
        super().__init__(ScopeStack(self.make_scope(symbols, None), symbol_map=symbol_map, scope_factory=self.make_scope))
        self.filename = filename
        self.flags = flags

    @property
    def skip_annotations(self) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(self.flags & CO_FUTURE_ANNOTATIONS)

    def error(self, names: List[str], lineno: int, col: int, filename: str) -> None:
        if False:
            return 10
        MSG: str = 'Class member conflicts with instance member: {names}'
        raise StrictModuleError(MSG.format(names=names), filename, lineno, col)

    def make_scope(self, symtable: SymbolTable, node: Optional[AST], vars: Optional[MutableMapping[str, object]]=None) -> SymbolScope[object, TransformerScope]:
        if False:
            print('Hello World!')
        if isinstance(node, FunctionDef):
            data = FunctionScope(node, self.scopes.scopes[-1].scope_data)
        elif isinstance(node, ClassDef):
            data = ClassScope()
        else:
            data = TransformerScope()
        return SymbolScope(symtable, data)

    def visit_Name(self, node: Name) -> None:
        if False:
            i = 10
            return i + 15
        scope = self.scope_for(node.id).scope_data
        if isinstance(node.ctx, ast.Load):
            scope.loaded(node.id)
        else:
            scope.stored(node.id)

    def visit_ExceptHandler(self, node: ExceptHandler) -> None:
        if False:
            print('Hello World!')
        self.generic_visit(node)
        name = node.name
        if name is not None:
            self.scope_for(name).scope_data.stored(name)

    def visit_Delete(self, node: Delete) -> None:
        if False:
            i = 10
            return i + 15
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.scope_for(target.id).scope_data.stored(target.id)

    def visit_Global(self, node: Global) -> None:
        if False:
            while True:
                i = 10
        if self.scopes.in_class_scope:
            for name in node.names:
                if name == '__annotations__':
                    self.error(['__annotations__'], node.lineno, node.col_offset, self.filename)

    def visit_ClassDef(self, node: ClassDef) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.visit_Class_Outer(node)
        class_scope = self.visit_Class_Inner(node).scope_data
        assert isinstance(class_scope, ClassScope)
        overlap = class_scope.instance_fields.intersection(class_scope.class_fields)
        if overlap:
            self.error(list(overlap), node.lineno, node.col_offset, self.filename)
        self.scope_for(node.name).scope_data.stored(node.name)

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.visit_Func_Outer(node)
        func_scope = self.visit_Func_Inner(node)
        self.scopes.current[node.name] = func_scope.scope_data
        self.scope_for(node.name).scope_data.stored(node.name)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        if False:
            print('Hello World!')
        self.visit_Func_Outer(node)
        self.visit_Func_Inner(node)
        self.scope_for(node.name).scope_data.stored(node.name)

    def visit_Import(self, node: Import) -> None:
        if False:
            print('Hello World!')
        for name in node.names:
            self.scope_for(imported_name(name)).scope_data.stored(imported_name(name))
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ImportFrom) -> None:
        if False:
            i = 10
            return i + 15
        if node.level == 0 and node.module is not None:
            for name in node.names:
                self.scope_for(name.asname or name.name).scope_data.stored(name.asname or name.name)

    def visit_Assign(self, node: Assign) -> None:
        if False:
            while True:
                i = 10
        self.scopes.scopes[-1].scope_data.visit_Assign(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: AnnAssign) -> None:
        if False:
            return 10
        self.scopes.scopes[-1].scope_data.visit_AnnAssign(node)
        value = node.value
        if value is not None:
            self.visit(node.target)
            self.visit(value)
            if not self.skip_annotations:
                self.visit(node.annotation)

    def visit_arg(self, node: ast.arg) -> None:
        if False:
            while True:
                i = 10
        if not self.skip_annotations:
            self.generic_visit(node)

def check_class_conflict(node: Module, filename: str, symbols: SymbolTable) -> None:
    if False:
        i = 10
        return i + 15
    symbol_map = get_symbol_map(node, symbols)
    flags = find_futures(0, node)
    visitor = ClassConflictChecker(symbols, symbol_map, filename=filename, flags=flags)
    visitor.visit(node)