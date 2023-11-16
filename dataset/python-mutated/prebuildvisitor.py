from __future__ import annotations
from mypy.nodes import Block, Decorator, Expression, FuncDef, FuncItem, Import, LambdaExpr, MemberExpr, MypyFile, NameExpr, Node, SymbolNode, Var
from mypy.traverser import ExtendedTraverserVisitor
from mypyc.errors import Errors

class PreBuildVisitor(ExtendedTraverserVisitor):
    """Mypy file AST visitor run before building the IR.

    This collects various things, including:

    * Determine relationships between nested functions and functions that
      contain nested functions
    * Find non-local variables (free variables)
    * Find property setters
    * Find decorators of functions
    * Find module import groups

    The main IR build pass uses this information.
    """

    def __init__(self, errors: Errors, current_file: MypyFile, decorators_to_remove: dict[FuncDef, list[int]]) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.free_variables: dict[FuncItem, set[SymbolNode]] = {}
        self.symbols_to_funcs: dict[SymbolNode, FuncItem] = {}
        self.funcs: list[FuncItem] = []
        self.prop_setters: set[FuncDef] = set()
        self.encapsulating_funcs: dict[FuncItem, list[FuncItem]] = {}
        self.nested_funcs: dict[FuncItem, FuncItem] = {}
        self.funcs_to_decorators: dict[FuncDef, list[Expression]] = {}
        self.decorators_to_remove: dict[FuncDef, list[int]] = decorators_to_remove
        self.module_import_groups: dict[Import, list[Import]] = {}
        self._current_import_group: Import | None = None
        self.errors: Errors = errors
        self.current_file: MypyFile = current_file

    def visit(self, o: Node) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(o, Import):
            self._current_import_group = None
        return True

    def visit_block(self, block: Block) -> None:
        if False:
            print('Hello World!')
        self._current_import_group = None
        super().visit_block(block)
        self._current_import_group = None

    def visit_decorator(self, dec: Decorator) -> None:
        if False:
            while True:
                i = 10
        if dec.decorators:
            if isinstance(dec.decorators[0], MemberExpr) and dec.decorators[0].name == 'setter':
                self.prop_setters.add(dec.func)
            else:
                decorators_to_store = dec.decorators.copy()
                if dec.func in self.decorators_to_remove:
                    to_remove = self.decorators_to_remove[dec.func]
                    for i in reversed(to_remove):
                        del decorators_to_store[i]
                    if not decorators_to_store:
                        return
                self.funcs_to_decorators[dec.func] = decorators_to_store
        super().visit_decorator(dec)

    def visit_func_def(self, fdef: FuncItem) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.visit_func(fdef)

    def visit_lambda_expr(self, expr: LambdaExpr) -> None:
        if False:
            print('Hello World!')
        self.visit_func(expr)

    def visit_func(self, func: FuncItem) -> None:
        if False:
            return 10
        if self.funcs:
            self.encapsulating_funcs.setdefault(self.funcs[-1], []).append(func)
            self.nested_funcs[func] = self.funcs[-1]
        self.funcs.append(func)
        super().visit_func(func)
        self.funcs.pop()

    def visit_import(self, imp: Import) -> None:
        if False:
            print('Hello World!')
        if self._current_import_group is not None:
            self.module_import_groups[self._current_import_group].append(imp)
        else:
            self.module_import_groups[imp] = [imp]
            self._current_import_group = imp
        super().visit_import(imp)

    def visit_name_expr(self, expr: NameExpr) -> None:
        if False:
            i = 10
            return i + 15
        if isinstance(expr.node, (Var, FuncDef)):
            self.visit_symbol_node(expr.node)

    def visit_var(self, var: Var) -> None:
        if False:
            while True:
                i = 10
        self.visit_symbol_node(var)

    def visit_symbol_node(self, symbol: SymbolNode) -> None:
        if False:
            while True:
                i = 10
        if not self.funcs:
            return
        if symbol in self.symbols_to_funcs:
            orig_func = self.symbols_to_funcs[symbol]
            if self.is_parent(self.funcs[-1], orig_func):
                self.symbols_to_funcs[symbol] = self.funcs[-1]
                self.free_variables.setdefault(self.funcs[-1], set()).add(symbol)
            elif self.is_parent(orig_func, self.funcs[-1]):
                self.add_free_variable(symbol)
        else:
            self.symbols_to_funcs[symbol] = self.funcs[-1]

    def is_parent(self, fitem: FuncItem, child: FuncItem) -> bool:
        if False:
            i = 10
            return i + 15
        if child not in self.nested_funcs:
            return False
        parent = self.nested_funcs[child]
        return parent == fitem or self.is_parent(fitem, parent)

    def add_free_variable(self, symbol: SymbolNode) -> None:
        if False:
            for i in range(10):
                print('nop')
        func = self.symbols_to_funcs[symbol]
        self.free_variables.setdefault(func, set()).add(symbol)