"""
Created on Aug 4, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ..visitors import Visitor, visit_children
from ..visitors.symbol_visitor import get_symbols
import ast
from ...utils import py2op

class ConditionalSymbolVisitor(Visitor):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._cond_lhs = set()
        self._stable_lhs = set()
        self._cond_rhs = set()
        self._stable_rhs = set()
        self.undefined = set()
        self.seen_break = False
    visitModule = visit_children
    visitPass = visit_children

    def update_stable_rhs(self, symbols):
        if False:
            return 10
        new_symbols = symbols - self._stable_rhs
        self._update_undefined(new_symbols)
        if self.seen_break:
            self._cond_rhs.update(new_symbols)
        else:
            self._cond_rhs -= new_symbols
            self._stable_rhs.update(new_symbols)

    def update_stable_lhs(self, symbols):
        if False:
            while True:
                i = 10
        new_symbols = symbols - self._stable_lhs
        if self.seen_break:
            self._cond_lhs.update(new_symbols)
        else:
            self._cond_lhs -= new_symbols
            self._stable_lhs.update(new_symbols)

    def update_cond_rhs(self, symbols):
        if False:
            while True:
                i = 10
        new_symbols = symbols - self._stable_rhs
        self._update_undefined(new_symbols)
        self._cond_rhs.update(new_symbols)

    def update_cond_lhs(self, symbols):
        if False:
            while True:
                i = 10
        self._cond_lhs.update(symbols - self._stable_lhs)

    def _update_undefined(self, symbols):
        if False:
            while True:
                i = 10
        self.undefined.update(symbols - self._stable_lhs)
    update_undefined = _update_undefined

    @property
    def stable_lhs(self):
        if False:
            return 10
        assert not self._stable_lhs & self._cond_lhs
        return self._stable_lhs

    @property
    def stable_rhs(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self._stable_rhs & self._cond_rhs
        return self._stable_rhs

    @property
    def cond_rhs(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self._stable_rhs & self._cond_rhs
        return self._cond_rhs

    @property
    def cond_lhs(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self._stable_lhs & self._cond_lhs
        return self._cond_lhs

    @property
    def lhs(self):
        if False:
            return 10
        assert not self._stable_lhs & self._cond_lhs
        return self._cond_lhs | self._stable_lhs

    @property
    def rhs(self):
        if False:
            while True:
                i = 10
        assert not self._stable_rhs & self._cond_rhs
        return self._cond_rhs | self._stable_rhs

    def visitAugAssign(self, node):
        if False:
            print('Hello World!')
        values = get_symbols(node.value)
        self.update_stable_rhs(values)
        targets = get_symbols(node.target)
        self.update_stable_rhs(targets)
        self.update_stable_lhs(targets)

    def visitAssign(self, node):
        if False:
            while True:
                i = 10
        ids = set()
        for target in node.targets:
            ids.update(get_symbols(target, ast.Store))
        rhs_ids = get_symbols(node.value, ast.Load)
        for target in node.targets:
            rhs_ids.update(get_symbols(target, ast.Load))
        self.update_stable_rhs(rhs_ids)
        self.update_stable_lhs(ids)

    def visitBreak(self, node):
        if False:
            return 10
        self.seen_break = True

    def visitContinue(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.seen_break = True

    def visit_loop(self, node):
        if False:
            for i in range(10):
                print('nop')
        gen = ConditionalSymbolVisitor()
        for stmnt in node.body:
            gen.visit(stmnt)
        self.update_cond_lhs(gen.cond_lhs)
        self.update_cond_rhs(gen.cond_rhs)
        outputs = gen.stable_lhs
        inputs = gen.stable_rhs
        gen = ConditionalSymbolVisitor()
        for stmnt in node.orelse:
            gen.visit(stmnt)
        self.update_cond_rhs(gen.cond_rhs)
        self.update_cond_lhs(gen.cond_lhs)
        orelse_outputs = gen.stable_lhs
        orelse_inputs = gen.stable_rhs
        self.update_stable_lhs(outputs.intersection(orelse_outputs))
        self.update_stable_rhs(inputs.intersection(orelse_inputs))
        self.update_cond_lhs(outputs.symmetric_difference(orelse_outputs))
        self.update_cond_rhs(inputs.symmetric_difference(orelse_inputs))

    def visitFor(self, node):
        if False:
            print('Hello World!')
        lhs_symbols = get_symbols(node.target, ast.Store)
        self.update_cond_lhs(lhs_symbols)
        rhs_symbols = get_symbols(node.iter, ast.Load)
        self.update_stable_rhs(rhs_symbols)
        remove_from_undef = lhs_symbols - self.undefined
        self.visit_loop(node)
        self.undefined -= remove_from_undef

    def visitExpr(self, node):
        if False:
            print('Hello World!')
        rhs_ids = get_symbols(node, ast.Load)
        self.update_stable_rhs(rhs_ids)

    def visitPrint(self, node):
        if False:
            while True:
                i = 10
        rhs_ids = get_symbols(node, ast.Load)
        self.update_stable_rhs(rhs_ids)

    def visitWhile(self, node):
        if False:
            return 10
        rhs_symbols = get_symbols(node.test, ast.Load)
        self.update_stable_rhs(rhs_symbols)
        self.visit_loop(node)

    def visitIf(self, node):
        if False:
            i = 10
            return i + 15
        rhs_symbols = get_symbols(node.test, ast.Load)
        self.update_stable_rhs(rhs_symbols)
        gen = ConditionalSymbolVisitor()
        for stmnt in node.body:
            gen.visit(stmnt)
        if gen.seen_break:
            self.seen_break = True
        self.update_cond_lhs(gen._cond_lhs)
        self.update_cond_rhs(gen._cond_rhs)
        outputs = gen.stable_lhs
        inputs = gen.stable_rhs
        gen = ConditionalSymbolVisitor()
        for stmnt in node.orelse:
            gen.visit(stmnt)
        self.update_cond_lhs(gen._cond_lhs)
        self.update_cond_rhs(gen._cond_rhs)
        orelse_outputs = gen.stable_lhs
        orelse_inputs = gen.stable_rhs
        self.update_stable_lhs(outputs.intersection(orelse_outputs))
        self.update_stable_rhs(inputs.intersection(orelse_inputs))
        self.update_cond_lhs(outputs.symmetric_difference(orelse_outputs))
        self.update_cond_rhs(inputs.symmetric_difference(orelse_inputs))

    @py2op
    def visitExec(self, node):
        if False:
            print('Hello World!')
        self.update_stable_rhs(get_symbols(node.body, ast.Load))
        if node.globals:
            self.update_stable_rhs(get_symbols(node.globals, ast.Load))
        if node.locals:
            self.update_stable_rhs(get_symbols(node.locals, ast.Load))

    def visitAssert(self, node):
        if False:
            while True:
                i = 10
        self.update_stable_rhs(get_symbols(node.test, ast.Load))
        if node.msg:
            self.update_stable_rhs(get_symbols(node.msg, ast.Load))

    @py2op
    def visitRaise(self, node):
        if False:
            return 10
        if node.type:
            self.update_stable_rhs(get_symbols(node.type, ast.Load))
        if node.inst:
            self.update_stable_rhs(get_symbols(node.inst, ast.Load))
        if node.tback:
            self.update_stable_rhs(get_symbols(node.tback, ast.Load))

    @visitRaise.py3op
    def visitRaise(self, node):
        if False:
            for i in range(10):
                print('nop')
        if node.exc:
            self.update_stable_rhs(get_symbols(node.exc, ast.Load))
        if node.cause:
            self.update_stable_rhs(get_symbols(node.cause, ast.Load))

    def visitTryExcept(self, node):
        if False:
            i = 10
            return i + 15
        gen = ConditionalSymbolVisitor()
        gen.visit_list(node.body)
        self.update_undefined(gen.undefined)
        handlers = [csv(hndlr) for hndlr in node.handlers]
        for g in handlers:
            self.update_undefined(g.undefined)
        stable_rhs = gen.stable_rhs.intersection(*[g.stable_rhs for g in handlers])
        self.update_stable_rhs(stable_rhs)
        all_rhs = gen.rhs.union(*[g.rhs for g in handlers])
        self.update_cond_rhs(all_rhs - stable_rhs)
        stable_lhs = gen.stable_lhs.intersection(*[g.stable_lhs for g in handlers])
        self.update_stable_lhs(stable_lhs)
        all_lhs = gen.lhs.union(*[g.lhs for g in handlers])
        self.update_cond_lhs(all_lhs - stable_lhs)
        gen = ConditionalSymbolVisitor()
        gen.visit_list(node.orelse)
        self.update_undefined(gen.undefined)
        self.update_cond_lhs(gen.lhs)
        self.update_cond_rhs(gen.rhs)

    @py2op
    def visitExceptHandler(self, node):
        if False:
            for i in range(10):
                print('nop')
        if node.type:
            self.update_stable_rhs(get_symbols(node.type, ast.Load))
        if node.name:
            self.update_stable_lhs(get_symbols(node.name, ast.Store))
        self.visit_list(node.body)

    @visitExceptHandler.py3op
    def visitExceptHandler(self, node):
        if False:
            for i in range(10):
                print('nop')
        if node.type:
            self.update_stable_rhs(get_symbols(node.type, ast.Load))
        if node.name:
            self.update_stable_lhs({node.name})
        self.visit_list(node.body)

    def visitTryFinally(self, node):
        if False:
            print('Hello World!')
        self.visit_list(node.body)
        self.visit_list(node.finalbody)

    def visitImportFrom(self, node):
        if False:
            print('Hello World!')
        symbols = get_symbols(node)
        self.update_stable_lhs(symbols)

    def visitImport(self, node):
        if False:
            return 10
        symbols = get_symbols(node)
        self.update_stable_lhs(symbols)

    def visitLambda(self, node):
        if False:
            return 10
        gen = ConditionalSymbolVisitor()
        gen.update_stable_lhs(symbols={arg for arg in node.args.args})
        gen.visit_list(node.body)
        self.update_stable_rhs(gen.undefined)

    def visitFunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        for decorator in node.decorator_list:
            self.update_stable_rhs(get_symbols(decorator, ast.Load))
        self.update_stable_lhs({node.name})
        gen = ConditionalSymbolVisitor()
        gen.update_stable_lhs(symbols={arg for arg in node.args.args})
        gen.visit_list(node.body)
        self.update_stable_rhs(gen.undefined)

    def visitGlobal(self, node):
        if False:
            for i in range(10):
                print('nop')
        pass

    def visitWith(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.update_stable_rhs(get_symbols(node.context_expr, ast.Load))
        if node.optional_vars:
            self.update_stable_lhs(get_symbols(node.optional_vars, ast.Load))
        self.visit_list(node.body)

    def visitReturn(self, node):
        if False:
            while True:
                i = 10
        self.update_stable_rhs(get_symbols(node.value, ast.Load))

def csv(node):
    if False:
        while True:
            i = 10
    gen = ConditionalSymbolVisitor()
    gen.visit(node)
    return gen

def lhs(node):
    if False:
        while True:
            i = 10
    '\n    Return a set of symbols in `node` that are assigned.\n\n    :param node: ast node\n\n    :returns: set of strings.\n    '
    gen = ConditionalSymbolVisitor()
    if isinstance(node, (list, tuple)):
        gen.visit_list(node)
    else:
        gen.visit(node)
    return gen.lhs

def rhs(node):
    if False:
        i = 10
        return i + 15
    '\n    Return a set of symbols in `node` that are used.\n\n    :param node: ast node\n\n    :returns: set of strings.\n    '
    gen = ConditionalSymbolVisitor()
    if isinstance(node, (list, tuple)):
        gen.visit_list(node)
    else:
        gen.visit(node)
    return gen.rhs

def conditional_lhs(node):
    if False:
        return 10
    '\n    Group outputs into conditional and stable\n    :param node: ast node\n\n    :returns: tuple of (conditional, stable)\n\n    '
    gen = ConditionalSymbolVisitor()
    gen.visit(node)
    return (gen.cond_lhs, gen.stable_lhs)

def conditional_symbols(node):
    if False:
        i = 10
        return i + 15
    '\n    Group lhs and rhs into conditional, stable and undefined\n    :param node: ast node\n\n    :returns: tuple of (conditional_lhs, stable_lhs),(conditional_rhs, stable_rhs), undefined\n\n    '
    gen = ConditionalSymbolVisitor()
    gen.visit(node)
    lhs = (gen.cond_lhs, gen.stable_lhs)
    rhs = (gen.cond_rhs, gen.stable_rhs)
    undefined = gen.undefined
    return (lhs, rhs, undefined)
if __name__ == '__main__':
    source = '\nwhile k:\n    a = 1\n    b = 1\n    break\n    d = 1\nelse:\n    a =2\n    c= 3\n    d = 1\n    '
    print(conditional_lhs(ast.parse(source)))