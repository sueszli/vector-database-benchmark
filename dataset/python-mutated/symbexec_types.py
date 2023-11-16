from __future__ import print_function
from future.utils import viewitems
from miasm.ir.symbexec import SymbolicExecutionEngine, StateEngine
from miasm.expression.simplifications import expr_simp
from miasm.expression.expression import ExprId, ExprMem

class SymbolicStateCTypes(StateEngine):
    """Store C types of symbols"""

    def __init__(self, symbols):
        if False:
            i = 10
            return i + 15
        tmp = {}
        for (expr, types) in viewitems(symbols):
            tmp[expr] = frozenset(types)
        self._symbols = frozenset(viewitems(tmp))

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((self.__class__, self._symbols))

    def __str__(self):
        if False:
            while True:
                i = 10
        out = []
        for (dst, src) in sorted(self._symbols):
            out.append('%s = %s' % (dst, src))
        return '\n'.join(out)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        return self.symbols == other.symbols

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self.__eq__(other)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for (dst, src) in self._symbols:
            yield (dst, src)

    def merge(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Merge two symbolic states\n        The resulting types are the union of types of both states.\n        @other: second symbolic state\n        '
        symb_a = self.symbols
        symb_b = other.symbols
        symbols = {}
        for expr in set(symb_a).union(set(symb_b)):
            ctypes = symb_a.get(expr, set()).union(symb_b.get(expr, set()))
            if ctypes:
                symbols[expr] = ctypes
        return self.__class__(symbols)

    @property
    def symbols(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the dictionary of known symbols'types"
        return dict(self._symbols)

class SymbExecCType(SymbolicExecutionEngine):
    """Engine of C types propagation
    WARNING: avoid memory aliases here!
    """
    StateEngine = SymbolicStateCTypes
    OBJC_INTERNAL = '___OBJC___'

    def __init__(self, lifter, symbols, chandler, sb_expr_simp=expr_simp):
        if False:
            for i in range(10):
                print('nop')
        self.chandler = chandler
        super(SymbExecCType, self).__init__(lifter, {}, sb_expr_simp)
        self.symbols = dict(symbols)

    def get_state(self):
        if False:
            return 10
        'Return the current state of the SymbolicEngine'
        return self.StateEngine(self.symbols)

    def eval_assignblk(self, assignblk):
        if False:
            return 10
        '\n        Evaluate AssignBlock on the current state\n        @assignblk: AssignBlock instance\n        '
        pool_out = {}
        for (dst, src) in viewitems(assignblk):
            objcs = self.chandler.expr_to_types(src, self.symbols)
            if isinstance(dst, ExprMem):
                continue
            elif isinstance(dst, ExprId):
                pool_out[dst] = frozenset(objcs)
            else:
                raise ValueError('Unsupported assignment', str(dst))
        return pool_out

    def eval_expr(self, expr, eval_cache=None):
        if False:
            print('Hello World!')
        return frozenset(self.chandler.expr_to_types(expr, self.symbols))

    def apply_change(self, dst, src):
        if False:
            return 10
        if src is None:
            if dst in self.symbols:
                del self.symbols[dst]
        else:
            self.symbols[dst] = src

    def del_mem_above_stack(self, stack_ptr):
        if False:
            return 10
        'No stack deletion'
        return

    def dump_id(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dump modififed registers symbols only\n        '
        for (expr, expr_types) in sorted(viewitems(self.symbols)):
            if not expr.is_mem():
                print(expr)
                for expr_type in expr_types:
                    print('\t', expr_type)

    def dump_mem(self):
        if False:
            print('Hello World!')
        '\n        Dump modififed memory symbols\n        '
        for (expr, value) in sorted(viewitems(self.symbols)):
            if expr.is_mem():
                print(expr, value)