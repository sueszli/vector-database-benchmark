from numba import njit
import numba
from numba.core import ir

@njit(inline='never')
def never_inline():
    if False:
        for i in range(10):
            print('nop')
    return 100

@njit(inline='always')
def always_inline():
    if False:
        for i in range(10):
            print('nop')
    return 200

def sentinel_cost_model(expr, caller_info, callee_info):
    if False:
        while True:
            i = 10
    for blk in callee_info.blocks.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Const):
                    if stmt.value.value == 37:
                        return True
    before_expr = True
    for blk in caller_info.blocks.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Expr):
                    if stmt.value == expr:
                        before_expr = False
                if isinstance(stmt.value, ir.Const):
                    if stmt.value.value == 13:
                        return True & before_expr
    return False

@njit(inline=sentinel_cost_model)
def maybe_inline1():
    if False:
        print('Hello World!')
    return 300

@njit(inline=sentinel_cost_model)
def maybe_inline2():
    if False:
        print('Hello World!')
    return 37

@njit
def foo():
    if False:
        return 10
    a = never_inline()
    b = always_inline()
    d = maybe_inline1()
    magic_const = 13
    e = maybe_inline1()
    c = maybe_inline2()
    return a + b + c + d + e + magic_const
foo()