from sympy.core.relational import Eq, is_eq
from sympy.core.basic import Basic
from sympy.core.logic import fuzzy_and, fuzzy_bool
from sympy.logic.boolalg import And
from sympy.multipledispatch import dispatch
from sympy.sets.sets import tfn, ProductSet, Interval, FiniteSet, Set

@dispatch(Interval, FiniteSet)
def _eval_is_eq(lhs, rhs):
    if False:
        i = 10
        return i + 15
    return False

@dispatch(FiniteSet, Interval)
def _eval_is_eq(lhs, rhs):
    if False:
        for i in range(10):
            print('nop')
    return False

@dispatch(Interval, Interval)
def _eval_is_eq(lhs, rhs):
    if False:
        for i in range(10):
            print('nop')
    return And(Eq(lhs.left, rhs.left), Eq(lhs.right, rhs.right), lhs.left_open == rhs.left_open, lhs.right_open == rhs.right_open)

@dispatch(FiniteSet, FiniteSet)
def _eval_is_eq(lhs, rhs):
    if False:
        for i in range(10):
            print('nop')

    def all_in_both():
        if False:
            i = 10
            return i + 15
        s_set = set(lhs.args)
        o_set = set(rhs.args)
        yield fuzzy_and((lhs._contains(e) for e in o_set - s_set))
        yield fuzzy_and((rhs._contains(e) for e in s_set - o_set))
    return tfn[fuzzy_and(all_in_both())]

@dispatch(ProductSet, ProductSet)
def _eval_is_eq(lhs, rhs):
    if False:
        while True:
            i = 10
    if len(lhs.sets) != len(rhs.sets):
        return False
    eqs = (is_eq(x, y) for (x, y) in zip(lhs.sets, rhs.sets))
    return tfn[fuzzy_and(map(fuzzy_bool, eqs))]

@dispatch(Set, Basic)
def _eval_is_eq(lhs, rhs):
    if False:
        for i in range(10):
            print('nop')
    return False

@dispatch(Set, Set)
def _eval_is_eq(lhs, rhs):
    if False:
        return 10
    return tfn[fuzzy_and((a.is_subset(b) for (a, b) in [(lhs, rhs), (rhs, lhs)]))]