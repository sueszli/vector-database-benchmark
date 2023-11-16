from collections import defaultdict
from sympy.assumptions.ask import Q
from sympy.core import Add, Mul, Pow, Number, NumberSymbol, Symbol
from sympy.core.numbers import ImaginaryUnit
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import Equivalent, And, Or, Implies
from sympy.matrices.expressions import MatMul

def allargs(symbol, fact, expr):
    if False:
        while True:
            i = 10
    '\n    Apply all arguments of the expression to the fact structure.\n\n    Parameters\n    ==========\n\n    symbol : Symbol\n        A placeholder symbol.\n\n    fact : Boolean\n        Resulting ``Boolean`` expression.\n\n    expr : Expr\n\n    Examples\n    ========\n\n    >>> from sympy import Q\n    >>> from sympy.assumptions.sathandlers import allargs\n    >>> from sympy.abc import x, y\n    >>> allargs(x, Q.negative(x) | Q.positive(x), x*y)\n    (Q.negative(x) | Q.positive(x)) & (Q.negative(y) | Q.positive(y))\n\n    '
    return And(*[fact.subs(symbol, arg) for arg in expr.args])

def anyarg(symbol, fact, expr):
    if False:
        print('Hello World!')
    '\n    Apply any argument of the expression to the fact structure.\n\n    Parameters\n    ==========\n\n    symbol : Symbol\n        A placeholder symbol.\n\n    fact : Boolean\n        Resulting ``Boolean`` expression.\n\n    expr : Expr\n\n    Examples\n    ========\n\n    >>> from sympy import Q\n    >>> from sympy.assumptions.sathandlers import anyarg\n    >>> from sympy.abc import x, y\n    >>> anyarg(x, Q.negative(x) & Q.positive(x), x*y)\n    (Q.negative(x) & Q.positive(x)) | (Q.negative(y) & Q.positive(y))\n\n    '
    return Or(*[fact.subs(symbol, arg) for arg in expr.args])

def exactlyonearg(symbol, fact, expr):
    if False:
        return 10
    '\n    Apply exactly one argument of the expression to the fact structure.\n\n    Parameters\n    ==========\n\n    symbol : Symbol\n        A placeholder symbol.\n\n    fact : Boolean\n        Resulting ``Boolean`` expression.\n\n    expr : Expr\n\n    Examples\n    ========\n\n    >>> from sympy import Q\n    >>> from sympy.assumptions.sathandlers import exactlyonearg\n    >>> from sympy.abc import x, y\n    >>> exactlyonearg(x, Q.positive(x), x*y)\n    (Q.positive(x) & ~Q.positive(y)) | (Q.positive(y) & ~Q.positive(x))\n\n    '
    pred_args = [fact.subs(symbol, arg) for arg in expr.args]
    res = Or(*[And(pred_args[i], *[~lit for lit in pred_args[:i] + pred_args[i + 1:]]) for i in range(len(pred_args))])
    return res

class ClassFactRegistry:
    """
    Register handlers against classes.

    Explanation
    ===========

    ``register`` method registers the handler function for a class. Here,
    handler function should return a single fact. ``multiregister`` method
    registers the handler function for multiple classes. Here, handler function
    should return a container of multiple facts.

    ``registry(expr)`` returns a set of facts for *expr*.

    Examples
    ========

    Here, we register the facts for ``Abs``.

    >>> from sympy import Abs, Equivalent, Q
    >>> from sympy.assumptions.sathandlers import ClassFactRegistry
    >>> reg = ClassFactRegistry()
    >>> @reg.register(Abs)
    ... def f1(expr):
    ...     return Q.nonnegative(expr)
    >>> @reg.register(Abs)
    ... def f2(expr):
    ...     arg = expr.args[0]
    ...     return Equivalent(~Q.zero(arg), ~Q.zero(expr))

    Calling the registry with expression returns the defined facts for the
    expression.

    >>> from sympy.abc import x
    >>> reg(Abs(x))
    {Q.nonnegative(Abs(x)), Equivalent(~Q.zero(x), ~Q.zero(Abs(x)))}

    Multiple facts can be registered at once by ``multiregister`` method.

    >>> reg2 = ClassFactRegistry()
    >>> @reg2.multiregister(Abs)
    ... def _(expr):
    ...     arg = expr.args[0]
    ...     return [Q.even(arg) >> Q.even(expr), Q.odd(arg) >> Q.odd(expr)]
    >>> reg2(Abs(x))
    {Implies(Q.even(x), Q.even(Abs(x))), Implies(Q.odd(x), Q.odd(Abs(x)))}

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.singlefacts = defaultdict(frozenset)
        self.multifacts = defaultdict(frozenset)

    def register(self, cls):
        if False:
            print('Hello World!')

        def _(func):
            if False:
                i = 10
                return i + 15
            self.singlefacts[cls] |= {func}
            return func
        return _

    def multiregister(self, *classes):
        if False:
            i = 10
            return i + 15

        def _(func):
            if False:
                while True:
                    i = 10
            for cls in classes:
                self.multifacts[cls] |= {func}
            return func
        return _

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        ret1 = self.singlefacts[key]
        for k in self.singlefacts:
            if issubclass(key, k):
                ret1 |= self.singlefacts[k]
        ret2 = self.multifacts[key]
        for k in self.multifacts:
            if issubclass(key, k):
                ret2 |= self.multifacts[k]
        return (ret1, ret2)

    def __call__(self, expr):
        if False:
            print('Hello World!')
        ret = set()
        (handlers1, handlers2) = self[type(expr)]
        for h in handlers1:
            ret.add(h(expr))
        for h in handlers2:
            ret.update(h(expr))
        return ret
class_fact_registry = ClassFactRegistry()
x = Symbol('x')

@class_fact_registry.multiregister(Abs)
def _(expr):
    if False:
        for i in range(10):
            print('nop')
    arg = expr.args[0]
    return [Q.nonnegative(expr), Equivalent(~Q.zero(arg), ~Q.zero(expr)), Q.even(arg) >> Q.even(expr), Q.odd(arg) >> Q.odd(expr), Q.integer(arg) >> Q.integer(expr)]

@class_fact_registry.multiregister(Add)
def _(expr):
    if False:
        for i in range(10):
            print('nop')
    return [allargs(x, Q.positive(x), expr) >> Q.positive(expr), allargs(x, Q.negative(x), expr) >> Q.negative(expr), allargs(x, Q.real(x), expr) >> Q.real(expr), allargs(x, Q.rational(x), expr) >> Q.rational(expr), allargs(x, Q.integer(x), expr) >> Q.integer(expr), exactlyonearg(x, ~Q.integer(x), expr) >> ~Q.integer(expr)]

@class_fact_registry.register(Add)
def _(expr):
    if False:
        return 10
    allargs_real = allargs(x, Q.real(x), expr)
    onearg_irrational = exactlyonearg(x, Q.irrational(x), expr)
    return Implies(allargs_real, Implies(onearg_irrational, Q.irrational(expr)))

@class_fact_registry.multiregister(Mul)
def _(expr):
    if False:
        return 10
    return [Equivalent(Q.zero(expr), anyarg(x, Q.zero(x), expr)), allargs(x, Q.positive(x), expr) >> Q.positive(expr), allargs(x, Q.real(x), expr) >> Q.real(expr), allargs(x, Q.rational(x), expr) >> Q.rational(expr), allargs(x, Q.integer(x), expr) >> Q.integer(expr), exactlyonearg(x, ~Q.rational(x), expr) >> ~Q.integer(expr), allargs(x, Q.commutative(x), expr) >> Q.commutative(expr)]

@class_fact_registry.register(Mul)
def _(expr):
    if False:
        while True:
            i = 10
    allargs_prime = allargs(x, Q.prime(x), expr)
    return Implies(allargs_prime, ~Q.prime(expr))

@class_fact_registry.register(Mul)
def _(expr):
    if False:
        return 10
    allargs_imag_or_real = allargs(x, Q.imaginary(x) | Q.real(x), expr)
    onearg_imaginary = exactlyonearg(x, Q.imaginary(x), expr)
    return Implies(allargs_imag_or_real, Implies(onearg_imaginary, Q.imaginary(expr)))

@class_fact_registry.register(Mul)
def _(expr):
    if False:
        print('Hello World!')
    allargs_real = allargs(x, Q.real(x), expr)
    onearg_irrational = exactlyonearg(x, Q.irrational(x), expr)
    return Implies(allargs_real, Implies(onearg_irrational, Q.irrational(expr)))

@class_fact_registry.register(Mul)
def _(expr):
    if False:
        while True:
            i = 10
    allargs_integer = allargs(x, Q.integer(x), expr)
    anyarg_even = anyarg(x, Q.even(x), expr)
    return Implies(allargs_integer, Equivalent(anyarg_even, Q.even(expr)))

@class_fact_registry.register(MatMul)
def _(expr):
    if False:
        print('Hello World!')
    allargs_square = allargs(x, Q.square(x), expr)
    allargs_invertible = allargs(x, Q.invertible(x), expr)
    return Implies(allargs_square, Equivalent(Q.invertible(expr), allargs_invertible))

@class_fact_registry.multiregister(Pow)
def _(expr):
    if False:
        return 10
    (base, exp) = (expr.base, expr.exp)
    return [(Q.real(base) & Q.even(exp) & Q.nonnegative(exp)) >> Q.nonnegative(expr), (Q.nonnegative(base) & Q.odd(exp) & Q.nonnegative(exp)) >> Q.nonnegative(expr), (Q.nonpositive(base) & Q.odd(exp) & Q.nonnegative(exp)) >> Q.nonpositive(expr), Equivalent(Q.zero(expr), Q.zero(base) & Q.positive(exp))]
_old_assump_getters = {Q.positive: lambda o: o.is_positive, Q.zero: lambda o: o.is_zero, Q.negative: lambda o: o.is_negative, Q.rational: lambda o: o.is_rational, Q.irrational: lambda o: o.is_irrational, Q.even: lambda o: o.is_even, Q.odd: lambda o: o.is_odd, Q.imaginary: lambda o: o.is_imaginary, Q.prime: lambda o: o.is_prime, Q.composite: lambda o: o.is_composite}

@class_fact_registry.multiregister(Number, NumberSymbol, ImaginaryUnit)
def _(expr):
    if False:
        return 10
    ret = []
    for (p, getter) in _old_assump_getters.items():
        pred = p(expr)
        prop = getter(expr)
        if prop is not None:
            ret.append(Equivalent(pred, prop))
    return ret