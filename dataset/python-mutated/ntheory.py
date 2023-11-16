"""
Handlers for keys related to number theory: prime, even, odd, etc.
"""
from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Float, Mul, Pow, S
from sympy.core.numbers import ImaginaryUnit, Infinity, Integer, NaN, NegativeInfinity, NumberSymbol, Rational, int_valued
from sympy.functions import Abs, im, re
from sympy.ntheory import isprime
from sympy.multipledispatch import MDNotImplementedError
from ..predicates.ntheory import PrimePredicate, CompositePredicate, EvenPredicate, OddPredicate

def _PrimePredicate_number(expr, assumptions):
    if False:
        while True:
            i = 10
    exact = not expr.atoms(Float)
    try:
        i = int(expr.round())
        if (expr - i).equals(0) is False:
            raise TypeError
    except TypeError:
        return False
    if exact:
        return isprime(i)

@PrimePredicate.register(Expr)
def _(expr, assumptions):
    if False:
        return 10
    ret = expr.is_prime
    if ret is None:
        raise MDNotImplementedError
    return ret

@PrimePredicate.register(Basic)
def _(expr, assumptions):
    if False:
        print('Hello World!')
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(Mul)
def _(expr, assumptions):
    if False:
        while True:
            i = 10
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)
    for arg in expr.args:
        if not ask(Q.integer(arg), assumptions):
            return None
    for arg in expr.args:
        if arg.is_number and arg.is_composite:
            return False

@PrimePredicate.register(Pow)
def _(expr, assumptions):
    if False:
        return 10
    '\n    Integer**Integer     -> !Prime\n    '
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)
    if ask(Q.integer(expr.exp), assumptions) and ask(Q.integer(expr.base), assumptions):
        return False

@PrimePredicate.register(Integer)
def _(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    return isprime(expr)

@PrimePredicate.register_many(Rational, Infinity, NegativeInfinity, ImaginaryUnit)
def _(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    return False

@PrimePredicate.register(Float)
def _(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(NumberSymbol)
def _(expr, assumptions):
    if False:
        print('Hello World!')
    return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(NaN)
def _(expr, assumptions):
    if False:
        while True:
            i = 10
    return None

@CompositePredicate.register(Expr)
def _(expr, assumptions):
    if False:
        return 10
    ret = expr.is_composite
    if ret is None:
        raise MDNotImplementedError
    return ret

@CompositePredicate.register(Basic)
def _(expr, assumptions):
    if False:
        print('Hello World!')
    _positive = ask(Q.positive(expr), assumptions)
    if _positive:
        _integer = ask(Q.integer(expr), assumptions)
        if _integer:
            _prime = ask(Q.prime(expr), assumptions)
            if _prime is None:
                return
            if expr.equals(1):
                return False
            return not _prime
        else:
            return _integer
    else:
        return _positive

def _EvenPredicate_number(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(expr, (float, Float)):
        if int_valued(expr):
            return None
        return False
    try:
        i = int(expr.round())
    except TypeError:
        return False
    if not (expr - i).equals(0):
        return False
    return i % 2 == 0

@EvenPredicate.register(Expr)
def _(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    ret = expr.is_even
    if ret is None:
        raise MDNotImplementedError
    return ret

@EvenPredicate.register(Basic)
def _(expr, assumptions):
    if False:
        while True:
            i = 10
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)

@EvenPredicate.register(Mul)
def _(expr, assumptions):
    if False:
        print('Hello World!')
    '\n    Even * Integer    -> Even\n    Even * Odd        -> Even\n    Integer * Odd     -> ?\n    Odd * Odd         -> Odd\n    Even * Even       -> Even\n    Integer * Integer -> Even if Integer + Integer = Odd\n    otherwise         -> ?\n    '
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    (even, odd, irrational, acc) = (False, 0, False, 1)
    for arg in expr.args:
        if ask(Q.integer(arg), assumptions):
            if ask(Q.even(arg), assumptions):
                even = True
            elif ask(Q.odd(arg), assumptions):
                odd += 1
            elif not even and acc != 1:
                if ask(Q.odd(acc + arg), assumptions):
                    even = True
        elif ask(Q.irrational(arg), assumptions):
            if irrational:
                break
            irrational = True
        else:
            break
        acc = arg
    else:
        if irrational:
            return False
        if even:
            return True
        if odd == len(expr.args):
            return False

@EvenPredicate.register(Add)
def _(expr, assumptions):
    if False:
        print('Hello World!')
    '\n    Even + Odd  -> Odd\n    Even + Even -> Even\n    Odd  + Odd  -> Even\n\n    '
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    _result = True
    for arg in expr.args:
        if ask(Q.even(arg), assumptions):
            pass
        elif ask(Q.odd(arg), assumptions):
            _result = not _result
        else:
            break
    else:
        return _result

@EvenPredicate.register(Pow)
def _(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    if ask(Q.integer(expr.exp), assumptions):
        if ask(Q.positive(expr.exp), assumptions):
            return ask(Q.even(expr.base), assumptions)
        elif ask(~Q.negative(expr.exp) & Q.odd(expr.base), assumptions):
            return False
        elif expr.base is S.NegativeOne:
            return False

@EvenPredicate.register(Integer)
def _(expr, assumptions):
    if False:
        return 10
    return not bool(expr.p & 1)

@EvenPredicate.register_many(Rational, Infinity, NegativeInfinity, ImaginaryUnit)
def _(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    return False

@EvenPredicate.register(NumberSymbol)
def _(expr, assumptions):
    if False:
        i = 10
        return i + 15
    return _EvenPredicate_number(expr, assumptions)

@EvenPredicate.register(Abs)
def _(expr, assumptions):
    if False:
        i = 10
        return i + 15
    if ask(Q.real(expr.args[0]), assumptions):
        return ask(Q.even(expr.args[0]), assumptions)

@EvenPredicate.register(re)
def _(expr, assumptions):
    if False:
        i = 10
        return i + 15
    if ask(Q.real(expr.args[0]), assumptions):
        return ask(Q.even(expr.args[0]), assumptions)

@EvenPredicate.register(im)
def _(expr, assumptions):
    if False:
        i = 10
        return i + 15
    if ask(Q.real(expr.args[0]), assumptions):
        return True

@EvenPredicate.register(NaN)
def _(expr, assumptions):
    if False:
        i = 10
        return i + 15
    return None

@OddPredicate.register(Expr)
def _(expr, assumptions):
    if False:
        i = 10
        return i + 15
    ret = expr.is_odd
    if ret is None:
        raise MDNotImplementedError
    return ret

@OddPredicate.register(Basic)
def _(expr, assumptions):
    if False:
        print('Hello World!')
    _integer = ask(Q.integer(expr), assumptions)
    if _integer:
        _even = ask(Q.even(expr), assumptions)
        if _even is None:
            return None
        return not _even
    return _integer