"""
This module contains query handlers responsible for calculus queries:
infinitesimal, finite, etc.
"""
from sympy.assumptions import Q, ask
from sympy.core import Add, Mul, Pow, Symbol
from sympy.core.numbers import NegativeInfinity, GoldenRatio, Infinity, Exp1, ComplexInfinity, ImaginaryUnit, NaN, Number, Pi, E, TribonacciConstant
from sympy.functions import cos, exp, log, sign, sin
from sympy.logic.boolalg import conjuncts
from ..predicates.calculus import FinitePredicate, InfinitePredicate, PositiveInfinitePredicate, NegativeInfinitePredicate

@FinitePredicate.register(Symbol)
def _(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    '\n    Handles Symbol.\n    '
    if expr.is_finite is not None:
        return expr.is_finite
    if Q.finite(expr) in conjuncts(assumptions):
        return True
    return None

@FinitePredicate.register(Add)
def _(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return True if expr is bounded, False if not and None if unknown.\n\n    Truth Table:\n\n    +-------+-----+-----------+-----------+\n    |       |     |           |           |\n    |       |  B  |     U     |     ?     |\n    |       |     |           |           |\n    +-------+-----+---+---+---+---+---+---+\n    |       |     |   |   |   |   |   |   |\n    |       |     |'+'|'-'|'x'|'+'|'-'|'x'|\n    |       |     |   |   |   |   |   |   |\n    +-------+-----+---+---+---+---+---+---+\n    |       |     |           |           |\n    |   B   |  B  |     U     |     ?     |\n    |       |     |           |           |\n    +---+---+-----+---+---+---+---+---+---+\n    |   |   |     |   |   |   |   |   |   |\n    |   |'+'|     | U | ? | ? | U | ? | ? |\n    |   |   |     |   |   |   |   |   |   |\n    |   +---+-----+---+---+---+---+---+---+\n    |   |   |     |   |   |   |   |   |   |\n    | U |'-'|     | ? | U | ? | ? | U | ? |\n    |   |   |     |   |   |   |   |   |   |\n    |   +---+-----+---+---+---+---+---+---+\n    |   |   |     |           |           |\n    |   |'x'|     |     ?     |     ?     |\n    |   |   |     |           |           |\n    +---+---+-----+---+---+---+---+---+---+\n    |       |     |           |           |\n    |   ?   |     |           |     ?     |\n    |       |     |           |           |\n    +-------+-----+-----------+---+---+---+\n\n        * 'B' = Bounded\n\n        * 'U' = Unbounded\n\n        * '?' = unknown boundedness\n\n        * '+' = positive sign\n\n        * '-' = negative sign\n\n        * 'x' = sign unknown\n\n        * All Bounded -> True\n\n        * 1 Unbounded and the rest Bounded -> False\n\n        * >1 Unbounded, all with same known sign -> False\n\n        * Any Unknown and unknown sign -> None\n\n        * Else -> None\n\n    When the signs are not the same you can have an undefined\n    result as in oo - oo, hence 'bounded' is also undefined.\n    "
    sign = -1
    result = True
    for arg in expr.args:
        _bounded = ask(Q.finite(arg), assumptions)
        if _bounded:
            continue
        s = ask(Q.extended_positive(arg), assumptions)
        if sign != -1 and s != sign or (s is None and None in (_bounded, sign)):
            return None
        else:
            sign = s
        if result is not False:
            result = _bounded
    return result

@FinitePredicate.register(Mul)
def _(expr, assumptions):
    if False:
        return 10
    '\n    Return True if expr is bounded, False if not and None if unknown.\n\n    Truth Table:\n\n    +---+---+---+--------+\n    |   |   |   |        |\n    |   | B | U |   ?    |\n    |   |   |   |        |\n    +---+---+---+---+----+\n    |   |   |   |   |    |\n    |   |   |   | s | /s |\n    |   |   |   |   |    |\n    +---+---+---+---+----+\n    |   |   |   |        |\n    | B | B | U |   ?    |\n    |   |   |   |        |\n    +---+---+---+---+----+\n    |   |   |   |   |    |\n    | U |   | U | U | ?  |\n    |   |   |   |   |    |\n    +---+---+---+---+----+\n    |   |   |   |        |\n    | ? |   |   |   ?    |\n    |   |   |   |        |\n    +---+---+---+---+----+\n\n        * B = Bounded\n\n        * U = Unbounded\n\n        * ? = unknown boundedness\n\n        * s = signed (hence nonzero)\n\n        * /s = not signed\n    '
    result = True
    for arg in expr.args:
        _bounded = ask(Q.finite(arg), assumptions)
        if _bounded:
            continue
        elif _bounded is None:
            if result is None:
                return None
            if ask(Q.extended_nonzero(arg), assumptions) is None:
                return None
            if result is not False:
                result = None
        else:
            result = False
    return result

@FinitePredicate.register(Pow)
def _(expr, assumptions):
    if False:
        print('Hello World!')
    '\n    * Unbounded ** NonZero -> Unbounded\n\n    * Bounded ** Bounded -> Bounded\n\n    * Abs()<=1 ** Positive -> Bounded\n\n    * Abs()>=1 ** Negative -> Bounded\n\n    * Otherwise unknown\n    '
    if expr.base == E:
        return ask(Q.finite(expr.exp), assumptions)
    base_bounded = ask(Q.finite(expr.base), assumptions)
    exp_bounded = ask(Q.finite(expr.exp), assumptions)
    if base_bounded is None and exp_bounded is None:
        return None
    if base_bounded is False and ask(Q.extended_nonzero(expr.exp), assumptions):
        return False
    if base_bounded and exp_bounded:
        return True
    if (abs(expr.base) <= 1) == True and ask(Q.extended_positive(expr.exp), assumptions):
        return True
    if (abs(expr.base) >= 1) == True and ask(Q.extended_negative(expr.exp), assumptions):
        return True
    if (abs(expr.base) >= 1) == True and exp_bounded is False:
        return False
    return None

@FinitePredicate.register(exp)
def _(expr, assumptions):
    if False:
        while True:
            i = 10
    return ask(Q.finite(expr.exp), assumptions)

@FinitePredicate.register(log)
def _(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    if ask(Q.infinite(expr.args[0]), assumptions):
        return False
    return ask(~Q.zero(expr.args[0]), assumptions)

@FinitePredicate.register_many(cos, sin, Number, Pi, Exp1, GoldenRatio, TribonacciConstant, ImaginaryUnit, sign)
def _(expr, assumptions):
    if False:
        return 10
    return True

@FinitePredicate.register_many(ComplexInfinity, Infinity, NegativeInfinity)
def _(expr, assumptions):
    if False:
        while True:
            i = 10
    return False

@FinitePredicate.register(NaN)
def _(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    return None

@InfinitePredicate.register_many(ComplexInfinity, Infinity, NegativeInfinity)
def _(expr, assumptions):
    if False:
        while True:
            i = 10
    return True

@PositiveInfinitePredicate.register(Infinity)
def _(expr, assumptions):
    if False:
        while True:
            i = 10
    return True

@PositiveInfinitePredicate.register_many(NegativeInfinity, ComplexInfinity)
def _(expr, assumptions):
    if False:
        print('Hello World!')
    return False

@NegativeInfinitePredicate.register(NegativeInfinity)
def _(expr, assumptions):
    if False:
        while True:
            i = 10
    return True

@NegativeInfinitePredicate.register_many(Infinity, ComplexInfinity)
def _(expr, assumptions):
    if False:
        print('Hello World!')
    return False