from __future__ import annotations
from typing import Callable
from sympy.core import S, Add, Expr, Basic, Mul, Pow, Rational
from sympy.core.logic import fuzzy_not
from sympy.logic.boolalg import Boolean
from sympy.assumptions import ask, Q

def refine(expr, assumptions=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Simplify an expression using assumptions.\n\n    Explanation\n    ===========\n\n    Unlike :func:`~.simplify()` which performs structural simplification\n    without any assumption, this function transforms the expression into\n    the form which is only valid under certain assumptions. Note that\n    ``simplify()`` is generally not done in refining process.\n\n    Refining boolean expression involves reducing it to ``S.true`` or\n    ``S.false``. Unlike :func:`~.ask()`, the expression will not be reduced\n    if the truth value cannot be determined.\n\n    Examples\n    ========\n\n    >>> from sympy import refine, sqrt, Q\n    >>> from sympy.abc import x\n    >>> refine(sqrt(x**2), Q.real(x))\n    Abs(x)\n    >>> refine(sqrt(x**2), Q.positive(x))\n    x\n\n    >>> refine(Q.real(x), Q.positive(x))\n    True\n    >>> refine(Q.positive(x), Q.real(x))\n    Q.positive(x)\n\n    See Also\n    ========\n\n    sympy.simplify.simplify.simplify : Structural simplification without assumptions.\n    sympy.assumptions.ask.ask : Query for boolean expressions using assumptions.\n    '
    if not isinstance(expr, Basic):
        return expr
    if not expr.is_Atom:
        args = [refine(arg, assumptions) for arg in expr.args]
        expr = expr.func(*args)
    if hasattr(expr, '_eval_refine'):
        ref_expr = expr._eval_refine(assumptions)
        if ref_expr is not None:
            return ref_expr
    name = expr.__class__.__name__
    handler = handlers_dict.get(name, None)
    if handler is None:
        return expr
    new_expr = handler(expr, assumptions)
    if new_expr is None or expr == new_expr:
        return expr
    if not isinstance(new_expr, Expr):
        return new_expr
    return refine(new_expr, assumptions)

def refine_abs(expr, assumptions):
    if False:
        return 10
    '\n    Handler for the absolute value.\n\n    Examples\n    ========\n\n    >>> from sympy import Q, Abs\n    >>> from sympy.assumptions.refine import refine_abs\n    >>> from sympy.abc import x\n    >>> refine_abs(Abs(x), Q.real(x))\n    >>> refine_abs(Abs(x), Q.positive(x))\n    x\n    >>> refine_abs(Abs(x), Q.negative(x))\n    -x\n\n    '
    from sympy.functions.elementary.complexes import Abs
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions) and fuzzy_not(ask(Q.negative(arg), assumptions)):
        return arg
    if ask(Q.negative(arg), assumptions):
        return -arg
    if isinstance(arg, Mul):
        r = [refine(abs(a), assumptions) for a in arg.args]
        non_abs = []
        in_abs = []
        for i in r:
            if isinstance(i, Abs):
                in_abs.append(i.args[0])
            else:
                non_abs.append(i)
        return Mul(*non_abs) * Abs(Mul(*in_abs))

def refine_Pow(expr, assumptions):
    if False:
        print('Hello World!')
    '\n    Handler for instances of Pow.\n\n    Examples\n    ========\n\n    >>> from sympy import Q\n    >>> from sympy.assumptions.refine import refine_Pow\n    >>> from sympy.abc import x,y,z\n    >>> refine_Pow((-1)**x, Q.real(x))\n    >>> refine_Pow((-1)**x, Q.even(x))\n    1\n    >>> refine_Pow((-1)**x, Q.odd(x))\n    -1\n\n    For powers of -1, even parts of the exponent can be simplified:\n\n    >>> refine_Pow((-1)**(x+y), Q.even(x))\n    (-1)**y\n    >>> refine_Pow((-1)**(x+y+z), Q.odd(x) & Q.odd(z))\n    (-1)**y\n    >>> refine_Pow((-1)**(x+y+2), Q.odd(x))\n    (-1)**(y + 1)\n    >>> refine_Pow((-1)**(x+3), True)\n    (-1)**(x + 1)\n\n    '
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions import sign
    if isinstance(expr.base, Abs):
        if ask(Q.real(expr.base.args[0]), assumptions) and ask(Q.even(expr.exp), assumptions):
            return expr.base.args[0] ** expr.exp
    if ask(Q.real(expr.base), assumptions):
        if expr.base.is_number:
            if ask(Q.even(expr.exp), assumptions):
                return abs(expr.base) ** expr.exp
            if ask(Q.odd(expr.exp), assumptions):
                return sign(expr.base) * abs(expr.base) ** expr.exp
        if isinstance(expr.exp, Rational):
            if isinstance(expr.base, Pow):
                return abs(expr.base.base) ** (expr.base.exp * expr.exp)
        if expr.base is S.NegativeOne:
            if expr.exp.is_Add:
                old = expr
                (coeff, terms) = expr.exp.as_coeff_add()
                terms = set(terms)
                even_terms = set()
                odd_terms = set()
                initial_number_of_terms = len(terms)
                for t in terms:
                    if ask(Q.even(t), assumptions):
                        even_terms.add(t)
                    elif ask(Q.odd(t), assumptions):
                        odd_terms.add(t)
                terms -= even_terms
                if len(odd_terms) % 2:
                    terms -= odd_terms
                    new_coeff = (coeff + S.One) % 2
                else:
                    terms -= odd_terms
                    new_coeff = coeff % 2
                if new_coeff != coeff or len(terms) < initial_number_of_terms:
                    terms.add(new_coeff)
                    expr = expr.base ** Add(*terms)
                e2 = 2 * expr.exp
                if ask(Q.even(e2), assumptions):
                    if e2.could_extract_minus_sign():
                        e2 *= expr.base
                if e2.is_Add:
                    (i, p) = e2.as_two_terms()
                    if p.is_Pow and p.base is S.NegativeOne:
                        if ask(Q.integer(p.exp), assumptions):
                            i = (i + 1) / 2
                            if ask(Q.even(i), assumptions):
                                return expr.base ** p.exp
                            elif ask(Q.odd(i), assumptions):
                                return expr.base ** (p.exp + 1)
                            else:
                                return expr.base ** (p.exp + i)
                if old != expr:
                    return expr

def refine_atan2(expr, assumptions):
    if False:
        print('Hello World!')
    '\n    Handler for the atan2 function.\n\n    Examples\n    ========\n\n    >>> from sympy import Q, atan2\n    >>> from sympy.assumptions.refine import refine_atan2\n    >>> from sympy.abc import x, y\n    >>> refine_atan2(atan2(y,x), Q.real(y) & Q.positive(x))\n    atan(y/x)\n    >>> refine_atan2(atan2(y,x), Q.negative(y) & Q.negative(x))\n    atan(y/x) - pi\n    >>> refine_atan2(atan2(y,x), Q.positive(y) & Q.negative(x))\n    atan(y/x) + pi\n    >>> refine_atan2(atan2(y,x), Q.zero(y) & Q.negative(x))\n    pi\n    >>> refine_atan2(atan2(y,x), Q.positive(y) & Q.zero(x))\n    pi/2\n    >>> refine_atan2(atan2(y,x), Q.negative(y) & Q.zero(x))\n    -pi/2\n    >>> refine_atan2(atan2(y,x), Q.zero(y) & Q.zero(x))\n    nan\n    '
    from sympy.functions.elementary.trigonometric import atan
    (y, x) = expr.args
    if ask(Q.real(y) & Q.positive(x), assumptions):
        return atan(y / x)
    elif ask(Q.negative(y) & Q.negative(x), assumptions):
        return atan(y / x) - S.Pi
    elif ask(Q.positive(y) & Q.negative(x), assumptions):
        return atan(y / x) + S.Pi
    elif ask(Q.zero(y) & Q.negative(x), assumptions):
        return S.Pi
    elif ask(Q.positive(y) & Q.zero(x), assumptions):
        return S.Pi / 2
    elif ask(Q.negative(y) & Q.zero(x), assumptions):
        return -S.Pi / 2
    elif ask(Q.zero(y) & Q.zero(x), assumptions):
        return S.NaN
    else:
        return expr

def refine_re(expr, assumptions):
    if False:
        i = 10
        return i + 15
    '\n    Handler for real part.\n\n    Examples\n    ========\n\n    >>> from sympy.assumptions.refine import refine_re\n    >>> from sympy import Q, re\n    >>> from sympy.abc import x\n    >>> refine_re(re(x), Q.real(x))\n    x\n    >>> refine_re(re(x), Q.imaginary(x))\n    0\n    '
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return arg
    if ask(Q.imaginary(arg), assumptions):
        return S.Zero
    return _refine_reim(expr, assumptions)

def refine_im(expr, assumptions):
    if False:
        print('Hello World!')
    '\n    Handler for imaginary part.\n\n    Explanation\n    ===========\n\n    >>> from sympy.assumptions.refine import refine_im\n    >>> from sympy import Q, im\n    >>> from sympy.abc import x\n    >>> refine_im(im(x), Q.real(x))\n    0\n    >>> refine_im(im(x), Q.imaginary(x))\n    -I*x\n    '
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return S.Zero
    if ask(Q.imaginary(arg), assumptions):
        return -S.ImaginaryUnit * arg
    return _refine_reim(expr, assumptions)

def refine_arg(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    '\n    Handler for complex argument\n\n    Explanation\n    ===========\n\n    >>> from sympy.assumptions.refine import refine_arg\n    >>> from sympy import Q, arg\n    >>> from sympy.abc import x\n    >>> refine_arg(arg(x), Q.positive(x))\n    0\n    >>> refine_arg(arg(x), Q.negative(x))\n    pi\n    '
    rg = expr.args[0]
    if ask(Q.positive(rg), assumptions):
        return S.Zero
    if ask(Q.negative(rg), assumptions):
        return S.Pi
    return None

def _refine_reim(expr, assumptions):
    if False:
        print('Hello World!')
    expanded = expr.expand(complex=True)
    if expanded != expr:
        refined = refine(expanded, assumptions)
        if refined != expanded:
            return refined
    return None

def refine_sign(expr, assumptions):
    if False:
        for i in range(10):
            print('nop')
    "\n    Handler for sign.\n\n    Examples\n    ========\n\n    >>> from sympy.assumptions.refine import refine_sign\n    >>> from sympy import Symbol, Q, sign, im\n    >>> x = Symbol('x', real = True)\n    >>> expr = sign(x)\n    >>> refine_sign(expr, Q.positive(x) & Q.nonzero(x))\n    1\n    >>> refine_sign(expr, Q.negative(x) & Q.nonzero(x))\n    -1\n    >>> refine_sign(expr, Q.zero(x))\n    0\n    >>> y = Symbol('y', imaginary = True)\n    >>> expr = sign(y)\n    >>> refine_sign(expr, Q.positive(im(y)))\n    I\n    >>> refine_sign(expr, Q.negative(im(y)))\n    -I\n    "
    arg = expr.args[0]
    if ask(Q.zero(arg), assumptions):
        return S.Zero
    if ask(Q.real(arg)):
        if ask(Q.positive(arg), assumptions):
            return S.One
        if ask(Q.negative(arg), assumptions):
            return S.NegativeOne
    if ask(Q.imaginary(arg)):
        (arg_re, arg_im) = arg.as_real_imag()
        if ask(Q.positive(arg_im), assumptions):
            return S.ImaginaryUnit
        if ask(Q.negative(arg_im), assumptions):
            return -S.ImaginaryUnit
    return expr

def refine_matrixelement(expr, assumptions):
    if False:
        i = 10
        return i + 15
    "\n    Handler for symmetric part.\n\n    Examples\n    ========\n\n    >>> from sympy.assumptions.refine import refine_matrixelement\n    >>> from sympy import MatrixSymbol, Q\n    >>> X = MatrixSymbol('X', 3, 3)\n    >>> refine_matrixelement(X[0, 1], Q.symmetric(X))\n    X[0, 1]\n    >>> refine_matrixelement(X[1, 0], Q.symmetric(X))\n    X[0, 1]\n    "
    from sympy.matrices.expressions.matexpr import MatrixElement
    (matrix, i, j) = expr.args
    if ask(Q.symmetric(matrix), assumptions):
        if (i - j).could_extract_minus_sign():
            return expr
        return MatrixElement(matrix, j, i)
handlers_dict: dict[str, Callable[[Expr, Boolean], Expr]] = {'Abs': refine_abs, 'Pow': refine_Pow, 'atan2': refine_atan2, 're': refine_re, 'im': refine_im, 'arg': refine_arg, 'sign': refine_sign, 'MatrixElement': refine_matrixelement}