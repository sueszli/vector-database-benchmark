from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions import DiracDelta, Heaviside
from .integrals import Integral, integrate

def change_mul(node, x):
    if False:
        return 10
    'change_mul(node, x)\n\n       Rearranges the operands of a product, bringing to front any simple\n       DiracDelta expression.\n\n       Explanation\n       ===========\n\n       If no simple DiracDelta expression was found, then all the DiracDelta\n       expressions are simplified (using DiracDelta.expand(diracdelta=True, wrt=x)).\n\n       Return: (dirac, new node)\n       Where:\n         o dirac is either a simple DiracDelta expression or None (if no simple\n           expression was found);\n         o new node is either a simplified DiracDelta expressions or None (if it\n           could not be simplified).\n\n       Examples\n       ========\n\n       >>> from sympy import DiracDelta, cos\n       >>> from sympy.integrals.deltafunctions import change_mul\n       >>> from sympy.abc import x, y\n       >>> change_mul(x*y*DiracDelta(x)*cos(x), x)\n       (DiracDelta(x), x*y*cos(x))\n       >>> change_mul(x*y*DiracDelta(x**2 - 1)*cos(x), x)\n       (None, x*y*cos(x)*DiracDelta(x - 1)/2 + x*y*cos(x)*DiracDelta(x + 1)/2)\n       >>> change_mul(x*y*DiracDelta(cos(x))*cos(x), x)\n       (None, None)\n\n       See Also\n       ========\n\n       sympy.functions.special.delta_functions.DiracDelta\n       deltaintegrate\n    '
    new_args = []
    dirac = None
    (c, nc) = node.args_cnc()
    sorted_args = sorted(c, key=default_sort_key)
    sorted_args.extend(nc)
    for arg in sorted_args:
        if arg.is_Pow and isinstance(arg.base, DiracDelta):
            new_args.append(arg.func(arg.base, arg.exp - 1))
            arg = arg.base
        if dirac is None and (isinstance(arg, DiracDelta) and arg.is_simple(x)):
            dirac = arg
        else:
            new_args.append(arg)
    if not dirac:
        new_args = []
        for arg in sorted_args:
            if isinstance(arg, DiracDelta):
                new_args.append(arg.expand(diracdelta=True, wrt=x))
            elif arg.is_Pow and isinstance(arg.base, DiracDelta):
                new_args.append(arg.func(arg.base.expand(diracdelta=True, wrt=x), arg.exp))
            else:
                new_args.append(arg)
        if new_args != sorted_args:
            nnode = Mul(*new_args).expand()
        else:
            nnode = None
        return (None, nnode)
    return (dirac, Mul(*new_args))

def deltaintegrate(f, x):
    if False:
        print('Hello World!')
    "\n    deltaintegrate(f, x)\n\n    Explanation\n    ===========\n\n    The idea for integration is the following:\n\n    - If we are dealing with a DiracDelta expression, i.e. DiracDelta(g(x)),\n      we try to simplify it.\n\n      If we could simplify it, then we integrate the resulting expression.\n      We already know we can integrate a simplified expression, because only\n      simple DiracDelta expressions are involved.\n\n      If we couldn't simplify it, there are two cases:\n\n      1) The expression is a simple expression: we return the integral,\n         taking care if we are dealing with a Derivative or with a proper\n         DiracDelta.\n\n      2) The expression is not simple (i.e. DiracDelta(cos(x))): we can do\n         nothing at all.\n\n    - If the node is a multiplication node having a DiracDelta term:\n\n      First we expand it.\n\n      If the expansion did work, then we try to integrate the expansion.\n\n      If not, we try to extract a simple DiracDelta term, then we have two\n      cases:\n\n      1) We have a simple DiracDelta term, so we return the integral.\n\n      2) We didn't have a simple term, but we do have an expression with\n         simplified DiracDelta terms, so we integrate this expression.\n\n    Examples\n    ========\n\n        >>> from sympy.abc import x, y, z\n        >>> from sympy.integrals.deltafunctions import deltaintegrate\n        >>> from sympy import sin, cos, DiracDelta\n        >>> deltaintegrate(x*sin(x)*cos(x)*DiracDelta(x - 1), x)\n        sin(1)*cos(1)*Heaviside(x - 1)\n        >>> deltaintegrate(y**2*DiracDelta(x - z)*DiracDelta(y - z), y)\n        z**2*DiracDelta(x - z)*Heaviside(y - z)\n\n    See Also\n    ========\n\n    sympy.functions.special.delta_functions.DiracDelta\n    sympy.integrals.integrals.Integral\n    "
    if not f.has(DiracDelta):
        return None
    if f.func == DiracDelta:
        h = f.expand(diracdelta=True, wrt=x)
        if h == f:
            if f.is_simple(x):
                if len(f.args) <= 1 or f.args[1] == 0:
                    return Heaviside(f.args[0])
                else:
                    return DiracDelta(f.args[0], f.args[1] - 1) / f.args[0].as_poly().LC()
        else:
            fh = integrate(h, x)
            return fh
    elif f.is_Mul or f.is_Pow:
        g = f.expand()
        if f != g:
            fh = integrate(g, x)
            if fh is not None and (not isinstance(fh, Integral)):
                return fh
        else:
            (deltaterm, rest_mult) = change_mul(f, x)
            if not deltaterm:
                if rest_mult:
                    fh = integrate(rest_mult, x)
                    return fh
            else:
                from sympy.solvers import solve
                deltaterm = deltaterm.expand(diracdelta=True, wrt=x)
                if deltaterm.is_Mul:
                    (deltaterm, rest_mult_2) = change_mul(deltaterm, x)
                    rest_mult = rest_mult * rest_mult_2
                point = solve(deltaterm.args[0], x)[0]
                n = 0 if len(deltaterm.args) == 1 else deltaterm.args[1]
                m = 0
                while n >= 0:
                    r = S.NegativeOne ** n * rest_mult.diff(x, n).subs(x, point)
                    if r.is_zero:
                        n -= 1
                        m += 1
                    elif m == 0:
                        return r * Heaviside(x - point)
                    else:
                        return r * DiracDelta(x, m - 1)
                return S.Zero
    return None