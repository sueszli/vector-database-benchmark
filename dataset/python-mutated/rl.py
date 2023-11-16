""" Generic Rules for SymPy

This file assumes knowledge of Basic and little else.
"""
from sympy.utilities.iterables import sift
from .util import new

def rm_id(isid, new=new):
    if False:
        return 10
    ' Create a rule to remove identities.\n\n    isid - fn :: x -> Bool  --- whether or not this element is an identity.\n\n    Examples\n    ========\n\n    >>> from sympy.strategies import rm_id\n    >>> from sympy import Basic, S\n    >>> remove_zeros = rm_id(lambda x: x==0)\n    >>> remove_zeros(Basic(S(1), S(0), S(2)))\n    Basic(1, 2)\n    >>> remove_zeros(Basic(S(0), S(0))) # If only identites then we keep one\n    Basic(0)\n\n    See Also:\n        unpack\n    '

    def ident_remove(expr):
        if False:
            for i in range(10):
                print('nop')
        ' Remove identities '
        ids = list(map(isid, expr.args))
        if sum(ids) == 0:
            return expr
        elif sum(ids) != len(ids):
            return new(expr.__class__, *[arg for (arg, x) in zip(expr.args, ids) if not x])
        else:
            return new(expr.__class__, expr.args[0])
    return ident_remove

def glom(key, count, combine):
    if False:
        return 10
    ' Create a rule to conglomerate identical args.\n\n    Examples\n    ========\n\n    >>> from sympy.strategies import glom\n    >>> from sympy import Add\n    >>> from sympy.abc import x\n\n    >>> key     = lambda x: x.as_coeff_Mul()[1]\n    >>> count   = lambda x: x.as_coeff_Mul()[0]\n    >>> combine = lambda cnt, arg: cnt * arg\n    >>> rl = glom(key, count, combine)\n\n    >>> rl(Add(x, -x, 3*x, 2, 3, evaluate=False))\n    3*x + 5\n\n    Wait, how are key, count and combine supposed to work?\n\n    >>> key(2*x)\n    x\n    >>> count(2*x)\n    2\n    >>> combine(2, x)\n    2*x\n    '

    def conglomerate(expr):
        if False:
            print('Hello World!')
        ' Conglomerate together identical args x + x -> 2x '
        groups = sift(expr.args, key)
        counts = {k: sum(map(count, args)) for (k, args) in groups.items()}
        newargs = [combine(cnt, mat) for (mat, cnt) in counts.items()]
        if set(newargs) != set(expr.args):
            return new(type(expr), *newargs)
        else:
            return expr
    return conglomerate

def sort(key, new=new):
    if False:
        i = 10
        return i + 15
    ' Create a rule to sort by a key function.\n\n    Examples\n    ========\n\n    >>> from sympy.strategies import sort\n    >>> from sympy import Basic, S\n    >>> sort_rl = sort(str)\n    >>> sort_rl(Basic(S(3), S(1), S(2)))\n    Basic(1, 2, 3)\n    '

    def sort_rl(expr):
        if False:
            return 10
        return new(expr.__class__, *sorted(expr.args, key=key))
    return sort_rl

def distribute(A, B):
    if False:
        print('Hello World!')
    " Turns an A containing Bs into a B of As\n\n    where A, B are container types\n\n    >>> from sympy.strategies import distribute\n    >>> from sympy import Add, Mul, symbols\n    >>> x, y = symbols('x,y')\n    >>> dist = distribute(Mul, Add)\n    >>> expr = Mul(2, x+y, evaluate=False)\n    >>> expr\n    2*(x + y)\n    >>> dist(expr)\n    2*x + 2*y\n    "

    def distribute_rl(expr):
        if False:
            return 10
        for (i, arg) in enumerate(expr.args):
            if isinstance(arg, B):
                (first, b, tail) = (expr.args[:i], expr.args[i], expr.args[i + 1:])
                return B(*[A(*first + (arg,) + tail) for arg in b.args])
        return expr
    return distribute_rl

def subs(a, b):
    if False:
        return 10
    ' Replace expressions exactly '

    def subs_rl(expr):
        if False:
            print('Hello World!')
        if expr == a:
            return b
        else:
            return expr
    return subs_rl

def unpack(expr):
    if False:
        for i in range(10):
            print('nop')
    ' Rule to unpack singleton args\n\n    >>> from sympy.strategies import unpack\n    >>> from sympy import Basic, S\n    >>> unpack(Basic(S(2)))\n    2\n    '
    if len(expr.args) == 1:
        return expr.args[0]
    else:
        return expr

def flatten(expr, new=new):
    if False:
        return 10
    ' Flatten T(a, b, T(c, d), T2(e)) to T(a, b, c, d, T2(e)) '
    cls = expr.__class__
    args = []
    for arg in expr.args:
        if arg.__class__ == cls:
            args.extend(arg.args)
        else:
            args.append(arg)
    return new(expr.__class__, *args)

def rebuild(expr):
    if False:
        while True:
            i = 10
    ' Rebuild a SymPy tree.\n\n    Explanation\n    ===========\n\n    This function recursively calls constructors in the expression tree.\n    This forces canonicalization and removes ugliness introduced by the use of\n    Basic.__new__\n    '
    if expr.is_Atom:
        return expr
    else:
        return expr.func(*list(map(rebuild, expr.args)))