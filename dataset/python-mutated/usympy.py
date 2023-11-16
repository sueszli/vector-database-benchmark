""" SymPy interface to Unification engine

See sympy.unify for module level docstring
See sympy.unify.core for algorithmic docstring """
from sympy.core import Basic, Add, Mul, Pow
from sympy.core.operations import AssocOp, LatticeOp
from sympy.matrices import MatAdd, MatMul, MatrixExpr
from sympy.sets.sets import Union, Intersection, FiniteSet
from sympy.unify.core import Compound, Variable, CondVariable
from sympy.unify import core
basic_new_legal = [MatrixExpr]
eval_false_legal = [AssocOp, Pow, FiniteSet]
illegal = [LatticeOp]

def sympy_associative(op):
    if False:
        print('Hello World!')
    assoc_ops = (AssocOp, MatAdd, MatMul, Union, Intersection, FiniteSet)
    return any((issubclass(op, aop) for aop in assoc_ops))

def sympy_commutative(op):
    if False:
        print('Hello World!')
    comm_ops = (Add, MatAdd, Union, Intersection, FiniteSet)
    return any((issubclass(op, cop) for cop in comm_ops))

def is_associative(x):
    if False:
        while True:
            i = 10
    return isinstance(x, Compound) and sympy_associative(x.op)

def is_commutative(x):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(x, Compound):
        return False
    if sympy_commutative(x.op):
        return True
    if issubclass(x.op, Mul):
        return all((construct(arg).is_commutative for arg in x.args))

def mk_matchtype(typ):
    if False:
        while True:
            i = 10

    def matchtype(x):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(x, typ) or (isinstance(x, Compound) and issubclass(x.op, typ))
    return matchtype

def deconstruct(s, variables=()):
    if False:
        for i in range(10):
            print('nop')
    ' Turn a SymPy object into a Compound '
    if s in variables:
        return Variable(s)
    if isinstance(s, (Variable, CondVariable)):
        return s
    if not isinstance(s, Basic) or s.is_Atom:
        return s
    return Compound(s.__class__, tuple((deconstruct(arg, variables) for arg in s.args)))

def construct(t):
    if False:
        for i in range(10):
            print('nop')
    ' Turn a Compound into a SymPy object '
    if isinstance(t, (Variable, CondVariable)):
        return t.arg
    if not isinstance(t, Compound):
        return t
    if any((issubclass(t.op, cls) for cls in eval_false_legal)):
        return t.op(*map(construct, t.args), evaluate=False)
    elif any((issubclass(t.op, cls) for cls in basic_new_legal)):
        return Basic.__new__(t.op, *map(construct, t.args))
    else:
        return t.op(*map(construct, t.args))

def rebuild(s):
    if False:
        while True:
            i = 10
    ' Rebuild a SymPy expression.\n\n    This removes harm caused by Expr-Rules interactions.\n    '
    return construct(deconstruct(s))

def unify(x, y, s=None, variables=(), **kwargs):
    if False:
        i = 10
        return i + 15
    ' Structural unification of two expressions/patterns.\n\n    Examples\n    ========\n\n    >>> from sympy.unify.usympy import unify\n    >>> from sympy import Basic, S\n    >>> from sympy.abc import x, y, z, p, q\n\n    >>> next(unify(Basic(S(1), S(2)), Basic(S(1), x), variables=[x]))\n    {x: 2}\n\n    >>> expr = 2*x + y + z\n    >>> pattern = 2*p + q\n    >>> next(unify(expr, pattern, {}, variables=(p, q)))\n    {p: x, q: y + z}\n\n    Unification supports commutative and associative matching\n\n    >>> expr = x + y + z\n    >>> pattern = p + q\n    >>> len(list(unify(expr, pattern, {}, variables=(p, q))))\n    12\n\n    Symbols not indicated to be variables are treated as literal,\n    else they are wild-like and match anything in a sub-expression.\n\n    >>> expr = x*y*z + 3\n    >>> pattern = x*y + 3\n    >>> next(unify(expr, pattern, {}, variables=[x, y]))\n    {x: y, y: x*z}\n\n    The x and y of the pattern above were in a Mul and matched factors\n    in the Mul of expr. Here, a single symbol matches an entire term:\n\n    >>> expr = x*y + 3\n    >>> pattern = p + 3\n    >>> next(unify(expr, pattern, {}, variables=[p]))\n    {p: x*y}\n\n    '
    decons = lambda x: deconstruct(x, variables)
    s = s or {}
    s = {decons(k): decons(v) for (k, v) in s.items()}
    ds = core.unify(decons(x), decons(y), s, is_associative=is_associative, is_commutative=is_commutative, **kwargs)
    for d in ds:
        yield {construct(k): construct(v) for (k, v) in d.items()}