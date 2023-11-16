"""
Boolean algebra module for SymPy
"""
from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent

def as_Boolean(e):
    if False:
        while True:
            i = 10
    'Like ``bool``, return the Boolean value of an expression, e,\n    which can be any instance of :py:class:`~.Boolean` or ``bool``.\n\n    Examples\n    ========\n\n    >>> from sympy import true, false, nan\n    >>> from sympy.logic.boolalg import as_Boolean\n    >>> from sympy.abc import x\n    >>> as_Boolean(0) is false\n    True\n    >>> as_Boolean(1) is true\n    True\n    >>> as_Boolean(x)\n    x\n    >>> as_Boolean(2)\n    Traceback (most recent call last):\n    ...\n    TypeError: expecting bool or Boolean, not `2`.\n    >>> as_Boolean(nan)\n    Traceback (most recent call last):\n    ...\n    TypeError: expecting bool or Boolean, not `nan`.\n\n    '
    from sympy.core.symbol import Symbol
    if e == True:
        return true
    if e == False:
        return false
    if isinstance(e, Symbol):
        z = e.is_zero
        if z is None:
            return e
        return false if z else true
    if isinstance(e, Boolean):
        return e
    raise TypeError('expecting bool or Boolean, not `%s`.' % e)

@sympify_method_args
class Boolean(Basic):
    """A Boolean object is an object for which logic operations make sense."""
    __slots__ = ()
    kind = BooleanKind

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __and__(self, other):
        if False:
            print('Hello World!')
        return And(self, other)
    __rand__ = __and__

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __or__(self, other):
        if False:
            print('Hello World!')
        return Or(self, other)
    __ror__ = __or__

    def __invert__(self):
        if False:
            return 10
        'Overloading for ~'
        return Not(self)

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __rshift__(self, other):
        if False:
            while True:
                i = 10
        return Implies(self, other)

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __lshift__(self, other):
        if False:
            return 10
        return Implies(other, self)
    __rrshift__ = __lshift__
    __rlshift__ = __rshift__

    @sympify_return([('other', 'Boolean')], NotImplemented)
    def __xor__(self, other):
        if False:
            return 10
        return Xor(self, other)
    __rxor__ = __xor__

    def equals(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Returns ``True`` if the given formulas have the same truth table.\n        For two formulas to be equal they must have the same literals.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import A, B, C\n        >>> from sympy import And, Or, Not\n        >>> (A >> B).equals(~B >> ~A)\n        True\n        >>> Not(And(A, B, C)).equals(And(Not(A), Not(B), Not(C)))\n        False\n        >>> Not(And(A, Not(A))).equals(Or(B, Not(B)))\n        False\n\n        '
        from sympy.logic.inference import satisfiable
        from sympy.core.relational import Relational
        if self.has(Relational) or other.has(Relational):
            raise NotImplementedError('handling of relationals')
        return self.atoms() == other.atoms() and (not satisfiable(Not(Equivalent(self, other))))

    def to_nnf(self, simplify=True):
        if False:
            i = 10
            return i + 15
        return self

    def as_set(self):
        if False:
            return 10
        "\n        Rewrites Boolean expression in terms of real sets.\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol, Eq, Or, And\n        >>> x = Symbol('x', real=True)\n        >>> Eq(x, 0).as_set()\n        {0}\n        >>> (x > 0).as_set()\n        Interval.open(0, oo)\n        >>> And(-2 < x, x < 2).as_set()\n        Interval.open(-2, 2)\n        >>> Or(x < -2, 2 < x).as_set()\n        Union(Interval.open(-oo, -2), Interval.open(2, oo))\n\n        "
        from sympy.calculus.util import periodicity
        from sympy.core.relational import Relational
        free = self.free_symbols
        if len(free) == 1:
            x = free.pop()
            if x.kind is NumberKind:
                reps = {}
                for r in self.atoms(Relational):
                    if periodicity(r, x) not in (0, None):
                        s = r._eval_as_set()
                        if s in (S.EmptySet, S.UniversalSet, S.Reals):
                            reps[r] = s.as_relational(x)
                            continue
                        raise NotImplementedError(filldedent('\n                            as_set is not implemented for relationals\n                            with periodic solutions\n                            '))
                new = self.subs(reps)
                if new.func != self.func:
                    return new.as_set()
                else:
                    return new._eval_as_set()
            return self._eval_as_set()
        else:
            raise NotImplementedError('Sorry, as_set has not yet been implemented for multivariate expressions')

    @property
    def binary_symbols(self):
        if False:
            return 10
        from sympy.core.relational import Eq, Ne
        return set().union(*[i.binary_symbols for i in self.args if i.is_Boolean or i.is_Symbol or isinstance(i, (Eq, Ne))])

    def _eval_refine(self, assumptions):
        if False:
            while True:
                i = 10
        from sympy.assumptions import ask
        ret = ask(self, assumptions)
        if ret is True:
            return true
        elif ret is False:
            return false
        return None

class BooleanAtom(Boolean):
    """
    Base class of :py:class:`~.BooleanTrue` and :py:class:`~.BooleanFalse`.
    """
    is_Boolean = True
    is_Atom = True
    _op_priority = 11

    def simplify(self, *a, **kw):
        if False:
            print('Hello World!')
        return self

    def expand(self, *a, **kw):
        if False:
            return 10
        return self

    @property
    def canonical(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def _noop(self, other=None):
        if False:
            for i in range(10):
                print('nop')
        raise TypeError('BooleanAtom not allowed in this context.')
    __add__ = _noop
    __radd__ = _noop
    __sub__ = _noop
    __rsub__ = _noop
    __mul__ = _noop
    __rmul__ = _noop
    __pow__ = _noop
    __rpow__ = _noop
    __truediv__ = _noop
    __rtruediv__ = _noop
    __mod__ = _noop
    __rmod__ = _noop
    _eval_power = _noop

    def __lt__(self, other):
        if False:
            print('Hello World!')
        raise TypeError(filldedent('\n            A Boolean argument can only be used in\n            Eq and Ne; all other relationals expect\n            real expressions.\n        '))
    __le__ = __lt__
    __gt__ = __lt__
    __ge__ = __lt__

    def _eval_simplify(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return self

class BooleanTrue(BooleanAtom, metaclass=Singleton):
    """
    SymPy version of ``True``, a singleton that can be accessed via ``S.true``.

    This is the SymPy version of ``True``, for use in the logic module. The
    primary advantage of using ``true`` instead of ``True`` is that shorthand Boolean
    operations like ``~`` and ``>>`` will work as expected on this class, whereas with
    True they act bitwise on 1. Functions in the logic module will return this
    class when they evaluate to true.

    Notes
    =====

    There is liable to be some confusion as to when ``True`` should
    be used and when ``S.true`` should be used in various contexts
    throughout SymPy. An important thing to remember is that
    ``sympify(True)`` returns ``S.true``. This means that for the most
    part, you can just use ``True`` and it will automatically be converted
    to ``S.true`` when necessary, similar to how you can generally use 1
    instead of ``S.One``.

    The rule of thumb is:

    "If the boolean in question can be replaced by an arbitrary symbolic
    ``Boolean``, like ``Or(x, y)`` or ``x > 1``, use ``S.true``.
    Otherwise, use ``True``"

    In other words, use ``S.true`` only on those contexts where the
    boolean is being used as a symbolic representation of truth.
    For example, if the object ends up in the ``.args`` of any expression,
    then it must necessarily be ``S.true`` instead of ``True``, as
    elements of ``.args`` must be ``Basic``. On the other hand,
    ``==`` is not a symbolic operation in SymPy, since it always returns
    ``True`` or ``False``, and does so in terms of structural equality
    rather than mathematical, so it should return ``True``. The assumptions
    system should use ``True`` and ``False``. Aside from not satisfying
    the above rule of thumb, the assumptions system uses a three-valued logic
    (``True``, ``False``, ``None``), whereas ``S.true`` and ``S.false``
    represent a two-valued logic. When in doubt, use ``True``.

    "``S.true == True is True``."

    While "``S.true is True``" is ``False``, "``S.true == True``"
    is ``True``, so if there is any doubt over whether a function or
    expression will return ``S.true`` or ``True``, just use ``==``
    instead of ``is`` to do the comparison, and it will work in either
    case.  Finally, for boolean flags, it's better to just use ``if x``
    instead of ``if x is True``. To quote PEP 8:

    Do not compare boolean values to ``True`` or ``False``
    using ``==``.

    * Yes:   ``if greeting:``
    * No:    ``if greeting == True:``
    * Worse: ``if greeting is True:``

    Examples
    ========

    >>> from sympy import sympify, true, false, Or
    >>> sympify(True)
    True
    >>> _ is True, _ is true
    (False, True)

    >>> Or(true, false)
    True
    >>> _ is true
    True

    Python operators give a boolean result for true but a
    bitwise result for True

    >>> ~true, ~True
    (False, -2)
    >>> true >> true, True >> True
    (True, 0)

    Python operators give a boolean result for true but a
    bitwise result for True

    >>> ~true, ~True
    (False, -2)
    >>> true >> true, True >> True
    (True, 0)

    See Also
    ========

    sympy.logic.boolalg.BooleanFalse

    """

    def __bool__(self):
        if False:
            return 10
        return True

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(True)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other is True:
            return True
        if other is False:
            return False
        return super().__eq__(other)

    @property
    def negated(self):
        if False:
            while True:
                i = 10
        return false

    def as_set(self):
        if False:
            while True:
                i = 10
        '\n        Rewrite logic operators and relationals in terms of real sets.\n\n        Examples\n        ========\n\n        >>> from sympy import true\n        >>> true.as_set()\n        UniversalSet\n\n        '
        return S.UniversalSet

class BooleanFalse(BooleanAtom, metaclass=Singleton):
    """
    SymPy version of ``False``, a singleton that can be accessed via ``S.false``.

    This is the SymPy version of ``False``, for use in the logic module. The
    primary advantage of using ``false`` instead of ``False`` is that shorthand
    Boolean operations like ``~`` and ``>>`` will work as expected on this class,
    whereas with ``False`` they act bitwise on 0. Functions in the logic module
    will return this class when they evaluate to false.

    Notes
    ======

    See the notes section in :py:class:`sympy.logic.boolalg.BooleanTrue`

    Examples
    ========

    >>> from sympy import sympify, true, false, Or
    >>> sympify(False)
    False
    >>> _ is False, _ is false
    (False, True)

    >>> Or(true, false)
    True
    >>> _ is true
    True

    Python operators give a boolean result for false but a
    bitwise result for False

    >>> ~false, ~False
    (True, -1)
    >>> false >> false, False >> False
    (True, 0)

    See Also
    ========

    sympy.logic.boolalg.BooleanTrue

    """

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(False)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other is True:
            return False
        if other is False:
            return True
        return super().__eq__(other)

    @property
    def negated(self):
        if False:
            while True:
                i = 10
        return true

    def as_set(self):
        if False:
            while True:
                i = 10
        '\n        Rewrite logic operators and relationals in terms of real sets.\n\n        Examples\n        ========\n\n        >>> from sympy import false\n        >>> false.as_set()\n        EmptySet\n        '
        return S.EmptySet
true = BooleanTrue()
false = BooleanFalse()
S.true = true
S.false = false
_sympy_converter[bool] = lambda x: true if x else false

class BooleanFunction(Application, Boolean):
    """Boolean function is a function that lives in a boolean space
    It is used as base class for :py:class:`~.And`, :py:class:`~.Or`,
    :py:class:`~.Not`, etc.
    """
    is_Boolean = True

    def _eval_simplify(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        rv = simplify_univariate(self)
        if not isinstance(rv, BooleanFunction):
            return rv.simplify(**kwargs)
        rv = rv.func(*[a.simplify(**kwargs) for a in rv.args])
        return simplify_logic(rv)

    def simplify(self, **kwargs):
        if False:
            i = 10
            return i + 15
        from sympy.simplify.simplify import simplify
        return simplify(self, **kwargs)

    def __lt__(self, other):
        if False:
            return 10
        raise TypeError(filldedent('\n            A Boolean argument can only be used in\n            Eq and Ne; all other relationals expect\n            real expressions.\n        '))
    __le__ = __lt__
    __ge__ = __lt__
    __gt__ = __lt__

    @classmethod
    def binary_check_and_simplify(self, *args):
        if False:
            while True:
                i = 10
        return [as_Boolean(i) for i in args]

    def to_nnf(self, simplify=True):
        if False:
            i = 10
            return i + 15
        return self._to_nnf(*self.args, simplify=simplify)

    def to_anf(self, deep=True):
        if False:
            print('Hello World!')
        return self._to_anf(*self.args, deep=deep)

    @classmethod
    def _to_nnf(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        simplify = kwargs.get('simplify', True)
        argset = set()
        for arg in args:
            if not is_literal(arg):
                arg = arg.to_nnf(simplify)
            if simplify:
                if isinstance(arg, cls):
                    arg = arg.args
                else:
                    arg = (arg,)
                for a in arg:
                    if Not(a) in argset:
                        return cls.zero
                    argset.add(a)
            else:
                argset.add(arg)
        return cls(*argset)

    @classmethod
    def _to_anf(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        deep = kwargs.get('deep', True)
        argset = set()
        for arg in args:
            if deep:
                if not is_literal(arg) or isinstance(arg, Not):
                    arg = arg.to_anf(deep=deep)
                argset.add(arg)
            else:
                argset.add(arg)
        return cls(*argset, remove_true=False)

    def diff(self, *symbols, **assumptions):
        if False:
            print('Hello World!')
        assumptions.setdefault('evaluate', True)
        return Derivative(self, *symbols, **assumptions)

    def _eval_derivative(self, x):
        if False:
            while True:
                i = 10
        if x in self.binary_symbols:
            from sympy.core.relational import Eq
            from sympy.functions.elementary.piecewise import Piecewise
            return Piecewise((0, Eq(self.subs(x, 0), self.subs(x, 1))), (1, True))
        elif x in self.free_symbols:
            pass
        else:
            return S.Zero

class And(LatticeOp, BooleanFunction):
    """
    Logical AND function.

    It evaluates its arguments in order, returning false immediately
    when an argument is false and true if they are all true.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import And
    >>> x & y
    x & y

    Notes
    =====

    The ``&`` operator is provided as a convenience, but note that its use
    here is different from its normal use in Python, which is bitwise
    and. Hence, ``And(a, b)`` and ``a & b`` will produce different results if
    ``a`` and ``b`` are integers.

    >>> And(x, y).subs(x, 1)
    y

    """
    zero = false
    identity = true
    nargs = None

    @classmethod
    def _new_args_filter(cls, args):
        if False:
            while True:
                i = 10
        args = BooleanFunction.binary_check_and_simplify(*args)
        args = LatticeOp._new_args_filter(args, And)
        newargs = []
        rel = set()
        for x in ordered(args):
            if x.is_Relational:
                c = x.canonical
                if c in rel:
                    continue
                elif c.negated.canonical in rel:
                    return [false]
                else:
                    rel.add(c)
            newargs.append(x)
        return newargs

    def _eval_subs(self, old, new):
        if False:
            return 10
        args = []
        bad = None
        for i in self.args:
            try:
                i = i.subs(old, new)
            except TypeError:
                if bad is None:
                    bad = i
                continue
            if i == False:
                return false
            elif i != True:
                args.append(i)
        if bad is not None:
            bad.subs(old, new)
        if isinstance(old, And):
            old_set = set(old.args)
            if old_set.issubset(args):
                args = set(args) - old_set
                args.add(new)
        return self.func(*args)

    def _eval_simplify(self, **kwargs):
        if False:
            while True:
                i = 10
        from sympy.core.relational import Equality, Relational
        from sympy.solvers.solveset import linear_coeffs
        rv = super()._eval_simplify(**kwargs)
        if not isinstance(rv, And):
            return rv
        (Rel, nonRel) = sift(rv.args, lambda i: isinstance(i, Relational), binary=True)
        if not Rel:
            return rv
        (eqs, other) = sift(Rel, lambda i: isinstance(i, Equality), binary=True)
        measure = kwargs['measure']
        if eqs:
            ratio = kwargs['ratio']
            reps = {}
            sifted = {}
            sifted = sift(ordered([(i.free_symbols, i) for i in eqs]), lambda x: len(x[0]))
            eqs = []
            nonlineqs = []
            while 1 in sifted:
                for (free, e) in sifted.pop(1):
                    x = free.pop()
                    if (e.lhs != x or x in e.rhs.free_symbols) and x not in reps:
                        try:
                            (m, b) = linear_coeffs(Add(e.lhs, -e.rhs, evaluate=False), x)
                            enew = e.func(x, -b / m)
                            if measure(enew) <= ratio * measure(e):
                                e = enew
                            else:
                                eqs.append(e)
                                continue
                        except ValueError:
                            pass
                    if x in reps:
                        eqs.append(e.subs(x, reps[x]))
                    elif e.lhs == x and x not in e.rhs.free_symbols:
                        reps[x] = e.rhs
                        eqs.append(e)
                    else:
                        nonlineqs.append(e)
                resifted = defaultdict(list)
                for k in sifted:
                    for (f, e) in sifted[k]:
                        e = e.xreplace(reps)
                        f = e.free_symbols
                        resifted[len(f)].append((f, e))
                sifted = resifted
            for k in sifted:
                eqs.extend([e for (f, e) in sifted[k]])
            nonlineqs = [ei.subs(reps) for ei in nonlineqs]
            other = [ei.subs(reps) for ei in other]
            rv = rv.func(*[i.canonical for i in eqs + nonlineqs + other] + nonRel)
        patterns = _simplify_patterns_and()
        threeterm_patterns = _simplify_patterns_and3()
        return _apply_patternbased_simplification(rv, patterns, measure, false, threeterm_patterns=threeterm_patterns)

    def _eval_as_set(self):
        if False:
            for i in range(10):
                print('nop')
        from sympy.sets.sets import Intersection
        return Intersection(*[arg.as_set() for arg in self.args])

    def _eval_rewrite_as_Nor(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return Nor(*[Not(arg) for arg in self.args])

    def to_anf(self, deep=True):
        if False:
            for i in range(10):
                print('nop')
        if deep:
            result = And._to_anf(*self.args, deep=deep)
            return distribute_xor_over_and(result)
        return self

class Or(LatticeOp, BooleanFunction):
    """
    Logical OR function

    It evaluates its arguments in order, returning true immediately
    when an  argument is true, and false if they are all false.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Or
    >>> x | y
    x | y

    Notes
    =====

    The ``|`` operator is provided as a convenience, but note that its use
    here is different from its normal use in Python, which is bitwise
    or. Hence, ``Or(a, b)`` and ``a | b`` will return different things if
    ``a`` and ``b`` are integers.

    >>> Or(x, y).subs(x, 0)
    y

    """
    zero = true
    identity = false

    @classmethod
    def _new_args_filter(cls, args):
        if False:
            return 10
        newargs = []
        rel = []
        args = BooleanFunction.binary_check_and_simplify(*args)
        for x in args:
            if x.is_Relational:
                c = x.canonical
                if c in rel:
                    continue
                nc = c.negated.canonical
                if any((r == nc for r in rel)):
                    return [true]
                rel.append(c)
            newargs.append(x)
        return LatticeOp._new_args_filter(newargs, Or)

    def _eval_subs(self, old, new):
        if False:
            return 10
        args = []
        bad = None
        for i in self.args:
            try:
                i = i.subs(old, new)
            except TypeError:
                if bad is None:
                    bad = i
                continue
            if i == True:
                return true
            elif i != False:
                args.append(i)
        if bad is not None:
            bad.subs(old, new)
        if isinstance(old, Or):
            old_set = set(old.args)
            if old_set.issubset(args):
                args = set(args) - old_set
                args.add(new)
        return self.func(*args)

    def _eval_as_set(self):
        if False:
            return 10
        from sympy.sets.sets import Union
        return Union(*[arg.as_set() for arg in self.args])

    def _eval_rewrite_as_Nand(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return Nand(*[Not(arg) for arg in self.args])

    def _eval_simplify(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        from sympy.core.relational import Le, Ge, Eq
        lege = self.atoms(Le, Ge)
        if lege:
            reps = {i: self.func(Eq(i.lhs, i.rhs), i.strict) for i in lege}
            return self.xreplace(reps)._eval_simplify(**kwargs)
        rv = super()._eval_simplify(**kwargs)
        if not isinstance(rv, Or):
            return rv
        patterns = _simplify_patterns_or()
        return _apply_patternbased_simplification(rv, patterns, kwargs['measure'], true)

    def to_anf(self, deep=True):
        if False:
            for i in range(10):
                print('nop')
        args = range(1, len(self.args) + 1)
        args = (combinations(self.args, j) for j in args)
        args = chain.from_iterable(args)
        args = (And(*arg) for arg in args)
        args = (to_anf(x, deep=deep) if deep else x for x in args)
        return Xor(*list(args), remove_true=False)

class Not(BooleanFunction):
    """
    Logical Not function (negation)


    Returns ``true`` if the statement is ``false`` or ``False``.
    Returns ``false`` if the statement is ``true`` or ``True``.

    Examples
    ========

    >>> from sympy import Not, And, Or
    >>> from sympy.abc import x, A, B
    >>> Not(True)
    False
    >>> Not(False)
    True
    >>> Not(And(True, False))
    True
    >>> Not(Or(True, False))
    False
    >>> Not(And(And(True, x), Or(x, False)))
    ~x
    >>> ~x
    ~x
    >>> Not(And(Or(A, B), Or(~A, ~B)))
    ~((A | B) & (~A | ~B))

    Notes
    =====

    - The ``~`` operator is provided as a convenience, but note that its use
      here is different from its normal use in Python, which is bitwise
      not. In particular, ``~a`` and ``Not(a)`` will be different if ``a`` is
      an integer. Furthermore, since bools in Python subclass from ``int``,
      ``~True`` is the same as ``~1`` which is ``-2``, which has a boolean
      value of True.  To avoid this issue, use the SymPy boolean types
      ``true`` and ``false``.

    >>> from sympy import true
    >>> ~True
    -2
    >>> ~true
    False

    """
    is_Not = True

    @classmethod
    def eval(cls, arg):
        if False:
            print('Hello World!')
        if isinstance(arg, Number) or arg in (True, False):
            return false if arg else true
        if arg.is_Not:
            return arg.args[0]
        if arg.is_Relational:
            return arg.negated

    def _eval_as_set(self):
        if False:
            while True:
                i = 10
        "\n        Rewrite logic operators and relationals in terms of real sets.\n\n        Examples\n        ========\n\n        >>> from sympy import Not, Symbol\n        >>> x = Symbol('x')\n        >>> Not(x > 0).as_set()\n        Interval(-oo, 0)\n        "
        return self.args[0].as_set().complement(S.Reals)

    def to_nnf(self, simplify=True):
        if False:
            return 10
        if is_literal(self):
            return self
        expr = self.args[0]
        (func, args) = (expr.func, expr.args)
        if func == And:
            return Or._to_nnf(*[Not(arg) for arg in args], simplify=simplify)
        if func == Or:
            return And._to_nnf(*[Not(arg) for arg in args], simplify=simplify)
        if func == Implies:
            (a, b) = args
            return And._to_nnf(a, Not(b), simplify=simplify)
        if func == Equivalent:
            return And._to_nnf(Or(*args), Or(*[Not(arg) for arg in args]), simplify=simplify)
        if func == Xor:
            result = []
            for i in range(1, len(args) + 1, 2):
                for neg in combinations(args, i):
                    clause = [Not(s) if s in neg else s for s in args]
                    result.append(Or(*clause))
            return And._to_nnf(*result, simplify=simplify)
        if func == ITE:
            (a, b, c) = args
            return And._to_nnf(Or(a, Not(c)), Or(Not(a), Not(b)), simplify=simplify)
        raise ValueError('Illegal operator %s in expression' % func)

    def to_anf(self, deep=True):
        if False:
            print('Hello World!')
        return Xor._to_anf(true, self.args[0], deep=deep)

class Xor(BooleanFunction):
    """
    Logical XOR (exclusive OR) function.


    Returns True if an odd number of the arguments are True and the rest are
    False.

    Returns False if an even number of the arguments are True and the rest are
    False.

    Examples
    ========

    >>> from sympy.logic.boolalg import Xor
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> Xor(True, False)
    True
    >>> Xor(True, True)
    False
    >>> Xor(True, False, True, True, False)
    True
    >>> Xor(True, False, True, False)
    False
    >>> x ^ y
    x ^ y

    Notes
    =====

    The ``^`` operator is provided as a convenience, but note that its use
    here is different from its normal use in Python, which is bitwise xor. In
    particular, ``a ^ b`` and ``Xor(a, b)`` will be different if ``a`` and
    ``b`` are integers.

    >>> Xor(x, y).subs(y, 0)
    x

    """

    def __new__(cls, *args, remove_true=True, **kwargs):
        if False:
            print('Hello World!')
        argset = set()
        obj = super().__new__(cls, *args, **kwargs)
        for arg in obj._args:
            if isinstance(arg, Number) or arg in (True, False):
                if arg:
                    arg = true
                else:
                    continue
            if isinstance(arg, Xor):
                for a in arg.args:
                    argset.remove(a) if a in argset else argset.add(a)
            elif arg in argset:
                argset.remove(arg)
            else:
                argset.add(arg)
        rel = [(r, r.canonical, r.negated.canonical) for r in argset if r.is_Relational]
        odd = False
        remove = []
        for (i, (r, c, nc)) in enumerate(rel):
            for j in range(i + 1, len(rel)):
                (rj, cj) = rel[j][:2]
                if cj == nc:
                    odd = not odd
                    break
                elif cj == c:
                    break
            else:
                continue
            remove.append((r, rj))
        if odd:
            argset.remove(true) if true in argset else argset.add(true)
        for (a, b) in remove:
            argset.remove(a)
            argset.remove(b)
        if len(argset) == 0:
            return false
        elif len(argset) == 1:
            return argset.pop()
        elif True in argset and remove_true:
            argset.remove(True)
            return Not(Xor(*argset))
        else:
            obj._args = tuple(ordered(argset))
            obj._argset = frozenset(argset)
            return obj

    @property
    @cacheit
    def args(self):
        if False:
            while True:
                i = 10
        return tuple(ordered(self._argset))

    def to_nnf(self, simplify=True):
        if False:
            i = 10
            return i + 15
        args = []
        for i in range(0, len(self.args) + 1, 2):
            for neg in combinations(self.args, i):
                clause = [Not(s) if s in neg else s for s in self.args]
                args.append(Or(*clause))
        return And._to_nnf(*args, simplify=simplify)

    def _eval_rewrite_as_Or(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        a = self.args
        return Or(*[_convert_to_varsSOP(x, self.args) for x in _get_odd_parity_terms(len(a))])

    def _eval_rewrite_as_And(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        a = self.args
        return And(*[_convert_to_varsPOS(x, self.args) for x in _get_even_parity_terms(len(a))])

    def _eval_simplify(self, **kwargs):
        if False:
            print('Hello World!')
        rv = self.func(*[a.simplify(**kwargs) for a in self.args])
        if not isinstance(rv, Xor):
            return rv
        patterns = _simplify_patterns_xor()
        return _apply_patternbased_simplification(rv, patterns, kwargs['measure'], None)

    def _eval_subs(self, old, new):
        if False:
            i = 10
            return i + 15
        if isinstance(old, Xor):
            old_set = set(old.args)
            if old_set.issubset(self.args):
                args = set(self.args) - old_set
                args.add(new)
                return self.func(*args)

class Nand(BooleanFunction):
    """
    Logical NAND function.

    It evaluates its arguments in order, giving True immediately if any
    of them are False, and False if they are all True.

    Returns True if any of the arguments are False
    Returns False if all arguments are True

    Examples
    ========

    >>> from sympy.logic.boolalg import Nand
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> Nand(False, True)
    True
    >>> Nand(True, True)
    False
    >>> Nand(x, y)
    ~(x & y)

    """

    @classmethod
    def eval(cls, *args):
        if False:
            return 10
        return Not(And(*args))

class Nor(BooleanFunction):
    """
    Logical NOR function.

    It evaluates its arguments in order, giving False immediately if any
    of them are True, and True if they are all False.

    Returns False if any argument is True
    Returns True if all arguments are False

    Examples
    ========

    >>> from sympy.logic.boolalg import Nor
    >>> from sympy import symbols
    >>> x, y = symbols('x y')

    >>> Nor(True, False)
    False
    >>> Nor(True, True)
    False
    >>> Nor(False, True)
    False
    >>> Nor(False, False)
    True
    >>> Nor(x, y)
    ~(x | y)

    """

    @classmethod
    def eval(cls, *args):
        if False:
            for i in range(10):
                print('nop')
        return Not(Or(*args))

class Xnor(BooleanFunction):
    """
    Logical XNOR function.

    Returns False if an odd number of the arguments are True and the rest are
    False.

    Returns True if an even number of the arguments are True and the rest are
    False.

    Examples
    ========

    >>> from sympy.logic.boolalg import Xnor
    >>> from sympy import symbols
    >>> x, y = symbols('x y')
    >>> Xnor(True, False)
    False
    >>> Xnor(True, True)
    True
    >>> Xnor(True, False, True, True, False)
    False
    >>> Xnor(True, False, True, False)
    True

    """

    @classmethod
    def eval(cls, *args):
        if False:
            print('Hello World!')
        return Not(Xor(*args))

class Implies(BooleanFunction):
    """
    Logical implication.

    A implies B is equivalent to if A then B. Mathematically, it is written
    as `A \\Rightarrow B` and is equivalent to `\\neg A \\vee B` or ``~A | B``.

    Accepts two Boolean arguments; A and B.
    Returns False if A is True and B is False
    Returns True otherwise.

    Examples
    ========

    >>> from sympy.logic.boolalg import Implies
    >>> from sympy import symbols
    >>> x, y = symbols('x y')

    >>> Implies(True, False)
    False
    >>> Implies(False, False)
    True
    >>> Implies(True, True)
    True
    >>> Implies(False, True)
    True
    >>> x >> y
    Implies(x, y)
    >>> y << x
    Implies(x, y)

    Notes
    =====

    The ``>>`` and ``<<`` operators are provided as a convenience, but note
    that their use here is different from their normal use in Python, which is
    bit shifts. Hence, ``Implies(a, b)`` and ``a >> b`` will return different
    things if ``a`` and ``b`` are integers.  In particular, since Python
    considers ``True`` and ``False`` to be integers, ``True >> True`` will be
    the same as ``1 >> 1``, i.e., 0, which has a truth value of False.  To
    avoid this issue, use the SymPy objects ``true`` and ``false``.

    >>> from sympy import true, false
    >>> True >> False
    1
    >>> true >> false
    False

    """

    @classmethod
    def eval(cls, *args):
        if False:
            while True:
                i = 10
        try:
            newargs = []
            for x in args:
                if isinstance(x, Number) or x in (0, 1):
                    newargs.append(bool(x))
                else:
                    newargs.append(x)
            (A, B) = newargs
        except ValueError:
            raise ValueError('%d operand(s) used for an Implies (pairs are required): %s' % (len(args), str(args)))
        if A in (True, False) or B in (True, False):
            return Or(Not(A), B)
        elif A == B:
            return true
        elif A.is_Relational and B.is_Relational:
            if A.canonical == B.canonical:
                return true
            if A.negated.canonical == B.canonical:
                return B
        else:
            return Basic.__new__(cls, *args)

    def to_nnf(self, simplify=True):
        if False:
            while True:
                i = 10
        (a, b) = self.args
        return Or._to_nnf(Not(a), b, simplify=simplify)

    def to_anf(self, deep=True):
        if False:
            print('Hello World!')
        (a, b) = self.args
        return Xor._to_anf(true, a, And(a, b), deep=deep)

class Equivalent(BooleanFunction):
    """
    Equivalence relation.

    ``Equivalent(A, B)`` is True iff A and B are both True or both False.

    Returns True if all of the arguments are logically equivalent.
    Returns False otherwise.

    For two arguments, this is equivalent to :py:class:`~.Xnor`.

    Examples
    ========

    >>> from sympy.logic.boolalg import Equivalent, And
    >>> from sympy.abc import x
    >>> Equivalent(False, False, False)
    True
    >>> Equivalent(True, False, False)
    False
    >>> Equivalent(x, And(x, True))
    True

    """

    def __new__(cls, *args, **options):
        if False:
            print('Hello World!')
        from sympy.core.relational import Relational
        args = [_sympify(arg) for arg in args]
        argset = set(args)
        for x in args:
            if isinstance(x, Number) or x in [True, False]:
                argset.discard(x)
                argset.add(bool(x))
        rel = []
        for r in argset:
            if isinstance(r, Relational):
                rel.append((r, r.canonical, r.negated.canonical))
        remove = []
        for (i, (r, c, nc)) in enumerate(rel):
            for j in range(i + 1, len(rel)):
                (rj, cj) = rel[j][:2]
                if cj == nc:
                    return false
                elif cj == c:
                    remove.append((r, rj))
                    break
        for (a, b) in remove:
            argset.remove(a)
            argset.remove(b)
            argset.add(True)
        if len(argset) <= 1:
            return true
        if True in argset:
            argset.discard(True)
            return And(*argset)
        if False in argset:
            argset.discard(False)
            return And(*[Not(arg) for arg in argset])
        _args = frozenset(argset)
        obj = super().__new__(cls, _args)
        obj._argset = _args
        return obj

    @property
    @cacheit
    def args(self):
        if False:
            print('Hello World!')
        return tuple(ordered(self._argset))

    def to_nnf(self, simplify=True):
        if False:
            while True:
                i = 10
        args = []
        for (a, b) in zip(self.args, self.args[1:]):
            args.append(Or(Not(a), b))
        args.append(Or(Not(self.args[-1]), self.args[0]))
        return And._to_nnf(*args, simplify=simplify)

    def to_anf(self, deep=True):
        if False:
            for i in range(10):
                print('nop')
        a = And(*self.args)
        b = And(*[to_anf(Not(arg), deep=False) for arg in self.args])
        b = distribute_xor_over_and(b)
        return Xor._to_anf(a, b, deep=deep)

class ITE(BooleanFunction):
    """
    If-then-else clause.

    ``ITE(A, B, C)`` evaluates and returns the result of B if A is true
    else it returns the result of C. All args must be Booleans.

    From a logic gate perspective, ITE corresponds to a 2-to-1 multiplexer,
    where A is the select signal.

    Examples
    ========

    >>> from sympy.logic.boolalg import ITE, And, Xor, Or
    >>> from sympy.abc import x, y, z
    >>> ITE(True, False, True)
    False
    >>> ITE(Or(True, False), And(True, True), Xor(True, True))
    True
    >>> ITE(x, y, z)
    ITE(x, y, z)
    >>> ITE(True, x, y)
    x
    >>> ITE(False, x, y)
    y
    >>> ITE(x, y, y)
    y

    Trying to use non-Boolean args will generate a TypeError:

    >>> ITE(True, [], ())
    Traceback (most recent call last):
    ...
    TypeError: expecting bool, Boolean or ITE, not `[]`

    """

    def __new__(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        from sympy.core.relational import Eq, Ne
        if len(args) != 3:
            raise ValueError('expecting exactly 3 args')
        (a, b, c) = args
        if isinstance(a, (Eq, Ne)):
            (b, c) = map(as_Boolean, (b, c))
            bin_syms = set().union(*[i.binary_symbols for i in (b, c)])
            if len(set(a.args) - bin_syms) == 1:
                _a = a
                if a.lhs is true:
                    a = a.rhs
                elif a.rhs is true:
                    a = a.lhs
                elif a.lhs is false:
                    a = Not(a.rhs)
                elif a.rhs is false:
                    a = Not(a.lhs)
                else:
                    a = false
                if isinstance(_a, Ne):
                    a = Not(a)
        else:
            (a, b, c) = BooleanFunction.binary_check_and_simplify(a, b, c)
        rv = None
        if kwargs.get('evaluate', True):
            rv = cls.eval(a, b, c)
        if rv is None:
            rv = BooleanFunction.__new__(cls, a, b, c, evaluate=False)
        return rv

    @classmethod
    def eval(cls, *args):
        if False:
            for i in range(10):
                print('nop')
        from sympy.core.relational import Eq, Ne
        (a, b, c) = args
        if isinstance(a, (Ne, Eq)):
            _a = a
            if true in a.args:
                a = a.lhs if a.rhs is true else a.rhs
            elif false in a.args:
                a = Not(a.lhs) if a.rhs is false else Not(a.rhs)
            else:
                _a = None
            if _a is not None and isinstance(_a, Ne):
                a = Not(a)
        if a is true:
            return b
        if a is false:
            return c
        if b == c:
            return b
        else:
            if b is true and c is false:
                return a
            if b is false and c is true:
                return Not(a)
        if [a, b, c] != args:
            return cls(a, b, c, evaluate=False)

    def to_nnf(self, simplify=True):
        if False:
            while True:
                i = 10
        (a, b, c) = self.args
        return And._to_nnf(Or(Not(a), b), Or(a, c), simplify=simplify)

    def _eval_as_set(self):
        if False:
            return 10
        return self.to_nnf().as_set()

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        if False:
            print('Hello World!')
        from sympy.functions.elementary.piecewise import Piecewise
        return Piecewise((args[1], args[0]), (args[2], True))

class Exclusive(BooleanFunction):
    """
    True if only one or no argument is true.

    ``Exclusive(A, B, C)`` is equivalent to ``~(A & B) & ~(A & C) & ~(B & C)``.

    For two arguments, this is equivalent to :py:class:`~.Xor`.

    Examples
    ========

    >>> from sympy.logic.boolalg import Exclusive
    >>> Exclusive(False, False, False)
    True
    >>> Exclusive(False, True, False)
    True
    >>> Exclusive(False, True, True)
    False

    """

    @classmethod
    def eval(cls, *args):
        if False:
            while True:
                i = 10
        and_args = []
        for (a, b) in combinations(args, 2):
            and_args.append(Not(And(a, b)))
        return And(*and_args)

def conjuncts(expr):
    if False:
        while True:
            i = 10
    'Return a list of the conjuncts in ``expr``.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import conjuncts\n    >>> from sympy.abc import A, B\n    >>> conjuncts(A & B)\n    frozenset({A, B})\n    >>> conjuncts(A | B)\n    frozenset({A | B})\n\n    '
    return And.make_args(expr)

def disjuncts(expr):
    if False:
        return 10
    'Return a list of the disjuncts in ``expr``.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import disjuncts\n    >>> from sympy.abc import A, B\n    >>> disjuncts(A | B)\n    frozenset({A, B})\n    >>> disjuncts(A & B)\n    frozenset({A & B})\n\n    '
    return Or.make_args(expr)

def distribute_and_over_or(expr):
    if False:
        while True:
            i = 10
    '\n    Given a sentence ``expr`` consisting of conjunctions and disjunctions\n    of literals, return an equivalent sentence in CNF.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import distribute_and_over_or, And, Or, Not\n    >>> from sympy.abc import A, B, C\n    >>> distribute_and_over_or(Or(A, And(Not(B), Not(C))))\n    (A | ~B) & (A | ~C)\n\n    '
    return _distribute((expr, And, Or))

def distribute_or_over_and(expr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a sentence ``expr`` consisting of conjunctions and disjunctions\n    of literals, return an equivalent sentence in DNF.\n\n    Note that the output is NOT simplified.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import distribute_or_over_and, And, Or, Not\n    >>> from sympy.abc import A, B, C\n    >>> distribute_or_over_and(And(Or(Not(A), B), C))\n    (B & C) | (C & ~A)\n\n    '
    return _distribute((expr, Or, And))

def distribute_xor_over_and(expr):
    if False:
        return 10
    '\n    Given a sentence ``expr`` consisting of conjunction and\n    exclusive disjunctions of literals, return an\n    equivalent exclusive disjunction.\n\n    Note that the output is NOT simplified.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import distribute_xor_over_and, And, Xor, Not\n    >>> from sympy.abc import A, B, C\n    >>> distribute_xor_over_and(And(Xor(Not(A), B), C))\n    (B & C) ^ (C & ~A)\n    '
    return _distribute((expr, Xor, And))

def _distribute(info):
    if False:
        print('Hello World!')
    '\n    Distributes ``info[1]`` over ``info[2]`` with respect to ``info[0]``.\n    '
    if isinstance(info[0], info[2]):
        for arg in info[0].args:
            if isinstance(arg, info[1]):
                conj = arg
                break
        else:
            return info[0]
        rest = info[2](*[a for a in info[0].args if a is not conj])
        return info[1](*list(map(_distribute, [(info[2](c, rest), info[1], info[2]) for c in conj.args])), remove_true=False)
    elif isinstance(info[0], info[1]):
        return info[1](*list(map(_distribute, [(x, info[1], info[2]) for x in info[0].args])), remove_true=False)
    else:
        return info[0]

def to_anf(expr, deep=True):
    if False:
        return 10
    '\n    Converts expr to Algebraic Normal Form (ANF).\n\n    ANF is a canonical normal form, which means that two\n    equivalent formulas will convert to the same ANF.\n\n    A logical expression is in ANF if it has the form\n\n    .. math:: 1 \\oplus a \\oplus b \\oplus ab \\oplus abc\n\n    i.e. it can be:\n        - purely true,\n        - purely false,\n        - conjunction of variables,\n        - exclusive disjunction.\n\n    The exclusive disjunction can only contain true, variables\n    or conjunction of variables. No negations are permitted.\n\n    If ``deep`` is ``False``, arguments of the boolean\n    expression are considered variables, i.e. only the\n    top-level expression is converted to ANF.\n\n    Examples\n    ========\n    >>> from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent\n    >>> from sympy.logic.boolalg import to_anf\n    >>> from sympy.abc import A, B, C\n    >>> to_anf(Not(A))\n    A ^ True\n    >>> to_anf(And(Or(A, B), Not(C)))\n    A ^ B ^ (A & B) ^ (A & C) ^ (B & C) ^ (A & B & C)\n    >>> to_anf(Implies(Not(A), Equivalent(B, C)), deep=False)\n    True ^ ~A ^ (~A & (Equivalent(B, C)))\n\n    '
    expr = sympify(expr)
    if is_anf(expr):
        return expr
    return expr.to_anf(deep=deep)

def to_nnf(expr, simplify=True):
    if False:
        while True:
            i = 10
    '\n    Converts ``expr`` to Negation Normal Form (NNF).\n\n    A logical expression is in NNF if it\n    contains only :py:class:`~.And`, :py:class:`~.Or` and :py:class:`~.Not`,\n    and :py:class:`~.Not` is applied only to literals.\n    If ``simplify`` is ``True``, the result contains no redundant clauses.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import A, B, C, D\n    >>> from sympy.logic.boolalg import Not, Equivalent, to_nnf\n    >>> to_nnf(Not((~A & ~B) | (C & D)))\n    (A | B) & (~C | ~D)\n    >>> to_nnf(Equivalent(A >> B, B >> A))\n    (A | ~B | (A & ~B)) & (B | ~A | (B & ~A))\n\n    '
    if is_nnf(expr, simplify):
        return expr
    return expr.to_nnf(simplify)

def to_cnf(expr, simplify=False, force=False):
    if False:
        return 10
    '\n    Convert a propositional logical sentence ``expr`` to conjunctive normal\n    form: ``((A | ~B | ...) & (B | C | ...) & ...)``.\n    If ``simplify`` is ``True``, ``expr`` is evaluated to its simplest CNF\n    form using the Quine-McCluskey algorithm; this may take a long\n    time. If there are more than 8 variables the ``force`` flag must be set\n    to ``True`` to simplify (default is ``False``).\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import to_cnf\n    >>> from sympy.abc import A, B, D\n    >>> to_cnf(~(A | B) | D)\n    (D | ~A) & (D | ~B)\n    >>> to_cnf((A | B) & (A | ~A), True)\n    A | B\n\n    '
    expr = sympify(expr)
    if not isinstance(expr, BooleanFunction):
        return expr
    if simplify:
        if not force and len(_find_predicates(expr)) > 8:
            raise ValueError(filldedent('\n            To simplify a logical expression with more\n            than 8 variables may take a long time and requires\n            the use of `force=True`.'))
        return simplify_logic(expr, 'cnf', True, force=force)
    if is_cnf(expr):
        return expr
    expr = eliminate_implications(expr)
    res = distribute_and_over_or(expr)
    return res

def to_dnf(expr, simplify=False, force=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a propositional logical sentence ``expr`` to disjunctive normal\n    form: ``((A & ~B & ...) | (B & C & ...) | ...)``.\n    If ``simplify`` is ``True``, ``expr`` is evaluated to its simplest DNF form using\n    the Quine-McCluskey algorithm; this may take a long\n    time. If there are more than 8 variables, the ``force`` flag must be set to\n    ``True`` to simplify (default is ``False``).\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import to_dnf\n    >>> from sympy.abc import A, B, C\n    >>> to_dnf(B & (A | C))\n    (A & B) | (B & C)\n    >>> to_dnf((A & B) | (A & ~B) | (B & C) | (~B & C), True)\n    A | C\n\n    '
    expr = sympify(expr)
    if not isinstance(expr, BooleanFunction):
        return expr
    if simplify:
        if not force and len(_find_predicates(expr)) > 8:
            raise ValueError(filldedent('\n            To simplify a logical expression with more\n            than 8 variables may take a long time and requires\n            the use of `force=True`.'))
        return simplify_logic(expr, 'dnf', True, force=force)
    if is_dnf(expr):
        return expr
    expr = eliminate_implications(expr)
    return distribute_or_over_and(expr)

def is_anf(expr):
    if False:
        print('Hello World!')
    '\n    Checks if ``expr``  is in Algebraic Normal Form (ANF).\n\n    A logical expression is in ANF if it has the form\n\n    .. math:: 1 \\oplus a \\oplus b \\oplus ab \\oplus abc\n\n    i.e. it is purely true, purely false, conjunction of\n    variables or exclusive disjunction. The exclusive\n    disjunction can only contain true, variables or\n    conjunction of variables. No negations are permitted.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import And, Not, Xor, true, is_anf\n    >>> from sympy.abc import A, B, C\n    >>> is_anf(true)\n    True\n    >>> is_anf(A)\n    True\n    >>> is_anf(And(A, B, C))\n    True\n    >>> is_anf(Xor(A, Not(B)))\n    False\n\n    '
    expr = sympify(expr)
    if is_literal(expr) and (not isinstance(expr, Not)):
        return True
    if isinstance(expr, And):
        for arg in expr.args:
            if not arg.is_Symbol:
                return False
        return True
    elif isinstance(expr, Xor):
        for arg in expr.args:
            if isinstance(arg, And):
                for a in arg.args:
                    if not a.is_Symbol:
                        return False
            elif is_literal(arg):
                if isinstance(arg, Not):
                    return False
            else:
                return False
        return True
    else:
        return False

def is_nnf(expr, simplified=True):
    if False:
        print('Hello World!')
    '\n    Checks if ``expr`` is in Negation Normal Form (NNF).\n\n    A logical expression is in NNF if it\n    contains only :py:class:`~.And`, :py:class:`~.Or` and :py:class:`~.Not`,\n    and :py:class:`~.Not` is applied only to literals.\n    If ``simplified`` is ``True``, checks if result contains no redundant clauses.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import A, B, C\n    >>> from sympy.logic.boolalg import Not, is_nnf\n    >>> is_nnf(A & B | ~C)\n    True\n    >>> is_nnf((A | ~A) & (B | C))\n    False\n    >>> is_nnf((A | ~A) & (B | C), False)\n    True\n    >>> is_nnf(Not(A & B) | C)\n    False\n    >>> is_nnf((A >> B) & (B >> A))\n    False\n\n    '
    expr = sympify(expr)
    if is_literal(expr):
        return True
    stack = [expr]
    while stack:
        expr = stack.pop()
        if expr.func in (And, Or):
            if simplified:
                args = expr.args
                for arg in args:
                    if Not(arg) in args:
                        return False
            stack.extend(expr.args)
        elif not is_literal(expr):
            return False
    return True

def is_cnf(expr):
    if False:
        print('Hello World!')
    '\n    Test whether or not an expression is in conjunctive normal form.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import is_cnf\n    >>> from sympy.abc import A, B, C\n    >>> is_cnf(A | B | C)\n    True\n    >>> is_cnf(A & B & C)\n    True\n    >>> is_cnf((A & B) | C)\n    False\n\n    '
    return _is_form(expr, And, Or)

def is_dnf(expr):
    if False:
        while True:
            i = 10
    '\n    Test whether or not an expression is in disjunctive normal form.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import is_dnf\n    >>> from sympy.abc import A, B, C\n    >>> is_dnf(A | B | C)\n    True\n    >>> is_dnf(A & B & C)\n    True\n    >>> is_dnf((A & B) | C)\n    True\n    >>> is_dnf(A & (B | C))\n    False\n\n    '
    return _is_form(expr, Or, And)

def _is_form(expr, function1, function2):
    if False:
        return 10
    '\n    Test whether or not an expression is of the required form.\n\n    '
    expr = sympify(expr)
    vals = function1.make_args(expr) if isinstance(expr, function1) else [expr]
    for lit in vals:
        if isinstance(lit, function2):
            vals2 = function2.make_args(lit) if isinstance(lit, function2) else [lit]
            for l in vals2:
                if is_literal(l) is False:
                    return False
        elif is_literal(lit) is False:
            return False
    return True

def eliminate_implications(expr):
    if False:
        while True:
            i = 10
    '\n    Change :py:class:`~.Implies` and :py:class:`~.Equivalent` into\n    :py:class:`~.And`, :py:class:`~.Or`, and :py:class:`~.Not`.\n    That is, return an expression that is equivalent to ``expr``, but has only\n    ``&``, ``|``, and ``~`` as logical\n    operators.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import Implies, Equivalent,          eliminate_implications\n    >>> from sympy.abc import A, B, C\n    >>> eliminate_implications(Implies(A, B))\n    B | ~A\n    >>> eliminate_implications(Equivalent(A, B))\n    (A | ~B) & (B | ~A)\n    >>> eliminate_implications(Equivalent(A, B, C))\n    (A | ~C) & (B | ~A) & (C | ~B)\n\n    '
    return to_nnf(expr, simplify=False)

def is_literal(expr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns True if expr is a literal, else False.\n\n    Examples\n    ========\n\n    >>> from sympy import Or, Q\n    >>> from sympy.abc import A, B\n    >>> from sympy.logic.boolalg import is_literal\n    >>> is_literal(A)\n    True\n    >>> is_literal(~A)\n    True\n    >>> is_literal(Q.zero(A))\n    True\n    >>> is_literal(A + B)\n    True\n    >>> is_literal(Or(A, B))\n    False\n\n    '
    from sympy.assumptions import AppliedPredicate
    if isinstance(expr, Not):
        return is_literal(expr.args[0])
    elif expr in (True, False) or isinstance(expr, AppliedPredicate) or expr.is_Atom:
        return True
    elif not isinstance(expr, BooleanFunction) and all((isinstance(expr, AppliedPredicate) or a.is_Atom for a in expr.args)):
        return True
    return False

def to_int_repr(clauses, symbols):
    if False:
        while True:
            i = 10
    '\n    Takes clauses in CNF format and puts them into an integer representation.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import to_int_repr\n    >>> from sympy.abc import x, y\n    >>> to_int_repr([x | y, y], [x, y]) == [{1, 2}, {2}]\n    True\n\n    '
    symbols = dict(zip(symbols, range(1, len(symbols) + 1)))

    def append_symbol(arg, symbols):
        if False:
            print('Hello World!')
        if isinstance(arg, Not):
            return -symbols[arg.args[0]]
        else:
            return symbols[arg]
    return [{append_symbol(arg, symbols) for arg in Or.make_args(c)} for c in clauses]

def term_to_integer(term):
    if False:
        i = 10
        return i + 15
    "\n    Return an integer corresponding to the base-2 digits given by *term*.\n\n    Parameters\n    ==========\n\n    term : a string or list of ones and zeros\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import term_to_integer\n    >>> term_to_integer([1, 0, 0])\n    4\n    >>> term_to_integer('100')\n    4\n\n    "
    return int(''.join(list(map(str, list(term)))), 2)
integer_to_term = ibin

def truth_table(expr, variables, input=True):
    if False:
        print('Hello World!')
    "\n    Return a generator of all possible configurations of the input variables,\n    and the result of the boolean expression for those values.\n\n    Parameters\n    ==========\n\n    expr : Boolean expression\n\n    variables : list of variables\n\n    input : bool (default ``True``)\n        Indicates whether to return the input combinations.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import truth_table\n    >>> from sympy.abc import x,y\n    >>> table = truth_table(x >> y, [x, y])\n    >>> for t in table:\n    ...     print('{0} -> {1}'.format(*t))\n    [0, 0] -> True\n    [0, 1] -> True\n    [1, 0] -> False\n    [1, 1] -> True\n\n    >>> table = truth_table(x | y, [x, y])\n    >>> list(table)\n    [([0, 0], False), ([0, 1], True), ([1, 0], True), ([1, 1], True)]\n\n    If ``input`` is ``False``, ``truth_table`` returns only a list of truth values.\n    In this case, the corresponding input values of variables can be\n    deduced from the index of a given output.\n\n    >>> from sympy.utilities.iterables import ibin\n    >>> vars = [y, x]\n    >>> values = truth_table(x >> y, vars, input=False)\n    >>> values = list(values)\n    >>> values\n    [True, False, True, True]\n\n    >>> for i, value in enumerate(values):\n    ...     print('{0} -> {1}'.format(list(zip(\n    ...     vars, ibin(i, len(vars)))), value))\n    [(y, 0), (x, 0)] -> True\n    [(y, 0), (x, 1)] -> False\n    [(y, 1), (x, 0)] -> True\n    [(y, 1), (x, 1)] -> True\n\n    "
    variables = [sympify(v) for v in variables]
    expr = sympify(expr)
    if not isinstance(expr, BooleanFunction) and (not is_literal(expr)):
        return
    table = product((0, 1), repeat=len(variables))
    for term in table:
        value = expr.xreplace(dict(zip(variables, term)))
        if input:
            yield (list(term), value)
        else:
            yield value

def _check_pair(minterm1, minterm2):
    if False:
        print('Hello World!')
    '\n    Checks if a pair of minterms differs by only one bit. If yes, returns\n    index, else returns `-1`.\n    '
    index = -1
    for (x, i) in enumerate(minterm1):
        if i != minterm2[x]:
            if index == -1:
                index = x
            else:
                return -1
    return index

def _convert_to_varsSOP(minterm, variables):
    if False:
        while True:
            i = 10
    '\n    Converts a term in the expansion of a function from binary to its\n    variable form (for SOP).\n    '
    temp = [variables[n] if val == 1 else Not(variables[n]) for (n, val) in enumerate(minterm) if val != 3]
    return And(*temp)

def _convert_to_varsPOS(maxterm, variables):
    if False:
        print('Hello World!')
    '\n    Converts a term in the expansion of a function from binary to its\n    variable form (for POS).\n    '
    temp = [variables[n] if val == 0 else Not(variables[n]) for (n, val) in enumerate(maxterm) if val != 3]
    return Or(*temp)

def _convert_to_varsANF(term, variables):
    if False:
        i = 10
        return i + 15
    "\n    Converts a term in the expansion of a function from binary to its\n    variable form (for ANF).\n\n    Parameters\n    ==========\n\n    term : list of 1's and 0's (complementation pattern)\n    variables : list of variables\n\n    "
    temp = [variables[n] for (n, t) in enumerate(term) if t == 1]
    if not temp:
        return true
    return And(*temp)

def _get_odd_parity_terms(n):
    if False:
        print('Hello World!')
    '\n    Returns a list of lists, with all possible combinations of n zeros and ones\n    with an odd number of ones.\n    '
    return [e for e in [ibin(i, n) for i in range(2 ** n)] if sum(e) % 2 == 1]

def _get_even_parity_terms(n):
    if False:
        return 10
    '\n    Returns a list of lists, with all possible combinations of n zeros and ones\n    with an even number of ones.\n    '
    return [e for e in [ibin(i, n) for i in range(2 ** n)] if sum(e) % 2 == 0]

def _simplified_pairs(terms):
    if False:
        print('Hello World!')
    '\n    Reduces a set of minterms, if possible, to a simplified set of minterms\n    with one less variable in the terms using QM method.\n    '
    if not terms:
        return []
    simplified_terms = []
    todo = list(range(len(terms)))
    termdict = defaultdict(list)
    for (n, term) in enumerate(terms):
        ones = sum([1 for t in term if t == 1])
        termdict[ones].append(n)
    variables = len(terms[0])
    for k in range(variables):
        for i in termdict[k]:
            for j in termdict[k + 1]:
                index = _check_pair(terms[i], terms[j])
                if index != -1:
                    todo[i] = todo[j] = None
                    newterm = terms[i][:]
                    newterm[index] = 3
                    if newterm not in simplified_terms:
                        simplified_terms.append(newterm)
    if simplified_terms:
        simplified_terms = _simplified_pairs(simplified_terms)
    simplified_terms.extend([terms[i] for i in todo if i is not None])
    return simplified_terms

def _rem_redundancy(l1, terms):
    if False:
        for i in range(10):
            print('nop')
    '\n    After the truth table has been sufficiently simplified, use the prime\n    implicant table method to recognize and eliminate redundant pairs,\n    and return the essential arguments.\n    '
    if not terms:
        return []
    nterms = len(terms)
    nl1 = len(l1)
    dommatrix = [[0] * nl1 for n in range(nterms)]
    colcount = [0] * nl1
    rowcount = [0] * nterms
    for (primei, prime) in enumerate(l1):
        for (termi, term) in enumerate(terms):
            if all((t == 3 or t == mt for (t, mt) in zip(prime, term))):
                dommatrix[termi][primei] = 1
                colcount[primei] += 1
                rowcount[termi] += 1
    anythingchanged = True
    while anythingchanged:
        anythingchanged = False
        for rowi in range(nterms):
            if rowcount[rowi]:
                row = dommatrix[rowi]
                for row2i in range(nterms):
                    if rowi != row2i and rowcount[rowi] and (rowcount[rowi] <= rowcount[row2i]):
                        row2 = dommatrix[row2i]
                        if all((row2[n] >= row[n] for n in range(nl1))):
                            rowcount[row2i] = 0
                            anythingchanged = True
                            for (primei, prime) in enumerate(row2):
                                if prime:
                                    dommatrix[row2i][primei] = 0
                                    colcount[primei] -= 1
        colcache = {}
        for coli in range(nl1):
            if colcount[coli]:
                if coli in colcache:
                    col = colcache[coli]
                else:
                    col = [dommatrix[i][coli] for i in range(nterms)]
                    colcache[coli] = col
                for col2i in range(nl1):
                    if coli != col2i and colcount[col2i] and (colcount[coli] >= colcount[col2i]):
                        if col2i in colcache:
                            col2 = colcache[col2i]
                        else:
                            col2 = [dommatrix[i][col2i] for i in range(nterms)]
                            colcache[col2i] = col2
                        if all((col[n] >= col2[n] for n in range(nterms))):
                            colcount[col2i] = 0
                            anythingchanged = True
                            for (termi, term) in enumerate(col2):
                                if term and dommatrix[termi][col2i]:
                                    dommatrix[termi][col2i] = 0
                                    rowcount[termi] -= 1
        if not anythingchanged:
            maxterms = 0
            bestcolidx = -1
            for coli in range(nl1):
                s = colcount[coli]
                if s > maxterms:
                    bestcolidx = coli
                    maxterms = s
            if bestcolidx != -1 and maxterms > 1:
                for (primei, prime) in enumerate(l1):
                    if primei != bestcolidx:
                        for (termi, term) in enumerate(colcache[bestcolidx]):
                            if term and dommatrix[termi][primei]:
                                dommatrix[termi][primei] = 0
                                anythingchanged = True
                                rowcount[termi] -= 1
                                colcount[primei] -= 1
    return [l1[i] for i in range(nl1) if colcount[i]]

def _input_to_binlist(inputlist, variables):
    if False:
        print('Hello World!')
    binlist = []
    bits = len(variables)
    for val in inputlist:
        if isinstance(val, int):
            binlist.append(ibin(val, bits))
        elif isinstance(val, dict):
            nonspecvars = list(variables)
            for key in val.keys():
                nonspecvars.remove(key)
            for t in product((0, 1), repeat=len(nonspecvars)):
                d = dict(zip(nonspecvars, t))
                d.update(val)
                binlist.append([d[v] for v in variables])
        elif isinstance(val, (list, tuple)):
            if len(val) != bits:
                raise ValueError('Each term must contain {bits} bits as there are\n{bits} variables (or be an integer).'.format(bits=bits))
            binlist.append(list(val))
        else:
            raise TypeError('A term list can only contain lists, ints or dicts.')
    return binlist

def SOPform(variables, minterms, dontcares=None):
    if False:
        i = 10
        return i + 15
    '\n    The SOPform function uses simplified_pairs and a redundant group-\n    eliminating algorithm to convert the list of all input combos that\n    generate \'1\' (the minterms) into the smallest sum-of-products form.\n\n    The variables must be given as the first argument.\n\n    Return a logical :py:class:`~.Or` function (i.e., the "sum of products" or\n    "SOP" form) that gives the desired outcome. If there are inputs that can\n    be ignored, pass them as a list, too.\n\n    The result will be one of the (perhaps many) functions that satisfy\n    the conditions.\n\n    Examples\n    ========\n\n    >>> from sympy.logic import SOPform\n    >>> from sympy import symbols\n    >>> w, x, y, z = symbols(\'w x y z\')\n    >>> minterms = [[0, 0, 0, 1], [0, 0, 1, 1],\n    ...             [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]\n    >>> dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]\n    >>> SOPform([w, x, y, z], minterms, dontcares)\n    (y & z) | (~w & ~x)\n\n    The terms can also be represented as integers:\n\n    >>> minterms = [1, 3, 7, 11, 15]\n    >>> dontcares = [0, 2, 5]\n    >>> SOPform([w, x, y, z], minterms, dontcares)\n    (y & z) | (~w & ~x)\n\n    They can also be specified using dicts, which does not have to be fully\n    specified:\n\n    >>> minterms = [{w: 0, x: 1}, {y: 1, z: 1, x: 0}]\n    >>> SOPform([w, x, y, z], minterms)\n    (x & ~w) | (y & z & ~x)\n\n    Or a combination:\n\n    >>> minterms = [4, 7, 11, [1, 1, 1, 1]]\n    >>> dontcares = [{w : 0, x : 0, y: 0}, 5]\n    >>> SOPform([w, x, y, z], minterms, dontcares)\n    (w & y & z) | (~w & ~y) | (x & z & ~w)\n\n    See also\n    ========\n\n    POSform\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Quine-McCluskey_algorithm\n    .. [2] https://en.wikipedia.org/wiki/Don%27t-care_term\n\n    '
    if not minterms:
        return false
    variables = tuple(map(sympify, variables))
    minterms = _input_to_binlist(minterms, variables)
    dontcares = _input_to_binlist(dontcares or [], variables)
    for d in dontcares:
        if d in minterms:
            raise ValueError('%s in minterms is also in dontcares' % d)
    return _sop_form(variables, minterms, dontcares)

def _sop_form(variables, minterms, dontcares):
    if False:
        for i in range(10):
            print('nop')
    new = _simplified_pairs(minterms + dontcares)
    essential = _rem_redundancy(new, minterms)
    return Or(*[_convert_to_varsSOP(x, variables) for x in essential])

def POSform(variables, minterms, dontcares=None):
    if False:
        i = 10
        return i + 15
    '\n    The POSform function uses simplified_pairs and a redundant-group\n    eliminating algorithm to convert the list of all input combinations\n    that generate \'1\' (the minterms) into the smallest product-of-sums form.\n\n    The variables must be given as the first argument.\n\n    Return a logical :py:class:`~.And` function (i.e., the "product of sums"\n    or "POS" form) that gives the desired outcome. If there are inputs that can\n    be ignored, pass them as a list, too.\n\n    The result will be one of the (perhaps many) functions that satisfy\n    the conditions.\n\n    Examples\n    ========\n\n    >>> from sympy.logic import POSform\n    >>> from sympy import symbols\n    >>> w, x, y, z = symbols(\'w x y z\')\n    >>> minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1],\n    ...             [1, 0, 1, 1], [1, 1, 1, 1]]\n    >>> dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]\n    >>> POSform([w, x, y, z], minterms, dontcares)\n    z & (y | ~w)\n\n    The terms can also be represented as integers:\n\n    >>> minterms = [1, 3, 7, 11, 15]\n    >>> dontcares = [0, 2, 5]\n    >>> POSform([w, x, y, z], minterms, dontcares)\n    z & (y | ~w)\n\n    They can also be specified using dicts, which does not have to be fully\n    specified:\n\n    >>> minterms = [{w: 0, x: 1}, {y: 1, z: 1, x: 0}]\n    >>> POSform([w, x, y, z], minterms)\n    (x | y) & (x | z) & (~w | ~x)\n\n    Or a combination:\n\n    >>> minterms = [4, 7, 11, [1, 1, 1, 1]]\n    >>> dontcares = [{w : 0, x : 0, y: 0}, 5]\n    >>> POSform([w, x, y, z], minterms, dontcares)\n    (w | x) & (y | ~w) & (z | ~y)\n\n    See also\n    ========\n\n    SOPform\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Quine-McCluskey_algorithm\n    .. [2] https://en.wikipedia.org/wiki/Don%27t-care_term\n\n    '
    if not minterms:
        return false
    variables = tuple(map(sympify, variables))
    minterms = _input_to_binlist(minterms, variables)
    dontcares = _input_to_binlist(dontcares or [], variables)
    for d in dontcares:
        if d in minterms:
            raise ValueError('%s in minterms is also in dontcares' % d)
    maxterms = []
    for t in product((0, 1), repeat=len(variables)):
        t = list(t)
        if t not in minterms and t not in dontcares:
            maxterms.append(t)
    new = _simplified_pairs(maxterms + dontcares)
    essential = _rem_redundancy(new, maxterms)
    return And(*[_convert_to_varsPOS(x, variables) for x in essential])

def ANFform(variables, truthvalues):
    if False:
        return 10
    '\n    The ANFform function converts the list of truth values to\n    Algebraic Normal Form (ANF).\n\n    The variables must be given as the first argument.\n\n    Return True, False, logical :py:class:`~.And` function (i.e., the\n    "Zhegalkin monomial") or logical :py:class:`~.Xor` function (i.e.,\n    the "Zhegalkin polynomial"). When True and False\n    are represented by 1 and 0, respectively, then\n    :py:class:`~.And` is multiplication and :py:class:`~.Xor` is addition.\n\n    Formally a "Zhegalkin monomial" is the product (logical\n    And) of a finite set of distinct variables, including\n    the empty set whose product is denoted 1 (True).\n    A "Zhegalkin polynomial" is the sum (logical Xor) of a\n    set of Zhegalkin monomials, with the empty set denoted\n    by 0 (False).\n\n    Parameters\n    ==========\n\n    variables : list of variables\n    truthvalues : list of 1\'s and 0\'s (result column of truth table)\n\n    Examples\n    ========\n    >>> from sympy.logic.boolalg import ANFform\n    >>> from sympy.abc import x, y\n    >>> ANFform([x], [1, 0])\n    x ^ True\n    >>> ANFform([x, y], [0, 1, 1, 1])\n    x ^ y ^ (x & y)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Zhegalkin_polynomial\n\n    '
    n_vars = len(variables)
    n_values = len(truthvalues)
    if n_values != 2 ** n_vars:
        raise ValueError('The number of truth values must be equal to 2^%d, got %d' % (n_vars, n_values))
    variables = tuple(map(sympify, variables))
    coeffs = anf_coeffs(truthvalues)
    terms = []
    for (i, t) in enumerate(product((0, 1), repeat=n_vars)):
        if coeffs[i] == 1:
            terms.append(t)
    return Xor(*[_convert_to_varsANF(x, variables) for x in terms], remove_true=False)

def anf_coeffs(truthvalues):
    if False:
        i = 10
        return i + 15
    '\n    Convert a list of truth values of some boolean expression\n    to the list of coefficients of the polynomial mod 2 (exclusive\n    disjunction) representing the boolean expression in ANF\n    (i.e., the "Zhegalkin polynomial").\n\n    There are `2^n` possible Zhegalkin monomials in `n` variables, since\n    each monomial is fully specified by the presence or absence of\n    each variable.\n\n    We can enumerate all the monomials. For example, boolean\n    function with four variables ``(a, b, c, d)`` can contain\n    up to `2^4 = 16` monomials. The 13-th monomial is the\n    product ``a & b & d``, because 13 in binary is 1, 1, 0, 1.\n\n    A given monomial\'s presence or absence in a polynomial corresponds\n    to that monomial\'s coefficient being 1 or 0 respectively.\n\n    Examples\n    ========\n    >>> from sympy.logic.boolalg import anf_coeffs, bool_monomial, Xor\n    >>> from sympy.abc import a, b, c\n    >>> truthvalues = [0, 1, 1, 0, 0, 1, 0, 1]\n    >>> coeffs = anf_coeffs(truthvalues)\n    >>> coeffs\n    [0, 1, 1, 0, 0, 0, 1, 0]\n    >>> polynomial = Xor(*[\n    ...     bool_monomial(k, [a, b, c])\n    ...     for k, coeff in enumerate(coeffs) if coeff == 1\n    ... ])\n    >>> polynomial\n    b ^ c ^ (a & b)\n\n    '
    s = '{:b}'.format(len(truthvalues))
    n = len(s) - 1
    if len(truthvalues) != 2 ** n:
        raise ValueError('The number of truth values must be a power of two, got %d' % len(truthvalues))
    coeffs = [[v] for v in truthvalues]
    for i in range(n):
        tmp = []
        for j in range(2 ** (n - i - 1)):
            tmp.append(coeffs[2 * j] + list(map(lambda x, y: x ^ y, coeffs[2 * j], coeffs[2 * j + 1])))
        coeffs = tmp
    return coeffs[0]

def bool_minterm(k, variables):
    if False:
        while True:
            i = 10
    "\n    Return the k-th minterm.\n\n    Minterms are numbered by a binary encoding of the complementation\n    pattern of the variables. This convention assigns the value 1 to\n    the direct form and 0 to the complemented form.\n\n    Parameters\n    ==========\n\n    k : int or list of 1's and 0's (complementation pattern)\n    variables : list of variables\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import bool_minterm\n    >>> from sympy.abc import x, y, z\n    >>> bool_minterm([1, 0, 1], [x, y, z])\n    x & z & ~y\n    >>> bool_minterm(6, [x, y, z])\n    x & y & ~z\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Canonical_normal_form#Indexing_minterms\n\n    "
    if isinstance(k, int):
        k = ibin(k, len(variables))
    variables = tuple(map(sympify, variables))
    return _convert_to_varsSOP(k, variables)

def bool_maxterm(k, variables):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the k-th maxterm.\n\n    Each maxterm is assigned an index based on the opposite\n    conventional binary encoding used for minterms. The maxterm\n    convention assigns the value 0 to the direct form and 1 to\n    the complemented form.\n\n    Parameters\n    ==========\n\n    k : int or list of 1's and 0's (complementation pattern)\n    variables : list of variables\n\n    Examples\n    ========\n    >>> from sympy.logic.boolalg import bool_maxterm\n    >>> from sympy.abc import x, y, z\n    >>> bool_maxterm([1, 0, 1], [x, y, z])\n    y | ~x | ~z\n    >>> bool_maxterm(6, [x, y, z])\n    z | ~x | ~y\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Canonical_normal_form#Indexing_maxterms\n\n    "
    if isinstance(k, int):
        k = ibin(k, len(variables))
    variables = tuple(map(sympify, variables))
    return _convert_to_varsPOS(k, variables)

def bool_monomial(k, variables):
    if False:
        return 10
    "\n    Return the k-th monomial.\n\n    Monomials are numbered by a binary encoding of the presence and\n    absences of the variables. This convention assigns the value\n    1 to the presence of variable and 0 to the absence of variable.\n\n    Each boolean function can be uniquely represented by a\n    Zhegalkin Polynomial (Algebraic Normal Form). The Zhegalkin\n    Polynomial of the boolean function with `n` variables can contain\n    up to `2^n` monomials. We can enumerate all the monomials.\n    Each monomial is fully specified by the presence or absence\n    of each variable.\n\n    For example, boolean function with four variables ``(a, b, c, d)``\n    can contain up to `2^4 = 16` monomials. The 13-th monomial is the\n    product ``a & b & d``, because 13 in binary is 1, 1, 0, 1.\n\n    Parameters\n    ==========\n\n    k : int or list of 1's and 0's\n    variables : list of variables\n\n    Examples\n    ========\n    >>> from sympy.logic.boolalg import bool_monomial\n    >>> from sympy.abc import x, y, z\n    >>> bool_monomial([1, 0, 1], [x, y, z])\n    x & z\n    >>> bool_monomial(6, [x, y, z])\n    x & y\n\n    "
    if isinstance(k, int):
        k = ibin(k, len(variables))
    variables = tuple(map(sympify, variables))
    return _convert_to_varsANF(k, variables)

def _find_predicates(expr):
    if False:
        while True:
            i = 10
    'Helper to find logical predicates in BooleanFunctions.\n\n    A logical predicate is defined here as anything within a BooleanFunction\n    that is not a BooleanFunction itself.\n\n    '
    if not isinstance(expr, BooleanFunction):
        return {expr}
    return set().union(*map(_find_predicates, expr.args))

def simplify_logic(expr, form=None, deep=True, force=False, dontcare=None):
    if False:
        return 10
    "\n    This function simplifies a boolean function to its simplified version\n    in SOP or POS form. The return type is an :py:class:`~.Or` or\n    :py:class:`~.And` object in SymPy.\n\n    Parameters\n    ==========\n\n    expr : Boolean\n\n    form : string (``'cnf'`` or ``'dnf'``) or ``None`` (default).\n        If ``'cnf'`` or ``'dnf'``, the simplest expression in the corresponding\n        normal form is returned; if ``None``, the answer is returned\n        according to the form with fewest args (in CNF by default).\n\n    deep : bool (default ``True``)\n        Indicates whether to recursively simplify any\n        non-boolean functions contained within the input.\n\n    force : bool (default ``False``)\n        As the simplifications require exponential time in the number\n        of variables, there is by default a limit on expressions with\n        8 variables. When the expression has more than 8 variables\n        only symbolical simplification (controlled by ``deep``) is\n        made. By setting ``force`` to ``True``, this limit is removed. Be\n        aware that this can lead to very long simplification times.\n\n    dontcare : Boolean\n        Optimize expression under the assumption that inputs where this\n        expression is true are don't care. This is useful in e.g. Piecewise\n        conditions, where later conditions do not need to consider inputs that\n        are converted by previous conditions. For example, if a previous\n        condition is ``And(A, B)``, the simplification of expr can be made\n        with don't cares for ``And(A, B)``.\n\n    Examples\n    ========\n\n    >>> from sympy.logic import simplify_logic\n    >>> from sympy.abc import x, y, z\n    >>> b = (~x & ~y & ~z) | ( ~x & ~y & z)\n    >>> simplify_logic(b)\n    ~x & ~y\n    >>> simplify_logic(x | y, dontcare=y)\n    x\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Don%27t-care_term\n\n    "
    if form not in (None, 'cnf', 'dnf'):
        raise ValueError('form can be cnf or dnf only')
    expr = sympify(expr)
    if form:
        form_ok = False
        if form == 'cnf':
            form_ok = is_cnf(expr)
        elif form == 'dnf':
            form_ok = is_dnf(expr)
        if form_ok and all((is_literal(a) for a in expr.args)):
            return expr
    from sympy.core.relational import Relational
    if deep:
        variables = expr.atoms(Relational)
        from sympy.simplify.simplify import simplify
        s = tuple(map(simplify, variables))
        expr = expr.xreplace(dict(zip(variables, s)))
    if not isinstance(expr, BooleanFunction):
        return expr
    repl = {}
    undo = {}
    from sympy.core.symbol import Dummy
    variables = expr.atoms(Relational)
    if dontcare is not None:
        dontcare = sympify(dontcare)
        variables.update(dontcare.atoms(Relational))
    while variables:
        var = variables.pop()
        if var.is_Relational:
            d = Dummy()
            undo[d] = var
            repl[var] = d
            nvar = var.negated
            if nvar in variables:
                repl[nvar] = Not(d)
                variables.remove(nvar)
    expr = expr.xreplace(repl)
    if dontcare is not None:
        dontcare = dontcare.xreplace(repl)
    variables = _find_predicates(expr)
    if not force and len(variables) > 8:
        return expr.xreplace(undo)
    if dontcare is not None:
        dcvariables = _find_predicates(dontcare)
        variables.update(dcvariables)
        if not force and len(variables) > 8:
            variables = _find_predicates(expr)
            dontcare = None
    (c, v) = sift(ordered(variables), lambda x: x in (True, False), binary=True)
    variables = c + v
    c = [1 if i == True else 0 for i in c]
    truthtable = _get_truthtable(v, expr, c)
    if dontcare is not None:
        dctruthtable = _get_truthtable(v, dontcare, c)
        truthtable = [t for t in truthtable if t not in dctruthtable]
    else:
        dctruthtable = []
    big = len(truthtable) >= 2 ** (len(variables) - 1)
    if form == 'dnf' or (form is None and big):
        return _sop_form(variables, truthtable, dctruthtable).xreplace(undo)
    return POSform(variables, truthtable, dctruthtable).xreplace(undo)

def _get_truthtable(variables, expr, const):
    if False:
        for i in range(10):
            print('nop')
    ' Return a list of all combinations leading to a True result for ``expr``.\n    '
    _variables = variables.copy()

    def _get_tt(inputs):
        if False:
            return 10
        if _variables:
            v = _variables.pop()
            tab = [[i[0].xreplace({v: false}), [0] + i[1]] for i in inputs if i[0] is not false]
            tab.extend([[i[0].xreplace({v: true}), [1] + i[1]] for i in inputs if i[0] is not false])
            return _get_tt(tab)
        return inputs
    res = [const + k[1] for k in _get_tt([[expr, []]]) if k[0]]
    if res == [[]]:
        return []
    else:
        return res

def _finger(eq):
    if False:
        print('Hello World!')
    "\n    Assign a 5-item fingerprint to each symbol in the equation:\n    [\n    # of times it appeared as a Symbol;\n    # of times it appeared as a Not(symbol);\n    # of times it appeared as a Symbol in an And or Or;\n    # of times it appeared as a Not(Symbol) in an And or Or;\n    a sorted tuple of tuples, (i, j, k), where i is the number of arguments\n    in an And or Or with which it appeared as a Symbol, and j is\n    the number of arguments that were Not(Symbol); k is the number\n    of times that (i, j) was seen.\n    ]\n\n    Examples\n    ========\n\n    >>> from sympy.logic.boolalg import _finger as finger\n    >>> from sympy import And, Or, Not, Xor, to_cnf, symbols\n    >>> from sympy.abc import a, b, x, y\n    >>> eq = Or(And(Not(y), a), And(Not(y), b), And(x, y))\n    >>> dict(finger(eq))\n    {(0, 0, 1, 0, ((2, 0, 1),)): [x],\n    (0, 0, 1, 0, ((2, 1, 1),)): [a, b],\n    (0, 0, 1, 2, ((2, 0, 1),)): [y]}\n    >>> dict(finger(x & ~y))\n    {(0, 1, 0, 0, ()): [y], (1, 0, 0, 0, ()): [x]}\n\n    In the following, the (5, 2, 6) means that there were 6 Or\n    functions in which a symbol appeared as itself amongst 5 arguments in\n    which there were also 2 negated symbols, e.g. ``(a0 | a1 | a2 | ~a3 | ~a4)``\n    is counted once for a0, a1 and a2.\n\n    >>> dict(finger(to_cnf(Xor(*symbols('a:5')))))\n    {(0, 0, 8, 8, ((5, 0, 1), (5, 2, 6), (5, 4, 1))): [a0, a1, a2, a3, a4]}\n\n    The equation must not have more than one level of nesting:\n\n    >>> dict(finger(And(Or(x, y), y)))\n    {(0, 0, 1, 0, ((2, 0, 1),)): [x], (1, 0, 1, 0, ((2, 0, 1),)): [y]}\n    >>> dict(finger(And(Or(x, And(a, x)), y)))\n    Traceback (most recent call last):\n    ...\n    NotImplementedError: unexpected level of nesting\n\n    So y and x have unique fingerprints, but a and b do not.\n    "
    f = eq.free_symbols
    d = dict(list(zip(f, [[0] * 4 + [defaultdict(int)] for fi in f])))
    for a in eq.args:
        if a.is_Symbol:
            d[a][0] += 1
        elif a.is_Not:
            d[a.args[0]][1] += 1
        else:
            o = (len(a.args), sum((isinstance(ai, Not) for ai in a.args)))
            for ai in a.args:
                if ai.is_Symbol:
                    d[ai][2] += 1
                    d[ai][-1][o] += 1
                elif ai.is_Not:
                    d[ai.args[0]][3] += 1
                else:
                    raise NotImplementedError('unexpected level of nesting')
    inv = defaultdict(list)
    for (k, v) in ordered(iter(d.items())):
        v[-1] = tuple(sorted([i + (j,) for (i, j) in v[-1].items()]))
        inv[tuple(v)].append(k)
    return inv

def bool_map(bool1, bool2):
    if False:
        print('Hello World!')
    '\n    Return the simplified version of *bool1*, and the mapping of variables\n    that makes the two expressions *bool1* and *bool2* represent the same\n    logical behaviour for some correspondence between the variables\n    of each.\n    If more than one mappings of this sort exist, one of them\n    is returned.\n\n    For example, ``And(x, y)`` is logically equivalent to ``And(a, b)`` for\n    the mapping ``{x: a, y: b}`` or ``{x: b, y: a}``.\n    If no such mapping exists, return ``False``.\n\n    Examples\n    ========\n\n    >>> from sympy import SOPform, bool_map, Or, And, Not, Xor\n    >>> from sympy.abc import w, x, y, z, a, b, c, d\n    >>> function1 = SOPform([x, z, y],[[1, 0, 1], [0, 0, 1]])\n    >>> function2 = SOPform([a, b, c],[[1, 0, 1], [1, 0, 0]])\n    >>> bool_map(function1, function2)\n    (y & ~z, {y: a, z: b})\n\n    The results are not necessarily unique, but they are canonical. Here,\n    ``(w, z)`` could be ``(a, d)`` or ``(d, a)``:\n\n    >>> eq =  Or(And(Not(y), w), And(Not(y), z), And(x, y))\n    >>> eq2 = Or(And(Not(c), a), And(Not(c), d), And(b, c))\n    >>> bool_map(eq, eq2)\n    ((x & y) | (w & ~y) | (z & ~y), {w: a, x: b, y: c, z: d})\n    >>> eq = And(Xor(a, b), c, And(c,d))\n    >>> bool_map(eq, eq.subs(c, x))\n    (c & d & (a | b) & (~a | ~b), {a: a, b: b, c: d, d: x})\n\n    '

    def match(function1, function2):
        if False:
            print('Hello World!')
        'Return the mapping that equates variables between two\n        simplified boolean expressions if possible.\n\n        By "simplified" we mean that a function has been denested\n        and is either an And (or an Or) whose arguments are either\n        symbols (x), negated symbols (Not(x)), or Or (or an And) whose\n        arguments are only symbols or negated symbols. For example,\n        ``And(x, Not(y), Or(w, Not(z)))``.\n\n        Basic.match is not robust enough (see issue 4835) so this is\n        a workaround that is valid for simplified boolean expressions\n        '
        if function1.__class__ != function2.__class__:
            return None
        if len(function1.args) != len(function2.args):
            return None
        if function1.is_Symbol:
            return {function1: function2}
        f1 = _finger(function1)
        f2 = _finger(function2)
        if len(f1) != len(f2):
            return False
        matchdict = {}
        for k in f1.keys():
            if k not in f2:
                return False
            if len(f1[k]) != len(f2[k]):
                return False
            for (i, x) in enumerate(f1[k]):
                matchdict[x] = f2[k][i]
        return matchdict
    a = simplify_logic(bool1)
    b = simplify_logic(bool2)
    m = match(a, b)
    if m:
        return (a, m)
    return m

def _apply_patternbased_simplification(rv, patterns, measure, dominatingvalue, replacementvalue=None, threeterm_patterns=None):
    if False:
        while True:
            i = 10
    '\n    Replace patterns of Relational\n\n    Parameters\n    ==========\n\n    rv : Expr\n        Boolean expression\n\n    patterns : tuple\n        Tuple of tuples, with (pattern to simplify, simplified pattern) with\n        two terms.\n\n    measure : function\n        Simplification measure.\n\n    dominatingvalue : Boolean or ``None``\n        The dominating value for the function of consideration.\n        For example, for :py:class:`~.And` ``S.false`` is dominating.\n        As soon as one expression is ``S.false`` in :py:class:`~.And`,\n        the whole expression is ``S.false``.\n\n    replacementvalue : Boolean or ``None``, optional\n        The resulting value for the whole expression if one argument\n        evaluates to ``dominatingvalue``.\n        For example, for :py:class:`~.Nand` ``S.false`` is dominating, but\n        in this case the resulting value is ``S.true``. Default is ``None``.\n        If ``replacementvalue`` is ``None`` and ``dominatingvalue`` is not\n        ``None``, ``replacementvalue = dominatingvalue``.\n\n    threeterm_patterns : tuple, optional\n        Tuple of tuples, with (pattern to simplify, simplified pattern) with\n        three terms.\n\n    '
    from sympy.core.relational import Relational, _canonical
    if replacementvalue is None and dominatingvalue is not None:
        replacementvalue = dominatingvalue
    (Rel, nonRel) = sift(rv.args, lambda i: isinstance(i, Relational), binary=True)
    if len(Rel) <= 1:
        return rv
    (Rel, nonRealRel) = sift(Rel, lambda i: not any((s.is_real is False for s in i.free_symbols)), binary=True)
    Rel = [i.canonical for i in Rel]
    if threeterm_patterns and len(Rel) >= 3:
        Rel = _apply_patternbased_threeterm_simplification(Rel, threeterm_patterns, rv.func, dominatingvalue, replacementvalue, measure)
    Rel = _apply_patternbased_twoterm_simplification(Rel, patterns, rv.func, dominatingvalue, replacementvalue, measure)
    rv = rv.func(*[_canonical(i) for i in ordered(Rel)] + nonRel + nonRealRel)
    return rv

def _apply_patternbased_twoterm_simplification(Rel, patterns, func, dominatingvalue, replacementvalue, measure):
    if False:
        return 10
    ' Apply pattern-based two-term simplification.'
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.core.relational import Ge, Gt, _Inequality
    changed = True
    while changed and len(Rel) >= 2:
        changed = False
        Rel = [r.reversed if isinstance(r, (Ge, Gt)) else r for r in Rel]
        Rel = list(ordered(Rel))
        rtmp = [(r,) if isinstance(r, _Inequality) else (r, r.reversed) for r in Rel]
        results = []
        for ((i, pi), (j, pj)) in combinations(enumerate(rtmp), 2):
            for (pattern, simp) in patterns:
                res = []
                for (p1, p2) in product(pi, pj):
                    oldexpr = Tuple(p1, p2)
                    tmpres = oldexpr.match(pattern)
                    if tmpres:
                        res.append((tmpres, oldexpr))
                if res:
                    for (tmpres, oldexpr) in res:
                        np = simp.xreplace(tmpres)
                        if np == dominatingvalue:
                            return [replacementvalue]
                        if not isinstance(np, ITE) and (not np.has(Min, Max)):
                            costsaving = measure(func(*oldexpr.args)) - measure(np)
                            if costsaving > 0:
                                results.append((costsaving, ([i, j], np)))
        if results:
            results = sorted(results, key=lambda pair: pair[0], reverse=True)
            replacement = results[0][1]
            (idx, newrel) = replacement
            idx.sort()
            for index in reversed(idx):
                del Rel[index]
            if dominatingvalue is None or newrel != Not(dominatingvalue):
                if newrel.func == func:
                    for a in newrel.args:
                        Rel.append(a)
                else:
                    Rel.append(newrel)
            changed = True
    return Rel

def _apply_patternbased_threeterm_simplification(Rel, patterns, func, dominatingvalue, replacementvalue, measure):
    if False:
        print('Hello World!')
    ' Apply pattern-based three-term simplification.'
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.core.relational import Le, Lt, _Inequality
    changed = True
    while changed and len(Rel) >= 3:
        changed = False
        Rel = [r.reversed if isinstance(r, (Le, Lt)) else r for r in Rel]
        Rel = list(ordered(Rel))
        results = []
        rtmp = [(r,) if isinstance(r, _Inequality) else (r, r.reversed) for r in Rel]
        for ((i, pi), (j, pj), (k, pk)) in permutations(enumerate(rtmp), 3):
            for (pattern, simp) in patterns:
                res = []
                for (p1, p2, p3) in product(pi, pj, pk):
                    oldexpr = Tuple(p1, p2, p3)
                    tmpres = oldexpr.match(pattern)
                    if tmpres:
                        res.append((tmpres, oldexpr))
                if res:
                    for (tmpres, oldexpr) in res:
                        np = simp.xreplace(tmpres)
                        if np == dominatingvalue:
                            return [replacementvalue]
                        if not isinstance(np, ITE) and (not np.has(Min, Max)):
                            costsaving = measure(func(*oldexpr.args)) - measure(np)
                            if costsaving > 0:
                                results.append((costsaving, ([i, j, k], np)))
        if results:
            results = sorted(results, key=lambda pair: pair[0], reverse=True)
            replacement = results[0][1]
            (idx, newrel) = replacement
            idx.sort()
            for index in reversed(idx):
                del Rel[index]
            if dominatingvalue is None or newrel != Not(dominatingvalue):
                if newrel.func == func:
                    for a in newrel.args:
                        Rel.append(a)
                else:
                    Rel.append(newrel)
            changed = True
    return Rel

@cacheit
def _simplify_patterns_and():
    if False:
        print('Hello World!')
    ' Two-term patterns for And.'
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.miscellaneous import Min, Max
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    _matchers_and = ((Tuple(Eq(a, b), Lt(a, b)), false), (Tuple(Lt(b, a), Lt(a, b)), false), (Tuple(Eq(a, b), Le(b, a)), Eq(a, b)), (Tuple(Le(b, a), Le(a, b)), Eq(a, b)), (Tuple(Le(a, b), Lt(a, b)), Lt(a, b)), (Tuple(Le(a, b), Ne(a, b)), Lt(a, b)), (Tuple(Lt(a, b), Ne(a, b)), Lt(a, b)), (Tuple(Eq(a, b), Eq(a, -b)), And(Eq(a, S.Zero), Eq(b, S.Zero))), (Tuple(Le(b, a), Le(c, a)), Ge(a, Max(b, c))), (Tuple(Le(b, a), Lt(c, a)), ITE(b > c, Ge(a, b), Gt(a, c))), (Tuple(Lt(b, a), Lt(c, a)), Gt(a, Max(b, c))), (Tuple(Le(a, b), Le(a, c)), Le(a, Min(b, c))), (Tuple(Le(a, b), Lt(a, c)), ITE(b < c, Le(a, b), Lt(a, c))), (Tuple(Lt(a, b), Lt(a, c)), Lt(a, Min(b, c))), (Tuple(Le(a, b), Le(c, a)), ITE(Eq(b, c), Eq(a, b), ITE(b < c, false, And(Le(a, b), Ge(a, c))))), (Tuple(Le(c, a), Le(a, b)), ITE(Eq(b, c), Eq(a, b), ITE(b < c, false, And(Le(a, b), Ge(a, c))))), (Tuple(Lt(a, b), Lt(c, a)), ITE(b < c, false, And(Lt(a, b), Gt(a, c)))), (Tuple(Lt(c, a), Lt(a, b)), ITE(b < c, false, And(Lt(a, b), Gt(a, c)))), (Tuple(Le(a, b), Lt(c, a)), ITE(b <= c, false, And(Le(a, b), Gt(a, c)))), (Tuple(Le(c, a), Lt(a, b)), ITE(b <= c, false, And(Lt(a, b), Ge(a, c)))), (Tuple(Eq(a, b), Eq(a, c)), ITE(Eq(b, c), Eq(a, b), false)), (Tuple(Lt(a, b), Lt(-b, a)), ITE(b > 0, Lt(Abs(a), b), false)), (Tuple(Le(a, b), Le(-b, a)), ITE(b >= 0, Le(Abs(a), b), false)))
    return _matchers_and

@cacheit
def _simplify_patterns_and3():
    if False:
        for i in range(10):
            print('nop')
    ' Three-term patterns for And.'
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ge, Gt
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    _matchers_and = ((Tuple(Ge(a, b), Ge(b, c), Gt(c, a)), false), (Tuple(Ge(a, b), Gt(b, c), Gt(c, a)), false), (Tuple(Gt(a, b), Gt(b, c), Gt(c, a)), false), (Tuple(Ge(a, b), Ge(a, c), Ge(b, c)), And(Ge(a, b), Ge(b, c))), (Tuple(Ge(a, b), Ge(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))), (Tuple(Ge(a, b), Gt(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))), (Tuple(Ge(a, c), Gt(a, b), Gt(b, c)), And(Gt(a, b), Gt(b, c))), (Tuple(Ge(b, c), Gt(a, b), Gt(a, c)), And(Gt(a, b), Ge(b, c))), (Tuple(Gt(a, b), Gt(a, c), Gt(b, c)), And(Gt(a, b), Gt(b, c))), (Tuple(Ge(b, a), Ge(c, a), Ge(b, c)), And(Ge(c, a), Ge(b, c))), (Tuple(Ge(b, a), Ge(c, a), Gt(b, c)), And(Ge(c, a), Gt(b, c))), (Tuple(Ge(b, a), Gt(c, a), Gt(b, c)), And(Gt(c, a), Gt(b, c))), (Tuple(Ge(c, a), Gt(b, a), Gt(b, c)), And(Ge(c, a), Gt(b, c))), (Tuple(Ge(b, c), Gt(b, a), Gt(c, a)), And(Gt(c, a), Ge(b, c))), (Tuple(Gt(b, a), Gt(c, a), Gt(b, c)), And(Gt(c, a), Gt(b, c))), (Tuple(Ge(a, b), Ge(b, c), Ge(c, a)), And(Eq(a, b), Eq(b, c))))
    return _matchers_and

@cacheit
def _simplify_patterns_or():
    if False:
        return 10
    ' Two-term patterns for Or.'
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.miscellaneous import Min, Max
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    _matchers_or = ((Tuple(Le(b, a), Le(a, b)), true), (Tuple(Le(b, a), Ne(a, b)), true), (Tuple(Eq(a, b), Le(a, b)), Le(a, b)), (Tuple(Eq(a, b), Lt(a, b)), Le(a, b)), (Tuple(Lt(b, a), Lt(a, b)), Ne(a, b)), (Tuple(Lt(b, a), Ne(a, b)), Ne(a, b)), (Tuple(Le(a, b), Lt(a, b)), Le(a, b)), (Tuple(Eq(a, b), Ne(a, c)), ITE(Eq(b, c), true, Ne(a, c))), (Tuple(Ne(a, b), Ne(a, c)), ITE(Eq(b, c), Ne(a, b), true)), (Tuple(Le(b, a), Le(c, a)), Ge(a, Min(b, c))), (Tuple(Le(b, a), Lt(c, a)), ITE(b > c, Lt(c, a), Le(b, a))), (Tuple(Lt(b, a), Lt(c, a)), Gt(a, Min(b, c))), (Tuple(Le(a, b), Le(a, c)), Le(a, Max(b, c))), (Tuple(Le(a, b), Lt(a, c)), ITE(b >= c, Le(a, b), Lt(a, c))), (Tuple(Lt(a, b), Lt(a, c)), Lt(a, Max(b, c))), (Tuple(Le(a, b), Le(c, a)), ITE(b >= c, true, Or(Le(a, b), Ge(a, c)))), (Tuple(Le(c, a), Le(a, b)), ITE(b >= c, true, Or(Le(a, b), Ge(a, c)))), (Tuple(Lt(a, b), Lt(c, a)), ITE(b > c, true, Or(Lt(a, b), Gt(a, c)))), (Tuple(Lt(c, a), Lt(a, b)), ITE(b > c, true, Or(Lt(a, b), Gt(a, c)))), (Tuple(Le(a, b), Lt(c, a)), ITE(b >= c, true, Or(Le(a, b), Gt(a, c)))), (Tuple(Le(c, a), Lt(a, b)), ITE(b >= c, true, Or(Lt(a, b), Ge(a, c)))), (Tuple(Lt(b, a), Lt(a, -b)), ITE(b >= 0, Gt(Abs(a), b), true)), (Tuple(Le(b, a), Le(a, -b)), ITE(b > 0, Ge(Abs(a), b), true)))
    return _matchers_or

@cacheit
def _simplify_patterns_xor():
    if False:
        i = 10
        return i + 15
    ' Two-term patterns for Xor.'
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    _matchers_xor = ((Tuple(Eq(a, b), Le(a, b)), Lt(a, b)), (Tuple(Eq(a, b), Lt(a, b)), Le(a, b)), (Tuple(Le(a, b), Lt(a, b)), Eq(a, b)), (Tuple(Le(a, b), Le(b, a)), Ne(a, b)), (Tuple(Le(b, a), Ne(a, b)), Le(a, b)), (Tuple(Lt(b, a), Ne(a, b)), Lt(a, b)), (Tuple(Le(b, a), Le(c, a)), And(Ge(a, Min(b, c)), Lt(a, Max(b, c)))), (Tuple(Le(b, a), Lt(c, a)), ITE(b > c, And(Gt(a, c), Lt(a, b)), And(Ge(a, b), Le(a, c)))), (Tuple(Lt(b, a), Lt(c, a)), And(Gt(a, Min(b, c)), Le(a, Max(b, c)))), (Tuple(Le(a, b), Le(a, c)), And(Le(a, Max(b, c)), Gt(a, Min(b, c)))), (Tuple(Le(a, b), Lt(a, c)), ITE(b < c, And(Lt(a, c), Gt(a, b)), And(Le(a, b), Ge(a, c)))), (Tuple(Lt(a, b), Lt(a, c)), And(Lt(a, Max(b, c)), Ge(a, Min(b, c)))))
    return _matchers_xor

def simplify_univariate(expr):
    if False:
        while True:
            i = 10
    'return a simplified version of univariate boolean expression, else ``expr``'
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.core.relational import Eq, Ne
    if not isinstance(expr, BooleanFunction):
        return expr
    if expr.atoms(Eq, Ne):
        return expr
    c = expr
    free = c.free_symbols
    if len(free) != 1:
        return c
    x = free.pop()
    (ok, i) = Piecewise((0, c), evaluate=False)._intervals(x, err_on_Eq=True)
    if not ok:
        return c
    if not i:
        return false
    args = []
    for (a, b, _, _) in i:
        if a is S.NegativeInfinity:
            if b is S.Infinity:
                c = true
            elif c.subs(x, b) == True:
                c = x <= b
            else:
                c = x < b
        else:
            incl_a = c.subs(x, a) == True
            incl_b = c.subs(x, b) == True
            if incl_a and incl_b:
                if b.is_infinite:
                    c = x >= a
                else:
                    c = And(a <= x, x <= b)
            elif incl_a:
                c = And(a <= x, x < b)
            elif incl_b:
                if b.is_infinite:
                    c = x > a
                else:
                    c = And(a < x, x <= b)
            else:
                c = And(a < x, x < b)
        args.append(c)
    return Or(*args)
BooleanGates = (And, Or, Xor, Nand, Nor, Not, Xnor, ITE)

def gateinputcount(expr):
    if False:
        i = 10
        return i + 15
    '\n    Return the total number of inputs for the logic gates realizing the\n    Boolean expression.\n\n    Returns\n    =======\n\n    int\n        Number of gate inputs\n\n    Note\n    ====\n\n    Not all Boolean functions count as gate here, only those that are\n    considered to be standard gates. These are: :py:class:`~.And`,\n    :py:class:`~.Or`, :py:class:`~.Xor`, :py:class:`~.Not`, and\n    :py:class:`~.ITE` (multiplexer). :py:class:`~.Nand`, :py:class:`~.Nor`,\n    and :py:class:`~.Xnor` will be evaluated to ``Not(And())`` etc.\n\n    Examples\n    ========\n\n    >>> from sympy.logic import And, Or, Nand, Not, gateinputcount\n    >>> from sympy.abc import x, y, z\n    >>> expr = And(x, y)\n    >>> gateinputcount(expr)\n    2\n    >>> gateinputcount(Or(expr, z))\n    4\n\n    Note that ``Nand`` is automatically evaluated to ``Not(And())`` so\n\n    >>> gateinputcount(Nand(x, y, z))\n    4\n    >>> gateinputcount(Not(And(x, y, z)))\n    4\n\n    Although this can be avoided by using ``evaluate=False``\n\n    >>> gateinputcount(Nand(x, y, z, evaluate=False))\n    3\n\n    Also note that a comparison will count as a Boolean variable:\n\n    >>> gateinputcount(And(x > z, y >= 2))\n    2\n\n    As will a symbol:\n    >>> gateinputcount(x)\n    0\n\n    '
    if not isinstance(expr, Boolean):
        raise TypeError('Expression must be Boolean')
    if isinstance(expr, BooleanGates):
        return len(expr.args) + sum((gateinputcount(x) for x in expr.args))
    return 0