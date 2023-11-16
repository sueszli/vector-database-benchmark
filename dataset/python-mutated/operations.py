from __future__ import annotations
from operator import attrgetter
from collections import defaultdict
from sympy.utilities.exceptions import sympy_deprecation_warning
from .sympify import _sympify as _sympify_, sympify
from .basic import Basic
from .cache import cacheit
from .sorting import ordered
from .logic import fuzzy_and
from .parameters import global_parameters
from sympy.utilities.iterables import sift
from sympy.multipledispatch.dispatcher import Dispatcher, ambiguity_register_error_ignore_dup, str_signature, RaiseNotImplementedError

class AssocOp(Basic):
    """ Associative operations, can separate noncommutative and
    commutative parts.

    (a op b) op c == a op (b op c) == a op b op c.

    Base class for Add and Mul.

    This is an abstract base class, concrete derived classes must define
    the attribute `identity`.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Parameters
    ==========

    *args :
        Arguments which are operated

    evaluate : bool, optional
        Evaluate the operation. If not passed, refer to ``global_parameters.evaluate``.
    """
    __slots__: tuple[str, ...] = ('is_commutative',)
    _args_type: type[Basic] | None = None

    @cacheit
    def __new__(cls, *args, evaluate=None, _sympify=True):
        if False:
            while True:
                i = 10
        if _sympify:
            args = list(map(_sympify_, args))
        typ = cls._args_type
        if typ is not None:
            from .relational import Relational
            if any((isinstance(arg, Relational) for arg in args)):
                raise TypeError('Relational cannot be used in %s' % cls.__name__)
            for arg in args:
                if not isinstance(arg, typ):
                    sympy_deprecation_warning(f'\n\nUsing non-Expr arguments in {cls.__name__} is deprecated (in this case, one of\nthe arguments has type {type(arg).__name__!r}).\n\nIf you really did intend to use a multiplication or addition operation with\nthis object, use the * or + operator instead.\n\n                        ', deprecated_since_version='1.7', active_deprecations_target='non-expr-args-deprecated', stacklevel=4)
        if evaluate is None:
            evaluate = global_parameters.evaluate
        if not evaluate:
            obj = cls._from_args(args)
            obj = cls._exec_constructor_postprocessors(obj)
            return obj
        args = [a for a in args if a is not cls.identity]
        if len(args) == 0:
            return cls.identity
        if len(args) == 1:
            return args[0]
        (c_part, nc_part, order_symbols) = cls.flatten(args)
        is_commutative = not nc_part
        obj = cls._from_args(c_part + nc_part, is_commutative)
        obj = cls._exec_constructor_postprocessors(obj)
        if order_symbols is not None:
            from sympy.series.order import Order
            return Order(obj, *order_symbols)
        return obj

    @classmethod
    def _from_args(cls, args, is_commutative=None):
        if False:
            print('Hello World!')
        'Create new instance with already-processed args.\n        If the args are not in canonical order, then a non-canonical\n        result will be returned, so use with caution. The order of\n        args may change if the sign of the args is changed.'
        if len(args) == 0:
            return cls.identity
        elif len(args) == 1:
            return args[0]
        obj = super().__new__(cls, *args)
        if is_commutative is None:
            is_commutative = fuzzy_and((a.is_commutative for a in args))
        obj.is_commutative = is_commutative
        return obj

    def _new_rawargs(self, *args, reeval=True, **kwargs):
        if False:
            while True:
                i = 10
        'Create new instance of own class with args exactly as provided by\n        caller but returning the self class identity if args is empty.\n\n        Examples\n        ========\n\n           This is handy when we want to optimize things, e.g.\n\n               >>> from sympy import Mul, S\n               >>> from sympy.abc import x, y\n               >>> e = Mul(3, x, y)\n               >>> e.args\n               (3, x, y)\n               >>> Mul(*e.args[1:])\n               x*y\n               >>> e._new_rawargs(*e.args[1:])  # the same as above, but faster\n               x*y\n\n           Note: use this with caution. There is no checking of arguments at\n           all. This is best used when you are rebuilding an Add or Mul after\n           simply removing one or more args. If, for example, modifications,\n           result in extra 1s being inserted they will show up in the result:\n\n               >>> m = (x*y)._new_rawargs(S.One, x); m\n               1*x\n               >>> m == x\n               False\n               >>> m.is_Mul\n               True\n\n           Another issue to be aware of is that the commutativity of the result\n           is based on the commutativity of self. If you are rebuilding the\n           terms that came from a commutative object then there will be no\n           problem, but if self was non-commutative then what you are\n           rebuilding may now be commutative.\n\n           Although this routine tries to do as little as possible with the\n           input, getting the commutativity right is important, so this level\n           of safety is enforced: commutativity will always be recomputed if\n           self is non-commutative and kwarg `reeval=False` has not been\n           passed.\n        '
        if reeval and self.is_commutative is False:
            is_commutative = None
        else:
            is_commutative = self.is_commutative
        return self._from_args(args, is_commutative)

    @classmethod
    def flatten(cls, seq):
        if False:
            for i in range(10):
                print('nop')
        'Return seq so that none of the elements are of type `cls`. This is\n        the vanilla routine that will be used if a class derived from AssocOp\n        does not define its own flatten routine.'
        new_seq = []
        while seq:
            o = seq.pop()
            if o.__class__ is cls:
                seq.extend(o.args)
            else:
                new_seq.append(o)
        new_seq.reverse()
        return ([], new_seq, None)

    def _matches_commutative(self, expr, repl_dict=None, old=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Matches Add/Mul "pattern" to an expression "expr".\n\n        repl_dict ... a dictionary of (wild: expression) pairs, that get\n                      returned with the results\n\n        This function is the main workhorse for Add/Mul.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols, Wild, sin\n        >>> a = Wild("a")\n        >>> b = Wild("b")\n        >>> c = Wild("c")\n        >>> x, y, z = symbols("x y z")\n        >>> (a+sin(b)*c)._matches_commutative(x+sin(y)*z)\n        {a_: x, b_: y, c_: z}\n\n        In the example above, "a+sin(b)*c" is the pattern, and "x+sin(y)*z" is\n        the expression.\n\n        The repl_dict contains parts that were already matched. For example\n        here:\n\n        >>> (x+sin(b)*c)._matches_commutative(x+sin(y)*z, repl_dict={a: x})\n        {a_: x, b_: y, c_: z}\n\n        the only function of the repl_dict is to return it in the\n        result, e.g. if you omit it:\n\n        >>> (x+sin(b)*c)._matches_commutative(x+sin(y)*z)\n        {b_: y, c_: z}\n\n        the "a: x" is not returned in the result, but otherwise it is\n        equivalent.\n\n        '
        from .function import _coeff_isneg
        from .expr import Expr
        if isinstance(self, Expr) and (not isinstance(expr, Expr)):
            return None
        if repl_dict is None:
            repl_dict = {}
        if self == expr:
            return repl_dict
        d = self._matches_simple(expr, repl_dict)
        if d is not None:
            return d
        from .function import WildFunction
        from .symbol import Wild
        (wild_part, exact_part) = sift(self.args, lambda p: p.has(Wild, WildFunction) and (not expr.has(p)), binary=True)
        if not exact_part:
            wild_part = list(ordered(wild_part))
            if self.is_Add:
                wild_part = sorted(wild_part, key=lambda x: x.args[0] if x.is_Mul and x.args[0].is_Number else 0)
        else:
            exact = self._new_rawargs(*exact_part)
            free = expr.free_symbols
            if free and exact.free_symbols - free:
                return None
            newexpr = self._combine_inverse(expr, exact)
            if not old and (expr.is_Add or expr.is_Mul):
                check = newexpr
                if _coeff_isneg(check):
                    check = -check
                if check.count_ops() > expr.count_ops():
                    return None
            newpattern = self._new_rawargs(*wild_part)
            return newpattern.matches(newexpr, repl_dict)
        i = 0
        saw = set()
        while expr not in saw:
            saw.add(expr)
            args = tuple(ordered(self.make_args(expr)))
            if self.is_Add and expr.is_Add:
                args = tuple(sorted(args, key=lambda x: x.args[0] if x.is_Mul and x.args[0].is_Number else 0))
            expr_list = (self.identity,) + args
            for last_op in reversed(expr_list):
                for w in reversed(wild_part):
                    d1 = w.matches(last_op, repl_dict)
                    if d1 is not None:
                        d2 = self.xreplace(d1).matches(expr, d1)
                        if d2 is not None:
                            return d2
            if i == 0:
                if self.is_Mul:
                    if expr.is_Pow and expr.exp.is_Integer:
                        from .mul import Mul
                        if expr.exp > 0:
                            expr = Mul(*[expr.base, expr.base ** (expr.exp - 1)], evaluate=False)
                        else:
                            expr = Mul(*[1 / expr.base, expr.base ** (expr.exp + 1)], evaluate=False)
                        i += 1
                        continue
                elif self.is_Add:
                    (c, e) = expr.as_coeff_Mul()
                    if abs(c) > 1:
                        from .add import Add
                        if c > 0:
                            expr = Add(*[e, (c - 1) * e], evaluate=False)
                        else:
                            expr = Add(*[-e, (c + 1) * e], evaluate=False)
                        i += 1
                        continue
                    from sympy.simplify.radsimp import collect
                    was = expr
                    did = set()
                    for w in reversed(wild_part):
                        (c, w) = w.as_coeff_mul(Wild)
                        free = c.free_symbols - did
                        if free:
                            did.update(free)
                            expr = collect(expr, free)
                    if expr != was:
                        i += 0
                        continue
                break
        return

    def _has_matcher(self):
        if False:
            while True:
                i = 10
        'Helper for .has() that checks for containment of\n        subexpressions within an expr by using sets of args\n        of similar nodes, e.g. x + 1 in x + y + 1 checks\n        to see that {x, 1} & {x, y, 1} == {x, 1}\n        '

        def _ncsplit(expr):
            if False:
                while True:
                    i = 10
            (cpart, ncpart) = sift(expr.args, lambda arg: arg.is_commutative is True, binary=True)
            return (set(cpart), ncpart)
        (c, nc) = _ncsplit(self)
        cls = self.__class__

        def is_in(expr):
            if False:
                return 10
            if isinstance(expr, cls):
                if expr == self:
                    return True
                (_c, _nc) = _ncsplit(expr)
                if c & _c == c:
                    if not nc:
                        return True
                    elif len(nc) <= len(_nc):
                        for i in range(len(_nc) - len(nc) + 1):
                            if _nc[i:i + len(nc)] == nc:
                                return True
            return False
        return is_in

    def _eval_evalf(self, prec):
        if False:
            while True:
                i = 10
        "\n        Evaluate the parts of self that are numbers; if the whole thing\n        was a number with no functions it would have been evaluated, but\n        it wasn't so we must judiciously extract the numbers and reconstruct\n        the object. This is *not* simply replacing numbers with evaluated\n        numbers. Numbers should be handled in the largest pure-number\n        expression as possible. So the code below separates ``self`` into\n        number and non-number parts and evaluates the number parts and\n        walks the args of the non-number part recursively (doing the same\n        thing).\n        "
        from .add import Add
        from .mul import Mul
        from .symbol import Symbol
        from .function import AppliedUndef
        if isinstance(self, (Mul, Add)):
            (x, tail) = self.as_independent(Symbol, AppliedUndef)
            if not (tail is self.identity or (isinstance(x, AssocOp) and x.is_Function) or (x is self.identity and isinstance(tail, AssocOp))):
                x = x._evalf(prec) if x is not self.identity else self.identity
                args = []
                tail_args = tuple(self.func.make_args(tail))
                for a in tail_args:
                    newa = a._eval_evalf(prec)
                    if newa is None:
                        args.append(a)
                    else:
                        args.append(newa)
                return self.func(x, *args)
        args = []
        for a in self.args:
            newa = a._eval_evalf(prec)
            if newa is None:
                args.append(a)
            else:
                args.append(newa)
        return self.func(*args)

    @classmethod
    def make_args(cls, expr):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a sequence of elements `args` such that cls(*args) == expr\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol, Mul, Add\n        >>> x, y = map(Symbol, 'xy')\n\n        >>> Mul.make_args(x*y)\n        (x, y)\n        >>> Add.make_args(x*y)\n        (x*y,)\n        >>> set(Add.make_args(x*y + y)) == set([y, x*y])\n        True\n\n        "
        if isinstance(expr, cls):
            return expr.args
        else:
            return (sympify(expr),)

    def doit(self, **hints):
        if False:
            while True:
                i = 10
        if hints.get('deep', True):
            terms = [term.doit(**hints) for term in self.args]
        else:
            terms = self.args
        return self.func(*terms, evaluate=True)

class ShortCircuit(Exception):
    pass

class LatticeOp(AssocOp):
    """
    Join/meet operations of an algebraic lattice[1].

    Explanation
    ===========

    These binary operations are associative (op(op(a, b), c) = op(a, op(b, c))),
    commutative (op(a, b) = op(b, a)) and idempotent (op(a, a) = op(a) = a).
    Common examples are AND, OR, Union, Intersection, max or min. They have an
    identity element (op(identity, a) = a) and an absorbing element
    conventionally called zero (op(zero, a) = zero).

    This is an abstract base class, concrete derived classes must declare
    attributes zero and identity. All defining properties are then respected.

    Examples
    ========

    >>> from sympy import Integer
    >>> from sympy.core.operations import LatticeOp
    >>> class my_join(LatticeOp):
    ...     zero = Integer(0)
    ...     identity = Integer(1)
    >>> my_join(2, 3) == my_join(3, 2)
    True
    >>> my_join(2, my_join(3, 4)) == my_join(2, 3, 4)
    True
    >>> my_join(0, 1, 4, 2, 3, 4)
    0
    >>> my_join(1, 2)
    2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lattice_%28order%29
    """
    is_commutative = True

    def __new__(cls, *args, **options):
        if False:
            return 10
        args = (_sympify_(arg) for arg in args)
        try:
            _args = frozenset(cls._new_args_filter(args))
        except ShortCircuit:
            return sympify(cls.zero)
        if not _args:
            return sympify(cls.identity)
        elif len(_args) == 1:
            return set(_args).pop()
        else:
            obj = super(AssocOp, cls).__new__(cls, *ordered(_args))
            obj._argset = _args
            return obj

    @classmethod
    def _new_args_filter(cls, arg_sequence, call_cls=None):
        if False:
            i = 10
            return i + 15
        'Generator filtering args'
        ncls = call_cls or cls
        for arg in arg_sequence:
            if arg == ncls.zero:
                raise ShortCircuit(arg)
            elif arg == ncls.identity:
                continue
            elif arg.func == ncls:
                yield from arg.args
            else:
                yield arg

    @classmethod
    def make_args(cls, expr):
        if False:
            print('Hello World!')
        '\n        Return a set of args such that cls(*arg_set) == expr.\n        '
        if isinstance(expr, cls):
            return expr._argset
        else:
            return frozenset([sympify(expr)])

class AssocOpDispatcher:
    """
    Handler dispatcher for associative operators

    .. notes::
       This approach is experimental, and can be replaced or deleted in the future.
       See https://github.com/sympy/sympy/pull/19463.

    Explanation
    ===========

    If arguments of different types are passed, the classes which handle the operation for each type
    are collected. Then, a class which performs the operation is selected by recursive binary dispatching.
    Dispatching relation can be registered by ``register_handlerclass`` method.

    Priority registration is unordered. You cannot make ``A*B`` and ``B*A`` refer to
    different handler classes. All logic dealing with the order of arguments must be implemented
    in the handler class.

    Examples
    ========

    >>> from sympy import Add, Expr, Symbol
    >>> from sympy.core.add import add

    >>> class NewExpr(Expr):
    ...     @property
    ...     def _add_handler(self):
    ...         return NewAdd
    >>> class NewAdd(NewExpr, Add):
    ...     pass
    >>> add.register_handlerclass((Add, NewAdd), NewAdd)

    >>> a, b = Symbol('a'), NewExpr()
    >>> add(a, b) == NewAdd(a, b)
    True

    """

    def __init__(self, name, doc=None):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.doc = doc
        self.handlerattr = '_%s_handler' % name
        self._handlergetter = attrgetter(self.handlerattr)
        self._dispatcher = Dispatcher(name)

    def __repr__(self):
        if False:
            return 10
        return '<dispatched %s>' % self.name

    def register_handlerclass(self, classes, typ, on_ambiguity=ambiguity_register_error_ignore_dup):
        if False:
            i = 10
            return i + 15
        '\n        Register the handler class for two classes, in both straight and reversed order.\n\n        Paramteters\n        ===========\n\n        classes : tuple of two types\n            Classes who are compared with each other.\n\n        typ:\n            Class which is registered to represent *cls1* and *cls2*.\n            Handler method of *self* must be implemented in this class.\n        '
        if not len(classes) == 2:
            raise RuntimeError('Only binary dispatch is supported, but got %s types: <%s>.' % (len(classes), str_signature(classes)))
        if len(set(classes)) == 1:
            raise RuntimeError('Duplicate types <%s> cannot be dispatched.' % str_signature(classes))
        self._dispatcher.add(tuple(classes), typ, on_ambiguity=on_ambiguity)
        self._dispatcher.add(tuple(reversed(classes)), typ, on_ambiguity=on_ambiguity)

    @cacheit
    def __call__(self, *args, _sympify=True, **kwargs):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ==========\n\n        *args :\n            Arguments which are operated\n        '
        if _sympify:
            args = tuple(map(_sympify_, args))
        handlers = frozenset(map(self._handlergetter, args))
        return self.dispatch(handlers)(*args, _sympify=False, **kwargs)

    @cacheit
    def dispatch(self, handlers):
        if False:
            print('Hello World!')
        '\n        Select the handler class, and return its handler method.\n        '
        if len(handlers) == 1:
            (h,) = handlers
            if not isinstance(h, type):
                raise RuntimeError('Handler {!r} is not a type.'.format(h))
            return h
        for (i, typ) in enumerate(handlers):
            if not isinstance(typ, type):
                raise RuntimeError('Handler {!r} is not a type.'.format(typ))
            if i == 0:
                handler = typ
            else:
                prev_handler = handler
                handler = self._dispatcher.dispatch(prev_handler, typ)
                if not isinstance(handler, type):
                    raise RuntimeError('Dispatcher for {!r} and {!r} must return a type, but got {!r}'.format(prev_handler, typ, handler))
        return handler

    @property
    def __doc__(self):
        if False:
            return 10
        docs = ['Multiply dispatched associative operator: %s' % self.name, 'Note that support for this is experimental, see the docs for :class:`AssocOpDispatcher` for details']
        if self.doc:
            docs.append(self.doc)
        s = 'Registered handler classes\n'
        s += '=' * len(s)
        docs.append(s)
        amb_sigs = []
        typ_sigs = defaultdict(list)
        for sigs in self._dispatcher.ordering[::-1]:
            key = self._dispatcher.funcs[sigs]
            typ_sigs[key].append(sigs)
        for (typ, sigs) in typ_sigs.items():
            sigs_str = ', '.join(('<%s>' % str_signature(sig) for sig in sigs))
            if isinstance(typ, RaiseNotImplementedError):
                amb_sigs.append(sigs_str)
                continue
            s = 'Inputs: %s\n' % sigs_str
            s += '-' * len(s) + '\n'
            s += typ.__name__
            docs.append(s)
        if amb_sigs:
            s = 'Ambiguous handler classes\n'
            s += '=' * len(s)
            docs.append(s)
            s = '\n'.join(amb_sigs)
            docs.append(s)
        return '\n\n'.join(docs)