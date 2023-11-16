"""A module which implements predicates and assumption context."""
from contextlib import contextmanager
import inspect
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import Boolean, false, true
from sympy.multipledispatch.dispatcher import Dispatcher, str_signature
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.source import get_class

class AssumptionsContext(set):
    """
    Set containing default assumptions which are applied to the ``ask()``
    function.

    Explanation
    ===========

    This is used to represent global assumptions, but you can also use this
    class to create your own local assumptions contexts. It is basically a thin
    wrapper to Python's set, so see its documentation for advanced usage.

    Examples
    ========

    The default assumption context is ``global_assumptions``, which is initially empty:

    >>> from sympy import ask, Q
    >>> from sympy.assumptions import global_assumptions
    >>> global_assumptions
    AssumptionsContext()

    You can add default assumptions:

    >>> from sympy.abc import x
    >>> global_assumptions.add(Q.real(x))
    >>> global_assumptions
    AssumptionsContext({Q.real(x)})
    >>> ask(Q.real(x))
    True

    And remove them:

    >>> global_assumptions.remove(Q.real(x))
    >>> print(ask(Q.real(x)))
    None

    The ``clear()`` method removes every assumption:

    >>> global_assumptions.add(Q.positive(x))
    >>> global_assumptions
    AssumptionsContext({Q.positive(x)})
    >>> global_assumptions.clear()
    >>> global_assumptions
    AssumptionsContext()

    See Also
    ========

    assuming

    """

    def add(self, *assumptions):
        if False:
            print('Hello World!')
        'Add assumptions.'
        for a in assumptions:
            super().add(a)

    def _sympystr(self, printer):
        if False:
            for i in range(10):
                print('nop')
        if not self:
            return '%s()' % self.__class__.__name__
        return '{}({})'.format(self.__class__.__name__, printer._print_set(self))
global_assumptions = AssumptionsContext()

class AppliedPredicate(Boolean):
    """
    The class of expressions resulting from applying ``Predicate`` to
    the arguments. ``AppliedPredicate`` merely wraps its argument and
    remain unevaluated. To evaluate it, use the ``ask()`` function.

    Examples
    ========

    >>> from sympy import Q, ask
    >>> Q.integer(1)
    Q.integer(1)

    The ``function`` attribute returns the predicate, and the ``arguments``
    attribute returns the tuple of arguments.

    >>> type(Q.integer(1))
    <class 'sympy.assumptions.assume.AppliedPredicate'>
    >>> Q.integer(1).function
    Q.integer
    >>> Q.integer(1).arguments
    (1,)

    Applied predicates can be evaluated to a boolean value with ``ask``:

    >>> ask(Q.integer(1))
    True

    """
    __slots__ = ()

    def __new__(cls, predicate, *args):
        if False:
            print('Hello World!')
        if not isinstance(predicate, Predicate):
            raise TypeError('%s is not a Predicate.' % predicate)
        args = map(_sympify, args)
        return super().__new__(cls, predicate, *args)

    @property
    def arg(self):
        if False:
            print('Hello World!')
        "\n        Return the expression used by this assumption.\n\n        Examples\n        ========\n\n        >>> from sympy import Q, Symbol\n        >>> x = Symbol('x')\n        >>> a = Q.integer(x + 1)\n        >>> a.arg\n        x + 1\n\n        "
        args = self._args
        if len(args) == 2:
            return args[1]
        raise TypeError("'arg' property is allowed only for unary predicates.")

    @property
    def function(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the predicate.\n        '
        return self._args[0]

    @property
    def arguments(self):
        if False:
            return 10
        '\n        Return the arguments which are applied to the predicate.\n        '
        return self._args[1:]

    def _eval_ask(self, assumptions):
        if False:
            i = 10
            return i + 15
        return self.function.eval(self.arguments, assumptions)

    @property
    def binary_symbols(self):
        if False:
            i = 10
            return i + 15
        from .ask import Q
        if self.function == Q.is_true:
            i = self.arguments[0]
            if i.is_Boolean or i.is_Symbol:
                return i.binary_symbols
        if self.function in (Q.eq, Q.ne):
            if true in self.arguments or false in self.arguments:
                if self.arguments[0].is_Symbol:
                    return {self.arguments[0]}
                elif self.arguments[1].is_Symbol:
                    return {self.arguments[1]}
        return set()

class PredicateMeta(type):

    def __new__(cls, clsname, bases, dct):
        if False:
            while True:
                i = 10
        if 'handler' not in dct:
            name = f'Ask{clsname.capitalize()}Handler'
            handler = Dispatcher(name, doc='Handler for key %s' % name)
            dct['handler'] = handler
        dct['_orig_doc'] = dct.get('__doc__', '')
        return super().__new__(cls, clsname, bases, dct)

    @property
    def __doc__(cls):
        if False:
            for i in range(10):
                print('nop')
        handler = cls.handler
        doc = cls._orig_doc
        if cls is not Predicate and handler is not None:
            doc += 'Handler\n'
            doc += '    =======\n\n'
            docs = ['    Multiply dispatched method: %s' % handler.name]
            if handler.doc:
                for line in handler.doc.splitlines():
                    if not line:
                        continue
                    docs.append('    %s' % line)
            other = []
            for sig in handler.ordering[::-1]:
                func = handler.funcs[sig]
                if func.__doc__:
                    s = '    Inputs: <%s>' % str_signature(sig)
                    lines = []
                    for line in func.__doc__.splitlines():
                        lines.append('    %s' % line)
                    s += '\n'.join(lines)
                    docs.append(s)
                else:
                    other.append(str_signature(sig))
            if other:
                othersig = '    Other signatures:'
                for line in other:
                    othersig += '\n        * %s' % line
                docs.append(othersig)
            doc += '\n\n'.join(docs)
        return doc

class Predicate(Boolean, metaclass=PredicateMeta):
    """
    Base class for mathematical predicates. It also serves as a
    constructor for undefined predicate objects.

    Explanation
    ===========

    Predicate is a function that returns a boolean value [1].

    Predicate function is object, and it is instance of predicate class.
    When a predicate is applied to arguments, ``AppliedPredicate``
    instance is returned. This merely wraps the argument and remain
    unevaluated. To obtain the truth value of applied predicate, use the
    function ``ask``.

    Evaluation of predicate is done by multiple dispatching. You can
    register new handler to the predicate to support new types.

    Every predicate in SymPy can be accessed via the property of ``Q``.
    For example, ``Q.even`` returns the predicate which checks if the
    argument is even number.

    To define a predicate which can be evaluated, you must subclass this
    class, make an instance of it, and register it to ``Q``. After then,
    dispatch the handler by argument types.

    If you directly construct predicate using this class, you will get
    ``UndefinedPredicate`` which cannot be dispatched. This is useful
    when you are building boolean expressions which do not need to be
    evaluated.

    Examples
    ========

    Applying and evaluating to boolean value:

    >>> from sympy import Q, ask
    >>> ask(Q.prime(7))
    True

    You can define a new predicate by subclassing and dispatching. Here,
    we define a predicate for sexy primes [2] as an example.

    >>> from sympy import Predicate, Integer
    >>> class SexyPrimePredicate(Predicate):
    ...     name = "sexyprime"
    >>> Q.sexyprime = SexyPrimePredicate()
    >>> @Q.sexyprime.register(Integer, Integer)
    ... def _(int1, int2, assumptions):
    ...     args = sorted([int1, int2])
    ...     if not all(ask(Q.prime(a), assumptions) for a in args):
    ...         return False
    ...     return args[1] - args[0] == 6
    >>> ask(Q.sexyprime(5, 11))
    True

    Direct constructing returns ``UndefinedPredicate``, which can be
    applied but cannot be dispatched.

    >>> from sympy import Predicate, Integer
    >>> Q.P = Predicate("P")
    >>> type(Q.P)
    <class 'sympy.assumptions.assume.UndefinedPredicate'>
    >>> Q.P(1)
    Q.P(1)
    >>> Q.P.register(Integer)(lambda expr, assump: True)
    Traceback (most recent call last):
      ...
    TypeError: <class 'sympy.assumptions.assume.UndefinedPredicate'> cannot be dispatched.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Predicate_%28mathematical_logic%29
    .. [2] https://en.wikipedia.org/wiki/Sexy_prime

    """
    is_Atom = True

    def __new__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if cls is Predicate:
            return UndefinedPredicate(*args, **kwargs)
        obj = super().__new__(cls, *args)
        return obj

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return type(self).__name__

    @classmethod
    def register(cls, *types, **kwargs):
        if False:
            print('Hello World!')
        '\n        Register the signature to the handler.\n        '
        if cls.handler is None:
            raise TypeError('%s cannot be dispatched.' % type(cls))
        return cls.handler.register(*types, **kwargs)

    @classmethod
    def register_many(cls, *types, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Register multiple signatures to same handler.\n        '

        def _(func):
            if False:
                return 10
            for t in types:
                if not is_sequence(t):
                    t = (t,)
                cls.register(*t, **kwargs)(func)
        return _

    def __call__(self, *args):
        if False:
            i = 10
            return i + 15
        return AppliedPredicate(self, *args)

    def eval(self, args, assumptions=True):
        if False:
            return 10
        '\n        Evaluate ``self(*args)`` under the given assumptions.\n\n        This uses only direct resolution methods, not logical inference.\n        '
        result = None
        try:
            result = self.handler(*args, assumptions=assumptions)
        except NotImplementedError:
            pass
        return result

    def _eval_refine(self, assumptions):
        if False:
            return 10
        return self

class UndefinedPredicate(Predicate):
    """
    Predicate without handler.

    Explanation
    ===========

    This predicate is generated by using ``Predicate`` directly for
    construction. It does not have a handler, and evaluating this with
    arguments is done by SAT solver.

    Examples
    ========

    >>> from sympy import Predicate, Q
    >>> Q.P = Predicate('P')
    >>> Q.P.func
    <class 'sympy.assumptions.assume.UndefinedPredicate'>
    >>> Q.P.name
    Str('P')

    """
    handler = None

    def __new__(cls, name, handlers=None):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(name, Str):
            name = Str(name)
        obj = super(Boolean, cls).__new__(cls, name)
        obj.handlers = handlers or []
        return obj

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return self.args[0]

    def _hashable_content(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.name,)

    def __getnewargs__(self):
        if False:
            return 10
        return (self.name,)

    def __call__(self, expr):
        if False:
            while True:
                i = 10
        return AppliedPredicate(self, expr)

    def add_handler(self, handler):
        if False:
            for i in range(10):
                print('nop')
        sympy_deprecation_warning('\n            The AskHandler system is deprecated. Predicate.add_handler()\n            should be replaced with the multipledispatch handler of Predicate.\n            ', deprecated_since_version='1.8', active_deprecations_target='deprecated-askhandler')
        self.handlers.append(handler)

    def remove_handler(self, handler):
        if False:
            for i in range(10):
                print('nop')
        sympy_deprecation_warning('\n            The AskHandler system is deprecated. Predicate.remove_handler()\n            should be replaced with the multipledispatch handler of Predicate.\n            ', deprecated_since_version='1.8', active_deprecations_target='deprecated-askhandler')
        self.handlers.remove(handler)

    def eval(self, args, assumptions=True):
        if False:
            for i in range(10):
                print('nop')
        sympy_deprecation_warning('\n            The AskHandler system is deprecated. Evaluating UndefinedPredicate\n            objects should be replaced with the multipledispatch handler of\n            Predicate.\n            ', deprecated_since_version='1.8', active_deprecations_target='deprecated-askhandler', stacklevel=5)
        (expr,) = args
        (res, _res) = (None, None)
        mro = inspect.getmro(type(expr))
        for handler in self.handlers:
            cls = get_class(handler)
            for subclass in mro:
                eval_ = getattr(cls, subclass.__name__, None)
                if eval_ is None:
                    continue
                res = eval_(expr, assumptions)
                if res is None:
                    continue
                if _res is None:
                    _res = res
                elif _res != res:
                    raise ValueError('incompatible resolutors')
                break
        return res

@contextmanager
def assuming(*assumptions):
    if False:
        for i in range(10):
            print('nop')
    '\n    Context manager for assumptions.\n\n    Examples\n    ========\n\n    >>> from sympy import assuming, Q, ask\n    >>> from sympy.abc import x, y\n    >>> print(ask(Q.integer(x + y)))\n    None\n    >>> with assuming(Q.integer(x), Q.integer(y)):\n    ...     print(ask(Q.integer(x + y)))\n    True\n    '
    old_global_assumptions = global_assumptions.copy()
    global_assumptions.update(assumptions)
    try:
        yield
    finally:
        global_assumptions.clear()
        global_assumptions.update(old_global_assumptions)