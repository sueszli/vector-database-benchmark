"""sympify -- convert objects SymPy internal format"""
from __future__ import annotations
from typing import Any, Callable
from inspect import getmro
import string
from sympy.core.random import choice
from .parameters import global_parameters
from sympy.utilities.iterables import iterable

class SympifyError(ValueError):

    def __init__(self, expr, base_exc=None):
        if False:
            for i in range(10):
                print('nop')
        self.expr = expr
        self.base_exc = base_exc

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if self.base_exc is None:
            return 'SympifyError: %r' % (self.expr,)
        return "Sympify of expression '%s' failed, because of exception being raised:\n%s: %s" % (self.expr, self.base_exc.__class__.__name__, str(self.base_exc))
converter: dict[type[Any], Callable[[Any], Basic]] = {}
_sympy_converter: dict[type[Any], Callable[[Any], Basic]] = {}
_external_converter = converter

class CantSympify:
    """
    Mix in this trait to a class to disallow sympification of its instances.

    Examples
    ========

    >>> from sympy import sympify
    >>> from sympy.core.sympify import CantSympify

    >>> class Something(dict):
    ...     pass
    ...
    >>> sympify(Something())
    {}

    >>> class Something(dict, CantSympify):
    ...     pass
    ...
    >>> sympify(Something())
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: {}

    """
    __slots__ = ()

def _is_numpy_instance(a):
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks if an object is an instance of a type from the numpy module.\n    '
    return any((type_.__module__ == 'numpy' for type_ in type(a).__mro__))

def _convert_numpy_types(a, **sympify_args):
    if False:
        while True:
            i = 10
    '\n    Converts a numpy datatype input to an appropriate SymPy type.\n    '
    import numpy as np
    if not isinstance(a, np.floating):
        if np.iscomplex(a):
            return _sympy_converter[complex](a.item())
        else:
            return sympify(a.item(), **sympify_args)
    else:
        from .numbers import Float
        prec = np.finfo(a).nmant + 1
        a = str(list(np.reshape(np.asarray(a), (1, np.size(a)))[0]))[1:-1]
        return Float(a, precision=prec)

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None):
    if False:
        return 10
    '\n    Converts an arbitrary expression to a type that can be used inside SymPy.\n\n    Explanation\n    ===========\n\n    It will convert Python ints into instances of :class:`~.Integer`, floats\n    into instances of :class:`~.Float`, etc. It is also able to coerce\n    symbolic expressions which inherit from :class:`~.Basic`. This can be\n    useful in cooperation with SAGE.\n\n    .. warning::\n        Note that this function uses ``eval``, and thus shouldn\'t be used on\n        unsanitized input.\n\n    If the argument is already a type that SymPy understands, it will do\n    nothing but return that value. This can be used at the beginning of a\n    function to ensure you are working with the correct type.\n\n    Examples\n    ========\n\n    >>> from sympy import sympify\n\n    >>> sympify(2).is_integer\n    True\n    >>> sympify(2).is_real\n    True\n\n    >>> sympify(2.0).is_real\n    True\n    >>> sympify("2.0").is_real\n    True\n    >>> sympify("2e-45").is_real\n    True\n\n    If the expression could not be converted, a SympifyError is raised.\n\n    >>> sympify("x***2")\n    Traceback (most recent call last):\n    ...\n    SympifyError: SympifyError: "could not parse \'x***2\'"\n\n    Locals\n    ------\n\n    The sympification happens with access to everything that is loaded\n    by ``from sympy import *``; anything used in a string that is not\n    defined by that import will be converted to a symbol. In the following,\n    the ``bitcount`` function is treated as a symbol and the ``O`` is\n    interpreted as the :class:`~.Order` object (used with series) and it raises\n    an error when used improperly:\n\n    >>> s = \'bitcount(42)\'\n    >>> sympify(s)\n    bitcount(42)\n    >>> sympify("O(x)")\n    O(x)\n    >>> sympify("O + 1")\n    Traceback (most recent call last):\n    ...\n    TypeError: unbound method...\n\n    In order to have ``bitcount`` be recognized it can be imported into a\n    namespace dictionary and passed as locals:\n\n    >>> ns = {}\n    >>> exec(\'from sympy.core.evalf import bitcount\', ns)\n    >>> sympify(s, locals=ns)\n    6\n\n    In order to have the ``O`` interpreted as a Symbol, identify it as such\n    in the namespace dictionary. This can be done in a variety of ways; all\n    three of the following are possibilities:\n\n    >>> from sympy import Symbol\n    >>> ns["O"] = Symbol("O")  # method 1\n    >>> exec(\'from sympy.abc import O\', ns)  # method 2\n    >>> ns.update(dict(O=Symbol("O")))  # method 3\n    >>> sympify("O + 1", locals=ns)\n    O + 1\n\n    If you want *all* single-letter and Greek-letter variables to be symbols\n    then you can use the clashing-symbols dictionaries that have been defined\n    there as private variables: ``_clash1`` (single-letter variables),\n    ``_clash2`` (the multi-letter Greek names) or ``_clash`` (both single and\n    multi-letter names that are defined in ``abc``).\n\n    >>> from sympy.abc import _clash1\n    >>> set(_clash1)  # if this fails, see issue #23903\n    {\'E\', \'I\', \'N\', \'O\', \'Q\', \'S\'}\n    >>> sympify(\'I & Q\', _clash1)\n    I & Q\n\n    Strict\n    ------\n\n    If the option ``strict`` is set to ``True``, only the types for which an\n    explicit conversion has been defined are converted. In the other\n    cases, a SympifyError is raised.\n\n    >>> print(sympify(None))\n    None\n    >>> sympify(None, strict=True)\n    Traceback (most recent call last):\n    ...\n    SympifyError: SympifyError: None\n\n    .. deprecated:: 1.6\n\n       ``sympify(obj)`` automatically falls back to ``str(obj)`` when all\n       other conversion methods fail, but this is deprecated. ``strict=True``\n       will disable this deprecated behavior. See\n       :ref:`deprecated-sympify-string-fallback`.\n\n    Evaluation\n    ----------\n\n    If the option ``evaluate`` is set to ``False``, then arithmetic and\n    operators will be converted into their SymPy equivalents and the\n    ``evaluate=False`` option will be added. Nested ``Add`` or ``Mul`` will\n    be denested first. This is done via an AST transformation that replaces\n    operators with their SymPy equivalents, so if an operand redefines any\n    of those operations, the redefined operators will not be used. If\n    argument a is not a string, the mathematical expression is evaluated\n    before being passed to sympify, so adding ``evaluate=False`` will still\n    return the evaluated result of expression.\n\n    >>> sympify(\'2**2 / 3 + 5\')\n    19/3\n    >>> sympify(\'2**2 / 3 + 5\', evaluate=False)\n    2**2/3 + 5\n    >>> sympify(\'4/2+7\', evaluate=True)\n    9\n    >>> sympify(\'4/2+7\', evaluate=False)\n    4/2 + 7\n    >>> sympify(4/2+7, evaluate=False)\n    9.00000000000000\n\n    Extending\n    ---------\n\n    To extend ``sympify`` to convert custom objects (not derived from ``Basic``),\n    just define a ``_sympy_`` method to your class. You can do that even to\n    classes that you do not own by subclassing or adding the method at runtime.\n\n    >>> from sympy import Matrix\n    >>> class MyList1(object):\n    ...     def __iter__(self):\n    ...         yield 1\n    ...         yield 2\n    ...         return\n    ...     def __getitem__(self, i): return list(self)[i]\n    ...     def _sympy_(self): return Matrix(self)\n    >>> sympify(MyList1())\n    Matrix([\n    [1],\n    [2]])\n\n    If you do not have control over the class definition you could also use the\n    ``converter`` global dictionary. The key is the class and the value is a\n    function that takes a single argument and returns the desired SymPy\n    object, e.g. ``converter[MyList] = lambda x: Matrix(x)``.\n\n    >>> class MyList2(object):   # XXX Do not do this if you control the class!\n    ...     def __iter__(self):  #     Use _sympy_!\n    ...         yield 1\n    ...         yield 2\n    ...         return\n    ...     def __getitem__(self, i): return list(self)[i]\n    >>> from sympy.core.sympify import converter\n    >>> converter[MyList2] = lambda x: Matrix(x)\n    >>> sympify(MyList2())\n    Matrix([\n    [1],\n    [2]])\n\n    Notes\n    =====\n\n    The keywords ``rational`` and ``convert_xor`` are only used\n    when the input is a string.\n\n    convert_xor\n    -----------\n\n    >>> sympify(\'x^y\',convert_xor=True)\n    x**y\n    >>> sympify(\'x^y\',convert_xor=False)\n    x ^ y\n\n    rational\n    --------\n\n    >>> sympify(\'0.1\',rational=False)\n    0.1\n    >>> sympify(\'0.1\',rational=True)\n    1/10\n\n    Sometimes autosimplification during sympification results in expressions\n    that are very different in structure than what was entered. Until such\n    autosimplification is no longer done, the ``kernS`` function might be of\n    some use. In the example below you can see how an expression reduces to\n    $-1$ by autosimplification, but does not do so when ``kernS`` is used.\n\n    >>> from sympy.core.sympify import kernS\n    >>> from sympy.abc import x\n    >>> -2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1\n    -1\n    >>> s = \'-2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1\'\n    >>> sympify(s)\n    -1\n    >>> kernS(s)\n    -2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1\n\n    Parameters\n    ==========\n\n    a :\n        - any object defined in SymPy\n        - standard numeric Python types: ``int``, ``long``, ``float``, ``Decimal``\n        - strings (like ``"0.09"``, ``"2e-19"`` or ``\'sin(x)\'``)\n        - booleans, including ``None`` (will leave ``None`` unchanged)\n        - dicts, lists, sets or tuples containing any of the above\n\n    convert_xor : bool, optional\n        If true, treats ``^`` as exponentiation.\n        If False, treats ``^`` as XOR itself.\n        Used only when input is a string.\n\n    locals : any object defined in SymPy, optional\n        In order to have strings be recognized it can be imported\n        into a namespace dictionary and passed as locals.\n\n    strict : bool, optional\n        If the option strict is set to ``True``, only the types for which\n        an explicit conversion has been defined are converted. In the\n        other cases, a SympifyError is raised.\n\n    rational : bool, optional\n        If ``True``, converts floats into :class:`~.Rational`.\n        If ``False``, it lets floats remain as it is.\n        Used only when input is a string.\n\n    evaluate : bool, optional\n        If False, then arithmetic and operators will be converted into\n        their SymPy equivalents. If True the expression will be evaluated\n        and the result will be returned.\n\n    '
    is_sympy = getattr(a, '__sympy__', None)
    if is_sympy is True:
        return a
    elif is_sympy is not None:
        if not strict:
            return a
        else:
            raise SympifyError(a)
    if isinstance(a, CantSympify):
        raise SympifyError(a)
    cls = getattr(a, '__class__', None)
    for superclass in getmro(cls):
        conv = _external_converter.get(superclass)
        if conv is None:
            conv = _sympy_converter.get(superclass)
        if conv is not None:
            return conv(a)
    if cls is type(None):
        if strict:
            raise SympifyError(a)
        else:
            return a
    if evaluate is None:
        evaluate = global_parameters.evaluate
    if _is_numpy_instance(a):
        import numpy as np
        if np.isscalar(a):
            return _convert_numpy_types(a, locals=locals, convert_xor=convert_xor, strict=strict, rational=rational, evaluate=evaluate)
    _sympy_ = getattr(a, '_sympy_', None)
    if _sympy_ is not None:
        return a._sympy_()
    if not strict:
        flat = getattr(a, 'flat', None)
        if flat is not None:
            shape = getattr(a, 'shape', None)
            if shape is not None:
                from sympy.tensor.array import Array
                return Array(a.flat, a.shape)
    if not isinstance(a, str):
        if _is_numpy_instance(a):
            import numpy as np
            assert not isinstance(a, np.number)
            if isinstance(a, np.ndarray):
                if a.ndim == 0:
                    try:
                        return sympify(a.item(), locals=locals, convert_xor=convert_xor, strict=strict, rational=rational, evaluate=evaluate)
                    except SympifyError:
                        pass
        elif hasattr(a, '__float__'):
            return sympify(float(a))
        elif hasattr(a, '__int__'):
            return sympify(int(a))
    if strict:
        raise SympifyError(a)
    if iterable(a):
        try:
            return type(a)([sympify(x, locals=locals, convert_xor=convert_xor, rational=rational, evaluate=evaluate) for x in a])
        except TypeError:
            pass
    if not isinstance(a, str):
        raise SympifyError('cannot sympify object of type %r' % type(a))
    from sympy.parsing.sympy_parser import parse_expr, TokenError, standard_transformations
    from sympy.parsing.sympy_parser import convert_xor as t_convert_xor
    from sympy.parsing.sympy_parser import rationalize as t_rationalize
    transformations = standard_transformations
    if rational:
        transformations += (t_rationalize,)
    if convert_xor:
        transformations += (t_convert_xor,)
    try:
        a = a.replace('\n', '')
        expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
    except (TokenError, SyntaxError) as exc:
        raise SympifyError('could not parse %r' % a, exc)
    return expr

def _sympify(a):
    if False:
        return 10
    "\n    Short version of :func:`~.sympify` for internal usage for ``__add__`` and\n    ``__eq__`` methods where it is ok to allow some things (like Python\n    integers and floats) in the expression. This excludes things (like strings)\n    that are unwise to allow into such an expression.\n\n    >>> from sympy import Integer\n    >>> Integer(1) == 1\n    True\n\n    >>> Integer(1) == '1'\n    False\n\n    >>> from sympy.abc import x\n    >>> x + 1\n    x + 1\n\n    >>> x + '1'\n    Traceback (most recent call last):\n    ...\n    TypeError: unsupported operand type(s) for +: 'Symbol' and 'str'\n\n    see: sympify\n\n    "
    return sympify(a, strict=True)

def kernS(s):
    if False:
        print('Hello World!')
    "Use a hack to try keep autosimplification from distributing a\n    a number into an Add; this modification does not\n    prevent the 2-arg Mul from becoming an Add, however.\n\n    Examples\n    ========\n\n    >>> from sympy.core.sympify import kernS\n    >>> from sympy.abc import x, y\n\n    The 2-arg Mul distributes a number (or minus sign) across the terms\n    of an expression, but kernS will prevent that:\n\n    >>> 2*(x + y), -(x + 1)\n    (2*x + 2*y, -x - 1)\n    >>> kernS('2*(x + y)')\n    2*(x + y)\n    >>> kernS('-(x + 1)')\n    -(x + 1)\n\n    If use of the hack fails, the un-hacked string will be passed to sympify...\n    and you get what you get.\n\n    XXX This hack should not be necessary once issue 4596 has been resolved.\n    "
    hit = False
    quoted = '"' in s or "'" in s
    if '(' in s and (not quoted):
        if s.count('(') != s.count(')'):
            raise SympifyError('unmatched left parenthesis')
        s = ''.join(s.split())
        olds = s
        s = s.replace('*(', '* *(')
        s = s.replace('** *', '**')
        target = '-( *('
        s = s.replace('-(', target)
        i = nest = 0
        assert target.endswith('(')
        while True:
            j = s.find(target, i)
            if j == -1:
                break
            j += len(target) - 1
            for j in range(j, len(s)):
                if s[j] == '(':
                    nest += 1
                elif s[j] == ')':
                    nest -= 1
                if nest == 0:
                    break
            s = s[:j] + ')' + s[j:]
            i = j + 2
        if ' ' in s:
            kern = '_'
            while kern in s:
                kern += choice(string.ascii_letters + string.digits)
            s = s.replace(' ', kern)
            hit = kern in s
        else:
            hit = False
    for i in range(2):
        try:
            expr = sympify(s)
            break
        except TypeError:
            if hit:
                s = olds
                hit = False
                continue
            expr = sympify(s)
    if not hit:
        return expr
    from .symbol import Symbol
    rep = {Symbol(kern): 1}

    def _clear(expr):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(expr, (list, tuple, set)):
            return type(expr)([_clear(e) for e in expr])
        if hasattr(expr, 'subs'):
            return expr.subs(rep, hack2=True)
        return expr
    expr = _clear(expr)
    return expr
from .basic import Basic