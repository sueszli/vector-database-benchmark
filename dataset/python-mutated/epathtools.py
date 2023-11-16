"""Tools for manipulation of expressions using paths. """
from sympy.core import Basic

class EPath:
    """
    Manipulate expressions using paths.

    EPath grammar in EBNF notation::

        literal   ::= /[A-Za-z_][A-Za-z_0-9]*/
        number    ::= /-?\\d+/
        type      ::= literal
        attribute ::= literal "?"
        all       ::= "*"
        slice     ::= "[" number? (":" number? (":" number?)?)? "]"
        range     ::= all | slice
        query     ::= (type | attribute) ("|" (type | attribute))*
        selector  ::= range | query range?
        path      ::= "/" selector ("/" selector)*

    See the docstring of the epath() function.

    """
    __slots__ = ('_path', '_epath')

    def __new__(cls, path):
        if False:
            while True:
                i = 10
        'Construct new EPath. '
        if isinstance(path, EPath):
            return path
        if not path:
            raise ValueError('empty EPath')
        _path = path
        if path[0] == '/':
            path = path[1:]
        else:
            raise NotImplementedError('non-root EPath')
        epath = []
        for selector in path.split('/'):
            selector = selector.strip()
            if not selector:
                raise ValueError('empty selector')
            index = 0
            for c in selector:
                if c.isalnum() or c in ('_', '|', '?'):
                    index += 1
                else:
                    break
            attrs = []
            types = []
            if index:
                elements = selector[:index]
                selector = selector[index:]
                for element in elements.split('|'):
                    element = element.strip()
                    if not element:
                        raise ValueError('empty element')
                    if element.endswith('?'):
                        attrs.append(element[:-1])
                    else:
                        types.append(element)
            span = None
            if selector == '*':
                pass
            else:
                if selector.startswith('['):
                    try:
                        i = selector.index(']')
                    except ValueError:
                        raise ValueError("expected ']', got EOL")
                    (_span, span) = (selector[1:i], [])
                    if ':' not in _span:
                        span = int(_span)
                    else:
                        for elt in _span.split(':', 3):
                            if not elt:
                                span.append(None)
                            else:
                                span.append(int(elt))
                        span = slice(*span)
                    selector = selector[i + 1:]
                if selector:
                    raise ValueError('trailing characters in selector')
            epath.append((attrs, types, span))
        obj = object.__new__(cls)
        obj._path = _path
        obj._epath = epath
        return obj

    def __repr__(self):
        if False:
            return 10
        return '%s(%r)' % (self.__class__.__name__, self._path)

    def _get_ordered_args(self, expr):
        if False:
            print('Hello World!')
        'Sort ``expr.args`` using printing order. '
        if expr.is_Add:
            return expr.as_ordered_terms()
        elif expr.is_Mul:
            return expr.as_ordered_factors()
        else:
            return expr.args

    def _hasattrs(self, expr, attrs):
        if False:
            return 10
        'Check if ``expr`` has any of ``attrs``. '
        for attr in attrs:
            if not hasattr(expr, attr):
                return False
        return True

    def _hastypes(self, expr, types):
        if False:
            i = 10
            return i + 15
        'Check if ``expr`` is any of ``types``. '
        _types = [cls.__name__ for cls in expr.__class__.mro()]
        return bool(set(_types).intersection(types))

    def _has(self, expr, attrs, types):
        if False:
            return 10
        'Apply ``_hasattrs`` and ``_hastypes`` to ``expr``. '
        if not (attrs or types):
            return True
        if attrs and self._hasattrs(expr, attrs):
            return True
        if types and self._hastypes(expr, types):
            return True
        return False

    def apply(self, expr, func, args=None, kwargs=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Modify parts of an expression selected by a path.\n\n        Examples\n        ========\n\n        >>> from sympy.simplify.epathtools import EPath\n        >>> from sympy import sin, cos, E\n        >>> from sympy.abc import x, y, z, t\n\n        >>> path = EPath("/*/[0]/Symbol")\n        >>> expr = [((x, 1), 2), ((3, y), z)]\n\n        >>> path.apply(expr, lambda expr: expr**2)\n        [((x**2, 1), 2), ((3, y**2), z)]\n\n        >>> path = EPath("/*/*/Symbol")\n        >>> expr = t + sin(x + 1) + cos(x + y + E)\n\n        >>> path.apply(expr, lambda expr: 2*expr)\n        t + sin(2*x + 1) + cos(2*x + 2*y + E)\n\n        '

        def _apply(path, expr, func):
            if False:
                while True:
                    i = 10
            if not path:
                return func(expr)
            else:
                (selector, path) = (path[0], path[1:])
                (attrs, types, span) = selector
                if isinstance(expr, Basic):
                    if not expr.is_Atom:
                        (args, basic) = (self._get_ordered_args(expr), True)
                    else:
                        return expr
                elif hasattr(expr, '__iter__'):
                    (args, basic) = (expr, False)
                else:
                    return expr
                args = list(args)
                if span is not None:
                    if isinstance(span, slice):
                        indices = range(*span.indices(len(args)))
                    else:
                        indices = [span]
                else:
                    indices = range(len(args))
                for i in indices:
                    try:
                        arg = args[i]
                    except IndexError:
                        continue
                    if self._has(arg, attrs, types):
                        args[i] = _apply(path, arg, func)
                if basic:
                    return expr.func(*args)
                else:
                    return expr.__class__(args)
        (_args, _kwargs) = (args or (), kwargs or {})
        _func = lambda expr: func(expr, *_args, **_kwargs)
        return _apply(self._epath, expr, _func)

    def select(self, expr):
        if False:
            print('Hello World!')
        '\n        Retrieve parts of an expression selected by a path.\n\n        Examples\n        ========\n\n        >>> from sympy.simplify.epathtools import EPath\n        >>> from sympy import sin, cos, E\n        >>> from sympy.abc import x, y, z, t\n\n        >>> path = EPath("/*/[0]/Symbol")\n        >>> expr = [((x, 1), 2), ((3, y), z)]\n\n        >>> path.select(expr)\n        [x, y]\n\n        >>> path = EPath("/*/*/Symbol")\n        >>> expr = t + sin(x + 1) + cos(x + y + E)\n\n        >>> path.select(expr)\n        [x, x, y]\n\n        '
        result = []

        def _select(path, expr):
            if False:
                print('Hello World!')
            if not path:
                result.append(expr)
            else:
                (selector, path) = (path[0], path[1:])
                (attrs, types, span) = selector
                if isinstance(expr, Basic):
                    args = self._get_ordered_args(expr)
                elif hasattr(expr, '__iter__'):
                    args = expr
                else:
                    return
                if span is not None:
                    if isinstance(span, slice):
                        args = args[span]
                    else:
                        try:
                            args = [args[span]]
                        except IndexError:
                            return
                for arg in args:
                    if self._has(arg, attrs, types):
                        _select(path, arg)
        _select(self._epath, expr)
        return result

def epath(path, expr=None, func=None, args=None, kwargs=None):
    if False:
        print('Hello World!')
    '\n    Manipulate parts of an expression selected by a path.\n\n    Explanation\n    ===========\n\n    This function allows to manipulate large nested expressions in single\n    line of code, utilizing techniques to those applied in XML processing\n    standards (e.g. XPath).\n\n    If ``func`` is ``None``, :func:`epath` retrieves elements selected by\n    the ``path``. Otherwise it applies ``func`` to each matching element.\n\n    Note that it is more efficient to create an EPath object and use the select\n    and apply methods of that object, since this will compile the path string\n    only once.  This function should only be used as a convenient shortcut for\n    interactive use.\n\n    This is the supported syntax:\n\n    * select all: ``/*``\n          Equivalent of ``for arg in args:``.\n    * select slice: ``/[0]`` or ``/[1:5]`` or ``/[1:5:2]``\n          Supports standard Python\'s slice syntax.\n    * select by type: ``/list`` or ``/list|tuple``\n          Emulates ``isinstance()``.\n    * select by attribute: ``/__iter__?``\n          Emulates ``hasattr()``.\n\n    Parameters\n    ==========\n\n    path : str | EPath\n        A path as a string or a compiled EPath.\n    expr : Basic | iterable\n        An expression or a container of expressions.\n    func : callable (optional)\n        A callable that will be applied to matching parts.\n    args : tuple (optional)\n        Additional positional arguments to ``func``.\n    kwargs : dict (optional)\n        Additional keyword arguments to ``func``.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.epathtools import epath\n    >>> from sympy import sin, cos, E\n    >>> from sympy.abc import x, y, z, t\n\n    >>> path = "/*/[0]/Symbol"\n    >>> expr = [((x, 1), 2), ((3, y), z)]\n\n    >>> epath(path, expr)\n    [x, y]\n    >>> epath(path, expr, lambda expr: expr**2)\n    [((x**2, 1), 2), ((3, y**2), z)]\n\n    >>> path = "/*/*/Symbol"\n    >>> expr = t + sin(x + 1) + cos(x + y + E)\n\n    >>> epath(path, expr)\n    [x, x, y]\n    >>> epath(path, expr, lambda expr: 2*expr)\n    t + sin(2*x + 1) + cos(2*x + 2*y + E)\n\n    '
    _epath = EPath(path)
    if expr is None:
        return _epath
    if func is None:
        return _epath.select(expr)
    else:
        return _epath.apply(expr, func, args, kwargs)