from collections import defaultdict
from .sympify import sympify, SympifyError
from sympy.utilities.iterables import iterable, uniq
__all__ = ['default_sort_key', 'ordered']

def default_sort_key(item, order=None):
    if False:
        for i in range(10):
            print('nop')
    "Return a key that can be used for sorting.\n\n    The key has the structure:\n\n    (class_key, (len(args), args), exponent.sort_key(), coefficient)\n\n    This key is supplied by the sort_key routine of Basic objects when\n    ``item`` is a Basic object or an object (other than a string) that\n    sympifies to a Basic object. Otherwise, this function produces the\n    key.\n\n    The ``order`` argument is passed along to the sort_key routine and is\n    used to determine how the terms *within* an expression are ordered.\n    (See examples below) ``order`` options are: 'lex', 'grlex', 'grevlex',\n    and reversed values of the same (e.g. 'rev-lex'). The default order\n    value is None (which translates to 'lex').\n\n    Examples\n    ========\n\n    >>> from sympy import S, I, default_sort_key, sin, cos, sqrt\n    >>> from sympy.core.function import UndefinedFunction\n    >>> from sympy.abc import x\n\n    The following are equivalent ways of getting the key for an object:\n\n    >>> x.sort_key() == default_sort_key(x)\n    True\n\n    Here are some examples of the key that is produced:\n\n    >>> default_sort_key(UndefinedFunction('f'))\n    ((0, 0, 'UndefinedFunction'), (1, ('f',)), ((1, 0, 'Number'),\n        (0, ()), (), 1), 1)\n    >>> default_sort_key('1')\n    ((0, 0, 'str'), (1, ('1',)), ((1, 0, 'Number'), (0, ()), (), 1), 1)\n    >>> default_sort_key(S.One)\n    ((1, 0, 'Number'), (0, ()), (), 1)\n    >>> default_sort_key(2)\n    ((1, 0, 'Number'), (0, ()), (), 2)\n\n    While sort_key is a method only defined for SymPy objects,\n    default_sort_key will accept anything as an argument so it is\n    more robust as a sorting key. For the following, using key=\n    lambda i: i.sort_key() would fail because 2 does not have a sort_key\n    method; that's why default_sort_key is used. Note, that it also\n    handles sympification of non-string items likes ints:\n\n    >>> a = [2, I, -I]\n    >>> sorted(a, key=default_sort_key)\n    [2, -I, I]\n\n    The returned key can be used anywhere that a key can be specified for\n    a function, e.g. sort, min, max, etc...:\n\n    >>> a.sort(key=default_sort_key); a[0]\n    2\n    >>> min(a, key=default_sort_key)\n    2\n\n    Notes\n    =====\n\n    The key returned is useful for getting items into a canonical order\n    that will be the same across platforms. It is not directly useful for\n    sorting lists of expressions:\n\n    >>> a, b = x, 1/x\n\n    Since ``a`` has only 1 term, its value of sort_key is unaffected by\n    ``order``:\n\n    >>> a.sort_key() == a.sort_key('rev-lex')\n    True\n\n    If ``a`` and ``b`` are combined then the key will differ because there\n    are terms that can be ordered:\n\n    >>> eq = a + b\n    >>> eq.sort_key() == eq.sort_key('rev-lex')\n    False\n    >>> eq.as_ordered_terms()\n    [x, 1/x]\n    >>> eq.as_ordered_terms('rev-lex')\n    [1/x, x]\n\n    But since the keys for each of these terms are independent of ``order``'s\n    value, they do not sort differently when they appear separately in a list:\n\n    >>> sorted(eq.args, key=default_sort_key)\n    [1/x, x]\n    >>> sorted(eq.args, key=lambda i: default_sort_key(i, order='rev-lex'))\n    [1/x, x]\n\n    The order of terms obtained when using these keys is the order that would\n    be obtained if those terms were *factors* in a product.\n\n    Although it is useful for quickly putting expressions in canonical order,\n    it does not sort expressions based on their complexity defined by the\n    number of operations, power of variables and others:\n\n    >>> sorted([sin(x)*cos(x), sin(x)], key=default_sort_key)\n    [sin(x)*cos(x), sin(x)]\n    >>> sorted([x, x**2, sqrt(x), x**3], key=default_sort_key)\n    [sqrt(x), x, x**2, x**3]\n\n    See Also\n    ========\n\n    ordered, sympy.core.expr.Expr.as_ordered_factors, sympy.core.expr.Expr.as_ordered_terms\n\n    "
    from .basic import Basic
    from .singleton import S
    if isinstance(item, Basic):
        return item.sort_key(order=order)
    if iterable(item, exclude=str):
        if isinstance(item, dict):
            args = item.items()
            unordered = True
        elif isinstance(item, set):
            args = item
            unordered = True
        else:
            args = list(item)
            unordered = False
        args = [default_sort_key(arg, order=order) for arg in args]
        if unordered:
            args = sorted(args)
        (cls_index, args) = (10, (len(args), tuple(args)))
    else:
        if not isinstance(item, str):
            try:
                item = sympify(item, strict=True)
            except SympifyError:
                pass
            else:
                if isinstance(item, Basic):
                    return default_sort_key(item)
        (cls_index, args) = (0, (1, (str(item),)))
    return ((cls_index, 0, item.__class__.__name__), args, S.One.sort_key(), S.One)

def _node_count(e):
    if False:
        while True:
            i = 10
    if e.is_Float:
        return 0.5
    return 1 + sum(map(_node_count, e.args))

def _nodes(e):
    if False:
        for i in range(10):
            print('nop')
    '\n    A helper for ordered() which returns the node count of ``e`` which\n    for Basic objects is the number of Basic nodes in the expression tree\n    but for other objects is 1 (unless the object is an iterable or dict\n    for which the sum of nodes is returned).\n    '
    from .basic import Basic
    from .function import Derivative
    if isinstance(e, Basic):
        if isinstance(e, Derivative):
            return _nodes(e.expr) + sum((i[1] if i[1].is_Number else _nodes(i[1]) for i in e.variable_count))
        return _node_count(e)
    elif iterable(e):
        return 1 + sum((_nodes(ei) for ei in e))
    elif isinstance(e, dict):
        return 1 + sum((_nodes(k) + _nodes(v) for (k, v) in e.items()))
    else:
        return 1

def ordered(seq, keys=None, default=True, warn=False):
    if False:
        i = 10
        return i + 15
    'Return an iterator of the seq where keys are used to break ties\n    in a conservative fashion: if, after applying a key, there are no\n    ties then no other keys will be computed.\n\n    Two default keys will be applied if 1) keys are not provided or\n    2) the given keys do not resolve all ties (but only if ``default``\n    is True). The two keys are ``_nodes`` (which places smaller\n    expressions before large) and ``default_sort_key`` which (if the\n    ``sort_key`` for an object is defined properly) should resolve\n    any ties. This strategy is similar to sorting done by\n    ``Basic.compare``, but differs in that ``ordered`` never makes a\n    decision based on an objects name.\n\n    If ``warn`` is True then an error will be raised if there were no\n    keys remaining to break ties. This can be used if it was expected that\n    there should be no ties between items that are not identical.\n\n    Examples\n    ========\n\n    >>> from sympy import ordered, count_ops\n    >>> from sympy.abc import x, y\n\n    The count_ops is not sufficient to break ties in this list and the first\n    two items appear in their original order (i.e. the sorting is stable):\n\n    >>> list(ordered([y + 2, x + 2, x**2 + y + 3],\n    ...    count_ops, default=False, warn=False))\n    ...\n    [y + 2, x + 2, x**2 + y + 3]\n\n    The default_sort_key allows the tie to be broken:\n\n    >>> list(ordered([y + 2, x + 2, x**2 + y + 3]))\n    ...\n    [x + 2, y + 2, x**2 + y + 3]\n\n    Here, sequences are sorted by length, then sum:\n\n    >>> seq, keys = [[[1, 2, 1], [0, 3, 1], [1, 1, 3], [2], [1]], [\n    ...    lambda x: len(x),\n    ...    lambda x: sum(x)]]\n    ...\n    >>> list(ordered(seq, keys, default=False, warn=False))\n    [[1], [2], [1, 2, 1], [0, 3, 1], [1, 1, 3]]\n\n    If ``warn`` is True, an error will be raised if there were not\n    enough keys to break ties:\n\n    >>> list(ordered(seq, keys, default=False, warn=True))\n    Traceback (most recent call last):\n    ...\n    ValueError: not enough keys to break ties\n\n\n    Notes\n    =====\n\n    The decorated sort is one of the fastest ways to sort a sequence for\n    which special item comparison is desired: the sequence is decorated,\n    sorted on the basis of the decoration (e.g. making all letters lower\n    case) and then undecorated. If one wants to break ties for items that\n    have the same decorated value, a second key can be used. But if the\n    second key is expensive to compute then it is inefficient to decorate\n    all items with both keys: only those items having identical first key\n    values need to be decorated. This function applies keys successively\n    only when needed to break ties. By yielding an iterator, use of the\n    tie-breaker is delayed as long as possible.\n\n    This function is best used in cases when use of the first key is\n    expected to be a good hashing function; if there are no unique hashes\n    from application of a key, then that key should not have been used. The\n    exception, however, is that even if there are many collisions, if the\n    first group is small and one does not need to process all items in the\n    list then time will not be wasted sorting what one was not interested\n    in. For example, if one were looking for the minimum in a list and\n    there were several criteria used to define the sort order, then this\n    function would be good at returning that quickly if the first group\n    of candidates is small relative to the number of items being processed.\n\n    '
    d = defaultdict(list)
    if keys:
        if isinstance(keys, (list, tuple)):
            keys = list(keys)
            f = keys.pop(0)
        else:
            f = keys
            keys = []
        for a in seq:
            d[f(a)].append(a)
    else:
        if not default:
            raise ValueError('if default=False then keys must be provided')
        d[None].extend(seq)
    for (k, value) in sorted(d.items()):
        if len(value) > 1:
            if keys:
                value = ordered(value, keys, default, warn)
            elif default:
                value = ordered(value, (_nodes, default_sort_key), default=False, warn=warn)
            elif warn:
                u = list(uniq(value))
                if len(u) > 1:
                    raise ValueError('not enough keys to break ties: %s' % u)
        yield from value