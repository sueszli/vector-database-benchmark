"""`PEP 3101`_ introduced the :meth:`str.format` method, and what
would later be called "new-style" string formatting. For the sake of
explicit correctness, it is probably best to refer to Python's dual
string formatting capabilities as *bracket-style* and
*percent-style*. There is overlap, but one does not replace the
other.

  * Bracket-style is more pluggable, slower, and uses a method.
  * Percent-style is simpler, faster, and uses an operator.

Bracket-style formatting brought with it a much more powerful toolbox,
but it was far from a full one. :meth:`str.format` uses `more powerful
syntax`_, but `the tools and idioms`_ for working with
that syntax are not well-developed nor well-advertised.

``formatutils`` adds several functions for working with bracket-style
format strings:

  * :class:`DeferredValue`: Defer fetching or calculating a value
    until format time.
  * :func:`get_format_args`: Parse the positional and keyword
    arguments out of a format string.
  * :func:`tokenize_format_str`: Tokenize a format string into
    literals and :class:`BaseFormatField` objects.
  * :func:`construct_format_field_str`: Assists in programmatic
    construction of format strings.
  * :func:`infer_positional_format_args`: Converts anonymous
    references in 2.7+ format strings to explicit positional arguments
    suitable for usage with Python 2.6.

.. _more powerful syntax: https://docs.python.org/2/library/string.html#format-string-syntax
.. _the tools and idioms: https://docs.python.org/2/library/string.html#string-formatting
.. _PEP 3101: https://www.python.org/dev/peps/pep-3101/
"""
from __future__ import print_function
import re
from string import Formatter
try:
    unicode
except NameError:
    unicode = str
__all__ = ['DeferredValue', 'get_format_args', 'tokenize_format_str', 'construct_format_field_str', 'infer_positional_format_args', 'BaseFormatField']
_pos_farg_re = re.compile('({{)|(}})|({[:!.\\[}])')

def construct_format_field_str(fname, fspec, conv):
    if False:
        while True:
            i = 10
    '\n    Constructs a format field string from the field name, spec, and\n    conversion character (``fname``, ``fspec``, ``conv``). See Python\n    String Formatting for more info.\n    '
    if fname is None:
        return ''
    ret = '{' + fname
    if conv:
        ret += '!' + conv
    if fspec:
        ret += ':' + fspec
    ret += '}'
    return ret

def split_format_str(fstr):
    if False:
        while True:
            i = 10
    'Does very basic splitting of a format string, returns a list of\n    strings. For full tokenization, see :func:`tokenize_format_str`.\n\n    '
    ret = []
    for (lit, fname, fspec, conv) in Formatter().parse(fstr):
        if fname is None:
            ret.append((lit, None))
            continue
        field_str = construct_format_field_str(fname, fspec, conv)
        ret.append((lit, field_str))
    return ret

def infer_positional_format_args(fstr):
    if False:
        for i in range(10):
            print('nop')
    'Takes format strings with anonymous positional arguments, (e.g.,\n    "{}" and {:d}), and converts them into numbered ones for explicitness and\n    compatibility with 2.6.\n\n    Returns a string with the inferred positional arguments.\n    '
    (ret, max_anon) = ('', 0)
    (start, end, prev_end) = (0, 0, 0)
    for match in _pos_farg_re.finditer(fstr):
        (start, end, group) = (match.start(), match.end(), match.group())
        if prev_end < start:
            ret += fstr[prev_end:start]
        prev_end = end
        if group == '{{' or group == '}}':
            ret += group
            continue
        ret += '{%s%s' % (max_anon, group[1:])
        max_anon += 1
    ret += fstr[prev_end:]
    return ret
_INTCHARS = 'bcdoxXn'
_FLOATCHARS = 'eEfFgGn%'
_TYPE_MAP = dict([(x, int) for x in _INTCHARS] + [(x, float) for x in _FLOATCHARS])
_TYPE_MAP['s'] = str

def get_format_args(fstr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Turn a format string into two lists of arguments referenced by the\n    format string. One is positional arguments, and the other is named\n    arguments. Each element of the list includes the name and the\n    nominal type of the field.\n\n    # >>> get_format_args("{noun} is {1:d} years old{punct}")\n    # ([(1, <type \'int\'>)], [(\'noun\', <type \'str\'>), (\'punct\', <type \'str\'>)])\n\n    # XXX: Py3k\n    >>> get_format_args("{noun} is {1:d} years old{punct}") ==         ([(1, int)], [(\'noun\', str), (\'punct\', str)])\n    True\n    '
    formatter = Formatter()
    (fargs, fkwargs, _dedup) = ([], [], set())

    def _add_arg(argname, type_char='s'):
        if False:
            i = 10
            return i + 15
        if argname not in _dedup:
            _dedup.add(argname)
            argtype = _TYPE_MAP.get(type_char, str)
            try:
                fargs.append((int(argname), argtype))
            except ValueError:
                fkwargs.append((argname, argtype))
    for (lit, fname, fspec, conv) in formatter.parse(fstr):
        if fname is not None:
            type_char = fspec[-1:]
            fname_list = re.split('[.[]', fname)
            if len(fname_list) > 1:
                raise ValueError('encountered compound format arg: %r' % fname)
            try:
                base_fname = fname_list[0]
                assert base_fname
            except (IndexError, AssertionError):
                raise ValueError('encountered anonymous positional argument')
            _add_arg(fname, type_char)
            for (sublit, subfname, _, _) in formatter.parse(fspec):
                if subfname is not None:
                    _add_arg(subfname)
    return (fargs, fkwargs)

def tokenize_format_str(fstr, resolve_pos=True):
    if False:
        return 10
    'Takes a format string, turns it into a list of alternating string\n    literals and :class:`BaseFormatField` tokens. By default, also\n    infers anonymous positional references into explicit, numbered\n    positional references. To disable this behavior set *resolve_pos*\n    to ``False``.\n    '
    ret = []
    if resolve_pos:
        fstr = infer_positional_format_args(fstr)
    formatter = Formatter()
    for (lit, fname, fspec, conv) in formatter.parse(fstr):
        if lit:
            ret.append(lit)
        if fname is None:
            continue
        ret.append(BaseFormatField(fname, fspec, conv))
    return ret

class BaseFormatField(object):
    """A class representing a reference to an argument inside of a
    bracket-style format string. For instance, in ``"{greeting},
    world!"``, there is a field named "greeting".

    These fields can have many options applied to them. See the
    Python docs on `Format String Syntax`_ for the full details.

    .. _Format String Syntax: https://docs.python.org/2/library/string.html#string-formatting
    """

    def __init__(self, fname, fspec='', conv=None):
        if False:
            print('Hello World!')
        self.set_fname(fname)
        self.set_fspec(fspec)
        self.set_conv(conv)

    def set_fname(self, fname):
        if False:
            for i in range(10):
                print('nop')
        'Set the field name.'
        path_list = re.split('[.[]', fname)
        self.base_name = path_list[0]
        self.fname = fname
        self.subpath = path_list[1:]
        self.is_positional = not self.base_name or self.base_name.isdigit()

    def set_fspec(self, fspec):
        if False:
            while True:
                i = 10
        'Set the field spec.'
        fspec = fspec or ''
        subfields = []
        for (sublit, subfname, _, _) in Formatter().parse(fspec):
            if subfname is not None:
                subfields.append(subfname)
        self.subfields = subfields
        self.fspec = fspec
        self.type_char = fspec[-1:]
        self.type_func = _TYPE_MAP.get(self.type_char, str)

    def set_conv(self, conv):
        if False:
            i = 10
            return i + 15
        'There are only two built-in converters: ``s`` and ``r``. They are\n        somewhat rare and appearlike ``"{ref!r}"``.'
        self.conv = conv
        self.conv_func = None

    @property
    def fstr(self):
        if False:
            print('Hello World!')
        'The current state of the field in string format.'
        return construct_format_field_str(self.fname, self.fspec, self.conv)

    def __repr__(self):
        if False:
            while True:
                i = 10
        cn = self.__class__.__name__
        args = [self.fname]
        if self.conv is not None:
            args.extend([self.fspec, self.conv])
        elif self.fspec != '':
            args.append(self.fspec)
        args_repr = ', '.join([repr(a) for a in args])
        return '%s(%s)' % (cn, args_repr)

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.fstr
_UNSET = object()

class DeferredValue(object):
    """:class:`DeferredValue` is a wrapper type, used to defer computing
    values which would otherwise be expensive to stringify and
    format. This is most valuable in areas like logging, where one
    would not want to waste time formatting a value for a log message
    which will subsequently be filtered because the message's log
    level was DEBUG and the logger was set to only emit CRITICAL
    messages.

    The :class:``DeferredValue`` is initialized with a callable that
    takes no arguments and returns the value, which can be of any
    type. By default DeferredValue only calls that callable once, and
    future references will get a cached value. This behavior can be
    disabled by setting *cache_value* to ``False``.

    Args:

        func (function): A callable that takes no arguments and
            computes the value being represented.
        cache_value (bool): Whether subsequent usages will call *func*
            again. Defaults to ``True``.

    >>> import sys
    >>> dv = DeferredValue(lambda: len(sys._current_frames()))
    >>> output = "works great in all {0} threads!".format(dv)

    PROTIP: To keep lines shorter, use: ``from formatutils import
    DeferredValue as DV``
    """

    def __init__(self, func, cache_value=True):
        if False:
            while True:
                i = 10
        self.func = func
        self.cache_value = cache_value
        self._value = _UNSET

    def get_value(self):
        if False:
            return 10
        'Computes, optionally caches, and returns the value of the\n        *func*. If ``get_value()`` has been called before, a cached\n        value may be returned depending on the *cache_value* option\n        passed to the constructor.\n        '
        if self._value is not _UNSET and self.cache_value:
            value = self._value
        else:
            value = self.func()
            if self.cache_value:
                self._value = value
        return value

    def __int__(self):
        if False:
            return 10
        return int(self.get_value())

    def __float__(self):
        if False:
            while True:
                i = 10
        return float(self.get_value())

    def __str__(self):
        if False:
            while True:
                i = 10
        return str(self.get_value())

    def __unicode__(self):
        if False:
            i = 10
            return i + 15
        return unicode(self.get_value())

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return repr(self.get_value())

    def __format__(self, fmt):
        if False:
            print('Hello World!')
        value = self.get_value()
        pt = fmt[-1:]
        type_conv = _TYPE_MAP.get(pt, str)
        try:
            return value.__format__(fmt)
        except (ValueError, TypeError):
            return type_conv(value).__format__(fmt)