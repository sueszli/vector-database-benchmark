"""
    tinycss.token_data
    ------------------

    Shared data for both implementations (Cython and Python) of the tokenizer.

    :copyright: (c) 2012 by Simon Sapin.
    :license: BSD, see LICENSE for more details.
"""
import re
import sys
import operator
import functools
import string
MACROS = '\n    nl\t\\n|\\r\\n|\\r|\\f\n    w\t[ \\t\\r\\n\\f]*\n    nonascii\t[^\\0-\\237]\n    unicode\t\\\\([0-9a-f]{{1,6}})(\\r\\n|[ \\n\\r\\t\\f])?\n    simple_escape\t[^\\n\\r\\f0-9a-f]\n    escape\t{unicode}|\\\\{simple_escape}\n    nmstart\t[_a-z]|{nonascii}|{escape}\n    nmchar\t[_a-z0-9-]|{nonascii}|{escape}\n    name\t{nmchar}+\n    ident\t[-]?{nmstart}{nmchar}*\n    num\t[-+]?(?:[0-9]*\\.[0-9]+|[0-9]+)\n    string1\t\\"([^\\n\\r\\f\\\\"]|\\\\{nl}|{escape})*\\"\n    string2\t\\\'([^\\n\\r\\f\\\\\']|\\\\{nl}|{escape})*\\\'\n    string\t{string1}|{string2}\n    badstring1\t\\"([^\\n\\r\\f\\\\"]|\\\\{nl}|{escape})*\\\\?\n    badstring2\t\\\'([^\\n\\r\\f\\\\\']|\\\\{nl}|{escape})*\\\\?\n    badstring\t{badstring1}|{badstring2}\n    badcomment1\t\\/\\*[^*]*\\*+([^/*][^*]*\\*+)*\n    badcomment2\t\\/\\*[^*]*(\\*+[^/*][^*]*)*\n    badcomment\t{badcomment1}|{badcomment2}\n    baduri1\turl\\({w}([!#$%&*-~]|{nonascii}|{escape})*{w}\n    baduri2\turl\\({w}{string}{w}\n    baduri3\turl\\({w}{badstring}\n    baduri\t{baduri1}|{baduri2}|{baduri3}\n'.replace('\\0', '\x00').replace('\\237', '\x9f')
TOKENS = '\n    S\t[ \\t\\r\\n\\f]+\n\n    URI\turl\\({w}({string}|([!#$%&*-\\[\\]-~]|{nonascii}|{escape})*){w}\\)\n    BAD_URI\t{baduri}\n    FUNCTION\t{ident}\\(\n    UNICODE-RANGE\tu\\+[0-9a-f?]{{1,6}}(-[0-9a-f]{{1,6}})?\n    IDENT\t{ident}\n\n    ATKEYWORD\t@{ident}\n    HASH\t#{name}\n\n    DIMENSION\t({num})({ident})\n    PERCENTAGE\t{num}%\n    NUMBER\t{num}\n\n    STRING\t{string}\n    BAD_STRING\t{badstring}\n\n    COMMENT\t\\/\\*[^*]*\\*+([^/*][^*]*\\*+)*\\/\n    BAD_COMMENT\t{badcomment}\n\n    :\t:\n    ;\t;\n    {\t\\{{\n    }\t\\}}\n    (\t\\(\n    )\t\\)\n    [\t\\[\n    ]\t\\]\n    CDO\t<!--\n    CDC\t-->\n'
COMPILED_MACROS = {}
COMPILED_TOKEN_REGEXPS = []
COMPILED_TOKEN_INDEXES = {}
TOKEN_DISPATCH = []
try:
    unichr
except NameError:
    unichr = chr
    unicode = str

def _init():
    if False:
        for i in range(10):
            print('nop')
    'Import-time initialization.'
    COMPILED_MACROS.clear()
    for line in MACROS.splitlines():
        if line.strip():
            (name, value) = line.split('\t')
            COMPILED_MACROS[name.strip()] = '(?:%s)' % value.format(**COMPILED_MACROS)
    COMPILED_TOKEN_REGEXPS[:] = ((name.strip(), re.compile(value.format(**COMPILED_MACROS), re.I).match) for line in TOKENS.splitlines() if line.strip() for (name, value) in [line.split('\t')])
    COMPILED_TOKEN_INDEXES.clear()
    for (i, (name, regexp)) in enumerate(COMPILED_TOKEN_REGEXPS):
        COMPILED_TOKEN_INDEXES[name] = i
    dispatch = [[] for i in range(161)]
    for (chars, names) in [(' \t\r\n\x0c', ['S']), ('uU', ['URI', 'BAD_URI', 'UNICODE-RANGE']), (string.ascii_letters + '\\_-' + unichr(160), ['FUNCTION', 'IDENT']), (string.digits + '.+-', ['DIMENSION', 'PERCENTAGE', 'NUMBER']), ('@', ['ATKEYWORD']), ('#', ['HASH']), ('\'"', ['STRING', 'BAD_STRING']), ('/', ['COMMENT', 'BAD_COMMENT']), ('<', ['CDO']), ('-', ['CDC'])]:
        for char in chars:
            dispatch[ord(char)].extend(names)
    for char in ':;{}()[]':
        dispatch[ord(char)] = [char]
    TOKEN_DISPATCH[:] = ([(index,) + COMPILED_TOKEN_REGEXPS[index] for name in names for index in [COMPILED_TOKEN_INDEXES[name]]] for names in dispatch)
_init()

def _unicode_replace(match, int=int, unichr=unichr, maxunicode=sys.maxunicode):
    if False:
        print('Hello World!')
    codepoint = int(match.group(1), 16)
    if codepoint <= maxunicode:
        return unichr(codepoint)
    else:
        return '�'
UNICODE_UNESCAPE = functools.partial(re.compile(COMPILED_MACROS['unicode'], re.I).sub, _unicode_replace)
NEWLINE_UNESCAPE = functools.partial(re.compile('()\\\\' + COMPILED_MACROS['nl']).sub, '')
SIMPLE_UNESCAPE = functools.partial(re.compile('\\\\(%s)' % COMPILED_MACROS['simple_escape'], re.I).sub, operator.methodcaller('group', 1))

def FIND_NEWLINES(x):
    if False:
        i = 10
        return i + 15
    return list(re.compile(COMPILED_MACROS['nl']).finditer(x))

class Token:
    """A single atomic token.

    .. attribute:: is_container

        Always ``False``.
        Helps to tell :class:`Token` apart from :class:`ContainerToken`.

    .. attribute:: type

        The type of token as a string:

        ``S``
            A sequence of white space

        ``IDENT``
            An identifier: a name that does not start with a digit.
            A name is a sequence of letters, digits, ``_``, ``-``, escaped
            characters and non-ASCII characters. Eg:\xa0``margin-left``

        ``HASH``
            ``#`` followed immediately by a name. Eg:\xa0``#ff8800``

        ``ATKEYWORD``
            ``@`` followed immediately by an identifier. Eg:\xa0``@page``

        ``URI``
            Eg:\xa0``url(foo)`` The content may or may not be quoted.

        ``UNICODE-RANGE``
            ``U+`` followed by one or two hexadecimal
            Unicode codepoints. Eg:\xa0``U+20-00FF``

        ``INTEGER``
            An integer with an optional ``+`` or ``-`` sign

        ``NUMBER``
            A non-integer number  with an optional ``+`` or ``-`` sign

        ``DIMENSION``
            An integer or number followed immediately by an
            identifier (the unit). Eg:\xa0``12px``

        ``PERCENTAGE``
            An integer or number followed immediately by ``%``

        ``STRING``
            A string, quoted with ``"`` or ``'``

        ``:`` or ``;``
            That character.

        ``DELIM``
            A single character not matched in another token. Eg:\xa0``,``

        See the source of the :mod:`.token_data` module for the precise
        regular expressions that match various tokens.

        Note that other token types exist in the early tokenization steps,
        but these are ignored, are syntax errors, or are later transformed
        into :class:`ContainerToken` or :class:`FunctionToken`.

    .. attribute:: value

        The parsed value:

        * INTEGER, NUMBER, PERCENTAGE or DIMENSION tokens: the numeric value
          as an int or float.
        * STRING tokens: the unescaped string without quotes
        * URI tokens: the unescaped URI without quotes or
          ``url(`` and ``)`` markers.
        * IDENT, ATKEYWORD or HASH tokens: the unescaped token,
          with ``@`` or ``#`` markers left as-is
        * Other tokens: same as :attr:`as_css`

        *Unescaped* refers to the various escaping methods based on the
        backslash ``\\`` character in CSS syntax.

    .. attribute:: unit

        * DIMENSION tokens: the normalized (unescaped, lower-case)
          unit name as a string. eg. ``'px'``
        * PERCENTAGE tokens: the string ``'%'``
        * Other tokens: ``None``

    .. attribute:: line

        The line number in the CSS source of the start of this token.

    .. attribute:: column

        The column number (inside a source line) of the start of this token.

    """
    is_container = False
    __slots__ = ('type', '_as_css', 'value', 'unit', 'line', 'column')

    def __init__(self, type_, css_value, value, unit, line, column):
        if False:
            while True:
                i = 10
        self.type = type_
        self._as_css = css_value
        self.value = value
        self.unit = unit
        self.line = line
        self.column = column

    def as_css(self):
        if False:
            i = 10
            return i + 15
        '\n        Return as an Unicode string the CSS representation of the token,\n        as parsed in the source.\n        '
        return self._as_css

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Token {0.type} at {0.line}:{0.column} {0.value!r}{1}>'.format(self, self.unit or '')

class ContainerToken:
    """A token that contains other (nested) tokens.

    .. attribute:: is_container

        Always ``True``.
        Helps to tell :class:`ContainerToken` apart from :class:`Token`.

    .. attribute:: type

        The type of token as a string. One of ``{``, ``(``, ``[`` or
        ``FUNCTION``. For ``FUNCTION``, the object is actually a
        :class:`FunctionToken`.

    .. attribute:: unit

        Always ``None``. Included to make :class:`ContainerToken` behave
        more like :class:`Token`.

    .. attribute:: content

        A list of :class:`Token` or nested :class:`ContainerToken`,
        not including the opening or closing token.

    .. attribute:: line

        The line number in the CSS source of the start of this token.

    .. attribute:: column

        The column number (inside a source line) of the start of this token.

    """
    is_container = True
    unit = None
    __slots__ = ('type', '_css_start', '_css_end', 'content', 'line', 'column')

    def __init__(self, type_, css_start, css_end, content, line, column):
        if False:
            while True:
                i = 10
        self.type = type_
        self._css_start = css_start
        self._css_end = css_end
        self.content = content
        self.line = line
        self.column = column

    def as_css(self):
        if False:
            print('Hello World!')
        '\n        Return as an Unicode string the CSS representation of the token,\n        as parsed in the source.\n        '
        parts = [self._css_start]
        parts.extend((token.as_css() for token in self.content))
        parts.append(self._css_end)
        return ''.join(parts)
    format_string = '<ContainerToken {0.type} at {0.line}:{0.column}>'

    def __repr__(self):
        if False:
            return 10
        return (self.format_string + ' {0.content}').format(self)

class FunctionToken(ContainerToken):
    """A specialized :class:`ContainerToken` for a ``FUNCTION`` group.
    Has an additional attribute:

    .. attribute:: function_name

        The unescaped name of the function, with the ``(`` marker removed.

    """
    __slots__ = ('function_name',)

    def __init__(self, type_, css_start, css_end, function_name, content, line, column):
        if False:
            i = 10
            return i + 15
        super(FunctionToken, self).__init__(type_, css_start, css_end, content, line, column)
        self.function_name = function_name[:-1]
    format_string = '<FunctionToken {0.function_name}() at {0.line}:{0.column}>'

class TokenList(list):
    """
    A mixed list of :class:`~.token_data.Token` and
    :class:`~.token_data.ContainerToken` objects.

    This is a subclass of the builtin :class:`~builtins.list` type.
    It can be iterated, indexed and sliced as usual, but also has some
    additional API:

    """

    @property
    def line(self):
        if False:
            while True:
                i = 10
        'The line number in the CSS source of the first token.'
        return self[0].line

    @property
    def column(self):
        if False:
            print('Hello World!')
        'The column number (inside a source line) of the first token.'
        return self[0].column

    def as_css(self):
        if False:
            while True:
                i = 10
        '\n        Return as an Unicode string the CSS representation of the tokens,\n        as parsed in the source.\n        '
        return ''.join((token.as_css() for token in self))

def load_c_tokenizer():
    if False:
        for i in range(10):
            print('nop')
    from calibre_extensions import tokenizer
    tokens = list(':;(){}[]') + ['DELIM', 'INTEGER', 'STRING']
    tokenizer.init(COMPILED_TOKEN_REGEXPS, UNICODE_UNESCAPE, NEWLINE_UNESCAPE, SIMPLE_UNESCAPE, FIND_NEWLINES, TOKEN_DISPATCH, COMPILED_TOKEN_INDEXES, *tokens)
    return tokenizer