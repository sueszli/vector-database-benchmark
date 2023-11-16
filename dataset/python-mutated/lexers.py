"""
Defines a variety of Pygments lexers for highlighting IPython code.

This includes:

    IPythonLexer, IPython3Lexer
        Lexers for pure IPython (python + magic/shell commands)

    IPythonPartialTracebackLexer, IPythonTracebackLexer
        Supports 2.x and 3.x via keyword `python3`.  The partial traceback
        lexer reads everything but the Python code appearing in a traceback.
        The full lexer combines the partial lexer with an IPython lexer.

    IPythonConsoleLexer
        A lexer for IPython console sessions, with support for tracebacks.

    IPyLexer
        A friendly lexer which examines the first line of text and from it,
        decides whether to use an IPython lexer or an IPython console lexer.
        This is probably the only lexer that needs to be explicitly added
        to Pygments.

"""
import re
from pygments.lexers import BashLexer, HtmlLexer, JavascriptLexer, RubyLexer, PerlLexer, PythonLexer, Python3Lexer, TexLexer
from pygments.lexer import Lexer, DelegatingLexer, RegexLexer, do_insertions, bygroups, using
from pygments.token import Generic, Keyword, Literal, Name, Operator, Other, Text, Error
from pygments.util import get_bool_opt
line_re = re.compile('.*?\n')
__all__ = ['build_ipy_lexer', 'IPython3Lexer', 'IPythonLexer', 'IPythonPartialTracebackLexer', 'IPythonTracebackLexer', 'IPythonConsoleLexer', 'IPyLexer']

def build_ipy_lexer(python3):
    if False:
        for i in range(10):
            print('nop')
    'Builds IPython lexers depending on the value of `python3`.\n\n    The lexer inherits from an appropriate Python lexer and then adds\n    information about IPython specific keywords (i.e. magic commands,\n    shell commands, etc.)\n\n    Parameters\n    ----------\n    python3 : bool\n        If `True`, then build an IPython lexer from a Python 3 lexer.\n\n    '
    if python3:
        PyLexer = Python3Lexer
        name = 'IPython3'
        aliases = ['ipython3']
        doc = 'IPython3 Lexer'
    else:
        PyLexer = PythonLexer
        name = 'IPython'
        aliases = ['ipython2', 'ipython']
        doc = 'IPython Lexer'
    ipython_tokens = [('(?s)(\\s*)(%%capture)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%debug)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?is)(\\s*)(%%html)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(HtmlLexer))), ('(?s)(\\s*)(%%javascript)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(JavascriptLexer))), ('(?s)(\\s*)(%%js)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(JavascriptLexer))), ('(?s)(\\s*)(%%latex)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(TexLexer))), ('(?s)(\\s*)(%%perl)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PerlLexer))), ('(?s)(\\s*)(%%prun)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%pypy)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%python)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%python2)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PythonLexer))), ('(?s)(\\s*)(%%python3)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(Python3Lexer))), ('(?s)(\\s*)(%%ruby)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(RubyLexer))), ('(?s)(\\s*)(%%time)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%timeit)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%writefile)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%file)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%)(\\w+)(.*)', bygroups(Text, Operator, Keyword, Text)), ('(?s)(^\\s*)(%%!)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(BashLexer))), ('(%%?)(\\w+)(\\?\\??)$', bygroups(Operator, Keyword, Operator)), ('\\b(\\?\\??)(\\s*)$', bygroups(Operator, Text)), ('(%)(sx|sc|system)(.*)(\\n)', bygroups(Operator, Keyword, using(BashLexer), Text)), ('(%)(\\w+)(.*\\n)', bygroups(Operator, Keyword, Text)), ('^(!!)(.+)(\\n)', bygroups(Operator, using(BashLexer), Text)), ('(!)(?!=)(.+)(\\n)', bygroups(Operator, using(BashLexer), Text)), ('^(\\s*)(\\?\\??)(\\s*%{0,2}[\\w\\.\\*]*)', bygroups(Text, Operator, Text)), ('(\\s*%{0,2}[\\w\\.\\*]*)(\\?\\??)(\\s*)$', bygroups(Text, Operator, Text))]
    tokens = PyLexer.tokens.copy()
    tokens['root'] = ipython_tokens + tokens['root']
    attrs = {'name': name, 'aliases': aliases, 'filenames': [], '__doc__': doc, 'tokens': tokens}
    return type(name, (PyLexer,), attrs)
IPython3Lexer = build_ipy_lexer(python3=True)
IPythonLexer = build_ipy_lexer(python3=False)

class IPythonPartialTracebackLexer(RegexLexer):
    """
    Partial lexer for IPython tracebacks.

    Handles all the non-python output.

    """
    name = 'IPython Partial Traceback'
    tokens = {'root': [('^(\\^C)?(-+\\n)', bygroups(Error, Generic.Traceback)), ('^(  File)(.*)(, line )(\\d+\\n)', bygroups(Generic.Traceback, Name.Namespace, Generic.Traceback, Literal.Number.Integer)), ('(?u)(^[^\\d\\W]\\w*)(\\s*)(Traceback.*?\\n)', bygroups(Name.Exception, Generic.Whitespace, Text)), ('(.*)( in )(.*)(\\(.*\\)\\n)', bygroups(Name.Namespace, Text, Name.Entity, Name.Tag)), ('(\\s*?)(\\d+)(.*?\\n)', bygroups(Generic.Whitespace, Literal.Number.Integer, Other)), ('(-*>?\\s?)(\\d+)(.*?\\n)', bygroups(Name.Exception, Literal.Number.Integer, Other)), ('(?u)(^[^\\d\\W]\\w*)(:.*?\\n)', bygroups(Name.Exception, Text)), ('.*\\n', Other)]}

class IPythonTracebackLexer(DelegatingLexer):
    """
    IPython traceback lexer.

    For doctests, the tracebacks can be snipped as much as desired with the
    exception to the lines that designate a traceback. For non-syntax error
    tracebacks, this is the line of hyphens. For syntax error tracebacks,
    this is the line which lists the File and line number.

    """
    name = 'IPython Traceback'
    aliases = ['ipythontb']

    def __init__(self, **options):
        if False:
            return 10
        '\n        A subclass of `DelegatingLexer` which delegates to the appropriate to either IPyLexer,\n        IPythonPartialTracebackLexer.\n        '
        self.python3 = get_bool_opt(options, 'python3', False)
        if self.python3:
            self.aliases = ['ipython3tb']
        else:
            self.aliases = ['ipython2tb', 'ipythontb']
        if self.python3:
            IPyLexer = IPython3Lexer
        else:
            IPyLexer = IPythonLexer
        DelegatingLexer.__init__(self, IPyLexer, IPythonPartialTracebackLexer, **options)

class IPythonConsoleLexer(Lexer):
    """
    An IPython console lexer for IPython code-blocks and doctests, such as:

    .. code-block:: rst

        .. code-block:: ipythonconsole

            In [1]: a = 'foo'

            In [2]: a
            Out[2]: 'foo'

            In [3]: print(a)
            foo


    Support is also provided for IPython exceptions:

    .. code-block:: rst

        .. code-block:: ipythonconsole

            In [1]: raise Exception
            Traceback (most recent call last):
            ...
            Exception

    """
    name = 'IPython console session'
    aliases = ['ipythonconsole']
    mimetypes = ['text/x-ipython-console']
    in1_regex = 'In \\[[0-9]+\\]: '
    in2_regex = '   \\.\\.+\\.: '
    out_regex = 'Out\\[[0-9]+\\]: '
    ipytb_start = re.compile('^(\\^C)?(-+\\n)|^(  File)(.*)(, line )(\\d+\\n)')

    def __init__(self, **options):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the IPython console lexer.\n\n        Parameters\n        ----------\n        python3 : bool\n            If `True`, then the console inputs are parsed using a Python 3\n            lexer. Otherwise, they are parsed using a Python 2 lexer.\n        in1_regex : RegexObject\n            The compiled regular expression used to detect the start\n            of inputs. Although the IPython configuration setting may have a\n            trailing whitespace, do not include it in the regex. If `None`,\n            then the default input prompt is assumed.\n        in2_regex : RegexObject\n            The compiled regular expression used to detect the continuation\n            of inputs. Although the IPython configuration setting may have a\n            trailing whitespace, do not include it in the regex. If `None`,\n            then the default input prompt is assumed.\n        out_regex : RegexObject\n            The compiled regular expression used to detect outputs. If `None`,\n            then the default output prompt is assumed.\n\n        '
        self.python3 = get_bool_opt(options, 'python3', False)
        if self.python3:
            self.aliases = ['ipython3console']
        else:
            self.aliases = ['ipython2console', 'ipythonconsole']
        in1_regex = options.get('in1_regex', self.in1_regex)
        in2_regex = options.get('in2_regex', self.in2_regex)
        out_regex = options.get('out_regex', self.out_regex)
        in1_regex_rstrip = in1_regex.rstrip() + '\n'
        in2_regex_rstrip = in2_regex.rstrip() + '\n'
        out_regex_rstrip = out_regex.rstrip() + '\n'
        attrs = ['in1_regex', 'in2_regex', 'out_regex', 'in1_regex_rstrip', 'in2_regex_rstrip', 'out_regex_rstrip']
        for attr in attrs:
            self.__setattr__(attr, re.compile(locals()[attr]))
        Lexer.__init__(self, **options)
        if self.python3:
            pylexer = IPython3Lexer
            tblexer = IPythonTracebackLexer
        else:
            pylexer = IPythonLexer
            tblexer = IPythonTracebackLexer
        self.pylexer = pylexer(**options)
        self.tblexer = tblexer(**options)
        self.reset()

    def reset(self):
        if False:
            while True:
                i = 10
        self.mode = 'output'
        self.index = 0
        self.buffer = u''
        self.insertions = []

    def buffered_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generator of unprocessed tokens after doing insertions and before\n        changing to a new state.\n\n        '
        if self.mode == 'output':
            tokens = [(0, Generic.Output, self.buffer)]
        elif self.mode == 'input':
            tokens = self.pylexer.get_tokens_unprocessed(self.buffer)
        else:
            tokens = self.tblexer.get_tokens_unprocessed(self.buffer)
        for (i, t, v) in do_insertions(self.insertions, tokens):
            yield (self.index + i, t, v)
        self.index += len(self.buffer)
        self.buffer = u''
        self.insertions = []

    def get_mci(self, line):
        if False:
            while True:
                i = 10
        '\n        Parses the line and returns a 3-tuple: (mode, code, insertion).\n\n        `mode` is the next mode (or state) of the lexer, and is always equal\n        to \'input\', \'output\', or \'tb\'.\n\n        `code` is a portion of the line that should be added to the buffer\n        corresponding to the next mode and eventually lexed by another lexer.\n        For example, `code` could be Python code if `mode` were \'input\'.\n\n        `insertion` is a 3-tuple (index, token, text) representing an\n        unprocessed "token" that will be inserted into the stream of tokens\n        that are created from the buffer once we change modes. This is usually\n        the input or output prompt.\n\n        In general, the next mode depends on current mode and on the contents\n        of `line`.\n\n        '
        in2_match = self.in2_regex.match(line)
        in2_match_rstrip = self.in2_regex_rstrip.match(line)
        if in2_match and in2_match.group().rstrip() == line.rstrip() or in2_match_rstrip:
            end_input = True
        else:
            end_input = False
        if end_input and self.mode != 'tb':
            mode = 'output'
            code = u''
            insertion = (0, Generic.Prompt, line)
            return (mode, code, insertion)
        out_match = self.out_regex.match(line)
        out_match_rstrip = self.out_regex_rstrip.match(line)
        if out_match or out_match_rstrip:
            mode = 'output'
            if out_match:
                idx = out_match.end()
            else:
                idx = out_match_rstrip.end()
            code = line[idx:]
            insertion = (0, Generic.Heading, line[:idx])
            return (mode, code, insertion)
        in1_match = self.in1_regex.match(line)
        if in1_match or (in2_match and self.mode != 'tb'):
            mode = 'input'
            if in1_match:
                idx = in1_match.end()
            else:
                idx = in2_match.end()
            code = line[idx:]
            insertion = (0, Generic.Prompt, line[:idx])
            return (mode, code, insertion)
        in1_match_rstrip = self.in1_regex_rstrip.match(line)
        if in1_match_rstrip or (in2_match_rstrip and self.mode != 'tb'):
            mode = 'input'
            if in1_match_rstrip:
                idx = in1_match_rstrip.end()
            else:
                idx = in2_match_rstrip.end()
            code = line[idx:]
            insertion = (0, Generic.Prompt, line[:idx])
            return (mode, code, insertion)
        if self.ipytb_start.match(line):
            mode = 'tb'
            code = line
            insertion = None
            return (mode, code, insertion)
        if self.mode in ('input', 'output'):
            mode = 'output'
        else:
            mode = 'tb'
        code = line
        insertion = None
        return (mode, code, insertion)

    def get_tokens_unprocessed(self, text):
        if False:
            while True:
                i = 10
        self.reset()
        for match in line_re.finditer(text):
            line = match.group()
            (mode, code, insertion) = self.get_mci(line)
            if mode != self.mode:
                for token in self.buffered_tokens():
                    yield token
                self.mode = mode
            if insertion:
                self.insertions.append((len(self.buffer), [insertion]))
            self.buffer += code
        for token in self.buffered_tokens():
            yield token

class IPyLexer(Lexer):
    """
    Primary lexer for all IPython-like code.

    This is a simple helper lexer.  If the first line of the text begins with
    "In \\[[0-9]+\\]:", then the entire text is parsed with an IPython console
    lexer. If not, then the entire text is parsed with an IPython lexer.

    The goal is to reduce the number of lexers that are registered
    with Pygments.

    """
    name = 'IPy session'
    aliases = ['ipy']

    def __init__(self, **options):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new IPyLexer instance which dispatch to either an\n        IPythonCOnsoleLexer (if In prompts are present) or and IPythonLexer (if\n        In prompts are not present).\n        '
        self.python3 = get_bool_opt(options, 'python3', False)
        if self.python3:
            self.aliases = ['ipy3']
        else:
            self.aliases = ['ipy2', 'ipy']
        Lexer.__init__(self, **options)
        self.IPythonLexer = IPythonLexer(**options)
        self.IPythonConsoleLexer = IPythonConsoleLexer(**options)

    def get_tokens_unprocessed(self, text):
        if False:
            return 10
        if re.match('.*(In \\[[0-9]+\\]:)', text.strip(), re.DOTALL):
            lex = self.IPythonConsoleLexer
        else:
            lex = self.IPythonLexer
        for token in lex.get_tokens_unprocessed(text):
            yield token