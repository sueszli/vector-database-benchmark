"""DEPRECATED: Input transformer classes to support IPython special syntax.

This module was deprecated in IPython 7.0, in favour of inputtransformer2.

This includes the machinery to recognise and transform ``%magic`` commands,
``!system`` commands, ``help?`` querying, prompt stripping, and so forth.
"""
import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
ESC_SHELL = '!'
ESC_SH_CAP = '!!'
ESC_HELP = '?'
ESC_HELP2 = '??'
ESC_MAGIC = '%'
ESC_MAGIC2 = '%%'
ESC_QUOTE = ','
ESC_QUOTE2 = ';'
ESC_PAREN = '/'
ESC_SEQUENCES = [ESC_SHELL, ESC_SH_CAP, ESC_HELP, ESC_HELP2, ESC_MAGIC, ESC_MAGIC2, ESC_QUOTE, ESC_QUOTE2, ESC_PAREN]

class InputTransformer(metaclass=abc.ABCMeta):
    """Abstract base class for line-based input transformers."""

    @abc.abstractmethod
    def push(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Send a line of input to the transformer, returning the transformed\n        input or None if the transformer is waiting for more input.\n\n        Must be overridden by subclasses.\n\n        Implementations may raise ``SyntaxError`` if the input is invalid. No\n        other exceptions may be raised.\n        '
        pass

    @abc.abstractmethod
    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        'Return, transformed any lines that the transformer has accumulated,\n        and reset its internal state.\n\n        Must be overridden by subclasses.\n        '
        pass

    @classmethod
    def wrap(cls, func):
        if False:
            print('Hello World!')
        'Can be used by subclasses as a decorator, to return a factory that\n        will allow instantiation with the decorated object.\n        '

        @functools.wraps(func)
        def transformer_factory(**kwargs):
            if False:
                print('Hello World!')
            return cls(func, **kwargs)
        return transformer_factory

class StatelessInputTransformer(InputTransformer):
    """Wrapper for a stateless input transformer implemented as a function."""

    def __init__(self, func):
        if False:
            for i in range(10):
                print('nop')
        self.func = func

    def __repr__(self):
        if False:
            return 10
        return 'StatelessInputTransformer(func={0!r})'.format(self.func)

    def push(self, line):
        if False:
            print('Hello World!')
        'Send a line of input to the transformer, returning the\n        transformed input.'
        return self.func(line)

    def reset(self):
        if False:
            return 10
        'No-op - exists for compatibility.'
        pass

class CoroutineInputTransformer(InputTransformer):
    """Wrapper for an input transformer implemented as a coroutine."""

    def __init__(self, coro, **kwargs):
        if False:
            i = 10
            return i + 15
        self.coro = coro(**kwargs)
        next(self.coro)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'CoroutineInputTransformer(coro={0!r})'.format(self.coro)

    def push(self, line):
        if False:
            i = 10
            return i + 15
        'Send a line of input to the transformer, returning the\n        transformed input or None if the transformer is waiting for more\n        input.\n        '
        return self.coro.send(line)

    def reset(self):
        if False:
            i = 10
            return i + 15
        'Return, transformed any lines that the transformer has\n        accumulated, and reset its internal state.\n        '
        return self.coro.send(None)

class TokenInputTransformer(InputTransformer):
    """Wrapper for a token-based input transformer.
    
    func should accept a list of tokens (5-tuples, see tokenize docs), and
    return an iterable which can be passed to tokenize.untokenize().
    """

    def __init__(self, func):
        if False:
            print('Hello World!')
        self.func = func
        self.buf = []
        self.reset_tokenizer()

    def reset_tokenizer(self):
        if False:
            i = 10
            return i + 15
        it = iter(self.buf)
        self.tokenizer = tokenutil.generate_tokens_catch_errors(it.__next__)

    def push(self, line):
        if False:
            i = 10
            return i + 15
        self.buf.append(line + '\n')
        if all((l.isspace() for l in self.buf)):
            return self.reset()
        tokens = []
        stop_at_NL = False
        try:
            for intok in self.tokenizer:
                tokens.append(intok)
                t = intok[0]
                if t == tokenize.NEWLINE or (stop_at_NL and t == tokenize.NL):
                    break
                elif t == tokenize.ERRORTOKEN:
                    stop_at_NL = True
        except TokenError:
            self.reset_tokenizer()
            return None
        return self.output(tokens)

    def output(self, tokens):
        if False:
            print('Hello World!')
        self.buf.clear()
        self.reset_tokenizer()
        return untokenize(self.func(tokens)).rstrip('\n')

    def reset(self):
        if False:
            while True:
                i = 10
        l = ''.join(self.buf)
        self.buf.clear()
        self.reset_tokenizer()
        if l:
            return l.rstrip('\n')

class assemble_python_lines(TokenInputTransformer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(assemble_python_lines, self).__init__(None)

    def output(self, tokens):
        if False:
            while True:
                i = 10
        return self.reset()

@CoroutineInputTransformer.wrap
def assemble_logical_lines():
    if False:
        for i in range(10):
            print('nop')
    'Join lines following explicit line continuations (\\)'
    line = ''
    while True:
        line = (yield line)
        if not line or line.isspace():
            continue
        parts = []
        while line is not None:
            if line.endswith('\\') and (not has_comment(line)):
                parts.append(line[:-1])
                line = (yield None)
            else:
                parts.append(line)
                break
        line = ''.join(parts)

def _make_help_call(target: str, esc: str, lspace: str) -> str:
    if False:
        i = 10
        return i + 15
    'Prepares a pinfo(2)/psearch call from a target name and the escape\n    (i.e. ? or ??)'
    method = 'pinfo2' if esc == '??' else 'psearch' if '*' in target else 'pinfo'
    arg = ' '.join([method, target])
    (t_magic_name, _, t_magic_arg_s) = arg.partition(' ')
    t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
    return '%sget_ipython().run_line_magic(%r, %r)' % (lspace, t_magic_name, t_magic_arg_s)

def _tr_system(line_info: LineInfo):
    if False:
        return 10
    'Translate lines escaped with: !'
    cmd = line_info.line.lstrip().lstrip(ESC_SHELL)
    return '%sget_ipython().system(%r)' % (line_info.pre, cmd)

def _tr_system2(line_info: LineInfo):
    if False:
        for i in range(10):
            print('nop')
    'Translate lines escaped with: !!'
    cmd = line_info.line.lstrip()[2:]
    return '%sget_ipython().getoutput(%r)' % (line_info.pre, cmd)

def _tr_help(line_info: LineInfo):
    if False:
        i = 10
        return i + 15
    'Translate lines escaped with: ?/??'
    if not line_info.line[1:]:
        return 'get_ipython().show_usage()'
    return _make_help_call(line_info.ifun, line_info.esc, line_info.pre)

def _tr_magic(line_info: LineInfo):
    if False:
        return 10
    'Translate lines escaped with: %'
    tpl = '%sget_ipython().run_line_magic(%r, %r)'
    if line_info.line.startswith(ESC_MAGIC2):
        return line_info.line
    cmd = ' '.join([line_info.ifun, line_info.the_rest]).strip()
    (t_magic_name, _, t_magic_arg_s) = cmd.partition(' ')
    t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
    return tpl % (line_info.pre, t_magic_name, t_magic_arg_s)

def _tr_quote(line_info: LineInfo):
    if False:
        i = 10
        return i + 15
    'Translate lines escaped with: ,'
    return '%s%s("%s")' % (line_info.pre, line_info.ifun, '", "'.join(line_info.the_rest.split()))

def _tr_quote2(line_info: LineInfo):
    if False:
        i = 10
        return i + 15
    'Translate lines escaped with: ;'
    return '%s%s("%s")' % (line_info.pre, line_info.ifun, line_info.the_rest)

def _tr_paren(line_info: LineInfo):
    if False:
        while True:
            i = 10
    'Translate lines escaped with: /'
    return '%s%s(%s)' % (line_info.pre, line_info.ifun, ', '.join(line_info.the_rest.split()))
tr = {ESC_SHELL: _tr_system, ESC_SH_CAP: _tr_system2, ESC_HELP: _tr_help, ESC_HELP2: _tr_help, ESC_MAGIC: _tr_magic, ESC_QUOTE: _tr_quote, ESC_QUOTE2: _tr_quote2, ESC_PAREN: _tr_paren}

@StatelessInputTransformer.wrap
def escaped_commands(line: str):
    if False:
        print('Hello World!')
    'Transform escaped commands - %magic, !system, ?help + various autocalls.'
    if not line or line.isspace():
        return line
    lineinf = LineInfo(line)
    if lineinf.esc not in tr:
        return line
    return tr[lineinf.esc](lineinf)
_initial_space_re = re.compile('\\s*')
_help_end_re = re.compile('(%{0,2}\n                              (?!\\d)[\\w*]+            # Variable name\n                              (\\.(?!\\d)[\\w*]+)*       # .etc.etc\n                              )\n                              (\\?\\??)$                # ? or ??\n                              ', re.VERBOSE)
_MULTILINE_STRING = object()
_MULTILINE_STRUCTURE = object()

def _line_tokens(line):
    if False:
        while True:
            i = 10
    'Helper for has_comment and ends_in_comment_or_string.'
    readline = StringIO(line).readline
    toktypes = set()
    try:
        for t in tokenutil.generate_tokens_catch_errors(readline):
            toktypes.add(t[0])
    except TokenError as e:
        if 'multi-line string' in e.args[0]:
            toktypes.add(_MULTILINE_STRING)
        else:
            toktypes.add(_MULTILINE_STRUCTURE)
    return toktypes

def has_comment(src):
    if False:
        print('Hello World!')
    'Indicate whether an input line has (i.e. ends in, or is) a comment.\n\n    This uses tokenize, so it can distinguish comments from # inside strings.\n\n    Parameters\n    ----------\n    src : string\n        A single line input string.\n\n    Returns\n    -------\n    comment : bool\n        True if source has a comment.\n    '
    return tokenize.COMMENT in _line_tokens(src)

def ends_in_comment_or_string(src):
    if False:
        i = 10
        return i + 15
    'Indicates whether or not an input line ends in a comment or within\n    a multiline string.\n\n    Parameters\n    ----------\n    src : string\n        A single line input string.\n\n    Returns\n    -------\n    comment : bool\n        True if source ends in a comment or multiline string.\n    '
    toktypes = _line_tokens(src)
    return tokenize.COMMENT in toktypes or _MULTILINE_STRING in toktypes

@StatelessInputTransformer.wrap
def help_end(line: str):
    if False:
        while True:
            i = 10
    'Translate lines with ?/?? at the end'
    m = _help_end_re.search(line)
    if m is None or ends_in_comment_or_string(line):
        return line
    target = m.group(1)
    esc = m.group(3)
    match = _initial_space_re.match(line)
    assert match is not None
    lspace = match.group(0)
    return _make_help_call(target, esc, lspace)

@CoroutineInputTransformer.wrap
def cellmagic(end_on_blank_line: bool=False):
    if False:
        i = 10
        return i + 15
    'Captures & transforms cell magics.\n\n    After a cell magic is started, this stores up any lines it gets until it is\n    reset (sent None).\n    '
    tpl = 'get_ipython().run_cell_magic(%r, %r, %r)'
    cellmagic_help_re = re.compile('%%\\w+\\?')
    line = ''
    while True:
        line = (yield line)
        while not line:
            line = (yield line)
        if not line.startswith(ESC_MAGIC2):
            while line is not None:
                line = (yield line)
            continue
        if cellmagic_help_re.match(line):
            continue
        first = line
        body = []
        line = (yield None)
        while line is not None and (line.strip() != '' or not end_on_blank_line):
            body.append(line)
            line = (yield None)
        (magic_name, _, first) = first.partition(' ')
        magic_name = magic_name.lstrip(ESC_MAGIC2)
        line = tpl % (magic_name, first, u'\n'.join(body))

def _strip_prompts(prompt_re, initial_re=None, turnoff_re=None):
    if False:
        i = 10
        return i + 15
    "Remove matching input prompts from a block of input.\n\n    Parameters\n    ----------\n    prompt_re : regular expression\n        A regular expression matching any input prompt (including continuation)\n    initial_re : regular expression, optional\n        A regular expression matching only the initial prompt, but not continuation.\n        If no initial expression is given, prompt_re will be used everywhere.\n        Used mainly for plain Python prompts, where the continuation prompt\n        ``...`` is a valid Python expression in Python 3, so shouldn't be stripped.\n\n    Notes\n    -----\n    If `initial_re` and `prompt_re differ`,\n    only `initial_re` will be tested against the first line.\n    If any prompt is found on the first two lines,\n    prompts will be stripped from the rest of the block.\n    "
    if initial_re is None:
        initial_re = prompt_re
    line = ''
    while True:
        line = (yield line)
        if line is None:
            continue
        (out, n1) = initial_re.subn('', line, count=1)
        if turnoff_re and (not n1):
            if turnoff_re.match(line):
                while line is not None:
                    line = (yield line)
                continue
        line = (yield out)
        if line is None:
            continue
        (out, n2) = prompt_re.subn('', line, count=1)
        line = (yield out)
        if n1 or n2:
            while line is not None:
                line = (yield prompt_re.sub('', line, count=1))
        else:
            while line is not None:
                line = (yield line)

@CoroutineInputTransformer.wrap
def classic_prompt():
    if False:
        i = 10
        return i + 15
    'Strip the >>>/... prompts of the Python interactive shell.'
    prompt_re = re.compile('^(>>>|\\.\\.\\.)( |$)')
    initial_re = re.compile('^>>>( |$)')
    turnoff_re = re.compile('^[%!]')
    return _strip_prompts(prompt_re, initial_re, turnoff_re)

@CoroutineInputTransformer.wrap
def ipy_prompt():
    if False:
        for i in range(10):
            print('nop')
    "Strip IPython's In [1]:/...: prompts."
    prompt_re = re.compile('^(In \\[\\d+\\]: |\\s*\\.{3,}: ?)')
    turnoff_re = re.compile('^%%')
    return _strip_prompts(prompt_re, turnoff_re=turnoff_re)

@CoroutineInputTransformer.wrap
def leading_indent():
    if False:
        i = 10
        return i + 15
    'Remove leading indentation.\n\n    If the first line starts with a spaces or tabs, the same whitespace will be\n    removed from each following line until it is reset.\n    '
    space_re = re.compile('^[ \\t]+')
    line = ''
    while True:
        line = (yield line)
        if line is None:
            continue
        m = space_re.match(line)
        if m:
            space = m.group(0)
            while line is not None:
                if line.startswith(space):
                    line = line[len(space):]
                line = (yield line)
        else:
            while line is not None:
                line = (yield line)
_assign_pat = '(?P<lhs>(\\s*)\n    ([\\w\\.]+)                # Initial identifier\n    (\\s*,\\s*\n        \\*?[\\w\\.]+)*         # Further identifiers for unpacking\n    \\s*?,?                   # Trailing comma\n    )\n    \\s*=\\s*\n'
assign_system_re = re.compile('{}!\\s*(?P<cmd>.*)'.format(_assign_pat), re.VERBOSE)
assign_system_template = '%s = get_ipython().getoutput(%r)'

@StatelessInputTransformer.wrap
def assign_from_system(line):
    if False:
        i = 10
        return i + 15
    'Transform assignment from system commands (e.g. files = !ls)'
    m = assign_system_re.match(line)
    if m is None:
        return line
    return assign_system_template % m.group('lhs', 'cmd')
assign_magic_re = re.compile('{}%\\s*(?P<cmd>.*)'.format(_assign_pat), re.VERBOSE)
assign_magic_template = '%s = get_ipython().run_line_magic(%r, %r)'

@StatelessInputTransformer.wrap
def assign_from_magic(line):
    if False:
        for i in range(10):
            print('nop')
    'Transform assignment from magic commands (e.g. a = %who_ls)'
    m = assign_magic_re.match(line)
    if m is None:
        return line
    (m_lhs, m_cmd) = m.group('lhs', 'cmd')
    (t_magic_name, _, t_magic_arg_s) = m_cmd.partition(' ')
    t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
    return assign_magic_template % (m_lhs, t_magic_name, t_magic_arg_s)