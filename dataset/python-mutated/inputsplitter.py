"""DEPRECATED: Input handling and transformation machinery.

This module was deprecated in IPython 7.0, in favour of inputtransformer2.

The first class in this module, :class:`InputSplitter`, is designed to tell when
input from a line-oriented frontend is complete and should be executed, and when
the user should be prompted for another line of code instead. The name 'input
splitter' is largely for historical reasons.

A companion, :class:`IPythonInputSplitter`, provides the same functionality but
with full support for the extended IPython syntax (magics, system calls, etc).
The code to actually do these transformations is in :mod:`IPython.core.inputtransformer`.
:class:`IPythonInputSplitter` feeds the raw code to the transformers in order
and stores the results.

For more details, see the class docstrings below.
"""
from warnings import warn
warn('IPython.core.inputsplitter is deprecated since IPython 7 in favor of `IPython.core.inputtransformer2`', DeprecationWarning)
import ast
import codeop
import io
import re
import sys
import tokenize
import warnings
from typing import List, Tuple, Union, Optional
from types import CodeType
from IPython.core.inputtransformer import leading_indent, classic_prompt, ipy_prompt, cellmagic, assemble_logical_lines, help_end, escaped_commands, assign_from_magic, assign_from_system, assemble_python_lines
from IPython.utils import tokenutil
from IPython.core.inputtransformer import ESC_SHELL, ESC_SH_CAP, ESC_HELP, ESC_HELP2, ESC_MAGIC, ESC_MAGIC2, ESC_QUOTE, ESC_QUOTE2, ESC_PAREN, ESC_SEQUENCES
dedent_re = re.compile('|'.join(['^\\s+raise(\\s.*)?$', '^\\s+raise\\([^\\)]*\\).*$', '^\\s+return(\\s.*)?$', '^\\s+return\\([^\\)]*\\).*$', '^\\s+pass\\s*$', '^\\s+break\\s*$', '^\\s+continue\\s*$']))
ini_spaces_re = re.compile('^([ \\t\\r\\f\\v]+)')
comment_line_re = re.compile('^\\s*\\#')

def num_ini_spaces(s):
    if False:
        for i in range(10):
            print('nop')
    "Return the number of initial spaces in a string.\n\n    Note that tabs are counted as a single space.  For now, we do *not* support\n    mixing of tabs and spaces in the user's input.\n\n    Parameters\n    ----------\n    s : string\n\n    Returns\n    -------\n    n : int\n    "
    warnings.warn('`num_ini_spaces` is Pending Deprecation since IPython 8.17.It is considered fro removal in in future version. Please open an issue if you believe it should be kept.', stacklevel=2, category=PendingDeprecationWarning)
    ini_spaces = ini_spaces_re.match(s)
    if ini_spaces:
        return ini_spaces.end()
    else:
        return 0
INCOMPLETE_STRING = tokenize.N_TOKENS
IN_MULTILINE_STATEMENT = tokenize.N_TOKENS + 1

class IncompleteString:
    type = exact_type = INCOMPLETE_STRING

    def __init__(self, s, start, end, line):
        if False:
            for i in range(10):
                print('nop')
        self.s = s
        self.start = start
        self.end = end
        self.line = line

class InMultilineStatement:
    type = exact_type = IN_MULTILINE_STATEMENT

    def __init__(self, pos, line):
        if False:
            print('Hello World!')
        self.s = ''
        self.start = self.end = pos
        self.line = line

def partial_tokens(s):
    if False:
        print('Hello World!')
    'Iterate over tokens from a possibly-incomplete string of code.\n\n    This adds two special token types: INCOMPLETE_STRING and\n    IN_MULTILINE_STATEMENT. These can only occur as the last token yielded, and\n    represent the two main ways for code to be incomplete.\n    '
    readline = io.StringIO(s).readline
    token = tokenize.TokenInfo(tokenize.NEWLINE, '', (1, 0), (1, 0), '')
    try:
        for token in tokenutil.generate_tokens_catch_errors(readline):
            yield token
    except tokenize.TokenError as e:
        lines = s.splitlines(keepends=True)
        end = (len(lines), len(lines[-1]))
        if 'multi-line string' in e.args[0]:
            (l, c) = start = token.end
            s = lines[l - 1][c:] + ''.join(lines[l:])
            yield IncompleteString(s, start, end, lines[-1])
        elif 'multi-line statement' in e.args[0]:
            yield InMultilineStatement(end, lines[-1])
        else:
            raise

def find_next_indent(code) -> int:
    if False:
        print('Hello World!')
    'Find the number of spaces for the next line of indentation'
    tokens = list(partial_tokens(code))
    if tokens[-1].type == tokenize.ENDMARKER:
        tokens.pop()
    if not tokens:
        return 0
    while tokens[-1].type in {tokenize.DEDENT, tokenize.NEWLINE, tokenize.COMMENT, tokenize.ERRORTOKEN}:
        tokens.pop()
    if tokens[-1].type == IN_MULTILINE_STATEMENT:
        while tokens[-2].type in {tokenize.NL}:
            tokens.pop(-2)
    if tokens[-1].type == INCOMPLETE_STRING:
        return 0
    prev_indents = [0]

    def _add_indent(n):
        if False:
            for i in range(10):
                print('nop')
        if n != prev_indents[-1]:
            prev_indents.append(n)
    tokiter = iter(tokens)
    for tok in tokiter:
        if tok.type in {tokenize.INDENT, tokenize.DEDENT}:
            _add_indent(tok.end[1])
        elif tok.type == tokenize.NL:
            try:
                _add_indent(next(tokiter).start[1])
            except StopIteration:
                break
    last_indent = prev_indents.pop()
    if tokens[-1].type == IN_MULTILINE_STATEMENT:
        if tokens[-2].exact_type in {tokenize.LPAR, tokenize.LSQB, tokenize.LBRACE}:
            return last_indent + 4
        return last_indent
    if tokens[-1].exact_type == tokenize.COLON:
        return last_indent + 4
    if last_indent:
        last_line_starts = 0
        for (i, tok) in enumerate(tokens):
            if tok.type == tokenize.NEWLINE:
                last_line_starts = i + 1
        last_line_tokens = tokens[last_line_starts:]
        names = [t.string for t in last_line_tokens if t.type == tokenize.NAME]
        if names and names[0] in {'raise', 'return', 'pass', 'break', 'continue'}:
            for indent in reversed(prev_indents):
                if indent < last_indent:
                    return indent
    return last_indent

def last_blank(src):
    if False:
        while True:
            i = 10
    'Determine if the input source ends in a blank.\n\n    A blank is either a newline or a line consisting of whitespace.\n\n    Parameters\n    ----------\n    src : string\n        A single or multiline string.\n    '
    if not src:
        return False
    ll = src.splitlines()[-1]
    return ll == '' or ll.isspace()
last_two_blanks_re = re.compile('\\n\\s*\\n\\s*$', re.MULTILINE)
last_two_blanks_re2 = re.compile('.+\\n\\s*\\n\\s+$', re.MULTILINE)

def last_two_blanks(src):
    if False:
        for i in range(10):
            print('nop')
    'Determine if the input source ends in two blanks.\n\n    A blank is either a newline or a line consisting of whitespace.\n\n    Parameters\n    ----------\n    src : string\n        A single or multiline string.\n    '
    if not src:
        return False
    new_src = '\n'.join(['###\n'] + src.splitlines()[-2:])
    return bool(last_two_blanks_re.match(new_src)) or bool(last_two_blanks_re2.match(new_src))

def remove_comments(src):
    if False:
        for i in range(10):
            print('nop')
    'Remove all comments from input source.\n\n    Note: comments are NOT recognized inside of strings!\n\n    Parameters\n    ----------\n    src : string\n        A single or multiline input string.\n\n    Returns\n    -------\n    String with all Python comments removed.\n    '
    return re.sub('#.*', '', src)

def get_input_encoding():
    if False:
        print('Hello World!')
    "Return the default standard input encoding.\n\n    If sys.stdin has no encoding, 'ascii' is returned."
    encoding = getattr(sys.stdin, 'encoding', None)
    if encoding is None:
        encoding = 'ascii'
    return encoding

class InputSplitter(object):
    """An object that can accumulate lines of Python source before execution.

    This object is designed to be fed python source line-by-line, using
    :meth:`push`. It will return on each push whether the currently pushed
    code could be executed already. In addition, it provides a method called
    :meth:`push_accepts_more` that can be used to query whether more input
    can be pushed into a single interactive block.

    This is a simple example of how an interactive terminal-based client can use
    this tool::

        isp = InputSplitter()
        while isp.push_accepts_more():
            indent = ' '*isp.indent_spaces
            prompt = '>>> ' + indent
            line = indent + raw_input(prompt)
            isp.push(line)
        print 'Input source was:\\n', isp.source_reset(),
    """
    _indent_spaces_cache: Union[Tuple[None, None], Tuple[str, int]] = (None, None)
    encoding = ''
    source: str = ''
    code: Optional[CodeType] = None
    _buffer: List[str]
    _compile: codeop.CommandCompiler
    _is_complete: Optional[bool] = None
    _is_invalid: bool = False

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        'Create a new InputSplitter instance.'
        self._buffer = []
        self._compile = codeop.CommandCompiler()
        self.encoding = get_input_encoding()

    def reset(self):
        if False:
            i = 10
            return i + 15
        'Reset the input buffer and associated state.'
        self._buffer[:] = []
        self.source = ''
        self.code = None
        self._is_complete = False
        self._is_invalid = False

    def source_reset(self):
        if False:
            return 10
        'Return the input source and perform a full reset.\n        '
        out = self.source
        self.reset()
        return out

    def check_complete(self, source):
        if False:
            return 10
        "Return whether a block of code is ready to execute, or should be continued\n\n        This is a non-stateful API, and will reset the state of this InputSplitter.\n\n        Parameters\n        ----------\n        source : string\n            Python input code, which can be multiline.\n\n        Returns\n        -------\n        status : str\n            One of 'complete', 'incomplete', or 'invalid' if source is not a\n            prefix of valid code.\n        indent_spaces : int or None\n            The number of spaces by which to indent the next line of code. If\n            status is not 'incomplete', this is None.\n        "
        self.reset()
        try:
            self.push(source)
        except SyntaxError:
            return ('invalid', None)
        else:
            if self._is_invalid:
                return ('invalid', None)
            elif self.push_accepts_more():
                return ('incomplete', self.get_indent_spaces())
            else:
                return ('complete', None)
        finally:
            self.reset()

    def push(self, lines: str) -> bool:
        if False:
            while True:
                i = 10
        'Push one or more lines of input.\n\n        This stores the given lines and returns a status code indicating\n        whether the code forms a complete Python block or not.\n\n        Any exceptions generated in compilation are swallowed, but if an\n        exception was produced, the method returns True.\n\n        Parameters\n        ----------\n        lines : string\n            One or more lines of Python input.\n\n        Returns\n        -------\n        is_complete : boolean\n            True if the current input source (the result of the current input\n            plus prior inputs) forms a complete Python execution block.  Note that\n            this value is also stored as a private attribute (``_is_complete``), so it\n            can be queried at any time.\n        '
        assert isinstance(lines, str)
        self._store(lines)
        source = self.source
        (self.code, self._is_complete) = (None, None)
        self._is_invalid = False
        if source.endswith('\\\n'):
            return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', SyntaxWarning)
                self.code = self._compile(source, symbol='exec')
        except (SyntaxError, OverflowError, ValueError, TypeError, MemoryError, SyntaxWarning):
            self._is_complete = True
            self._is_invalid = True
        else:
            self._is_complete = self.code is not None
        return self._is_complete

    def push_accepts_more(self):
        if False:
            print('Hello World!')
        'Return whether a block of interactive input can accept more input.\n\n        This method is meant to be used by line-oriented frontends, who need to\n        guess whether a block is complete or not based solely on prior and\n        current input lines.  The InputSplitter considers it has a complete\n        interactive block and will not accept more input when either:\n\n        * A SyntaxError is raised\n\n        * The code is complete and consists of a single line or a single\n          non-compound statement\n\n        * The code is complete and has a blank line at the end\n\n        If the current input produces a syntax error, this method immediately\n        returns False but does *not* raise the syntax error exception, as\n        typically clients will want to send invalid syntax to an execution\n        backend which might convert the invalid syntax into valid Python via\n        one of the dynamic IPython mechanisms.\n        '
        if not self._is_complete:
            return True
        last_line = self.source.splitlines()[-1]
        if not last_line or last_line.isspace():
            return False
        if self.get_indent_spaces() == 0:
            if len(self.source.splitlines()) <= 1:
                return False
            try:
                code_ast = ast.parse(''.join(self._buffer))
            except Exception:
                return False
            else:
                if len(code_ast.body) == 1 and (not hasattr(code_ast.body[0], 'body')):
                    return False
        return True

    def get_indent_spaces(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        (sourcefor, n) = self._indent_spaces_cache
        if sourcefor == self.source:
            assert n is not None
            return n
        n = find_next_indent(self.source[:-1])
        self._indent_spaces_cache = (self.source, n)
        return n
    indent_spaces = property(get_indent_spaces)

    def _store(self, lines, buffer=None, store='source'):
        if False:
            i = 10
            return i + 15
        'Store one or more lines of input.\n\n        If input lines are not newline-terminated, a newline is automatically\n        appended.'
        if buffer is None:
            buffer = self._buffer
        if lines.endswith('\n'):
            buffer.append(lines)
        else:
            buffer.append(lines + '\n')
        setattr(self, store, self._set_source(buffer))

    def _set_source(self, buffer):
        if False:
            for i in range(10):
                print('nop')
        return u''.join(buffer)

class IPythonInputSplitter(InputSplitter):
    """An input splitter that recognizes all of IPython's special syntax."""
    source_raw = ''
    transformer_accumulating = False
    within_python_line = False
    _buffer_raw: List[str]

    def __init__(self, line_input_checker=True, physical_line_transforms=None, logical_line_transforms=None, python_line_transforms=None):
        if False:
            for i in range(10):
                print('nop')
        super(IPythonInputSplitter, self).__init__()
        self._buffer_raw = []
        self._validate = True
        if physical_line_transforms is not None:
            self.physical_line_transforms = physical_line_transforms
        else:
            self.physical_line_transforms = [leading_indent(), classic_prompt(), ipy_prompt(), cellmagic(end_on_blank_line=line_input_checker)]
        self.assemble_logical_lines = assemble_logical_lines()
        if logical_line_transforms is not None:
            self.logical_line_transforms = logical_line_transforms
        else:
            self.logical_line_transforms = [help_end(), escaped_commands(), assign_from_magic(), assign_from_system()]
        self.assemble_python_lines = assemble_python_lines()
        if python_line_transforms is not None:
            self.python_line_transforms = python_line_transforms
        else:
            self.python_line_transforms = []

    @property
    def transforms(self):
        if False:
            while True:
                i = 10
        'Quick access to all transformers.'
        return self.physical_line_transforms + [self.assemble_logical_lines] + self.logical_line_transforms + [self.assemble_python_lines] + self.python_line_transforms

    @property
    def transforms_in_use(self):
        if False:
            print('Hello World!')
        "Transformers, excluding logical line transformers if we're in a\n        Python line."
        t = self.physical_line_transforms[:]
        if not self.within_python_line:
            t += [self.assemble_logical_lines] + self.logical_line_transforms
        return t + [self.assemble_python_lines] + self.python_line_transforms

    def reset(self):
        if False:
            while True:
                i = 10
        'Reset the input buffer and associated state.'
        super(IPythonInputSplitter, self).reset()
        self._buffer_raw[:] = []
        self.source_raw = ''
        self.transformer_accumulating = False
        self.within_python_line = False
        for t in self.transforms:
            try:
                t.reset()
            except SyntaxError:
                pass

    def flush_transformers(self):
        if False:
            for i in range(10):
                print('nop')

        def _flush(transform, outs):
            if False:
                return 10
            'yield transformed lines\n\n            always strings, never None\n\n            transform: the current transform\n            outs: an iterable of previously transformed inputs.\n                 Each may be multiline, which will be passed\n                 one line at a time to transform.\n            '
            for out in outs:
                for line in out.splitlines():
                    tmp = transform.push(line)
                    if tmp is not None:
                        yield tmp
            tmp = transform.reset()
            if tmp is not None:
                yield tmp
        out: List[str] = []
        for t in self.transforms_in_use:
            out = _flush(t, out)
        out = list(out)
        if out:
            self._store('\n'.join(out))

    def raw_reset(self):
        if False:
            i = 10
            return i + 15
        'Return raw input only and perform a full reset.\n        '
        out = self.source_raw
        self.reset()
        return out

    def source_reset(self):
        if False:
            while True:
                i = 10
        try:
            self.flush_transformers()
            return self.source
        finally:
            self.reset()

    def push_accepts_more(self):
        if False:
            print('Hello World!')
        if self.transformer_accumulating:
            return True
        else:
            return super(IPythonInputSplitter, self).push_accepts_more()

    def transform_cell(self, cell):
        if False:
            while True:
                i = 10
        'Process and translate a cell of input.\n        '
        self.reset()
        try:
            self.push(cell)
            self.flush_transformers()
            return self.source
        finally:
            self.reset()

    def push(self, lines: str) -> bool:
        if False:
            print('Hello World!')
        'Push one or more lines of IPython input.\n\n        This stores the given lines and returns a status code indicating\n        whether the code forms a complete Python block or not, after processing\n        all input lines for special IPython syntax.\n\n        Any exceptions generated in compilation are swallowed, but if an\n        exception was produced, the method returns True.\n\n        Parameters\n        ----------\n        lines : string\n            One or more lines of Python input.\n\n        Returns\n        -------\n        is_complete : boolean\n            True if the current input source (the result of the current input\n            plus prior inputs) forms a complete Python execution block.  Note that\n            this value is also stored as a private attribute (_is_complete), so it\n            can be queried at any time.\n        '
        assert isinstance(lines, str)
        lines_list = lines.splitlines()
        if not lines_list:
            lines_list = ['']
        self._store(lines, self._buffer_raw, 'source_raw')
        transformed_lines_list = []
        for line in lines_list:
            transformed = self._transform_line(line)
            if transformed is not None:
                transformed_lines_list.append(transformed)
        if transformed_lines_list:
            transformed_lines = '\n'.join(transformed_lines_list)
            return super(IPythonInputSplitter, self).push(transformed_lines)
        else:
            return False

    def _transform_line(self, line):
        if False:
            print('Hello World!')
        'Push a line of input code through the various transformers.\n\n        Returns any output from the transformers, or None if a transformer\n        is accumulating lines.\n\n        Sets self.transformer_accumulating as a side effect.\n        '

        def _accumulating(dbg):
            if False:
                while True:
                    i = 10
            self.transformer_accumulating = True
            return None
        for transformer in self.physical_line_transforms:
            line = transformer.push(line)
            if line is None:
                return _accumulating(transformer)
        if not self.within_python_line:
            line = self.assemble_logical_lines.push(line)
            if line is None:
                return _accumulating('acc logical line')
            for transformer in self.logical_line_transforms:
                line = transformer.push(line)
                if line is None:
                    return _accumulating(transformer)
        line = self.assemble_python_lines.push(line)
        if line is None:
            self.within_python_line = True
            return _accumulating('acc python line')
        else:
            self.within_python_line = False
        for transformer in self.python_line_transforms:
            line = transformer.push(line)
            if line is None:
                return _accumulating(transformer)
        self.transformer_accumulating = False
        return line