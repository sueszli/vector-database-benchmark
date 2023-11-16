"""Implements the xonsh executer."""
import builtins
import collections.abc as cabc
import inspect
import sys
import types
from xonsh.ast import CtxAwareTransformer
from xonsh.parser import Parser
from xonsh.tools import balanced_parens, ends_with_colon_token, find_next_break, get_logical_line, replace_logical_line, starting_whitespace, subproc_toks

class Execer:
    """Executes xonsh code in a context."""

    def __init__(self, filename='<xonsh-code>', debug_level=0, parser_args=None, scriptcache=True, cacheall=False):
        if False:
            for i in range(10):
                print('nop')
        'Parameters\n        ----------\n        filename : str, optional\n            File we are to execute.\n        debug_level : int, optional\n            Debugging level to use in lexing and parsing.\n        parser_args : dict, optional\n            Arguments to pass down to the parser.\n        scriptcache : bool, optional\n            Whether or not to use a precompiled bytecode cache when execing\n            code, default: True.\n        cacheall : bool, optional\n            Whether or not to cache all xonsh code, and not just files. If this\n            is set to true, it will cache command line input too, default: False.\n        '
        parser_args = parser_args or {}
        self.parser = Parser(**parser_args)
        self.filename = filename
        self._default_filename = filename
        self.debug_level = debug_level
        self.scriptcache = scriptcache
        self.cacheall = cacheall
        self.ctxtransformer = CtxAwareTransformer(self.parser)

    def parse(self, input, ctx, mode='exec', filename=None, transform=True):
        if False:
            print('Hello World!')
        'Parses xonsh code in a context-aware fashion. For context-free\n        parsing, please use the Parser class directly or pass in\n        transform=False.\n        '
        if filename is None:
            filename = self.filename
        if not transform:
            return self.parser.parse(input, filename=filename, mode=mode, debug_level=self.debug_level >= 2)
        (tree, input) = self._parse_ctx_free(input, mode=mode, filename=filename)
        if tree is None:
            return None
        if ctx is None:
            ctx = set()
        elif isinstance(ctx, cabc.Mapping):
            ctx = set(ctx.keys())
        tree = self.ctxtransformer.ctxvisit(tree, input, ctx, mode=mode, debug_level=self.debug_level)
        return tree

    def compile(self, input, mode='exec', glbs=None, locs=None, stacklevel=2, filename=None, transform=True, compile_empty_tree=True):
        if False:
            print('Hello World!')
        'Compiles xonsh code into a Python code object, which may then\n        be execed or evaled.\n        '
        if filename is None:
            filename = self.filename
            self.filename = self._default_filename
        if glbs is None or locs is None:
            frame = inspect.currentframe()
            for _ in range(stacklevel):
                frame = frame.f_back
            glbs = frame.f_globals if glbs is None else glbs
            locs = frame.f_locals if locs is None else locs
        ctx = set(dir(builtins)) | set(glbs.keys()) | set(locs.keys())
        tree = self.parse(input, ctx, mode=mode, filename=filename, transform=transform)
        if tree is None:
            return compile('pass', filename, mode) if compile_empty_tree else None
        try:
            code = compile(tree, filename, mode)
        except SyntaxError as e:
            if e.text is None:
                lines = input.splitlines()
                i = max(0, min(e.lineno - 1, len(lines) - 1))
                e.text = lines[i]
            raise e
        return code

    def eval(self, input, glbs=None, locs=None, stacklevel=2, filename=None, transform=True):
        if False:
            for i in range(10):
                print('nop')
        'Evaluates (and returns) xonsh code.'
        if glbs is None:
            glbs = {}
        if isinstance(input, types.CodeType):
            code = input
        else:
            input = input.rstrip('\n')
            if filename is None:
                filename = self.filename
            code = self.compile(input=input, glbs=glbs, locs=locs, mode='eval', stacklevel=stacklevel, filename=filename, transform=transform)
            if code is None:
                return None
        return eval(code, glbs, locs)

    def exec(self, input, mode='exec', glbs=None, locs=None, stacklevel=2, filename=None, transform=True):
        if False:
            for i in range(10):
                print('nop')
        'Execute xonsh code.'
        if glbs is None:
            glbs = {}
        if isinstance(input, types.CodeType):
            code = input
        else:
            if not input.endswith('\n'):
                input += '\n'
            if filename is None:
                filename = self.filename
            code = self.compile(input=input, glbs=glbs, locs=locs, mode=mode, stacklevel=stacklevel, filename=filename, transform=transform)
            if code is None:
                return None
        return exec(code, glbs, locs)

    def _print_debug_wrapping(self, line, sbpline, last_error_line, last_error_col, maxcol=None):
        if False:
            for i in range(10):
                print('nop')
        'print some debugging info if asked for.'
        if self.debug_level >= 1:
            msg = '{0}:{1}:{2}{3} - {4}\n{0}:{1}:{2}{3} + {5}'
            mstr = '' if maxcol is None else ':' + str(maxcol)
            msg = msg.format(self.filename, last_error_line, last_error_col, mstr, line, sbpline)
            print(msg, file=sys.stderr)

    def _parse_ctx_free(self, input, mode='exec', filename=None, logical_input=False):
        if False:
            for i in range(10):
                print('nop')
        if filename is None:
            filename = self.filename

        def _try_parse(input, greedy):
            if False:
                for i in range(10):
                    print('nop')
            last_error_line = last_error_col = -1
            parsed = False
            original_error = None
            if logical_input:
                beg_spaces = starting_whitespace(input)
                input = input[len(beg_spaces):]
            while not parsed:
                try:
                    tree = self.parser.parse(input, filename=filename, mode=mode, debug_level=self.debug_level >= 2)
                    parsed = True
                except IndentationError as e:
                    if original_error is None:
                        raise e
                    else:
                        raise original_error from None
                except SyntaxError as e:
                    if original_error is None:
                        original_error = e
                    if e.loc is None or (last_error_line == e.loc.lineno and last_error_col in (e.loc.column + 1, e.loc.column)):
                        raise original_error from None
                    elif last_error_line != e.loc.lineno:
                        original_error = e
                    last_error_col = e.loc.column
                    last_error_line = e.loc.lineno
                    idx = last_error_line - 1
                    lines = input.splitlines()
                    if input.endswith('\n'):
                        lines.append('')
                    (line, nlogical, idx) = get_logical_line(lines, idx)
                    if nlogical > 1 and (not logical_input):
                        (_, sbpline) = self._parse_ctx_free(line, mode=mode, filename=filename, logical_input=True)
                        self._print_debug_wrapping(line, sbpline, last_error_line, last_error_col, maxcol=None)
                        replace_logical_line(lines, sbpline, idx, nlogical)
                        last_error_col += 3
                        input = '\n'.join(lines)
                        continue
                    if len(line.strip()) == 0:
                        del lines[idx]
                        last_error_line = last_error_col = -1
                        input = '\n'.join(lines)
                        continue
                    if last_error_line > 1 and ends_with_colon_token(lines[idx - 1]):
                        prev_indent = len(lines[idx - 1]) - len(lines[idx - 1].lstrip())
                        curr_indent = len(lines[idx]) - len(lines[idx].lstrip())
                        if prev_indent == curr_indent:
                            raise original_error from None
                    lexer = self.parser.lexer
                    maxcol = None if greedy else find_next_break(line, mincol=last_error_col, lexer=lexer)
                    if not greedy and maxcol in (e.loc.column + 1, e.loc.column):
                        if not balanced_parens(line, maxcol=maxcol):
                            greedy = True
                            maxcol = None
                    sbpline = subproc_toks(line, returnline=True, greedy=greedy, maxcol=maxcol, lexer=lexer)
                    if sbpline is None:
                        if len(line.partition('#')[0].strip()) == 0:
                            del lines[idx]
                            last_error_line = last_error_col = -1
                            input = '\n'.join(lines)
                            continue
                        elif not greedy:
                            greedy = True
                            continue
                        else:
                            raise original_error from None
                    elif sbpline[last_error_col:].startswith('![![') or sbpline.lstrip().startswith('![!['):
                        if not greedy:
                            greedy = True
                            continue
                        else:
                            raise original_error from None
                    self._print_debug_wrapping(line, sbpline, last_error_line, last_error_col, maxcol=maxcol)
                    replace_logical_line(lines, sbpline, idx, nlogical)
                    last_error_col += 3
                    input = '\n'.join(lines)
            if logical_input:
                input = beg_spaces + input
            return (tree, input)
        try:
            return _try_parse(input, greedy=False)
        except SyntaxError:
            return _try_parse(input, greedy=True)