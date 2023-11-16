"""Token-related utilities"""
from collections import namedtuple
from io import StringIO
from keyword import iskeyword
import tokenize
from tokenize import TokenInfo
from typing import List, Optional
Token = namedtuple('Token', ['token', 'text', 'start', 'end', 'line'])

def generate_tokens(readline):
    if False:
        for i in range(10):
            print('nop')
    'wrap generate_tkens to catch EOF errors'
    try:
        for token in tokenize.generate_tokens(readline):
            yield token
    except tokenize.TokenError:
        return

def generate_tokens_catch_errors(readline, extra_errors_to_catch: Optional[List[str]]=None):
    if False:
        i = 10
        return i + 15
    default_errors_to_catch = ['unterminated string literal', 'invalid non-printable character', 'after line continuation character']
    assert extra_errors_to_catch is None or isinstance(extra_errors_to_catch, list)
    errors_to_catch = default_errors_to_catch + (extra_errors_to_catch or [])
    tokens: List[TokenInfo] = []
    try:
        for token in tokenize.generate_tokens(readline):
            tokens.append(token)
            yield token
    except tokenize.TokenError as exc:
        if any((error in exc.args[0] for error in errors_to_catch)):
            if tokens:
                start = (tokens[-1].start[0], tokens[-1].end[0])
                end = start
                line = tokens[-1].line
            else:
                start = end = (1, 0)
                line = ''
            yield tokenize.TokenInfo(tokenize.ERRORTOKEN, '', start, end, line)
        else:
            raise

def line_at_cursor(cell, cursor_pos=0):
    if False:
        while True:
            i = 10
    "Return the line in a cell at a given cursor position\n\n    Used for calling line-based APIs that don't support multi-line input, yet.\n\n    Parameters\n    ----------\n    cell : str\n        multiline block of text\n    cursor_pos : integer\n        the cursor position\n\n    Returns\n    -------\n    (line, offset): (string, integer)\n        The line with the current cursor, and the character offset of the start of the line.\n    "
    offset = 0
    lines = cell.splitlines(True)
    for line in lines:
        next_offset = offset + len(line)
        if not line.endswith('\n'):
            next_offset += 1
        if next_offset > cursor_pos:
            break
        offset = next_offset
    else:
        line = ''
    return (line, offset)

def token_at_cursor(cell: str, cursor_pos: int=0):
    if False:
        return 10
    'Get the token at a given cursor\n\n    Used for introspection.\n\n    Function calls are prioritized, so the token for the callable will be returned\n    if the cursor is anywhere inside the call.\n\n    Parameters\n    ----------\n    cell : str\n        A block of Python code\n    cursor_pos : int\n        The location of the cursor in the block where the token should be found\n    '
    names: List[str] = []
    tokens: List[Token] = []
    call_names = []
    offsets = {1: 0}
    for tup in generate_tokens(StringIO(cell).readline):
        tok = Token(*tup)
        (start_line, start_col) = tok.start
        (end_line, end_col) = tok.end
        if end_line + 1 not in offsets:
            lines = tok.line.splitlines(True)
            for (lineno, line) in enumerate(lines, start_line + 1):
                if lineno not in offsets:
                    offsets[lineno] = offsets[lineno - 1] + len(line)
        offset = offsets[start_line]
        boundary = cursor_pos + 1 if start_col == 0 else cursor_pos
        if offset + start_col >= boundary:
            break
        if tok.token == tokenize.NAME and (not iskeyword(tok.text)):
            if names and tokens and (tokens[-1].token == tokenize.OP) and (tokens[-1].text == '.'):
                names[-1] = '%s.%s' % (names[-1], tok.text)
            else:
                names.append(tok.text)
        elif tok.token == tokenize.OP:
            if tok.text == '=' and names:
                names.pop(-1)
            if tok.text == '(' and names:
                call_names.append(names[-1])
            elif tok.text == ')' and call_names:
                call_names.pop(-1)
        tokens.append(tok)
        if offsets[end_line] + end_col > cursor_pos:
            break
    if call_names:
        return call_names[-1]
    elif names:
        return names[-1]
    else:
        return ''