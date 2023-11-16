"""
FIXME:https://github.com/twisted/twisted/issues/3843
This can be removed once t.persisted.aot is removed.
New code should not make use of this.

Tokenization help for Python programs.
vendored from https://github.com/python/cpython/blob/6b825c1b8a14460641ca6f1647d83005c68199aa/Lib/tokenize.py
Licence: https://docs.python.org/3/license.html

tokenize(readline) is a generator that breaks a stream of bytes into
Python tokens.  It decodes the bytes according to PEP-0263 for
determining source file encoding.

It accepts a readline-like method which is called repeatedly to get the
next line of input (or b"" for EOF).  It generates 5-tuples with these
members:

    the token type (see token.py)
    the token (a string)
    the starting (row, column) indices of the token (a 2-tuple of ints)
    the ending (row, column) indices of the token (a 2-tuple of ints)
    the original line (string)

It is designed to match the working of the Python tokenizer exactly, except
that it produces COMMENT tokens for comments and gives type OP for all
operators.  Additionally, all token lists start with an ENCODING token
which tells you which encoding was used to decode the bytes stream.
"""
__author__ = 'Ka-Ping Yee <ping@lfw.org>'
__credits__ = 'GvR, ESR, Tim Peters, Thomas Wouters, Fred Drake, Skip Montanaro, Raymond Hettinger, Trent Nelson, Michael Foord'
import collections
import functools
import itertools as _itertools
import re
import sys
from builtins import open as _builtin_open
from codecs import BOM_UTF8, lookup
from io import TextIOWrapper
from ._token import AMPER, AMPEREQUAL, ASYNC, AT, ATEQUAL, AWAIT, CIRCUMFLEX, CIRCUMFLEXEQUAL, COLON, COLONEQUAL, COMMA, COMMENT, DEDENT, DOT, DOUBLESLASH, DOUBLESLASHEQUAL, DOUBLESTAR, DOUBLESTAREQUAL, ELLIPSIS, ENCODING, ENDMARKER, EQEQUAL, EQUAL, ERRORTOKEN, EXACT_TOKEN_TYPES, GREATER, GREATEREQUAL, INDENT, ISEOF, ISNONTERMINAL, ISTERMINAL, LBRACE, LEFTSHIFT, LEFTSHIFTEQUAL, LESS, LESSEQUAL, LPAR, LSQB, MINEQUAL, MINUS, N_TOKENS, NAME, NEWLINE, NL, NOTEQUAL, NT_OFFSET, NUMBER, OP, PERCENT, PERCENTEQUAL, PLUS, PLUSEQUAL, RARROW, RBRACE, RIGHTSHIFT, RIGHTSHIFTEQUAL, RPAR, RSQB, SEMI, SLASH, SLASHEQUAL, SOFT_KEYWORD, STAR, STAREQUAL, STRING, TILDE, TYPE_COMMENT, TYPE_IGNORE, VBAR, VBAREQUAL, tok_name
__all__ = ['tok_name', 'ISTERMINAL', 'ISNONTERMINAL', 'ISEOF', 'ENDMARKER', 'NAME', 'NUMBER', 'STRING', 'NEWLINE', 'INDENT', 'DEDENT', 'LPAR', 'RPAR', 'LSQB', 'RSQB', 'COLON', 'COMMA', 'SEMI', 'PLUS', 'MINUS', 'STAR', 'SLASH', 'VBAR', 'AMPER', 'LESS', 'GREATER', 'EQUAL', 'DOT', 'PERCENT', 'LBRACE', 'RBRACE', 'EQEQUAL', 'NOTEQUAL', 'LESSEQUAL', 'GREATEREQUAL', 'TILDE', 'CIRCUMFLEX', 'LEFTSHIFT', 'RIGHTSHIFT', 'DOUBLESTAR', 'PLUSEQUAL', 'MINEQUAL', 'STAREQUAL', 'SLASHEQUAL', 'PERCENTEQUAL', 'AMPEREQUAL', 'VBAREQUAL', 'CIRCUMFLEXEQUAL', 'LEFTSHIFTEQUAL', 'RIGHTSHIFTEQUAL', 'DOUBLESTAREQUAL', 'DOUBLESLASH', 'DOUBLESLASHEQUAL', 'AT', 'ATEQUAL', 'RARROW', 'ELLIPSIS', 'COLONEQUAL', 'OP', 'AWAIT', 'ASYNC', 'TYPE_IGNORE', 'TYPE_COMMENT', 'SOFT_KEYWORD', 'ERRORTOKEN', 'COMMENT', 'NL', 'ENCODING', 'N_TOKENS', 'NT_OFFSET', 'tokenize', 'generate_tokens', 'detect_encoding', 'untokenize', 'TokenInfo']
cookie_re = re.compile('^[ \\t\\f]*#.*?coding[:=][ \\t]*([-\\w.]+)', re.ASCII)
blank_re = re.compile(b'^[ \\t\\f]*(?:[#\\r\\n]|$)', re.ASCII)

class TokenInfo(collections.namedtuple('TokenInfo', 'type string start end line')):

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        annotated_type = '%d (%s)' % (self.type, tok_name[self.type])
        return 'TokenInfo(type=%s, string=%r, start=%r, end=%r, line=%r)' % self._replace(type=annotated_type)

    @property
    def exact_type(self):
        if False:
            i = 10
            return i + 15
        if self.type == OP and self.string in EXACT_TOKEN_TYPES:
            return EXACT_TOKEN_TYPES[self.string]
        else:
            return self.type

def group(*choices):
    if False:
        while True:
            i = 10
    return '(' + '|'.join(choices) + ')'

def any(*choices):
    if False:
        for i in range(10):
            print('nop')
    return group(*choices) + '*'

def maybe(*choices):
    if False:
        for i in range(10):
            print('nop')
    return group(*choices) + '?'
Whitespace = '[ \\f\\t]*'
Comment = '#[^\\r\\n]*'
Ignore = Whitespace + any('\\\\\\r?\\n' + Whitespace) + maybe(Comment)
Name = '\\w+'
Hexnumber = '0[xX](?:_?[0-9a-fA-F])+'
Binnumber = '0[bB](?:_?[01])+'
Octnumber = '0[oO](?:_?[0-7])+'
Decnumber = '(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
Intnumber = group(Hexnumber, Binnumber, Octnumber, Decnumber)
Exponent = '[eE][-+]?[0-9](?:_?[0-9])*'
Pointfloat = group('[0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?', '\\.[0-9](?:_?[0-9])*') + maybe(Exponent)
Expfloat = '[0-9](?:_?[0-9])*' + Exponent
Floatnumber = group(Pointfloat, Expfloat)
Imagnumber = group('[0-9](?:_?[0-9])*[jJ]', Floatnumber + '[jJ]')
Number = group(Imagnumber, Floatnumber, Intnumber)

def _all_string_prefixes():
    if False:
        i = 10
        return i + 15
    _valid_string_prefixes = ['b', 'r', 'u', 'f', 'br', 'fr']
    result = {''}
    for prefix in _valid_string_prefixes:
        for t in _itertools.permutations(prefix):
            for u in _itertools.product(*[(c, c.upper()) for c in t]):
                result.add(''.join(u))
    return result

@functools.lru_cache(None)
def _compile(expr):
    if False:
        for i in range(10):
            print('nop')
    return re.compile(expr, re.UNICODE)
StringPrefix = group(*_all_string_prefixes())
Single = "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
Double = '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"'
Single3 = "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''"
Double3 = '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'
Triple = group(StringPrefix + "'''", StringPrefix + '"""')
String = group(StringPrefix + "'[^\\n'\\\\]*(?:\\\\.[^\\n'\\\\]*)*'", StringPrefix + '"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*"')
Special = group(*(re.escape(x) for x in sorted(EXACT_TOKEN_TYPES, reverse=True)))
Funny = group('\\r?\\n', Special)
PlainToken = group(Number, Funny, String, Name)
Token = Ignore + PlainToken
ContStr = group(StringPrefix + "'[^\\n'\\\\]*(?:\\\\.[^\\n'\\\\]*)*" + group("'", '\\\\\\r?\\n'), StringPrefix + '"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*' + group('"', '\\\\\\r?\\n'))
PseudoExtras = group('\\\\\\r?\\n|\\Z', Comment, Triple)
PseudoToken = Whitespace + group(PseudoExtras, Number, Funny, ContStr, Name)
endpats = {}
for _prefix in _all_string_prefixes():
    endpats[_prefix + "'"] = Single
    endpats[_prefix + '"'] = Double
    endpats[_prefix + "'''"] = Single3
    endpats[_prefix + '"""'] = Double3
del _prefix
single_quoted = set()
triple_quoted = set()
for t in _all_string_prefixes():
    for u in (t + '"', t + "'"):
        single_quoted.add(u)
    for u in (t + '"""', t + "'''"):
        triple_quoted.add(u)
del t, u
tabsize = 8

class TokenError(Exception):
    pass

class StopTokenizing(Exception):
    pass

class Untokenizer:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.tokens = []
        self.prev_row = 1
        self.prev_col = 0
        self.encoding = None

    def add_whitespace(self, start):
        if False:
            for i in range(10):
                print('nop')
        (row, col) = start
        if row < self.prev_row or (row == self.prev_row and col < self.prev_col):
            raise ValueError('start ({},{}) precedes previous end ({},{})'.format(row, col, self.prev_row, self.prev_col))
        row_offset = row - self.prev_row
        if row_offset:
            self.tokens.append('\\\n' * row_offset)
            self.prev_col = 0
        col_offset = col - self.prev_col
        if col_offset:
            self.tokens.append(' ' * col_offset)

    def untokenize(self, iterable):
        if False:
            return 10
        it = iter(iterable)
        indents = []
        startline = False
        for t in it:
            if len(t) == 2:
                self.compat(t, it)
                break
            (tok_type, token, start, end, line) = t
            if tok_type == ENCODING:
                self.encoding = token
                continue
            if tok_type == ENDMARKER:
                break
            if tok_type == INDENT:
                indents.append(token)
                continue
            elif tok_type == DEDENT:
                indents.pop()
                (self.prev_row, self.prev_col) = end
                continue
            elif tok_type in (NEWLINE, NL):
                startline = True
            elif startline and indents:
                indent = indents[-1]
                if start[1] >= len(indent):
                    self.tokens.append(indent)
                    self.prev_col = len(indent)
                startline = False
            self.add_whitespace(start)
            self.tokens.append(token)
            (self.prev_row, self.prev_col) = end
            if tok_type in (NEWLINE, NL):
                self.prev_row += 1
                self.prev_col = 0
        return ''.join(self.tokens)

    def compat(self, token, iterable):
        if False:
            i = 10
            return i + 15
        indents = []
        toks_append = self.tokens.append
        startline = token[0] in (NEWLINE, NL)
        prevstring = False
        for tok in _itertools.chain([token], iterable):
            (toknum, tokval) = tok[:2]
            if toknum == ENCODING:
                self.encoding = tokval
                continue
            if toknum in (NAME, NUMBER):
                tokval += ' '
            if toknum == STRING:
                if prevstring:
                    tokval = ' ' + tokval
                prevstring = True
            else:
                prevstring = False
            if toknum == INDENT:
                indents.append(tokval)
                continue
            elif toknum == DEDENT:
                indents.pop()
                continue
            elif toknum in (NEWLINE, NL):
                startline = True
            elif startline and indents:
                toks_append(indents[-1])
                startline = False
            toks_append(tokval)

def untokenize(iterable):
    if False:
        return 10
    'Transform tokens back into Python source code.\n    It returns a bytes object, encoded using the ENCODING\n    token, which is the first token sequence output by tokenize.\n\n    Each element returned by the iterable must be a token sequence\n    with at least two elements, a token number and token value.  If\n    only two tokens are passed, the resulting output is poor.\n\n    Round-trip invariant for full input:\n        Untokenized source will match input source exactly\n\n    Round-trip invariant for limited input:\n        # Output bytes will tokenize back to the input\n        t1 = [tok[:2] for tok in tokenize(f.readline)]\n        newcode = untokenize(t1)\n        readline = BytesIO(newcode).readline\n        t2 = [tok[:2] for tok in tokenize(readline)]\n        assert t1 == t2\n    '
    ut = Untokenizer()
    out = ut.untokenize(iterable)
    if ut.encoding is not None:
        out = out.encode(ut.encoding)
    return out

def _get_normal_name(orig_enc):
    if False:
        while True:
            i = 10
    'Imitates get_normal_name in tokenizer.c.'
    enc = orig_enc[:12].lower().replace('_', '-')
    if enc == 'utf-8' or enc.startswith('utf-8-'):
        return 'utf-8'
    if enc in ('latin-1', 'iso-8859-1', 'iso-latin-1') or enc.startswith(('latin-1-', 'iso-8859-1-', 'iso-latin-1-')):
        return 'iso-8859-1'
    return orig_enc

def detect_encoding(readline):
    if False:
        print('Hello World!')
    "\n    The detect_encoding() function is used to detect the encoding that should\n    be used to decode a Python source file.  It requires one argument, readline,\n    in the same way as the tokenize() generator.\n\n    It will call readline a maximum of twice, and return the encoding used\n    (as a string) and a list of any lines (left as bytes) it has read in.\n\n    It detects the encoding from the presence of a utf-8 bom or an encoding\n    cookie as specified in pep-0263.  If both a bom and a cookie are present,\n    but disagree, a SyntaxError will be raised.  If the encoding cookie is an\n    invalid charset, raise a SyntaxError.  Note that if a utf-8 bom is found,\n    'utf-8-sig' is returned.\n\n    If no encoding is specified, then the default of 'utf-8' will be returned.\n    "
    try:
        filename = readline.__self__.name
    except AttributeError:
        filename = None
    bom_found = False
    encoding = None
    default = 'utf-8'

    def read_or_stop():
        if False:
            i = 10
            return i + 15
        try:
            return readline()
        except StopIteration:
            return b''

    def find_cookie(line):
        if False:
            print('Hello World!')
        try:
            line_string = line.decode('utf-8')
        except UnicodeDecodeError:
            msg = 'invalid or missing encoding declaration'
            if filename is not None:
                msg = '{} for {!r}'.format(msg, filename)
            raise SyntaxError(msg)
        match = cookie_re.match(line_string)
        if not match:
            return None
        encoding = _get_normal_name(match.group(1))
        try:
            lookup(encoding)
        except LookupError:
            if filename is None:
                msg = 'unknown encoding: ' + encoding
            else:
                msg = 'unknown encoding for {!r}: {}'.format(filename, encoding)
            raise SyntaxError(msg)
        if bom_found:
            if encoding != 'utf-8':
                if filename is None:
                    msg = 'encoding problem: utf-8'
                else:
                    msg = 'encoding problem for {!r}: utf-8'.format(filename)
                raise SyntaxError(msg)
            encoding += '-sig'
        return encoding
    first = read_or_stop()
    if first.startswith(BOM_UTF8):
        bom_found = True
        first = first[3:]
        default = 'utf-8-sig'
    if not first:
        return (default, [])
    encoding = find_cookie(first)
    if encoding:
        return (encoding, [first])
    if not blank_re.match(first):
        return (default, [first])
    second = read_or_stop()
    if not second:
        return (default, [first])
    encoding = find_cookie(second)
    if encoding:
        return (encoding, [first, second])
    return (default, [first, second])

def open(filename):
    if False:
        return 10
    'Open a file in read only mode using the encoding detected by\n    detect_encoding().\n    '
    buffer = _builtin_open(filename, 'rb')
    try:
        (encoding, lines) = detect_encoding(buffer.readline)
        buffer.seek(0)
        text = TextIOWrapper(buffer, encoding, line_buffering=True)
        text.mode = 'r'
        return text
    except BaseException:
        buffer.close()
        raise

def tokenize(readline):
    if False:
        for i in range(10):
            print('nop')
    "\n    The tokenize() generator requires one argument, readline, which\n    must be a callable object which provides the same interface as the\n    readline() method of built-in file objects.  Each call to the function\n    should return one line of input as bytes.  Alternatively, readline\n    can be a callable function terminating with StopIteration:\n        readline = open(myfile, 'rb').__next__  # Example of alternate readline\n\n    The generator produces 5-tuples with these members: the token type; the\n    token string; a 2-tuple (srow, scol) of ints specifying the row and\n    column where the token begins in the source; a 2-tuple (erow, ecol) of\n    ints specifying the row and column where the token ends in the source;\n    and the line on which the token was found.  The line passed is the\n    physical line.\n\n    The first token sequence will always be an ENCODING token\n    which tells you which encoding was used to decode the bytes stream.\n    "
    (encoding, consumed) = detect_encoding(readline)
    empty = _itertools.repeat(b'')
    rl_gen = _itertools.chain(consumed, iter(readline, b''), empty)
    return _tokenize(rl_gen.__next__, encoding)

def _tokenize(readline, encoding):
    if False:
        while True:
            i = 10
    strstart = None
    endprog = None
    lnum = parenlev = continued = 0
    numchars = '0123456789'
    (contstr, needcont) = ('', 0)
    contline = None
    indents = [0]
    if encoding is not None:
        if encoding == 'utf-8-sig':
            encoding = 'utf-8'
        yield TokenInfo(ENCODING, encoding, (0, 0), (0, 0), '')
    last_line = b''
    line = b''
    while True:
        try:
            last_line = line
            line = readline()
        except StopIteration:
            line = b''
        if encoding is not None:
            line = line.decode(encoding)
        lnum += 1
        (pos, max) = (0, len(line))
        if contstr:
            if not line:
                raise TokenError('EOF in multi-line string', strstart)
            endmatch = endprog.match(line)
            if endmatch:
                pos = end = endmatch.end(0)
                yield TokenInfo(STRING, contstr + line[:end], strstart, (lnum, end), contline + line)
                (contstr, needcont) = ('', 0)
                contline = None
            elif needcont and line[-2:] != '\\\n' and (line[-3:] != '\\\r\n'):
                yield TokenInfo(ERRORTOKEN, contstr + line, strstart, (lnum, len(line)), contline)
                contstr = ''
                contline = None
                continue
            else:
                contstr = contstr + line
                contline = contline + line
                continue
        elif parenlev == 0 and (not continued):
            if not line:
                break
            column = 0
            while pos < max:
                if line[pos] == ' ':
                    column += 1
                elif line[pos] == '\t':
                    column = (column // tabsize + 1) * tabsize
                elif line[pos] == '\x0c':
                    column = 0
                else:
                    break
                pos += 1
            if pos == max:
                break
            if line[pos] in '#\r\n':
                if line[pos] == '#':
                    comment_token = line[pos:].rstrip('\r\n')
                    yield TokenInfo(COMMENT, comment_token, (lnum, pos), (lnum, pos + len(comment_token)), line)
                    pos += len(comment_token)
                yield TokenInfo(NL, line[pos:], (lnum, pos), (lnum, len(line)), line)
                continue
            if column > indents[-1]:
                indents.append(column)
                yield TokenInfo(INDENT, line[:pos], (lnum, 0), (lnum, pos), line)
            while column < indents[-1]:
                if column not in indents:
                    raise IndentationError('unindent does not match any outer indentation level', ('<tokenize>', lnum, pos, line))
                indents = indents[:-1]
                yield TokenInfo(DEDENT, '', (lnum, pos), (lnum, pos), line)
        else:
            if not line:
                raise TokenError('EOF in multi-line statement', (lnum, 0))
            continued = 0
        while pos < max:
            pseudomatch = _compile(PseudoToken).match(line, pos)
            if pseudomatch:
                (start, end) = pseudomatch.span(1)
                (spos, epos, pos) = ((lnum, start), (lnum, end), end)
                if start == end:
                    continue
                (token, initial) = (line[start:end], line[start])
                if initial in numchars or (initial == '.' and token != '.' and (token != '...')):
                    yield TokenInfo(NUMBER, token, spos, epos, line)
                elif initial in '\r\n':
                    if parenlev > 0:
                        yield TokenInfo(NL, token, spos, epos, line)
                    else:
                        yield TokenInfo(NEWLINE, token, spos, epos, line)
                elif initial == '#':
                    assert not token.endswith('\n')
                    yield TokenInfo(COMMENT, token, spos, epos, line)
                elif token in triple_quoted:
                    endprog = _compile(endpats[token])
                    endmatch = endprog.match(line, pos)
                    if endmatch:
                        pos = endmatch.end(0)
                        token = line[start:pos]
                        yield TokenInfo(STRING, token, spos, (lnum, pos), line)
                    else:
                        strstart = (lnum, start)
                        contstr = line[start:]
                        contline = line
                        break
                elif initial in single_quoted or token[:2] in single_quoted or token[:3] in single_quoted:
                    if token[-1] == '\n':
                        strstart = (lnum, start)
                        endprog = _compile(endpats.get(initial) or endpats.get(token[1]) or endpats.get(token[2]))
                        (contstr, needcont) = (line[start:], 1)
                        contline = line
                        break
                    else:
                        yield TokenInfo(STRING, token, spos, epos, line)
                elif initial.isidentifier():
                    yield TokenInfo(NAME, token, spos, epos, line)
                elif initial == '\\':
                    continued = 1
                else:
                    if initial in '([{':
                        parenlev += 1
                    elif initial in ')]}':
                        parenlev -= 1
                    yield TokenInfo(OP, token, spos, epos, line)
            else:
                yield TokenInfo(ERRORTOKEN, line[pos], (lnum, pos), (lnum, pos + 1), line)
                pos += 1
    if last_line and last_line[-1] not in '\r\n' and (not last_line.strip().startswith('#')):
        yield TokenInfo(NEWLINE, '', (lnum - 1, len(last_line)), (lnum - 1, len(last_line) + 1), '')
    for indent in indents[1:]:
        yield TokenInfo(DEDENT, '', (lnum, 0), (lnum, 0), '')
    yield TokenInfo(ENDMARKER, '', (lnum, 0), (lnum, 0), '')

def generate_tokens(readline):
    if False:
        return 10
    'Tokenize a source reading Python code as unicode strings.\n\n    This has the same API as tokenize(), except that it expects the *readline*\n    callable to return str objects instead of bytes.\n    '
    return _tokenize(readline, None)

def main():
    if False:
        for i in range(10):
            print('nop')
    import argparse

    def perror(message):
        if False:
            return 10
        sys.stderr.write(message)
        sys.stderr.write('\n')

    def error(message, filename=None, location=None):
        if False:
            i = 10
            return i + 15
        if location:
            args = (filename,) + location + (message,)
            perror('%s:%d:%d: error: %s' % args)
        elif filename:
            perror('%s: error: %s' % (filename, message))
        else:
            perror('error: %s' % message)
        sys.exit(1)
    parser = argparse.ArgumentParser(prog='python -m tokenize')
    parser.add_argument(dest='filename', nargs='?', metavar='filename.py', help='the file to tokenize; defaults to stdin')
    parser.add_argument('-e', '--exact', dest='exact', action='store_true', help='display token names using the exact type')
    args = parser.parse_args()
    try:
        if args.filename:
            filename = args.filename
            with _builtin_open(filename, 'rb') as f:
                tokens = list(tokenize(f.readline))
        else:
            filename = '<stdin>'
            tokens = _tokenize(sys.stdin.readline, None)
        for token in tokens:
            token_type = token.type
            if args.exact:
                token_type = token.exact_type
            token_range = '%d,%d-%d,%d:' % (token.start + token.end)
            print('%-20s%-15s%-15r' % (token_range, tok_name[token_type], token.string))
    except IndentationError as err:
        (line, column) = err.args[1][1:3]
        error(err.args[0], filename, (line, column))
    except TokenError as err:
        (line, column) = err.args[1]
        error(err.args[0], filename, (line, column))
    except SyntaxError as err:
        error(err, filename)
    except OSError as err:
        error(err)
    except KeyboardInterrupt:
        print('interrupted\n')
    except Exception as err:
        perror('unexpected error: %s' % err)
        raise
if __name__ == '__main__':
    main()