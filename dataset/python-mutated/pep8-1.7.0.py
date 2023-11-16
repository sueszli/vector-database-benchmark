"""
Check Python source code formatting, according to PEP 8.

For usage and a list of options, try this:
$ python pep8.py -h

This program and its regression test suite live here:
https://github.com/pycqa/pep8

Groups of errors and warnings:
E errors
W warnings
100 indentation
200 whitespace
300 blank lines
400 imports
500 line length
600 deprecation
700 statements
900 syntax error
"""
from __future__ import with_statement
import os
import sys
import re
import time
import inspect
import keyword
import tokenize
from optparse import OptionParser
from fnmatch import fnmatch
try:
    from configparser import RawConfigParser
    from io import TextIOWrapper
except ImportError:
    from ConfigParser import RawConfigParser
__version__ = '1.7.0'
DEFAULT_EXCLUDE = '.svn,CVS,.bzr,.hg,.git,__pycache__,.tox'
DEFAULT_IGNORE = 'E121,E123,E126,E226,E24,E704'
try:
    if sys.platform == 'win32':
        USER_CONFIG = os.path.expanduser('~\\.pep8')
    else:
        USER_CONFIG = os.path.join(os.getenv('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'), 'pep8')
except ImportError:
    USER_CONFIG = None
PROJECT_CONFIG = ('setup.cfg', 'tox.ini', '.pep8')
TESTSUITE_PATH = os.path.join(os.path.dirname(__file__), 'testsuite')
MAX_LINE_LENGTH = 100
REPORT_FORMAT = {'default': '%(path)s:%(row)d:%(col)d: %(code)s %(text)s', 'pylint': '%(path)s:%(row)d: [%(code)s] %(text)s'}
PyCF_ONLY_AST = 1024
SINGLETONS = frozenset(['False', 'None', 'True'])
KEYWORDS = frozenset(keyword.kwlist + ['print']) - SINGLETONS
UNARY_OPERATORS = frozenset(['>>', '**', '*', '+', '-'])
ARITHMETIC_OP = frozenset(['**', '*', '/', '//', '+', '-'])
WS_OPTIONAL_OPERATORS = ARITHMETIC_OP.union(['^', '&', '|', '<<', '>>', '%'])
WS_NEEDED_OPERATORS = frozenset(['**=', '*=', '/=', '//=', '+=', '-=', '!=', '<>', '<', '>', '%=', '^=', '&=', '|=', '==', '<=', '>=', '<<=', '>>=', '='])
WHITESPACE = frozenset(' \t')
NEWLINE = frozenset([tokenize.NL, tokenize.NEWLINE])
SKIP_TOKENS = NEWLINE.union([tokenize.INDENT, tokenize.DEDENT])
SKIP_COMMENTS = SKIP_TOKENS.union([tokenize.COMMENT, tokenize.ERRORTOKEN])
BENCHMARK_KEYS = ['directories', 'files', 'logical lines', 'physical lines']
INDENT_REGEX = re.compile('([ \\t]*)')
RAISE_COMMA_REGEX = re.compile('raise\\s+\\w+\\s*,')
RERAISE_COMMA_REGEX = re.compile('raise\\s+\\w+\\s*,.*,\\s*\\w+\\s*$')
ERRORCODE_REGEX = re.compile('\\b[A-Z]\\d{3}\\b')
DOCSTRING_REGEX = re.compile('u?r?["\\\']')
EXTRANEOUS_WHITESPACE_REGEX = re.compile('[\\[({] | [\\]}),;:]')
WHITESPACE_AFTER_COMMA_REGEX = re.compile('[,;:]\\s*(?:  |\\t)')
COMPARE_SINGLETON_REGEX = re.compile('(\\bNone|\\bFalse|\\bTrue)?\\s*([=!]=)\\s*(?(1)|(None|False|True))\\b')
COMPARE_NEGATIVE_REGEX = re.compile('\\b(not)\\s+[^][)(}{ ]+\\s+(in|is)\\s')
COMPARE_TYPE_REGEX = re.compile('(?:[=!]=|is(?:\\s+not)?)\\s*type(?:s.\\w+Type|\\s*\\(\\s*([^)]*[^ )])\\s*\\))')
KEYWORD_REGEX = re.compile('(\\s*)\\b(?:%s)\\b(\\s*)' % '|'.join(KEYWORDS))
OPERATOR_REGEX = re.compile('(?:[^,\\s])(\\s*)(?:[-+*/|!<=>%&^]+)(\\s*)')
LAMBDA_REGEX = re.compile('\\blambda\\b')
HUNK_REGEX = re.compile('^@@ -\\d+(?:,\\d+)? \\+(\\d+)(?:,(\\d+))? @@.*$')
STARTSWITH_DEF_REGEX = re.compile('^(async\\s+def|def)\\b')
STARTSWITH_INDENT_STATEMENT_REGEX = re.compile('^\\s*({})\\b'.format('|'.join((s.replace(' ', '\\s+') for s in ('def', 'async def', 'for', 'async for', 'if', 'elif', 'else', 'try', 'except', 'finally', 'with', 'async with', 'class', 'while')))))
COMMENT_WITH_NL = tokenize.generate_tokens(['#\n'].pop).send(None)[1] == '#\n'

def tabs_or_spaces(physical_line, indent_char):
    if False:
        i = 10
        return i + 15
    'Never mix tabs and spaces.\n\n    The most popular way of indenting Python is with spaces only.  The\n    second-most popular way is with tabs only.  Code indented with a mixture\n    of tabs and spaces should be converted to using spaces exclusively.  When\n    invoking the Python command line interpreter with the -t option, it issues\n    warnings about code that illegally mixes tabs and spaces.  When using -tt\n    these warnings become errors.  These options are highly recommended!\n\n    Okay: if a == 0:\\n        a = 1\\n        b = 1\n    E101: if a == 0:\\n        a = 1\\n\\tb = 1\n    '
    indent = INDENT_REGEX.match(physical_line).group(1)
    for (offset, char) in enumerate(indent):
        if char != indent_char:
            return (offset, 'E101 indentation contains mixed spaces and tabs')

def tabs_obsolete(physical_line):
    if False:
        i = 10
        return i + 15
    'For new projects, spaces-only are strongly recommended over tabs.\n\n    Okay: if True:\\n    return\n    W191: if True:\\n\\treturn\n    '
    indent = INDENT_REGEX.match(physical_line).group(1)
    if '\t' in indent:
        return (indent.index('\t'), 'W191 indentation contains tabs')

def trailing_whitespace(physical_line):
    if False:
        print('Hello World!')
    'Trailing whitespace is superfluous.\n\n    The warning returned varies on whether the line itself is blank, for easier\n    filtering for those who want to indent their blank lines.\n\n    Okay: spam(1)\\n#\n    W291: spam(1) \\n#\n    W293: class Foo(object):\\n    \\n    bang = 12\n    '
    physical_line = physical_line.rstrip('\n')
    physical_line = physical_line.rstrip('\r')
    physical_line = physical_line.rstrip('\x0c')
    stripped = physical_line.rstrip(' \t\x0b')
    if physical_line != stripped:
        if stripped:
            return (len(stripped), 'W291 trailing whitespace')
        else:
            return (0, 'W293 blank line contains whitespace')

def trailing_blacklist_words(physical_line):
    if False:
        i = 10
        return i + 15
    if physical_line.find('assert ') != -1:
        return (0, "Please don't use assert, use log4Error.invalidInputError instead")
    if physical_line.find('raise ') != -1:
        return (0, "Please don't use raise, use log4Error.invalidInputError instead")

def trailing_blank_lines(physical_line, lines, line_number, total_lines):
    if False:
        for i in range(10):
            print('nop')
    'Trailing blank lines are superfluous.\n\n    Okay: spam(1)\n    W391: spam(1)\\n\n\n    However the last line should end with a new line (warning W292).\n    '
    if line_number == total_lines:
        stripped_last_line = physical_line.rstrip()
        if not stripped_last_line:
            return (0, 'W391 blank line at end of file')
        if stripped_last_line == physical_line:
            return (len(physical_line), 'W292 no newline at end of file')

def maximum_line_length(physical_line, max_line_length, multiline):
    if False:
        print('Hello World!')
    'Limit all lines to a maximum of 79 characters.\n\n    There are still many devices around that are limited to 80 character\n    lines; plus, limiting windows to 80 characters makes it possible to have\n    several windows side-by-side.  The default wrapping on such devices looks\n    ugly.  Therefore, please limit all lines to a maximum of 79 characters.\n    For flowing long blocks of text (docstrings or comments), limiting the\n    length to 72 characters is recommended.\n\n    Reports error E501.\n    '
    line = physical_line.rstrip()
    length = len(line)
    if length > max_line_length and (not noqa(line)):
        chunks = line.split()
        if (len(chunks) == 1 and multiline or (len(chunks) == 2 and chunks[0] == '#')) and len(line) - len(chunks[-1]) < max_line_length - 7:
            return
        if hasattr(line, 'decode'):
            try:
                length = len(line.decode('utf-8'))
            except UnicodeError:
                pass
        if length > max_line_length:
            return (max_line_length, 'E501 line too long (%d > %d characters)' % (length, max_line_length))

def blank_lines(logical_line, blank_lines, indent_level, line_number, blank_before, previous_logical, previous_indent_level):
    if False:
        while True:
            i = 10
    'Separate top-level function and class definitions with two blank lines.\n\n    Method definitions inside a class are separated by a single blank line.\n\n    Extra blank lines may be used (sparingly) to separate groups of related\n    functions.  Blank lines may be omitted between a bunch of related\n    one-liners (e.g. a set of dummy implementations).\n\n    Use blank lines in functions, sparingly, to indicate logical sections.\n\n    Okay: def a():\\n    pass\\n\\n\\ndef b():\\n    pass\n    Okay: def a():\\n    pass\\n\\n\\n# Foo\\n# Bar\\n\\ndef b():\\n    pass\n\n    E301: class Foo:\\n    b = 0\\n    def bar():\\n        pass\n    E302: def a():\\n    pass\\n\\ndef b(n):\\n    pass\n    E303: def a():\\n    pass\\n\\n\\n\\ndef b(n):\\n    pass\n    E303: def a():\\n\\n\\n\\n    pass\n    E304: @decorator\\n\\ndef a():\\n    pass\n    '
    if line_number < 3 and (not previous_logical):
        return
    if previous_logical.startswith('@'):
        if blank_lines:
            yield (0, 'E304 blank lines found after function decorator')
    elif blank_lines > 2 or (indent_level and blank_lines == 2):
        yield (0, 'E303 too many blank lines (%d)' % blank_lines)
    elif logical_line.startswith(('def ', 'class ', '@')):
        if indent_level:
            if not (blank_before or previous_indent_level < indent_level or DOCSTRING_REGEX.match(previous_logical)):
                yield (0, 'E301 expected 1 blank line, found 0')
        elif blank_before != 2:
            yield (0, 'E302 expected 2 blank lines, found %d' % blank_before)

def extraneous_whitespace(logical_line):
    if False:
        i = 10
        return i + 15
    'Avoid extraneous whitespace.\n\n    Avoid extraneous whitespace in these situations:\n    - Immediately inside parentheses, brackets or braces.\n    - Immediately before a comma, semicolon, or colon.\n\n    Okay: spam(ham[1], {eggs: 2})\n    E201: spam( ham[1], {eggs: 2})\n    E201: spam(ham[ 1], {eggs: 2})\n    E201: spam(ham[1], { eggs: 2})\n    E202: spam(ham[1], {eggs: 2} )\n    E202: spam(ham[1 ], {eggs: 2})\n    E202: spam(ham[1], {eggs: 2 })\n\n    E203: if x == 4: print x, y; x, y = y , x\n    E203: if x == 4: print x, y ; x, y = y, x\n    E203: if x == 4 : print x, y; x, y = y, x\n    '
    line = logical_line
    for match in EXTRANEOUS_WHITESPACE_REGEX.finditer(line):
        text = match.group()
        char = text.strip()
        found = match.start()
        if text == char + ' ':
            yield (found + 1, "E201 whitespace after '%s'" % char)
        elif line[found - 1] != ',':
            code = 'E202' if char in '}])' else 'E203'
            yield (found, "%s whitespace before '%s'" % (code, char))

def whitespace_around_keywords(logical_line):
    if False:
        i = 10
        return i + 15
    'Avoid extraneous whitespace around keywords.\n\n    Okay: True and False\n    E271: True and  False\n    E272: True  and False\n    E273: True and\\tFalse\n    E274: True\\tand False\n    '
    for match in KEYWORD_REGEX.finditer(logical_line):
        (before, after) = match.groups()
        if '\t' in before:
            yield (match.start(1), 'E274 tab before keyword')
        elif len(before) > 1:
            yield (match.start(1), 'E272 multiple spaces before keyword')
        if '\t' in after:
            yield (match.start(2), 'E273 tab after keyword')
        elif len(after) > 1:
            yield (match.start(2), 'E271 multiple spaces after keyword')

def missing_whitespace(logical_line):
    if False:
        return 10
    "Each comma, semicolon or colon should be followed by whitespace.\n\n    Okay: [a, b]\n    Okay: (3,)\n    Okay: a[1:4]\n    Okay: a[:4]\n    Okay: a[1:]\n    Okay: a[1:4:2]\n    E231: ['a','b']\n    E231: foo(bar,baz)\n    E231: [{'a':'b'}]\n    "
    line = logical_line
    for index in range(len(line) - 1):
        char = line[index]
        if char in ',;:' and line[index + 1] not in WHITESPACE:
            before = line[:index]
            if char == ':' and before.count('[') > before.count(']') and (before.rfind('{') < before.rfind('[')):
                continue
            if char == ',' and line[index + 1] == ')':
                continue
            yield (index, "E231 missing whitespace after '%s'" % char)

def indentation(logical_line, previous_logical, indent_char, indent_level, previous_indent_level):
    if False:
        return 10
    "Use 4 spaces per indentation level.\n\n    For really old code that you don't want to mess up, you can continue to\n    use 8-space tabs.\n\n    Okay: a = 1\n    Okay: if a == 0:\\n    a = 1\n    E111:   a = 1\n    E114:   # a = 1\n\n    Okay: for item in items:\\n    pass\n    E112: for item in items:\\npass\n    E115: for item in items:\\n# Hi\\n    pass\n\n    Okay: a = 1\\nb = 2\n    E113: a = 1\\n    b = 2\n    E116: a = 1\\n    # b = 2\n    "
    c = 0 if logical_line else 3
    tmpl = 'E11%d %s' if logical_line else 'E11%d %s (comment)'
    if indent_level % 4:
        yield (0, tmpl % (1 + c, 'indentation is not a multiple of four'))
    indent_expect = previous_logical.endswith(':')
    if indent_expect and indent_level <= previous_indent_level:
        yield (0, tmpl % (2 + c, 'expected an indented block'))
    elif not indent_expect and indent_level > previous_indent_level:
        yield (0, tmpl % (3 + c, 'unexpected indentation'))

def continued_indentation(logical_line, tokens, indent_level, hang_closing, indent_char, noqa, verbose):
    if False:
        for i in range(10):
            print('nop')
    "Continuation lines indentation.\n\n    Continuation lines should align wrapped elements either vertically\n    using Python's implicit line joining inside parentheses, brackets\n    and braces, or using a hanging indent.\n\n    When using a hanging indent these considerations should be applied:\n    - there should be no arguments on the first line, and\n    - further indentation should be used to clearly distinguish itself as a\n      continuation line.\n\n    Okay: a = (\\n)\n    E123: a = (\\n    )\n\n    Okay: a = (\\n    42)\n    E121: a = (\\n   42)\n    E122: a = (\\n42)\n    E123: a = (\\n    42\\n    )\n    E124: a = (24,\\n     42\\n)\n    E125: if (\\n    b):\\n    pass\n    E126: a = (\\n        42)\n    E127: a = (24,\\n      42)\n    E128: a = (24,\\n    42)\n    E129: if (a or\\n    b):\\n    pass\n    E131: a = (\\n    42\\n 24)\n    "
    first_row = tokens[0][2][0]
    nrows = 1 + tokens[-1][2][0] - first_row
    if noqa or nrows == 1:
        return
    indent_next = logical_line.endswith(':')
    row = depth = 0
    valid_hangs = (4,) if indent_char != '\t' else (4, 8)
    parens = [0] * nrows
    rel_indent = [0] * nrows
    open_rows = [[0]]
    hangs = [None]
    indent_chances = {}
    last_indent = tokens[0][2]
    visual_indent = None
    last_token_multiline = False
    indent = [last_indent[1]]
    if verbose >= 3:
        print('>>> ' + tokens[0][4].rstrip())
    for (token_type, text, start, end, line) in tokens:
        newline = row < start[0] - first_row
        if newline:
            row = start[0] - first_row
            newline = not last_token_multiline and token_type not in NEWLINE
        if newline:
            last_indent = start
            if verbose >= 3:
                print('... ' + line.rstrip())
            rel_indent[row] = expand_indent(line) - indent_level
            close_bracket = token_type == tokenize.OP and text in ']})'
            for open_row in reversed(open_rows[depth]):
                hang = rel_indent[row] - rel_indent[open_row]
                hanging_indent = hang in valid_hangs
                if hanging_indent:
                    break
            if hangs[depth]:
                hanging_indent = hang == hangs[depth]
            visual_indent = not close_bracket and hang > 0 and indent_chances.get(start[1])
            if close_bracket and indent[depth]:
                if start[1] != indent[depth]:
                    yield (start, 'E124 closing bracket does not match visual indentation')
            elif close_bracket and (not hang):
                if hang_closing:
                    yield (start, 'E133 closing bracket is missing indentation')
            elif indent[depth] and start[1] < indent[depth]:
                if visual_indent is not True:
                    yield (start, 'E128 continuation line under-indented for visual indent')
            elif hanging_indent or (indent_next and rel_indent[row] == 8):
                if close_bracket and (not hang_closing):
                    yield (start, "E123 closing bracket does not match indentation of opening bracket's line")
                hangs[depth] = hang
            elif visual_indent is True:
                indent[depth] = start[1]
            elif visual_indent in (text, str):
                pass
            else:
                if hang <= 0:
                    error = ('E122', 'missing indentation or outdented')
                elif indent[depth]:
                    error = ('E127', 'over-indented for visual indent')
                elif not close_bracket and hangs[depth]:
                    error = ('E131', 'unaligned for hanging indent')
                else:
                    hangs[depth] = hang
                    if hang > 4:
                        error = ('E126', 'over-indented for hanging indent')
                    else:
                        error = ('E121', 'under-indented for hanging indent')
                yield (start, '%s continuation line %s' % error)
        if parens[row] and token_type not in (tokenize.NL, tokenize.COMMENT) and (not indent[depth]):
            indent[depth] = start[1]
            indent_chances[start[1]] = True
            if verbose >= 4:
                print('bracket depth %s indent to %s' % (depth, start[1]))
        elif token_type in (tokenize.STRING, tokenize.COMMENT) or text in ('u', 'ur', 'b', 'br'):
            indent_chances[start[1]] = str
        elif not indent_chances and (not row) and (not depth) and (text == 'if'):
            indent_chances[end[1] + 1] = True
        elif text == ':' and line[end[1]:].isspace():
            open_rows[depth].append(row)
        if token_type == tokenize.OP:
            if text in '([{':
                depth += 1
                indent.append(0)
                hangs.append(None)
                if len(open_rows) == depth:
                    open_rows.append([])
                open_rows[depth].append(row)
                parens[row] += 1
                if verbose >= 4:
                    print('bracket depth %s seen, col %s, visual min = %s' % (depth, start[1], indent[depth]))
            elif text in ')]}' and depth > 0:
                prev_indent = indent.pop() or last_indent[1]
                hangs.pop()
                for d in range(depth):
                    if indent[d] > prev_indent:
                        indent[d] = 0
                for ind in list(indent_chances):
                    if ind >= prev_indent:
                        del indent_chances[ind]
                del open_rows[depth + 1:]
                depth -= 1
                if depth:
                    indent_chances[indent[depth]] = True
                for idx in range(row, -1, -1):
                    if parens[idx]:
                        parens[idx] -= 1
                        break
            assert len(indent) == depth + 1
            if start[1] not in indent_chances:
                indent_chances[start[1]] = text
        last_token_multiline = start[0] != end[0]
        if last_token_multiline:
            rel_indent[end[0] - first_row] = rel_indent[row]
    if indent_next and expand_indent(line) == indent_level + 4:
        pos = (start[0], indent[0] + 4)
        if visual_indent:
            code = 'E129 visually indented line'
        else:
            code = 'E125 continuation line'
        yield (pos, '%s with same indent as next logical line' % code)

def whitespace_before_parameters(logical_line, tokens):
    if False:
        i = 10
        return i + 15
    "Avoid extraneous whitespace.\n\n    Avoid extraneous whitespace in the following situations:\n    - before the open parenthesis that starts the argument list of a\n      function call.\n    - before the open parenthesis that starts an indexing or slicing.\n\n    Okay: spam(1)\n    E211: spam (1)\n\n    Okay: dict['key'] = list[index]\n    E211: dict ['key'] = list[index]\n    E211: dict['key'] = list [index]\n    "
    (prev_type, prev_text, __, prev_end, __) = tokens[0]
    for index in range(1, len(tokens)):
        (token_type, text, start, end, __) = tokens[index]
        if token_type == tokenize.OP and text in '([' and (start != prev_end) and (prev_type == tokenize.NAME or prev_text in '}])') and (index < 2 or tokens[index - 2][1] != 'class') and (not keyword.iskeyword(prev_text)):
            yield (prev_end, "E211 whitespace before '%s'" % text)
        prev_type = token_type
        prev_text = text
        prev_end = end

def whitespace_around_operator(logical_line):
    if False:
        while True:
            i = 10
    'Avoid extraneous whitespace around an operator.\n\n    Okay: a = 12 + 3\n    E221: a = 4  + 5\n    E222: a = 4 +  5\n    E223: a = 4\\t+ 5\n    E224: a = 4 +\\t5\n    '
    for match in OPERATOR_REGEX.finditer(logical_line):
        (before, after) = match.groups()
        if '\t' in before:
            yield (match.start(1), 'E223 tab before operator')
        elif len(before) > 1:
            yield (match.start(1), 'E221 multiple spaces before operator')
        if '\t' in after:
            yield (match.start(2), 'E224 tab after operator')
        elif len(after) > 1:
            yield (match.start(2), 'E222 multiple spaces after operator')

def missing_whitespace_around_operator(logical_line, tokens):
    if False:
        for i in range(10):
            print('nop')
    "Surround operators with a single space on either side.\n\n    - Always surround these binary operators with a single space on\n      either side: assignment (=), augmented assignment (+=, -= etc.),\n      comparisons (==, <, >, !=, <=, >=, in, not in, is, is not),\n      Booleans (and, or, not).\n\n    - If operators with different priorities are used, consider adding\n      whitespace around the operators with the lowest priorities.\n\n    Okay: i = i + 1\n    Okay: submitted += 1\n    Okay: x = x * 2 - 1\n    Okay: hypot2 = x * x + y * y\n    Okay: c = (a + b) * (a - b)\n    Okay: foo(bar, key='word', *args, **kwargs)\n    Okay: alpha[:-i]\n\n    E225: i=i+1\n    E225: submitted +=1\n    E225: x = x /2 - 1\n    E225: z = x **y\n    E226: c = (a+b) * (a-b)\n    E226: hypot2 = x*x + y*y\n    E227: c = a|b\n    E228: msg = fmt%(errno, errmsg)\n    "
    parens = 0
    need_space = False
    prev_type = tokenize.OP
    prev_text = prev_end = None
    for (token_type, text, start, end, line) in tokens:
        if token_type in SKIP_COMMENTS:
            continue
        if text in ('(', 'lambda'):
            parens += 1
        elif text == ')':
            parens -= 1
        if need_space:
            if start != prev_end:
                if need_space is not True and (not need_space[1]):
                    yield (need_space[0], 'E225 missing whitespace around operator')
                need_space = False
            elif text == '>' and prev_text in ('<', '-'):
                pass
            else:
                if need_space is True or need_space[1]:
                    yield (prev_end, 'E225 missing whitespace around operator')
                elif prev_text != '**':
                    (code, optype) = ('E226', 'arithmetic')
                    if prev_text == '%':
                        (code, optype) = ('E228', 'modulo')
                    elif prev_text not in ARITHMETIC_OP:
                        (code, optype) = ('E227', 'bitwise or shift')
                    yield (need_space[0], '%s missing whitespace around %s operator' % (code, optype))
                need_space = False
        elif token_type == tokenize.OP and prev_end is not None:
            if text == '=' and parens:
                pass
            elif text in WS_NEEDED_OPERATORS:
                need_space = True
            elif text in UNARY_OPERATORS:
                if prev_text in '}])' if prev_type == tokenize.OP else prev_text not in KEYWORDS:
                    need_space = None
            elif text in WS_OPTIONAL_OPERATORS:
                need_space = None
            if need_space is None:
                need_space = (prev_end, start != prev_end)
            elif need_space and start == prev_end:
                yield (prev_end, 'E225 missing whitespace around operator')
                need_space = False
        prev_type = token_type
        prev_text = text
        prev_end = end

def whitespace_around_comma(logical_line):
    if False:
        while True:
            i = 10
    'Avoid extraneous whitespace after a comma or a colon.\n\n    Note: these checks are disabled by default\n\n    Okay: a = (1, 2)\n    E241: a = (1,  2)\n    E242: a = (1,\\t2)\n    '
    line = logical_line
    for m in WHITESPACE_AFTER_COMMA_REGEX.finditer(line):
        found = m.start() + 1
        if '\t' in m.group():
            yield (found, "E242 tab after '%s'" % m.group()[0])
        else:
            yield (found, "E241 multiple spaces after '%s'" % m.group()[0])

def whitespace_around_named_parameter_equals(logical_line, tokens):
    if False:
        while True:
            i = 10
    "Don't use spaces around the '=' sign in function arguments.\n\n    Don't use spaces around the '=' sign when used to indicate a\n    keyword argument or a default parameter value.\n\n    Okay: def complex(real, imag=0.0):\n    Okay: return magic(r=real, i=imag)\n    Okay: boolean(a == b)\n    Okay: boolean(a != b)\n    Okay: boolean(a <= b)\n    Okay: boolean(a >= b)\n    Okay: def foo(arg: int = 42):\n\n    E251: def complex(real, imag = 0.0):\n    E251: return magic(r = real, i = imag)\n    "
    parens = 0
    no_space = False
    prev_end = None
    annotated_func_arg = False
    in_def = logical_line.startswith('def')
    message = 'E251 unexpected spaces around keyword / parameter equals'
    for (token_type, text, start, end, line) in tokens:
        if token_type == tokenize.NL:
            continue
        if no_space:
            no_space = False
            if start != prev_end:
                yield (prev_end, message)
        if token_type == tokenize.OP:
            if text == '(':
                parens += 1
            elif text == ')':
                parens -= 1
            elif in_def and text == ':' and (parens == 1):
                annotated_func_arg = True
            elif parens and text == ',' and (parens == 1):
                annotated_func_arg = False
            elif parens and text == '=' and (not annotated_func_arg):
                no_space = True
                if start != prev_end:
                    yield (prev_end, message)
            if not parens:
                annotated_func_arg = False
        prev_end = end

def whitespace_before_comment(logical_line, tokens):
    if False:
        i = 10
        return i + 15
    'Separate inline comments by at least two spaces.\n\n    An inline comment is a comment on the same line as a statement.  Inline\n    comments should be separated by at least two spaces from the statement.\n    They should start with a # and a single space.\n\n    Each line of a block comment starts with a # and a single space\n    (unless it is indented text inside the comment).\n\n    Okay: x = x + 1  # Increment x\n    Okay: x = x + 1    # Increment x\n    Okay: # Block comment\n    E261: x = x + 1 # Increment x\n    E262: x = x + 1  #Increment x\n    E262: x = x + 1  #  Increment x\n    E265: #Block comment\n    E266: ### Block comment\n    '
    prev_end = (0, 0)
    for (token_type, text, start, end, line) in tokens:
        if token_type == tokenize.COMMENT:
            inline_comment = line[:start[1]].strip()
            if inline_comment:
                if prev_end[0] == start[0] and start[1] < prev_end[1] + 2:
                    yield (prev_end, 'E261 at least two spaces before inline comment')
            (symbol, sp, comment) = text.partition(' ')
            bad_prefix = symbol not in '#:' and (symbol.lstrip('#')[:1] or '#')
            if inline_comment:
                if bad_prefix or comment[:1] in WHITESPACE:
                    yield (start, "E262 inline comment should start with '# '")
            elif bad_prefix and (bad_prefix != '!' or start[0] > 1):
                if bad_prefix != '#':
                    yield (start, "E265 block comment should start with '# '")
                elif comment:
                    yield (start, "E266 too many leading '#' for block comment")
        elif token_type != tokenize.NL:
            prev_end = end

def imports_on_separate_lines(logical_line):
    if False:
        print('Hello World!')
    'Imports should usually be on separate lines.\n\n    Okay: import os\\nimport sys\n    E401: import sys, os\n\n    Okay: from subprocess import Popen, PIPE\n    Okay: from myclas import MyClass\n    Okay: from foo.bar.yourclass import YourClass\n    Okay: import myclass\n    Okay: import foo.bar.yourclass\n    '
    line = logical_line
    if line.startswith('import '):
        found = line.find(',')
        if -1 < found and ';' not in line[:found]:
            yield (found, 'E401 multiple imports on one line')

def module_imports_on_top_of_file(logical_line, indent_level, checker_state, noqa):
    if False:
        for i in range(10):
            print('nop')
    'Imports are always put at the top of the file, just after any module\n    comments and docstrings, and before module globals and constants.\n\n    Okay: import os\n    Okay: # this is a comment\\nimport os\n    Okay: \'\'\'this is a module docstring\'\'\'\\nimport os\n    Okay: r\'\'\'this is a module docstring\'\'\'\\nimport os\n    Okay: try:\\n    import x\\nexcept:\\n    pass\\nelse:\\n    pass\\nimport y\n    Okay: try:\\n    import x\\nexcept:\\n    pass\\nfinally:\\n    pass\\nimport y\n    E402: a=1\\nimport os\n    E402: \'One string\'\\n"Two string"\\nimport os\n    E402: a=1\\nfrom sys import x\n\n    Okay: if x:\\n    import os\n    '

    def is_string_literal(line):
        if False:
            for i in range(10):
                print('nop')
        if line[0] in 'uUbB':
            line = line[1:]
        if line and line[0] in 'rR':
            line = line[1:]
        return line and (line[0] == '"' or line[0] == "'")
    allowed_try_keywords = ('try', 'except', 'else', 'finally')
    if indent_level:
        return
    if not logical_line:
        return
    if noqa:
        return
    line = logical_line
    if line.startswith('import ') or line.startswith('from '):
        if checker_state.get('seen_non_imports', False):
            yield (0, 'E402 module level import not at top of file')
    elif any((line.startswith(kw) for kw in allowed_try_keywords)):
        return
    elif is_string_literal(line):
        if checker_state.get('seen_docstring', False):
            checker_state['seen_non_imports'] = True
        else:
            checker_state['seen_docstring'] = True
    else:
        checker_state['seen_non_imports'] = True

def compound_statements(logical_line):
    if False:
        for i in range(10):
            print('nop')
    "Compound statements (on the same line) are generally\n    discouraged.\n\n    While sometimes it's okay to put an if/for/while with a small body\n    on the same line, never do this for multi-clause statements.\n    Also avoid folding such long lines!\n\n    Always use a def statement instead of an assignment statement that\n    binds a lambda expression directly to a name.\n\n    Okay: if foo == 'blah':\\n    do_blah_thing()\n    Okay: do_one()\n    Okay: do_two()\n    Okay: do_three()\n\n    E701: if foo == 'blah': do_blah_thing()\n    E701: for x in lst: total += x\n    E701: while t < 10: t = delay()\n    E701: if foo == 'blah': do_blah_thing()\n    E701: else: do_non_blah_thing()\n    E701: try: something()\n    E701: finally: cleanup()\n    E701: if foo == 'blah': one(); two(); three()\n    E702: do_one(); do_two(); do_three()\n    E703: do_four();  # useless semicolon\n    E704: def f(x): return 2*x\n    E731: f = lambda x: 2*x\n    "
    line = logical_line
    last_char = len(line) - 1
    found = line.find(':')
    prev_found = 0
    counts = {char: 0 for char in '{}[]()'}
    while -1 < found < last_char:
        update_counts(line[prev_found:found], counts)
        if (counts['{'] <= counts['}'] and counts['['] <= counts[']'] and (counts['('] <= counts[')'])) and (not (sys.version_info >= (3, 8) and line[found + 1] == '=')):
            lambda_kw = LAMBDA_REGEX.search(line, 0, found)
            if lambda_kw:
                before = line[:lambda_kw.start()].rstrip()
                if before[-1:] == '=' and before[:-1].strip().isidentifier():
                    yield (0, 'E731 do not assign a lambda expression, use a def')
                break
            if STARTSWITH_DEF_REGEX.match(line):
                yield (0, 'E704 multiple statements on one line (def)')
            elif STARTSWITH_INDENT_STATEMENT_REGEX.match(line):
                yield (found, 'E701 multiple statements on one line (colon)')
        prev_found = found
        found = line.find(':', found + 1)
    found = line.find(';')
    while -1 < found:
        if found < last_char:
            yield (found, 'E702 multiple statements on one line (semicolon)')
        else:
            yield (found, 'E703 statement ends with a semicolon')
        found = line.find(';', found + 1)

def explicit_line_join(logical_line, tokens):
    if False:
        print('Hello World!')
    'Avoid explicit line join between brackets.\n\n    The preferred way of wrapping long lines is by using Python\'s implied line\n    continuation inside parentheses, brackets and braces.  Long lines can be\n    broken over multiple lines by wrapping expressions in parentheses.  These\n    should be used in preference to using a backslash for line continuation.\n\n    E502: aaa = [123, \\\\n       123]\n    E502: aaa = ("bbb " \\\\n       "ccc")\n\n    Okay: aaa = [123,\\n       123]\n    Okay: aaa = ("bbb "\\n       "ccc")\n    Okay: aaa = "bbb " \\\\n    "ccc"\n    Okay: aaa = 123  # \\\\\n    '
    prev_start = prev_end = parens = 0
    comment = False
    backslash = None
    for (token_type, text, start, end, line) in tokens:
        if token_type == tokenize.COMMENT:
            comment = True
        if start[0] != prev_start and parens and backslash and (not comment):
            yield (backslash, 'E502 the backslash is redundant between brackets')
        if end[0] != prev_end:
            if line.rstrip('\r\n').endswith('\\'):
                backslash = (end[0], len(line.splitlines()[-1]) - 1)
            else:
                backslash = None
            prev_start = prev_end = end[0]
        else:
            prev_start = start[0]
        if token_type == tokenize.OP:
            if text in '([{':
                parens += 1
            elif text in ')]}':
                parens -= 1

def break_around_binary_operator(logical_line, tokens):
    if False:
        for i in range(10):
            print('nop')
    "\n    Avoid breaks before binary operators.\n\n    The preferred place to break around a binary operator is after the\n    operator, not before it.\n\n    W503: (width == 0\\n + height == 0)\n    W503: (width == 0\\n and height == 0)\n\n    Okay: (width == 0 +\\n height == 0)\n    Okay: foo(\\n    -x)\n    Okay: foo(x\\n    [])\n    Okay: x = '''\\n''' + ''\n    Okay: foo(x,\\n    -y)\n    Okay: foo(x,  # comment\\n    -y)\n    "

    def is_binary_operator(token_type, text):
        if False:
            for i in range(10):
                print('nop')
        return (token_type == tokenize.OP or text in ['and', 'or']) and text not in '()[]{},:.;@=%'
    line_break = False
    unary_context = True
    for (token_type, text, start, end, line) in tokens:
        if token_type == tokenize.COMMENT:
            continue
        if ('\n' in text or '\r' in text) and token_type != tokenize.STRING:
            line_break = True
        else:
            if is_binary_operator(token_type, text) and line_break and (not unary_context):
                yield (start, 'W503 line break before binary operator')
            unary_context = text in '([{,;'
            line_break = False

def comparison_to_singleton(logical_line, noqa):
    if False:
        return 10
    'Comparison to singletons should use "is" or "is not".\n\n    Comparisons to singletons like None should always be done\n    with "is" or "is not", never the equality operators.\n\n    Okay: if arg is not None:\n    E711: if arg != None:\n    E711: if None == arg:\n    E712: if arg == True:\n    E712: if False == arg:\n\n    Also, beware of writing if x when you really mean if x is not None --\n    e.g. when testing whether a variable or argument that defaults to None was\n    set to some other value.  The other value might have a type (such as a\n    container) that could be false in a boolean context!\n    '
    match = not noqa and COMPARE_SINGLETON_REGEX.search(logical_line)
    if match:
        singleton = match.group(1) or match.group(3)
        same = match.group(2) == '=='
        msg = "'if cond is %s:'" % (('' if same else 'not ') + singleton)
        if singleton in ('None',):
            code = 'E711'
        else:
            code = 'E712'
            nonzero = singleton == 'True' and same or (singleton == 'False' and (not same))
            msg += " or 'if %scond:'" % ('' if nonzero else 'not ')
        yield (match.start(2), '%s comparison to %s should be %s' % (code, singleton, msg))

def comparison_negative(logical_line):
    if False:
        i = 10
        return i + 15
    'Negative comparison should be done using "not in" and "is not".\n\n    Okay: if x not in y:\\n    pass\n    Okay: assert (X in Y or X is Z)\n    Okay: if not (X in Y):\\n    pass\n    Okay: zz = x is not y\n    E713: Z = not X in Y\n    E713: if not X.B in Y:\\n    pass\n    E714: if not X is Y:\\n    pass\n    E714: Z = not X.B is Y\n    '
    match = COMPARE_NEGATIVE_REGEX.search(logical_line)
    if match:
        pos = match.start(1)
        if match.group(2) == 'in':
            yield (pos, "E713 test for membership should be 'not in'")
        else:
            yield (pos, "E714 test for object identity should be 'is not'")

def comparison_type(logical_line, noqa):
    if False:
        print('Hello World!')
    'Object type comparisons should always use isinstance().\n\n    Do not compare types directly.\n\n    Okay: if isinstance(obj, int):\n    E721: if type(obj) is type(1):\n\n    When checking if an object is a string, keep in mind that it might be a\n    unicode string too! In Python 2.3, str and unicode have a common base\n    class, basestring, so you can do:\n\n    Okay: if isinstance(obj, basestring):\n    Okay: if type(a1) is type(b1):\n    '
    match = COMPARE_TYPE_REGEX.search(logical_line)
    if match and (not noqa):
        inst = match.group(1)
        if inst and isidentifier(inst) and (inst not in SINGLETONS):
            return
        yield (match.start(), "E721 do not compare types, use 'isinstance()'")

def python_3000_has_key(logical_line, noqa):
    if False:
        print('Hello World!')
    'The {}.has_key() method is removed in Python 3: use the \'in\' operator.\n\n    Okay: if "alph" in d:\\n    print d["alph"]\n    W601: assert d.has_key(\'alph\')\n    '
    pos = logical_line.find('.has_key(')
    if pos > -1 and (not noqa):
        yield (pos, "W601 .has_key() is deprecated, use 'in'")

def python_3000_raise_comma(logical_line):
    if False:
        return 10
    'When raising an exception, use "raise ValueError(\'message\')".\n\n    The older form is removed in Python 3.\n\n    Okay: raise DummyError("Message")\n    W602: raise DummyError, "Message"\n    '
    match = RAISE_COMMA_REGEX.match(logical_line)
    if match and (not RERAISE_COMMA_REGEX.match(logical_line)):
        yield (match.end() - 1, 'W602 deprecated form of raising exception')

def python_3000_not_equal(logical_line):
    if False:
        print('Hello World!')
    "New code should always use != instead of <>.\n\n    The older syntax is removed in Python 3.\n\n    Okay: if a != 'no':\n    W603: if a <> 'no':\n    "
    pos = logical_line.find('<>')
    if pos > -1:
        yield (pos, "W603 '<>' is deprecated, use '!='")

def python_3000_backticks(logical_line):
    if False:
        return 10
    'Backticks are removed in Python 3: use repr() instead.\n\n    Okay: val = repr(1 + 2)\n    W604: val = `1 + 2`\n    '
    pos = logical_line.find('`')
    if pos > -1:
        yield (pos, "W604 backticks are deprecated, use 'repr()'")
if sys.version_info < (3,):

    def readlines(filename):
        if False:
            i = 10
            return i + 15
        'Read the source code.'
        with open(filename, 'rU') as f:
            return f.readlines()
    isidentifier = re.compile('[a-zA-Z_]\\w*$').match
    stdin_get_value = sys.stdin.read
else:

    def readlines(filename):
        if False:
            print('Hello World!')
        'Read the source code.'
        try:
            with open(filename, 'rb') as f:
                (coding, lines) = tokenize.detect_encoding(f.readline)
                f = TextIOWrapper(f, coding, line_buffering=True)
                return [l.decode(coding) for l in lines] + f.readlines()
        except (LookupError, SyntaxError, UnicodeError):
            with open(filename, encoding='latin-1') as f:
                return f.readlines()
    isidentifier = str.isidentifier

    def stdin_get_value():
        if False:
            print('Hello World!')
        return TextIOWrapper(sys.stdin.buffer, errors='ignore').read()
noqa = re.compile('# no(?:qa|pep8)\\b', re.I).search

def expand_indent(line):
    if False:
        return 10
    "Return the amount of indentation.\n\n    Tabs are expanded to the next multiple of 8.\n\n    >>> expand_indent('    ')\n    4\n    >>> expand_indent('\\t')\n    8\n    >>> expand_indent('       \\t')\n    8\n    >>> expand_indent('        \\t')\n    16\n    "
    if '\t' not in line:
        return len(line) - len(line.lstrip())
    result = 0
    for char in line:
        if char == '\t':
            result = result // 8 * 8 + 8
        elif char == ' ':
            result += 1
        else:
            break
    return result

def mute_string(text):
    if False:
        print('Hello World!')
    'Replace contents with \'xxx\' to prevent syntax matching.\n\n    >>> mute_string(\'"abc"\')\n    \'"xxx"\'\n    >>> mute_string("\'\'\'abc\'\'\'")\n    "\'\'\'xxx\'\'\'"\n    >>> mute_string("r\'abc\'")\n    "r\'xxx\'"\n    '
    start = text.index(text[-1]) + 1
    end = len(text) - 1
    if text[-3:] in ('"""', "'''"):
        start += 2
        end -= 2
    return text[:start] + 'x' * (end - start) + text[end:]

def parse_udiff(diff, patterns=None, parent='.'):
    if False:
        print('Hello World!')
    'Return a dictionary of matching lines.'
    rv = {}
    path = nrows = None
    for line in diff.splitlines():
        if nrows:
            if line[:1] != '-':
                nrows -= 1
            continue
        if line[:3] == '@@ ':
            hunk_match = HUNK_REGEX.match(line)
            (row, nrows) = [int(g or '1') for g in hunk_match.groups()]
            rv[path].update(range(row, row + nrows))
        elif line[:3] == '+++':
            path = line[4:].split('\t', 1)[0]
            if path[:2] == 'b/':
                path = path[2:]
            rv[path] = set()
    return dict([(os.path.join(parent, path), rows) for (path, rows) in rv.items() if rows and filename_match(path, patterns)])

def normalize_paths(value, parent=os.curdir):
    if False:
        for i in range(10):
            print('nop')
    'Parse a comma-separated list of paths.\n\n    Return a list of absolute paths.\n    '
    if not value:
        return []
    if isinstance(value, list):
        return value
    paths = []
    for path in value.split(','):
        path = path.strip()
        if '/' in path:
            path = os.path.abspath(os.path.join(parent, path))
        paths.append(path.rstrip('/'))
    return paths

def filename_match(filename, patterns, default=True):
    if False:
        return 10
    'Check if patterns contains a pattern that matches filename.\n\n    If patterns is unspecified, this always returns True.\n    '
    if not patterns:
        return default
    return any((fnmatch(filename, pattern) for pattern in patterns))

def update_counts(s, counts):
    if False:
        return 10
    'Adds one to the counts of each appearance of characters in s,\n        for characters in counts'
    for char in s:
        if char in counts:
            counts[char] += 1

def _is_eol_token(token):
    if False:
        i = 10
        return i + 15
    return token[0] in NEWLINE or token[4][token[3][1]:].lstrip() == '\\\n'
if COMMENT_WITH_NL:

    def _is_eol_token(token, _eol_token=_is_eol_token):
        if False:
            for i in range(10):
                print('nop')
        return _eol_token(token) or (token[0] == tokenize.COMMENT and token[1] == token[4])
_checks = {'physical_line': {}, 'logical_line': {}, 'tree': {}}

def _get_parameters(function):
    if False:
        print('Hello World!')
    if sys.version_info >= (3, 3):
        return [parameter.name for parameter in inspect.signature(function).parameters.values() if parameter.kind == parameter.POSITIONAL_OR_KEYWORD]
    else:
        return inspect.getargspec(function)[0]

def register_check(check, codes=None):
    if False:
        for i in range(10):
            print('nop')
    'Register a new check object.'

    def _add_check(check, kind, codes, args):
        if False:
            return 10
        if check in _checks[kind]:
            _checks[kind][check][0].extend(codes or [])
        else:
            _checks[kind][check] = (codes or [''], args)
    if inspect.isfunction(check):
        args = _get_parameters(check)
        if args and args[0] in ('physical_line', 'logical_line'):
            if codes is None:
                codes = ERRORCODE_REGEX.findall(check.__doc__ or '')
            _add_check(check, args[0], codes, args)
    elif inspect.isclass(check):
        if _get_parameters(check.__init__)[:2] == ['self', 'tree']:
            _add_check(check, 'tree', codes, None)

def init_checks_registry():
    if False:
        i = 10
        return i + 15
    "Register all globally visible functions.\n\n    The first argument name is either 'physical_line' or 'logical_line'.\n    "
    mod = inspect.getmodule(register_check)
    for (name, function) in inspect.getmembers(mod, inspect.isfunction):
        register_check(function)
init_checks_registry()

class Checker(object):
    """Load a Python source file, tokenize it, check coding style."""

    def __init__(self, filename=None, lines=None, options=None, report=None, **kwargs):
        if False:
            return 10
        if options is None:
            options = StyleGuide(kwargs).options
        else:
            assert not kwargs
        self._io_error = None
        self._physical_checks = options.physical_checks
        self._logical_checks = options.logical_checks
        self._ast_checks = options.ast_checks
        self.max_line_length = options.max_line_length
        self.multiline = False
        self.hang_closing = options.hang_closing
        self.verbose = options.verbose
        self.filename = filename
        self._checker_states = {}
        if filename is None:
            self.filename = 'stdin'
            self.lines = lines or []
        elif filename == '-':
            self.filename = 'stdin'
            self.lines = stdin_get_value().splitlines(True)
        elif lines is None:
            try:
                self.lines = readlines(filename)
            except IOError:
                (exc_type, exc) = sys.exc_info()[:2]
                self._io_error = '%s: %s' % (exc_type.__name__, exc)
                self.lines = []
        else:
            self.lines = lines
        if self.lines:
            ord0 = ord(self.lines[0][0])
            if ord0 in (239, 65279):
                if ord0 == 65279:
                    self.lines[0] = self.lines[0][1:]
                elif self.lines[0][:3] == 'ï»¿':
                    self.lines[0] = self.lines[0][3:]
        self.report = report or options.report
        self.report_error = self.report.error

    def report_invalid_syntax(self):
        if False:
            return 10
        'Check if the syntax is valid.'
        (exc_type, exc) = sys.exc_info()[:2]
        if len(exc.args) > 1:
            offset = exc.args[1]
            if len(offset) > 2:
                offset = offset[1:3]
        else:
            offset = (1, 0)
        self.report_error(offset[0], offset[1] or 0, 'E901 %s: %s' % (exc_type.__name__, exc.args[0]), self.report_invalid_syntax)

    def readline(self):
        if False:
            print('Hello World!')
        'Get the next line from the input buffer.'
        if self.line_number >= self.total_lines:
            return ''
        line = self.lines[self.line_number]
        self.line_number += 1
        if self.indent_char is None and line[:1] in WHITESPACE:
            self.indent_char = line[0]
        return line

    def run_check(self, check, argument_names):
        if False:
            return 10
        'Run a check plugin.'
        arguments = []
        for name in argument_names:
            arguments.append(getattr(self, name))
        return check(*arguments)

    def init_checker_state(self, name, argument_names):
        if False:
            i = 10
            return i + 15
        ' Prepares a custom state for the specific checker plugin.'
        if 'checker_state' in argument_names:
            self.checker_state = self._checker_states.setdefault(name, {})

    def check_physical(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Run all physical checks on a raw input line.'
        self.physical_line = line
        for (name, check, argument_names) in self._physical_checks:
            self.init_checker_state(name, argument_names)
            result = self.run_check(check, argument_names)
            if result is not None:
                (offset, text) = result
                self.report_error(self.line_number, offset, text, check)
                if text[:4] == 'E101':
                    self.indent_char = line[0]

    def build_tokens_line(self):
        if False:
            return 10
        'Build a logical line from tokens.'
        logical = []
        comments = []
        length = 0
        prev_row = prev_col = mapping = None
        for (token_type, text, start, end, line) in self.tokens:
            if token_type in SKIP_TOKENS:
                continue
            if not mapping:
                mapping = [(0, start)]
            if token_type == tokenize.COMMENT:
                comments.append(text)
                continue
            if token_type == tokenize.STRING:
                text = mute_string(text)
            if prev_row:
                (start_row, start_col) = start
                if prev_row != start_row:
                    prev_text = self.lines[prev_row - 1][prev_col - 1]
                    if prev_text == ',' or (prev_text not in '{[(' and text not in '}])'):
                        text = ' ' + text
                elif prev_col != start_col:
                    text = line[prev_col:start_col] + text
            logical.append(text)
            length += len(text)
            mapping.append((length, end))
            (prev_row, prev_col) = end
        self.logical_line = ''.join(logical)
        self.noqa = comments and noqa(''.join(comments))
        return mapping

    def check_logical(self):
        if False:
            return 10
        'Build a line from tokens and run all logical checks on it.'
        self.report.increment_logical_line()
        mapping = self.build_tokens_line()
        if not mapping:
            return
        (start_row, start_col) = mapping[0][1]
        start_line = self.lines[start_row - 1]
        self.indent_level = expand_indent(start_line[:start_col])
        if self.blank_before < self.blank_lines:
            self.blank_before = self.blank_lines
        if self.verbose >= 2:
            print(self.logical_line[:80].rstrip())
        for (name, check, argument_names) in self._logical_checks:
            if self.verbose >= 4:
                print('   ' + name)
            self.init_checker_state(name, argument_names)
            for (offset, text) in self.run_check(check, argument_names) or ():
                if not isinstance(offset, tuple):
                    for (token_offset, pos) in mapping:
                        if offset <= token_offset:
                            break
                    offset = (pos[0], pos[1] + offset - token_offset)
                self.report_error(offset[0], offset[1], text, check)
        if self.logical_line:
            self.previous_indent_level = self.indent_level
            self.previous_logical = self.logical_line
        self.blank_lines = 0
        self.tokens = []

    def check_ast(self):
        if False:
            while True:
                i = 10
        "Build the file's AST and run all AST checks."
        try:
            tree = compile(''.join(self.lines), '', 'exec', PyCF_ONLY_AST)
        except (ValueError, SyntaxError, TypeError):
            return self.report_invalid_syntax()
        for (name, cls, __) in self._ast_checks:
            checker = cls(tree, self.filename)
            for (lineno, offset, text, check) in checker.run():
                if not self.lines or not noqa(self.lines[lineno - 1]):
                    self.report_error(lineno, offset, text, check)

    def generate_tokens(self):
        if False:
            print('Hello World!')
        'Tokenize the file, run physical line checks and yield tokens.'
        if self._io_error:
            self.report_error(1, 0, 'E902 %s' % self._io_error, readlines)
        tokengen = tokenize.generate_tokens(self.readline)
        try:
            for token in tokengen:
                if token[2][0] > self.total_lines:
                    return
                self.maybe_check_physical(token)
                yield token
        except (SyntaxError, tokenize.TokenError):
            self.report_invalid_syntax()

    def maybe_check_physical(self, token):
        if False:
            while True:
                i = 10
        'If appropriate (based on token), check current physical line(s).'
        if _is_eol_token(token):
            self.check_physical(token[4])
        elif token[0] == tokenize.STRING and '\n' in token[1]:
            if noqa(token[4]):
                return
            self.multiline = True
            self.line_number = token[2][0]
            for line in token[1].split('\n')[:-1]:
                self.check_physical(line + '\n')
                self.line_number += 1
            self.multiline = False

    def check_all(self, expected=None, line_offset=0):
        if False:
            print('Hello World!')
        'Run all checks on the input file.'
        self.report.init_file(self.filename, self.lines, expected, line_offset)
        self.total_lines = len(self.lines)
        if self._ast_checks:
            self.check_ast()
        self.line_number = 0
        self.indent_char = None
        self.indent_level = self.previous_indent_level = 0
        self.previous_logical = ''
        self.tokens = []
        self.blank_lines = self.blank_before = 0
        parens = 0
        for token in self.generate_tokens():
            self.tokens.append(token)
            (token_type, text) = token[0:2]
            if self.verbose >= 3:
                if token[2][0] == token[3][0]:
                    pos = '[%s:%s]' % (token[2][1] or '', token[3][1])
                else:
                    pos = 'l.%s' % token[3][0]
                print('l.%s\t%s\t%s\t%r' % (token[2][0], pos, tokenize.tok_name[token[0]], text))
            if token_type == tokenize.OP:
                if text in '([{':
                    parens += 1
                elif text in '}])':
                    parens -= 1
            elif not parens:
                if token_type in NEWLINE:
                    if token_type == tokenize.NEWLINE:
                        self.check_logical()
                        self.blank_before = 0
                    elif len(self.tokens) == 1:
                        self.blank_lines += 1
                        del self.tokens[0]
                    else:
                        self.check_logical()
                elif COMMENT_WITH_NL and token_type == tokenize.COMMENT:
                    if len(self.tokens) == 1:
                        token = list(token)
                        token[1] = text.rstrip('\r\n')
                        token[3] = (token[2][0], token[2][1] + len(token[1]))
                        self.tokens = [tuple(token)]
                        self.check_logical()
        if self.tokens:
            self.check_physical(self.lines[-1])
            self.check_logical()
        return self.report.get_file_results()

class BaseReport(object):
    """Collect the results of the checks."""
    print_filename = False

    def __init__(self, options):
        if False:
            for i in range(10):
                print('nop')
        self._benchmark_keys = options.benchmark_keys
        self._ignore_code = options.ignore_code
        self.elapsed = 0
        self.total_errors = 0
        self.counters = dict.fromkeys(self._benchmark_keys, 0)
        self.messages = {}

    def start(self):
        if False:
            i = 10
            return i + 15
        'Start the timer.'
        self._start_time = time.time()

    def stop(self):
        if False:
            return 10
        'Stop the timer.'
        self.elapsed = time.time() - self._start_time

    def init_file(self, filename, lines, expected, line_offset):
        if False:
            while True:
                i = 10
        'Signal a new file.'
        self.filename = filename
        self.lines = lines
        self.expected = expected or ()
        self.line_offset = line_offset
        self.file_errors = 0
        self.counters['files'] += 1
        self.counters['physical lines'] += len(lines)

    def increment_logical_line(self):
        if False:
            i = 10
            return i + 15
        'Signal a new logical line.'
        self.counters['logical lines'] += 1

    def error(self, line_number, offset, text, check):
        if False:
            return 10
        'Report an error, according to options.'
        code = text[:4]
        if self._ignore_code(code):
            return
        if code in self.counters:
            self.counters[code] += 1
        else:
            self.counters[code] = 1
            self.messages[code] = text[5:]
        if code in self.expected:
            return
        if self.print_filename and (not self.file_errors):
            print(self.filename)
        self.file_errors += 1
        self.total_errors += 1
        return code

    def get_file_results(self):
        if False:
            return 10
        'Return the count of errors and warnings for this file.'
        return self.file_errors

    def get_count(self, prefix=''):
        if False:
            print('Hello World!')
        'Return the total count of errors and warnings.'
        return sum([self.counters[key] for key in self.messages if key.startswith(prefix)])

    def get_statistics(self, prefix=''):
        if False:
            for i in range(10):
                print('nop')
        "Get statistics for message codes that start with the prefix.\n\n        prefix='' matches all errors and warnings\n        prefix='E' matches all errors\n        prefix='W' matches all warnings\n        prefix='E4' matches all errors that have to do with imports\n        "
        return ['%-7s %s %s' % (self.counters[key], key, self.messages[key]) for key in sorted(self.messages) if key.startswith(prefix)]

    def print_statistics(self, prefix=''):
        if False:
            i = 10
            return i + 15
        'Print overall statistics (number of errors and warnings).'
        for line in self.get_statistics(prefix):
            print(line)

    def print_benchmark(self):
        if False:
            while True:
                i = 10
        'Print benchmark numbers.'
        print('%-7.2f %s' % (self.elapsed, 'seconds elapsed'))
        if self.elapsed:
            for key in self._benchmark_keys:
                print('%-7d %s per second (%d total)' % (self.counters[key] / self.elapsed, key, self.counters[key]))

class FileReport(BaseReport):
    """Collect the results of the checks and print only the filenames."""
    print_filename = True

class StandardReport(BaseReport):
    """Collect and print the results of the checks."""

    def __init__(self, options):
        if False:
            i = 10
            return i + 15
        super(StandardReport, self).__init__(options)
        self._fmt = REPORT_FORMAT.get(options.format.lower(), options.format)
        self._repeat = options.repeat
        self._show_source = options.show_source
        self._show_pep8 = options.show_pep8

    def init_file(self, filename, lines, expected, line_offset):
        if False:
            return 10
        'Signal a new file.'
        self._deferred_print = []
        return super(StandardReport, self).init_file(filename, lines, expected, line_offset)

    def error(self, line_number, offset, text, check):
        if False:
            while True:
                i = 10
        'Report an error, according to options.'
        code = super(StandardReport, self).error(line_number, offset, text, check)
        if code and (self.counters[code] == 1 or self._repeat):
            self._deferred_print.append((line_number, offset, code, text[5:], check.__doc__))
        return code

    def get_file_results(self):
        if False:
            while True:
                i = 10
        'Print the result and return the overall count for this file.'
        self._deferred_print.sort()
        for (line_number, offset, code, text, doc) in self._deferred_print:
            print(self._fmt % {'path': self.filename, 'row': self.line_offset + line_number, 'col': offset + 1, 'code': code, 'text': text})
            if self._show_source:
                if line_number > len(self.lines):
                    line = ''
                else:
                    line = self.lines[line_number - 1]
                print(line.rstrip())
                print(re.sub('\\S', ' ', line[:offset]) + '^')
            if self._show_pep8 and doc:
                print('    ' + doc.strip())
            sys.stdout.flush()
        return self.file_errors

class DiffReport(StandardReport):
    """Collect and print the results for the changed lines only."""

    def __init__(self, options):
        if False:
            while True:
                i = 10
        super(DiffReport, self).__init__(options)
        self._selected = options.selected_lines

    def error(self, line_number, offset, text, check):
        if False:
            for i in range(10):
                print('nop')
        if line_number not in self._selected[self.filename]:
            return
        return super(DiffReport, self).error(line_number, offset, text, check)

class StyleGuide(object):
    """Initialize a PEP-8 instance with few options."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.checker_class = kwargs.pop('checker_class', Checker)
        parse_argv = kwargs.pop('parse_argv', False)
        config_file = kwargs.pop('config_file', False)
        parser = kwargs.pop('parser', None)
        options_dict = dict(*args, **kwargs)
        arglist = None if parse_argv else options_dict.get('paths', None)
        (options, self.paths) = process_options(arglist, parse_argv, config_file, parser)
        if options_dict:
            options.__dict__.update(options_dict)
            if 'paths' in options_dict:
                self.paths = options_dict['paths']
        self.runner = self.input_file
        self.options = options
        if not options.reporter:
            options.reporter = BaseReport if options.quiet else StandardReport
        options.select = tuple(options.select or ())
        if not (options.select or options.ignore or options.testsuite or options.doctest) and DEFAULT_IGNORE:
            options.ignore = tuple(DEFAULT_IGNORE.split(','))
        else:
            options.ignore = ('',) if options.select else tuple(options.ignore)
        options.benchmark_keys = BENCHMARK_KEYS[:]
        options.ignore_code = self.ignore_code
        options.physical_checks = self.get_checks('physical_line')
        options.logical_checks = self.get_checks('logical_line')
        options.ast_checks = self.get_checks('tree')
        self.init_report()

    def init_report(self, reporter=None):
        if False:
            print('Hello World!')
        'Initialize the report instance.'
        self.options.report = (reporter or self.options.reporter)(self.options)
        return self.options.report

    def check_files(self, paths=None):
        if False:
            print('Hello World!')
        'Run all checks on the paths.'
        if paths is None:
            paths = self.paths
        report = self.options.report
        runner = self.runner
        report.start()
        try:
            for path in paths:
                if os.path.isdir(path):
                    self.input_dir(path)
                elif not self.excluded(path):
                    runner(path)
        except KeyboardInterrupt:
            print('... stopped')
        report.stop()
        return report

    def input_file(self, filename, lines=None, expected=None, line_offset=0):
        if False:
            for i in range(10):
                print('nop')
        'Run all checks on a Python source file.'
        if self.options.verbose:
            print('checking %s' % filename)
        fchecker = self.checker_class(filename, lines=lines, options=self.options)
        return fchecker.check_all(expected=expected, line_offset=line_offset)

    def input_dir(self, dirname):
        if False:
            i = 10
            return i + 15
        'Check all files in this directory and all subdirectories.'
        dirname = dirname.rstrip('/')
        if self.excluded(dirname):
            return 0
        counters = self.options.report.counters
        verbose = self.options.verbose
        filepatterns = self.options.filename
        runner = self.runner
        for (root, dirs, files) in os.walk(dirname):
            if verbose:
                print('directory ' + root)
            counters['directories'] += 1
            for subdir in sorted(dirs):
                if self.excluded(subdir, root):
                    dirs.remove(subdir)
            for filename in sorted(files):
                if filename_match(filename, filepatterns) and (not self.excluded(filename, root)):
                    runner(os.path.join(root, filename))

    def excluded(self, filename, parent=None):
        if False:
            while True:
                i = 10
        "Check if the file should be excluded.\n\n        Check if 'options.exclude' contains a pattern that matches filename.\n        "
        if not self.options.exclude:
            return False
        basename = os.path.basename(filename)
        if filename_match(basename, self.options.exclude):
            return True
        if parent:
            filename = os.path.join(parent, filename)
        filename = os.path.abspath(filename)
        return filename_match(filename, self.options.exclude)

    def ignore_code(self, code):
        if False:
            return 10
        "Check if the error code should be ignored.\n\n        If 'options.select' contains a prefix of the error code,\n        return False.  Else, if 'options.ignore' contains a prefix of\n        the error code, return True.\n        "
        if len(code) < 4 and any((s.startswith(code) for s in self.options.select)):
            return False
        return code.startswith(self.options.ignore) and (not code.startswith(self.options.select))

    def get_checks(self, argument_name):
        if False:
            print('Hello World!')
        'Get all the checks for this category.\n\n        Find all globally visible functions where the first argument name\n        starts with argument_name and which contain selected tests.\n        '
        checks = []
        for (check, attrs) in _checks[argument_name].items():
            (codes, args) = attrs
            if any((not (code and self.ignore_code(code)) for code in codes)):
                checks.append((check.__name__, check, args))
        return sorted(checks)

def get_parser(prog='pep8', version=__version__):
    if False:
        while True:
            i = 10
    parser = OptionParser(prog=prog, version=version, usage='%prog [options] input ...')
    parser.config_options = ['exclude', 'filename', 'select', 'ignore', 'max-line-length', 'hang-closing', 'count', 'format', 'quiet', 'show-pep8', 'show-source', 'statistics', 'verbose']
    parser.add_option('-v', '--verbose', default=0, action='count', help='print status messages, or debug with -vv')
    parser.add_option('-q', '--quiet', default=0, action='count', help='report only file names, or nothing with -qq')
    parser.add_option('-r', '--repeat', default=True, action='store_true', help='(obsolete) show all occurrences of the same error')
    parser.add_option('--first', action='store_false', dest='repeat', help='show first occurrence of each error')
    parser.add_option('--exclude', metavar='patterns', default=DEFAULT_EXCLUDE, help='exclude files or directories which match these comma separated patterns (default: %default)')
    parser.add_option('--filename', metavar='patterns', default='*.py', help='when parsing directories, only check filenames matching these comma separated patterns (default: %default)')
    parser.add_option('--select', metavar='errors', default='', help='select errors and warnings (e.g. E,W6)')
    parser.add_option('--ignore', metavar='errors', default='', help='skip errors and warnings (e.g. E4,W) (default: %s)' % DEFAULT_IGNORE)
    parser.add_option('--show-source', action='store_true', help='show source code for each error')
    parser.add_option('--show-pep8', action='store_true', help='show text of PEP 8 for each error (implies --first)')
    parser.add_option('--statistics', action='store_true', help='count errors and warnings')
    parser.add_option('--count', action='store_true', help='print total number of errors and warnings to standard error and set exit code to 1 if total is not null')
    parser.add_option('--max-line-length', type='int', metavar='n', default=MAX_LINE_LENGTH, help='set maximum allowed line length (default: %default)')
    parser.add_option('--hang-closing', action='store_true', help="hang closing bracket instead of matching indentation of opening bracket's line")
    parser.add_option('--format', metavar='format', default='default', help='set the error format [default|pylint|<custom>]')
    parser.add_option('--diff', action='store_true', help='report changes only within line number ranges in the unified diff received on STDIN')
    group = parser.add_option_group('Testing Options')
    if os.path.exists(TESTSUITE_PATH):
        group.add_option('--testsuite', metavar='dir', help='run regression tests from dir')
        group.add_option('--doctest', action='store_true', help='run doctest on myself')
    group.add_option('--benchmark', action='store_true', help='measure processing speed')
    return parser

def read_config(options, args, arglist, parser):
    if False:
        return 10
    'Read and parse configurations\n\n    If a config file is specified on the command line with the "--config"\n    option, then only it is used for configuration.\n\n    Otherwise, the user configuration (~/.config/pep8) and any local\n    configurations in the current directory or above will be merged together\n    (in that order) using the read method of ConfigParser.\n    '
    config = RawConfigParser()
    cli_conf = options.config
    local_dir = os.curdir
    if USER_CONFIG and os.path.isfile(USER_CONFIG):
        if options.verbose:
            print('user configuration: %s' % USER_CONFIG)
        config.read(USER_CONFIG)
    parent = tail = args and os.path.abspath(os.path.commonprefix(args))
    while tail:
        if config.read((os.path.join(parent, fn) for fn in PROJECT_CONFIG)):
            local_dir = parent
            if options.verbose:
                print('local configuration: in %s' % parent)
            break
        (parent, tail) = os.path.split(parent)
    if cli_conf and os.path.isfile(cli_conf):
        if options.verbose:
            print('cli configuration: %s' % cli_conf)
        config.read(cli_conf)
    pep8_section = parser.prog
    if config.has_section(pep8_section):
        option_list = dict([(o.dest, o.type or o.action) for o in parser.option_list])
        (new_options, __) = parser.parse_args([])
        for opt in config.options(pep8_section):
            if opt.replace('_', '-') not in parser.config_options:
                print("  unknown option '%s' ignored" % opt)
                continue
            if options.verbose > 1:
                print('  %s = %s' % (opt, config.get(pep8_section, opt)))
            normalized_opt = opt.replace('-', '_')
            opt_type = option_list[normalized_opt]
            if opt_type in ('int', 'count'):
                value = config.getint(pep8_section, opt)
            elif opt_type == 'string':
                value = config.get(pep8_section, opt)
                if normalized_opt == 'exclude':
                    value = normalize_paths(value, local_dir)
            else:
                assert opt_type in ('store_true', 'store_false')
                value = config.getboolean(pep8_section, opt)
            setattr(new_options, normalized_opt, value)
        (options, __) = parser.parse_args(arglist, values=new_options)
    options.doctest = options.testsuite = False
    return options

def process_options(arglist=None, parse_argv=False, config_file=None, parser=None):
    if False:
        print('Hello World!')
    'Process options passed either via arglist or via command line args.\n\n    Passing in the ``config_file`` parameter allows other tools, such as flake8\n    to specify their own options to be processed in pep8.\n    '
    if not parser:
        parser = get_parser()
    if not parser.has_option('--config'):
        group = parser.add_option_group('Configuration', description='The project options are read from the [%s] section of the tox.ini file or the setup.cfg file located in any parent folder of the path(s) being processed.  Allowed options are: %s.' % (parser.prog, ', '.join(parser.config_options)))
        group.add_option('--config', metavar='path', default=config_file, help='user config file location')
    if not arglist and (not parse_argv):
        arglist = []
    (options, args) = parser.parse_args(arglist)
    options.reporter = None
    if options.ensure_value('testsuite', False):
        args.append(options.testsuite)
    elif not options.ensure_value('doctest', False):
        if parse_argv and (not args):
            if options.diff or any((os.path.exists(name) for name in PROJECT_CONFIG)):
                args = ['.']
            else:
                parser.error('input not specified')
        options = read_config(options, args, arglist, parser)
        options.reporter = parse_argv and options.quiet == 1 and FileReport
    options.filename = _parse_multi_options(options.filename)
    options.exclude = normalize_paths(options.exclude)
    options.select = _parse_multi_options(options.select)
    options.ignore = _parse_multi_options(options.ignore)
    if options.diff:
        options.reporter = DiffReport
        stdin = stdin_get_value()
        options.selected_lines = parse_udiff(stdin, options.filename, args[0])
        args = sorted(options.selected_lines)
    return (options, args)

def _parse_multi_options(options, split_token=','):
    if False:
        return 10
    'Split and strip and discard empties.\n\n    Turns the following:\n\n    A,\n    B,\n\n    into ["A", "B"]\n    '
    if options:
        return [o.strip() for o in options.split(split_token) if o.strip()]
    else:
        return options

def _main():
    if False:
        while True:
            i = 10
    'Parse options and run checks on Python source.'
    import signal
    try:
        signal.signal(signal.SIGPIPE, lambda signum, frame: sys.exit(1))
    except AttributeError:
        pass
    pep8style = StyleGuide(parse_argv=True)
    options = pep8style.options
    if options.doctest or options.testsuite:
        from testsuite.support import run_tests
        report = run_tests(pep8style)
    else:
        report = pep8style.check_files()
    if options.statistics:
        report.print_statistics()
    if options.benchmark:
        report.print_benchmark()
    if options.testsuite and (not options.quiet):
        report.print_results()
    if report.total_errors:
        if options.count:
            sys.stderr.write(str(report.total_errors) + '\n')
        sys.exit(1)
if __name__ == '__main__':
    _main()