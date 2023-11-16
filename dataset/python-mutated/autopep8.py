"""Automatically formats Python code to conform to the PEP 8 style guide.

Fixes that only need be done once can be added by adding a function of the form
"fix_<code>(source)" to this module. They should return the fixed source code.
These fixes are picked up by apply_global_fixes().

Fixes that depend on pycodestyle should be added as methods to FixPEP8. See the
class documentation for more information.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import codecs
import collections
import copy
import difflib
import fnmatch
import inspect
import io
import itertools
import keyword
import locale
import os
import re
import signal
import sys
import textwrap
import token
import tokenize
import warnings
import ast
from configparser import ConfigParser as SafeConfigParser, Error
import pycodestyle
from pycodestyle import STARTSWITH_INDENT_STATEMENT_REGEX
__version__ = '2.0.4'
CR = '\r'
LF = '\n'
CRLF = '\r\n'
PYTHON_SHEBANG_REGEX = re.compile('^#!.*\\bpython[23]?\\b\\s*$')
LAMBDA_REGEX = re.compile('([\\w.]+)\\s=\\slambda\\s*([)(=\\w,\\s.]*):')
COMPARE_NEGATIVE_REGEX = re.compile('\\b(not)\\s+([^][)(}{]+?)\\s+(in|is)\\s')
COMPARE_NEGATIVE_REGEX_THROUGH = re.compile('\\b(not\\s+in|is\\s+not)\\s')
BARE_EXCEPT_REGEX = re.compile('except\\s*:')
STARTSWITH_DEF_REGEX = re.compile('^(async\\s+def|def)\\s.*\\):')
DOCSTRING_START_REGEX = re.compile('^u?r?(?P<kind>["\\\']{3})')
ENABLE_REGEX = re.compile('# *(fmt|autopep8): *on')
DISABLE_REGEX = re.compile('# *(fmt|autopep8): *off')
EXIT_CODE_OK = 0
EXIT_CODE_ERROR = 1
EXIT_CODE_EXISTS_DIFF = 2
EXIT_CODE_ARGPARSE_ERROR = 99
SHORTEN_OPERATOR_GROUPS = frozenset([frozenset([',']), frozenset(['%']), frozenset([',', '(', '[', '{']), frozenset(['%', '(', '[', '{']), frozenset([',', '(', '[', '{', '%', '+', '-', '*', '/', '//']), frozenset(['%', '+', '-', '*', '/', '//'])])
DEFAULT_IGNORE = 'E226,E24,W50,W690'
DEFAULT_INDENT_SIZE = 4
CONFLICTING_CODES = ('W503', 'W504')
CODE_TO_2TO3 = {'E231': ['ws_comma'], 'E721': ['idioms'], 'W690': ['apply', 'except', 'exitfunc', 'numliterals', 'operator', 'paren', 'reduce', 'renames', 'standarderror', 'sys_exc', 'throw', 'tuple_params', 'xreadlines']}
if sys.platform == 'win32':
    DEFAULT_CONFIG = os.path.expanduser('~\\.pycodestyle')
else:
    DEFAULT_CONFIG = os.path.join(os.getenv('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'), 'pycodestyle')
if not os.path.exists(DEFAULT_CONFIG):
    if sys.platform == 'win32':
        DEFAULT_CONFIG = os.path.expanduser('~\\.pep8')
    else:
        DEFAULT_CONFIG = os.path.join(os.path.expanduser('~/.config'), 'pep8')
PROJECT_CONFIG = ('setup.cfg', 'tox.ini', '.pep8', '.flake8')
MAX_PYTHON_FILE_DETECTION_BYTES = 1024

def open_with_encoding(filename, mode='r', encoding=None, limit_byte_check=-1):
    if False:
        print('Hello World!')
    'Return opened file with a specific encoding.'
    if not encoding:
        encoding = detect_encoding(filename, limit_byte_check=limit_byte_check)
    return io.open(filename, mode=mode, encoding=encoding, newline='')

def detect_encoding(filename, limit_byte_check=-1):
    if False:
        while True:
            i = 10
    'Return file encoding.'
    try:
        with open(filename, 'rb') as input_file:
            from lib2to3.pgen2 import tokenize as lib2to3_tokenize
            encoding = lib2to3_tokenize.detect_encoding(input_file.readline)[0]
        with open_with_encoding(filename, encoding=encoding) as test_file:
            test_file.read(limit_byte_check)
        return encoding
    except (LookupError, SyntaxError, UnicodeDecodeError):
        return 'latin-1'

def readlines_from_file(filename):
    if False:
        i = 10
        return i + 15
    'Return contents of file.'
    with open_with_encoding(filename) as input_file:
        return input_file.readlines()

def extended_blank_lines(logical_line, blank_lines, blank_before, indent_level, previous_logical):
    if False:
        print('Hello World!')
    'Check for missing blank lines after class declaration.'
    if previous_logical.startswith('def '):
        if blank_lines and pycodestyle.DOCSTRING_REGEX.match(logical_line):
            yield (0, 'E303 too many blank lines ({})'.format(blank_lines))
    elif pycodestyle.DOCSTRING_REGEX.match(previous_logical):
        if indent_level and (not blank_lines) and (not blank_before) and logical_line.startswith('def ') and ('(self' in logical_line):
            yield (0, 'E301 expected 1 blank line, found 0')
pycodestyle.register_check(extended_blank_lines)

def continued_indentation(logical_line, tokens, indent_level, hang_closing, indent_char, noqa):
    if False:
        for i in range(10):
            print('nop')
    "Override pycodestyle's function to provide indentation information."
    first_row = tokens[0][2][0]
    nrows = 1 + tokens[-1][2][0] - first_row
    if noqa or nrows == 1:
        return
    indent_next = logical_line.endswith(':')
    row = depth = 0
    valid_hangs = (DEFAULT_INDENT_SIZE,) if indent_char != '\t' else (DEFAULT_INDENT_SIZE, 2 * DEFAULT_INDENT_SIZE)
    parens = [0] * nrows
    rel_indent = [0] * nrows
    open_rows = [[0]]
    hangs = [None]
    indent_chances = {}
    last_indent = tokens[0][2]
    indent = [last_indent[1]]
    last_token_multiline = None
    line = None
    last_line = ''
    last_line_begins_with_multiline = False
    for (token_type, text, start, end, line) in tokens:
        newline = row < start[0] - first_row
        if newline:
            row = start[0] - first_row
            newline = not last_token_multiline and token_type not in (tokenize.NL, tokenize.NEWLINE)
            last_line_begins_with_multiline = last_token_multiline
        if newline:
            last_indent = start
            rel_indent[row] = pycodestyle.expand_indent(line) - indent_level
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
                    yield (start, 'E124 {}'.format(indent[depth]))
            elif close_bracket and (not hang):
                if hang_closing:
                    yield (start, 'E133 {}'.format(indent[depth]))
            elif indent[depth] and start[1] < indent[depth]:
                if visual_indent is not True:
                    yield (start, 'E128 {}'.format(indent[depth]))
            elif hanging_indent or (indent_next and rel_indent[row] == 2 * DEFAULT_INDENT_SIZE):
                if close_bracket and (not hang_closing):
                    yield (start, 'E123 {}'.format(indent_level + rel_indent[open_row]))
                hangs[depth] = hang
            elif visual_indent is True:
                indent[depth] = start[1]
            elif visual_indent in (text, str):
                pass
            else:
                one_indented = indent_level + rel_indent[open_row] + DEFAULT_INDENT_SIZE
                if hang <= 0:
                    error = ('E122', one_indented)
                elif indent[depth]:
                    error = ('E127', indent[depth])
                elif not close_bracket and hangs[depth]:
                    error = ('E131', one_indented)
                elif hang > DEFAULT_INDENT_SIZE:
                    error = ('E126', one_indented)
                else:
                    hangs[depth] = hang
                    error = ('E121', one_indented)
                yield (start, '{} {}'.format(*error))
        if parens[row] and token_type not in (tokenize.NL, tokenize.COMMENT) and (not indent[depth]):
            indent[depth] = start[1]
            indent_chances[start[1]] = True
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
            if start[1] not in indent_chances and (not last_line.rstrip().endswith(',')):
                indent_chances[start[1]] = text
        last_token_multiline = start[0] != end[0]
        if last_token_multiline:
            rel_indent[end[0] - first_row] = rel_indent[row]
        last_line = line
    if indent_next and (not last_line_begins_with_multiline) and (pycodestyle.expand_indent(line) == indent_level + DEFAULT_INDENT_SIZE):
        pos = (start[0], indent[0] + 4)
        desired_indent = indent_level + 2 * DEFAULT_INDENT_SIZE
        if visual_indent:
            yield (pos, 'E129 {}'.format(desired_indent))
        else:
            yield (pos, 'E125 {}'.format(desired_indent))
del pycodestyle._checks['logical_line'][pycodestyle.continued_indentation]
pycodestyle.register_check(continued_indentation)

class FixPEP8(object):
    """Fix invalid code.

    Fixer methods are prefixed "fix_". The _fix_source() method looks for these
    automatically.

    The fixer method can take either one or two arguments (in addition to
    self). The first argument is "result", which is the error information from
    pycodestyle. The second argument, "logical", is required only for
    logical-line fixes.

    The fixer method can return the list of modified lines or None. An empty
    list would mean that no changes were made. None would mean that only the
    line reported in the pycodestyle error was modified. Note that the modified
    line numbers that are returned are indexed at 1. This typically would
    correspond with the line number reported in the pycodestyle error
    information.

    [fixed method list]
        - e111,e114,e115,e116
        - e121,e122,e123,e124,e125,e126,e127,e128,e129
        - e201,e202,e203
        - e211
        - e221,e222,e223,e224,e225
        - e231
        - e251,e252
        - e261,e262
        - e271,e272,e273,e274,e275
        - e301,e302,e303,e304,e305,e306
        - e401,e402
        - e502
        - e701,e702,e703,e704
        - e711,e712,e713,e714
        - e722
        - e731
        - w291
        - w503,504

    """

    def __init__(self, filename, options, contents=None, long_line_ignore_cache=None):
        if False:
            while True:
                i = 10
        self.filename = filename
        if contents is None:
            self.source = readlines_from_file(filename)
        else:
            sio = io.StringIO(contents)
            self.source = sio.readlines()
        self.options = options
        self.indent_word = _get_indentword(''.join(self.source))
        self.imports = {}
        for (i, line) in enumerate(self.source):
            if (line.find('import ') == 0 or line.find('from ') == 0) and line not in self.imports:
                self.imports[line] = i
        self.long_line_ignore_cache = set() if long_line_ignore_cache is None else long_line_ignore_cache
        self.fix_e115 = self.fix_e112
        self.fix_e121 = self._fix_reindent
        self.fix_e122 = self._fix_reindent
        self.fix_e123 = self._fix_reindent
        self.fix_e124 = self._fix_reindent
        self.fix_e126 = self._fix_reindent
        self.fix_e127 = self._fix_reindent
        self.fix_e128 = self._fix_reindent
        self.fix_e129 = self._fix_reindent
        self.fix_e133 = self.fix_e131
        self.fix_e202 = self.fix_e201
        self.fix_e203 = self.fix_e201
        self.fix_e211 = self.fix_e201
        self.fix_e221 = self.fix_e271
        self.fix_e222 = self.fix_e271
        self.fix_e223 = self.fix_e271
        self.fix_e226 = self.fix_e225
        self.fix_e227 = self.fix_e225
        self.fix_e228 = self.fix_e225
        self.fix_e241 = self.fix_e271
        self.fix_e242 = self.fix_e224
        self.fix_e252 = self.fix_e225
        self.fix_e261 = self.fix_e262
        self.fix_e272 = self.fix_e271
        self.fix_e273 = self.fix_e271
        self.fix_e274 = self.fix_e271
        self.fix_e275 = self.fix_e271
        self.fix_e306 = self.fix_e301
        self.fix_e501 = self.fix_long_line_logically if options and (options.aggressive >= 2 or options.experimental) else self.fix_long_line_physically
        self.fix_e703 = self.fix_e702
        self.fix_w292 = self.fix_w291
        self.fix_w293 = self.fix_w291

    def _fix_source(self, results):
        if False:
            while True:
                i = 10
        try:
            (logical_start, logical_end) = _find_logical(self.source)
            logical_support = True
        except (SyntaxError, tokenize.TokenError):
            logical_support = False
        completed_lines = set()
        for result in sorted(results, key=_priority_key):
            if result['line'] in completed_lines:
                continue
            fixed_methodname = 'fix_' + result['id'].lower()
            if hasattr(self, fixed_methodname):
                fix = getattr(self, fixed_methodname)
                line_index = result['line'] - 1
                original_line = self.source[line_index]
                is_logical_fix = len(_get_parameters(fix)) > 2
                if is_logical_fix:
                    logical = None
                    if logical_support:
                        logical = _get_logical(self.source, result, logical_start, logical_end)
                        if logical and set(range(logical[0][0] + 1, logical[1][0] + 1)).intersection(completed_lines):
                            continue
                    modified_lines = fix(result, logical)
                else:
                    modified_lines = fix(result)
                if modified_lines is None:
                    assert not is_logical_fix
                    if self.source[line_index] == original_line:
                        modified_lines = []
                if modified_lines:
                    completed_lines.update(modified_lines)
                elif modified_lines == []:
                    if self.options.verbose >= 2:
                        print('--->  Not fixing {error} on line {line}'.format(error=result['id'], line=result['line']), file=sys.stderr)
                else:
                    completed_lines.add(result['line'])
            elif self.options.verbose >= 3:
                print("--->  '{}' is not defined.".format(fixed_methodname), file=sys.stderr)
                info = result['info'].strip()
                print('--->  {}:{}:{}:{}'.format(self.filename, result['line'], result['column'], info), file=sys.stderr)

    def fix(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a version of the source code with PEP 8 violations fixed.'
        pep8_options = {'ignore': self.options.ignore, 'select': self.options.select, 'max_line_length': self.options.max_line_length, 'hang_closing': self.options.hang_closing}
        results = _execute_pep8(pep8_options, self.source)
        if self.options.verbose:
            progress = {}
            for r in results:
                if r['id'] not in progress:
                    progress[r['id']] = set()
                progress[r['id']].add(r['line'])
            print('--->  {n} issue(s) to fix {progress}'.format(n=len(results), progress=progress), file=sys.stderr)
        if self.options.line_range:
            (start, end) = self.options.line_range
            results = [r for r in results if start <= r['line'] <= end]
        self._fix_source(filter_results(source=''.join(self.source), results=results, aggressive=self.options.aggressive))
        if self.options.line_range:
            count = sum((sline.count('\n') for sline in self.source[start - 1:end]))
            self.options.line_range[1] = start + count - 1
        return ''.join(self.source)

    def _fix_reindent(self, result):
        if False:
            return 10
        'Fix a badly indented line.\n\n        This is done by adding or removing from its initial indent only.\n\n        '
        num_indent_spaces = int(result['info'].split()[1])
        line_index = result['line'] - 1
        target = self.source[line_index]
        self.source[line_index] = ' ' * num_indent_spaces + target.lstrip()

    def fix_e112(self, result):
        if False:
            print('Hello World!')
        'Fix under-indented comments.'
        line_index = result['line'] - 1
        target = self.source[line_index]
        if not target.lstrip().startswith('#'):
            return []
        self.source[line_index] = self.indent_word + target

    def fix_e113(self, result):
        if False:
            while True:
                i = 10
        'Fix unexpected indentation.'
        line_index = result['line'] - 1
        target = self.source[line_index]
        indent = _get_indentation(target)
        stripped = target.lstrip()
        self.source[line_index] = indent[1:] + stripped

    def fix_e116(self, result):
        if False:
            while True:
                i = 10
        'Fix over-indented comments.'
        line_index = result['line'] - 1
        target = self.source[line_index]
        indent = _get_indentation(target)
        stripped = target.lstrip()
        if not stripped.startswith('#'):
            return []
        self.source[line_index] = indent[1:] + stripped

    def fix_e117(self, result):
        if False:
            for i in range(10):
                print('nop')
        'Fix over-indented.'
        line_index = result['line'] - 1
        target = self.source[line_index]
        indent = _get_indentation(target)
        if indent == '\t':
            return []
        stripped = target.lstrip()
        self.source[line_index] = indent[1:] + stripped

    def fix_e125(self, result):
        if False:
            while True:
                i = 10
        'Fix indentation undistinguish from the next logical line.'
        num_indent_spaces = int(result['info'].split()[1])
        line_index = result['line'] - 1
        target = self.source[line_index]
        spaces_to_add = num_indent_spaces - len(_get_indentation(target))
        indent = len(_get_indentation(target))
        modified_lines = []
        while len(_get_indentation(self.source[line_index])) >= indent:
            self.source[line_index] = ' ' * spaces_to_add + self.source[line_index]
            modified_lines.append(1 + line_index)
            line_index -= 1
        return modified_lines

    def fix_e131(self, result):
        if False:
            for i in range(10):
                print('nop')
        'Fix indentation undistinguish from the next logical line.'
        num_indent_spaces = int(result['info'].split()[1])
        line_index = result['line'] - 1
        target = self.source[line_index]
        spaces_to_add = num_indent_spaces - len(_get_indentation(target))
        indent_length = len(_get_indentation(target))
        spaces_to_add = num_indent_spaces - indent_length
        if num_indent_spaces == 0 and indent_length == 0:
            spaces_to_add = 4
        if spaces_to_add >= 0:
            self.source[line_index] = ' ' * spaces_to_add + self.source[line_index]
        else:
            offset = abs(spaces_to_add)
            self.source[line_index] = self.source[line_index][offset:]

    def fix_e201(self, result):
        if False:
            i = 10
            return i + 15
        'Remove extraneous whitespace.'
        line_index = result['line'] - 1
        target = self.source[line_index]
        offset = result['column'] - 1
        fixed = fix_whitespace(target, offset=offset, replacement='')
        self.source[line_index] = fixed

    def fix_e224(self, result):
        if False:
            while True:
                i = 10
        'Remove extraneous whitespace around operator.'
        target = self.source[result['line'] - 1]
        offset = result['column'] - 1
        fixed = target[:offset] + target[offset:].replace('\t', ' ')
        self.source[result['line'] - 1] = fixed

    def fix_e225(self, result):
        if False:
            while True:
                i = 10
        'Fix missing whitespace around operator.'
        target = self.source[result['line'] - 1]
        offset = result['column'] - 1
        fixed = target[:offset] + ' ' + target[offset:]
        if fixed.replace(' ', '') == target.replace(' ', '') and _get_indentation(fixed) == _get_indentation(target):
            self.source[result['line'] - 1] = fixed
            error_code = result.get('id', 0)
            try:
                ts = generate_tokens(fixed)
            except (SyntaxError, tokenize.TokenError):
                return
            if not check_syntax(fixed.lstrip()):
                return
            try:
                _missing_whitespace = pycodestyle.missing_whitespace_around_operator
            except AttributeError:
                _missing_whitespace = pycodestyle.missing_whitespace
            errors = list(_missing_whitespace(fixed, ts))
            for e in reversed(errors):
                if error_code != e[1].split()[0]:
                    continue
                offset = e[0][1]
                fixed = fixed[:offset] + ' ' + fixed[offset:]
            self.source[result['line'] - 1] = fixed
        else:
            return []

    def fix_e231(self, result):
        if False:
            while True:
                i = 10
        'Add missing whitespace.'
        line_index = result['line'] - 1
        target = self.source[line_index]
        offset = result['column']
        fixed = target[:offset].rstrip() + ' ' + target[offset:].lstrip()
        self.source[line_index] = fixed

    def fix_e251(self, result):
        if False:
            return 10
        "Remove whitespace around parameter '=' sign."
        line_index = result['line'] - 1
        target = self.source[line_index]
        c = min(result['column'] - 1, len(target) - 1)
        if target[c].strip():
            fixed = target
        else:
            fixed = target[:c].rstrip() + target[c:].lstrip()
        if fixed.endswith(('=\\\n', '=\\\r\n', '=\\\r')):
            self.source[line_index] = fixed.rstrip('\n\r \t\\')
            self.source[line_index + 1] = self.source[line_index + 1].lstrip()
            return [line_index + 1, line_index + 2]
        self.source[result['line'] - 1] = fixed

    def fix_e262(self, result):
        if False:
            for i in range(10):
                print('nop')
        'Fix spacing after inline comment hash.'
        target = self.source[result['line'] - 1]
        offset = result['column']
        code = target[:offset].rstrip(' \t#')
        comment = target[offset:].lstrip(' \t#')
        fixed = code + ('  # ' + comment if comment.strip() else '\n')
        self.source[result['line'] - 1] = fixed

    def fix_e265(self, result):
        if False:
            for i in range(10):
                print('nop')
        'Fix spacing after block comment hash.'
        target = self.source[result['line'] - 1]
        indent = _get_indentation(target)
        line = target.lstrip(' \t')
        pos = next((index for (index, c) in enumerate(line) if c != '#'))
        hashes = line[:pos]
        comment = line[pos:].lstrip(' \t')
        if comment.startswith('!'):
            return
        fixed = indent + hashes + (' ' + comment if comment.strip() else '\n')
        self.source[result['line'] - 1] = fixed

    def fix_e266(self, result):
        if False:
            while True:
                i = 10
        'Fix too many block comment hashes.'
        target = self.source[result['line'] - 1]
        if target.strip().endswith('#'):
            return
        indentation = _get_indentation(target)
        fixed = indentation + '# ' + target.lstrip('# \t')
        self.source[result['line'] - 1] = fixed

    def fix_e271(self, result):
        if False:
            for i in range(10):
                print('nop')
        'Fix extraneous whitespace around keywords.'
        line_index = result['line'] - 1
        target = self.source[line_index]
        offset = result['column'] - 1
        fixed = fix_whitespace(target, offset=offset, replacement=' ')
        if fixed == target:
            return []
        else:
            self.source[line_index] = fixed

    def fix_e301(self, result):
        if False:
            print('Hello World!')
        'Add missing blank line.'
        cr = '\n'
        self.source[result['line'] - 1] = cr + self.source[result['line'] - 1]

    def fix_e302(self, result):
        if False:
            for i in range(10):
                print('nop')
        'Add missing 2 blank lines.'
        add_linenum = 2 - int(result['info'].split()[-1])
        offset = 1
        if self.source[result['line'] - 2].strip() == '\\':
            offset = 2
        cr = '\n' * add_linenum
        self.source[result['line'] - offset] = cr + self.source[result['line'] - offset]

    def fix_e303(self, result):
        if False:
            for i in range(10):
                print('nop')
        'Remove extra blank lines.'
        delete_linenum = int(result['info'].split('(')[1].split(')')[0]) - 2
        delete_linenum = max(1, delete_linenum)
        cnt = 0
        line = result['line'] - 2
        modified_lines = []
        while cnt < delete_linenum and line >= 0:
            if not self.source[line].strip():
                self.source[line] = ''
                modified_lines.append(1 + line)
                cnt += 1
            line -= 1
        return modified_lines

    def fix_e304(self, result):
        if False:
            for i in range(10):
                print('nop')
        'Remove blank line following function decorator.'
        line = result['line'] - 2
        if not self.source[line].strip():
            self.source[line] = ''

    def fix_e305(self, result):
        if False:
            print('Hello World!')
        'Add missing 2 blank lines after end of function or class.'
        add_delete_linenum = 2 - int(result['info'].split()[-1])
        cnt = 0
        offset = result['line'] - 2
        modified_lines = []
        if add_delete_linenum < 0:
            add_delete_linenum = abs(add_delete_linenum)
            while cnt < add_delete_linenum and offset >= 0:
                if not self.source[offset].strip():
                    self.source[offset] = ''
                    modified_lines.append(1 + offset)
                    cnt += 1
                offset -= 1
        else:
            cr = '\n'
            while True:
                if offset < 0:
                    break
                line = self.source[offset].lstrip()
                if not line:
                    break
                if line[0] != '#':
                    break
                offset -= 1
            offset += 1
            self.source[offset] = cr + self.source[offset]
            modified_lines.append(1 + offset)
        return modified_lines

    def fix_e401(self, result):
        if False:
            while True:
                i = 10
        'Put imports on separate lines.'
        line_index = result['line'] - 1
        target = self.source[line_index]
        offset = result['column'] - 1
        if not target.lstrip().startswith('import'):
            return []
        indentation = re.split(pattern='\\bimport\\b', string=target, maxsplit=1)[0]
        fixed = target[:offset].rstrip('\t ,') + '\n' + indentation + 'import ' + target[offset:].lstrip('\t ,')
        self.source[line_index] = fixed

    def fix_e402(self, result):
        if False:
            while True:
                i = 10
        (line_index, offset, target) = get_index_offset_contents(result, self.source)
        for i in range(1, 100):
            line = ''.join(self.source[line_index:line_index + i])
            try:
                generate_tokens(''.join(line))
            except (SyntaxError, tokenize.TokenError):
                continue
            break
        if not (target in self.imports and self.imports[target] != line_index):
            mod_offset = get_module_imports_on_top_of_file(self.source, line_index)
            self.source[mod_offset] = line + self.source[mod_offset]
        for offset in range(i):
            self.source[line_index + offset] = ''

    def fix_long_line_logically(self, result, logical):
        if False:
            return 10
        'Try to make lines fit within --max-line-length characters.'
        if not logical or len(logical[2]) == 1 or self.source[result['line'] - 1].lstrip().startswith('#'):
            return self.fix_long_line_physically(result)
        start_line_index = logical[0][0]
        end_line_index = logical[1][0]
        logical_lines = logical[2]
        previous_line = get_item(self.source, start_line_index - 1, default='')
        next_line = get_item(self.source, end_line_index + 1, default='')
        single_line = join_logical_line(''.join(logical_lines))
        try:
            fixed = self.fix_long_line(target=single_line, previous_line=previous_line, next_line=next_line, original=''.join(logical_lines))
        except (SyntaxError, tokenize.TokenError):
            return self.fix_long_line_physically(result)
        if fixed:
            for line_index in range(start_line_index, end_line_index + 1):
                self.source[line_index] = ''
            self.source[start_line_index] = fixed
            return range(start_line_index + 1, end_line_index + 1)
        return []

    def fix_long_line_physically(self, result):
        if False:
            i = 10
            return i + 15
        'Try to make lines fit within --max-line-length characters.'
        line_index = result['line'] - 1
        target = self.source[line_index]
        previous_line = get_item(self.source, line_index - 1, default='')
        next_line = get_item(self.source, line_index + 1, default='')
        try:
            fixed = self.fix_long_line(target=target, previous_line=previous_line, next_line=next_line, original=target)
        except (SyntaxError, tokenize.TokenError):
            return []
        if fixed:
            self.source[line_index] = fixed
            return [line_index + 1]
        return []

    def fix_long_line(self, target, previous_line, next_line, original):
        if False:
            return 10
        cache_entry = (target, previous_line, next_line)
        if cache_entry in self.long_line_ignore_cache:
            return []
        if target.lstrip().startswith('#'):
            if self.options.aggressive:
                return shorten_comment(line=target, max_line_length=self.options.max_line_length, last_comment=not next_line.lstrip().startswith('#'))
            return []
        fixed = get_fixed_long_line(target=target, previous_line=previous_line, original=original, indent_word=self.indent_word, max_line_length=self.options.max_line_length, aggressive=self.options.aggressive, experimental=self.options.experimental, verbose=self.options.verbose)
        if fixed and (not code_almost_equal(original, fixed)):
            return fixed
        self.long_line_ignore_cache.add(cache_entry)
        return None

    def fix_e502(self, result):
        if False:
            i = 10
            return i + 15
        'Remove extraneous escape of newline.'
        (line_index, _, target) = get_index_offset_contents(result, self.source)
        self.source[line_index] = target.rstrip('\n\r \t\\') + '\n'

    def fix_e701(self, result):
        if False:
            return 10
        'Put colon-separated compound statement on separate lines.'
        line_index = result['line'] - 1
        target = self.source[line_index]
        c = result['column']
        fixed_source = target[:c] + '\n' + _get_indentation(target) + self.indent_word + target[c:].lstrip('\n\r \t\\')
        self.source[result['line'] - 1] = fixed_source
        return [result['line'], result['line'] + 1]

    def fix_e702(self, result, logical):
        if False:
            return 10
        'Put semicolon-separated compound statement on separate lines.'
        if not logical:
            return []
        logical_lines = logical[2]
        for line in logical_lines:
            if result['id'] == 'E702' and ':' in line and STARTSWITH_INDENT_STATEMENT_REGEX.match(line):
                if self.options.verbose:
                    print('---> avoid fixing {error} with other compound statements'.format(error=result['id']), file=sys.stderr)
                return []
        line_index = result['line'] - 1
        target = self.source[line_index]
        if target.rstrip().endswith('\\'):
            self.source[line_index] = target.rstrip('\n \r\t\\')
            self.source[line_index + 1] = self.source[line_index + 1].lstrip()
            return [line_index + 1, line_index + 2]
        if target.rstrip().endswith(';'):
            self.source[line_index] = target.rstrip('\n \r\t;') + '\n'
            return [line_index + 1]
        offset = result['column'] - 1
        first = target[:offset].rstrip(';').rstrip()
        second = _get_indentation(logical_lines[0]) + target[offset:].lstrip(';').lstrip()
        inline_comment = None
        if target[offset:].lstrip(';').lstrip()[:2] == '# ':
            inline_comment = target[offset:].lstrip(';')
        if inline_comment:
            self.source[line_index] = first + inline_comment
        else:
            self.source[line_index] = first + '\n' + second
        return [line_index + 1]

    def fix_e704(self, result):
        if False:
            return 10
        'Fix multiple statements on one line def'
        (line_index, _, target) = get_index_offset_contents(result, self.source)
        match = STARTSWITH_DEF_REGEX.match(target)
        if match:
            self.source[line_index] = '{}\n{}{}'.format(match.group(0), _get_indentation(target) + self.indent_word, target[match.end(0):].lstrip())

    def fix_e711(self, result):
        if False:
            return 10
        'Fix comparison with None.'
        (line_index, offset, target) = get_index_offset_contents(result, self.source)
        right_offset = offset + 2
        if right_offset >= len(target):
            return []
        left = target[:offset].rstrip()
        center = target[offset:right_offset]
        right = target[right_offset:].lstrip()
        if center.strip() == '==':
            new_center = 'is'
        elif center.strip() == '!=':
            new_center = 'is not'
        else:
            return []
        self.source[line_index] = ' '.join([left, new_center, right])

    def fix_e712(self, result):
        if False:
            print('Hello World!')
        'Fix (trivial case of) comparison with boolean.'
        (line_index, offset, target) = get_index_offset_contents(result, self.source)
        if re.match('^\\s*if [\\w."\\\'\\[\\]]+ == False:$', target):
            self.source[line_index] = re.sub('if ([\\w."\\\'\\[\\]]+) == False:', 'if not \\1:', target, count=1)
        elif re.match('^\\s*if [\\w."\\\'\\[\\]]+ != True:$', target):
            self.source[line_index] = re.sub('if ([\\w."\\\'\\[\\]]+) != True:', 'if not \\1:', target, count=1)
        else:
            right_offset = offset + 2
            if right_offset >= len(target):
                return []
            left = target[:offset].rstrip()
            center = target[offset:right_offset]
            right = target[right_offset:].lstrip()
            new_right = None
            if center.strip() == '==':
                if re.match('\\bTrue\\b', right):
                    new_right = re.sub('\\bTrue\\b *', '', right, count=1)
            elif center.strip() == '!=':
                if re.match('\\bFalse\\b', right):
                    new_right = re.sub('\\bFalse\\b *', '', right, count=1)
            if new_right is None:
                return []
            if new_right[0].isalnum():
                new_right = ' ' + new_right
            self.source[line_index] = left + new_right

    def fix_e713(self, result):
        if False:
            i = 10
            return i + 15
        'Fix (trivial case of) non-membership check.'
        (line_index, offset, target) = get_index_offset_contents(result, self.source)
        before_target = target[:offset]
        target = target[offset:]
        match_notin = COMPARE_NEGATIVE_REGEX_THROUGH.search(target)
        (notin_pos_start, notin_pos_end) = (0, 0)
        if match_notin:
            notin_pos_start = match_notin.start(1)
            notin_pos_end = match_notin.end()
            target = '{}{} {}'.format(target[:notin_pos_start], 'in', target[notin_pos_end:])
        match = COMPARE_NEGATIVE_REGEX.search(target)
        if match:
            if match.group(3) == 'in':
                pos_start = match.start(1)
                new_target = '{5}{0}{1} {2} {3} {4}'.format(target[:pos_start], match.group(2), match.group(1), match.group(3), target[match.end():], before_target)
                if match_notin:
                    pos_start = notin_pos_start + offset
                    pos_end = notin_pos_end + offset - 4
                    new_target = '{}{} {}'.format(new_target[:pos_start], 'not in', new_target[pos_end:])
                self.source[line_index] = new_target

    def fix_e714(self, result):
        if False:
            i = 10
            return i + 15
        "Fix object identity should be 'is not' case."
        (line_index, offset, target) = get_index_offset_contents(result, self.source)
        before_target = target[:offset]
        target = target[offset:]
        match_isnot = COMPARE_NEGATIVE_REGEX_THROUGH.search(target)
        (isnot_pos_start, isnot_pos_end) = (0, 0)
        if match_isnot:
            isnot_pos_start = match_isnot.start(1)
            isnot_pos_end = match_isnot.end()
            target = '{}{} {}'.format(target[:isnot_pos_start], 'in', target[isnot_pos_end:])
        match = COMPARE_NEGATIVE_REGEX.search(target)
        if match:
            if match.group(3).startswith('is'):
                pos_start = match.start(1)
                new_target = '{5}{0}{1} {2} {3} {4}'.format(target[:pos_start], match.group(2), match.group(3), match.group(1), target[match.end():], before_target)
                if match_isnot:
                    pos_start = isnot_pos_start + offset
                    pos_end = isnot_pos_end + offset - 4
                    new_target = '{}{} {}'.format(new_target[:pos_start], 'is not', new_target[pos_end:])
                self.source[line_index] = new_target

    def fix_e722(self, result):
        if False:
            i = 10
            return i + 15
        'fix bare except'
        (line_index, _, target) = get_index_offset_contents(result, self.source)
        match = BARE_EXCEPT_REGEX.search(target)
        if match:
            self.source[line_index] = '{}{}{}'.format(target[:result['column'] - 1], 'except BaseException:', target[match.end():])

    def fix_e731(self, result):
        if False:
            for i in range(10):
                print('nop')
        'Fix do not assign a lambda expression check.'
        (line_index, _, target) = get_index_offset_contents(result, self.source)
        match = LAMBDA_REGEX.search(target)
        if match:
            end = match.end()
            self.source[line_index] = '{}def {}({}): return {}'.format(target[:match.start(0)], match.group(1), match.group(2), target[end:].lstrip())

    def fix_w291(self, result):
        if False:
            print('Hello World!')
        'Remove trailing whitespace.'
        fixed_line = self.source[result['line'] - 1].rstrip()
        self.source[result['line'] - 1] = fixed_line + '\n'

    def fix_w391(self, _):
        if False:
            return 10
        'Remove trailing blank lines.'
        blank_count = 0
        for line in reversed(self.source):
            line = line.rstrip()
            if line:
                break
            else:
                blank_count += 1
        original_length = len(self.source)
        self.source = self.source[:original_length - blank_count]
        return range(1, 1 + original_length)

    def fix_w503(self, result):
        if False:
            for i in range(10):
                print('nop')
        (line_index, _, target) = get_index_offset_contents(result, self.source)
        one_string_token = target.split()[0]
        try:
            ts = generate_tokens(one_string_token)
        except (SyntaxError, tokenize.TokenError):
            return
        if not _is_binary_operator(ts[0][0], one_string_token):
            return
        comment_index = 0
        found_not_comment_only_line = False
        comment_only_linenum = 0
        for i in range(5):
            if line_index - i < 0:
                break
            from_index = line_index - i - 1
            if from_index < 0 or len(self.source) <= from_index:
                break
            to_index = line_index + 1
            strip_line = self.source[from_index].lstrip()
            if not found_not_comment_only_line and strip_line and (strip_line[0] == '#'):
                comment_only_linenum += 1
                continue
            found_not_comment_only_line = True
            try:
                ts = generate_tokens(''.join(self.source[from_index:to_index]))
            except (SyntaxError, tokenize.TokenError):
                continue
            newline_count = 0
            newline_index = []
            for (index, t) in enumerate(ts):
                if t[0] in (tokenize.NEWLINE, tokenize.NL):
                    newline_index.append(index)
                    newline_count += 1
            if newline_count > 2:
                tts = ts[newline_index[-3]:]
            else:
                tts = ts
            old = []
            for t in tts:
                if t[0] in (tokenize.NEWLINE, tokenize.NL):
                    newline_count -= 1
                if newline_count <= 1:
                    break
                if tokenize.COMMENT == t[0] and old and (old[0] != tokenize.NL):
                    comment_index = old[3][1]
                    break
                old = t
            break
        i = target.index(one_string_token)
        fix_target_line = line_index - 1 - comment_only_linenum
        self.source[line_index] = '{}{}'.format(target[:i], target[i + len(one_string_token):].lstrip())
        nl = find_newline(self.source[fix_target_line:line_index])
        before_line = self.source[fix_target_line]
        bl = before_line.index(nl)
        if comment_index:
            self.source[fix_target_line] = '{} {} {}'.format(before_line[:comment_index], one_string_token, before_line[comment_index + 1:])
        elif before_line[:bl].endswith('#'):
            self.source[fix_target_line] = '{}{} {}'.format(before_line[:bl - 2], one_string_token, before_line[bl - 2:])
        else:
            self.source[fix_target_line] = '{} {}{}'.format(before_line[:bl], one_string_token, before_line[bl:])

    def fix_w504(self, result):
        if False:
            print('Hello World!')
        (line_index, _, target) = get_index_offset_contents(result, self.source)
        comment_index = 0
        operator_position = None
        for i in range(1, 6):
            to_index = line_index + i
            try:
                ts = generate_tokens(''.join(self.source[line_index:to_index]))
            except (SyntaxError, tokenize.TokenError):
                continue
            newline_count = 0
            newline_index = []
            for (index, t) in enumerate(ts):
                if _is_binary_operator(t[0], t[1]):
                    if t[2][0] == 1 and t[3][0] == 1:
                        operator_position = (t[2][1], t[3][1])
                elif t[0] == tokenize.NAME and t[1] in ('and', 'or'):
                    if t[2][0] == 1 and t[3][0] == 1:
                        operator_position = (t[2][1], t[3][1])
                elif t[0] in (tokenize.NEWLINE, tokenize.NL):
                    newline_index.append(index)
                    newline_count += 1
            if newline_count > 2:
                tts = ts[:newline_index[-3]]
            else:
                tts = ts
            old = []
            for t in tts:
                if tokenize.COMMENT == t[0] and old:
                    (comment_row, comment_index) = old[3]
                    break
                old = t
            break
        if not operator_position:
            return
        target_operator = target[operator_position[0]:operator_position[1]]
        if comment_index and comment_row == 1:
            self.source[line_index] = '{}{}'.format(target[:operator_position[0]].rstrip(), target[comment_index:])
        else:
            self.source[line_index] = '{}{}{}'.format(target[:operator_position[0]].rstrip(), target[operator_position[1]:].lstrip(), target[operator_position[1]:])
        next_line = self.source[line_index + 1]
        next_line_indent = 0
        m = re.match('\\s*', next_line)
        if m:
            next_line_indent = m.span()[1]
        self.source[line_index + 1] = '{}{} {}'.format(next_line[:next_line_indent], target_operator, next_line[next_line_indent:])

    def fix_w605(self, result):
        if False:
            i = 10
            return i + 15
        (line_index, offset, target) = get_index_offset_contents(result, self.source)
        self.source[line_index] = '{}\\{}'.format(target[:offset + 1], target[offset + 1:])

def get_module_imports_on_top_of_file(source, import_line_index):
    if False:
        for i in range(10):
            print('nop')
    'return import or from keyword position\n\n    example:\n      > 0: import sys\n        1: import os\n        2:\n        3: def function():\n    '

    def is_string_literal(line):
        if False:
            print('Hello World!')
        if line[0] in 'uUbB':
            line = line[1:]
        if line and line[0] in 'rR':
            line = line[1:]
        return line and (line[0] == '"' or line[0] == "'")

    def is_future_import(line):
        if False:
            i = 10
            return i + 15
        nodes = ast.parse(line)
        for n in nodes.body:
            if isinstance(n, ast.ImportFrom) and n.module == '__future__':
                return True
        return False

    def has_future_import(source):
        if False:
            print('Hello World!')
        offset = 0
        line = ''
        for (_, next_line) in source:
            for line_part in next_line.strip().splitlines(True):
                line = line + line_part
                try:
                    return (is_future_import(line), offset)
                except SyntaxError:
                    continue
            offset += 1
        return (False, offset)
    allowed_try_keywords = ('try', 'except', 'else', 'finally')
    in_docstring = False
    docstring_kind = '"""'
    source_stream = iter(enumerate(source))
    for (cnt, line) in source_stream:
        if not in_docstring:
            m = DOCSTRING_START_REGEX.match(line.lstrip())
            if m is not None:
                in_docstring = True
                docstring_kind = m.group('kind')
                remain = line[m.end():m.endpos].rstrip()
                if remain[-3:] == docstring_kind:
                    in_docstring = False
                continue
        if in_docstring:
            if line.rstrip()[-3:] == docstring_kind:
                in_docstring = False
            continue
        if not line.rstrip():
            continue
        elif line.startswith('#'):
            continue
        if line.startswith('import '):
            if cnt == import_line_index:
                continue
            return cnt
        elif line.startswith('from '):
            if cnt == import_line_index:
                continue
            (hit, offset) = has_future_import(itertools.chain([(cnt, line)], source_stream))
            if hit:
                return cnt + offset + 1
            return cnt
        elif pycodestyle.DUNDER_REGEX.match(line):
            return cnt
        elif any((line.startswith(kw) for kw in allowed_try_keywords)):
            continue
        elif is_string_literal(line):
            return cnt
        else:
            return cnt
    return 0

def get_index_offset_contents(result, source):
    if False:
        return 10
    'Return (line_index, column_offset, line_contents).'
    line_index = result['line'] - 1
    return (line_index, result['column'] - 1, source[line_index])

def get_fixed_long_line(target, previous_line, original, indent_word='    ', max_line_length=79, aggressive=False, experimental=False, verbose=False):
    if False:
        i = 10
        return i + 15
    'Break up long line and return result.\n\n    Do this by generating multiple reformatted candidates and then\n    ranking the candidates to heuristically select the best option.\n\n    '
    indent = _get_indentation(target)
    source = target[len(indent):]
    assert source.lstrip() == source
    assert not target.lstrip().startswith('#')
    tokens = list(generate_tokens(source))
    candidates = shorten_line(tokens, source, indent, indent_word, max_line_length, aggressive=aggressive, experimental=experimental, previous_line=previous_line)
    candidates = sorted(sorted(set(candidates).union([target, original])), key=lambda x: line_shortening_rank(x, indent_word, max_line_length, experimental=experimental))
    if verbose >= 4:
        print(('-' * 79 + '\n').join([''] + candidates + ['']), file=wrap_output(sys.stderr, 'utf-8'))
    if candidates:
        best_candidate = candidates[0]
        if longest_line_length(best_candidate) > longest_line_length(original):
            return None
        return best_candidate

def longest_line_length(code):
    if False:
        i = 10
        return i + 15
    'Return length of longest line.'
    if len(code) == 0:
        return 0
    return max((len(line) for line in code.splitlines()))

def join_logical_line(logical_line):
    if False:
        for i in range(10):
            print('nop')
    'Return single line based on logical line input.'
    indentation = _get_indentation(logical_line)
    return indentation + untokenize_without_newlines(generate_tokens(logical_line.lstrip())) + '\n'

def untokenize_without_newlines(tokens):
    if False:
        while True:
            i = 10
    'Return source code based on tokens.'
    text = ''
    last_row = 0
    last_column = -1
    for t in tokens:
        token_string = t[1]
        (start_row, start_column) = t[2]
        (end_row, end_column) = t[3]
        if start_row > last_row:
            last_column = 0
        if (start_column > last_column or token_string == '\n') and (not text.endswith(' ')):
            text += ' '
        if token_string != '\n':
            text += token_string
        last_row = end_row
        last_column = end_column
    return text.rstrip()

def _find_logical(source_lines):
    if False:
        i = 10
        return i + 15
    logical_start = []
    logical_end = []
    last_newline = True
    parens = 0
    for t in generate_tokens(''.join(source_lines)):
        if t[0] in [tokenize.COMMENT, tokenize.DEDENT, tokenize.INDENT, tokenize.NL, tokenize.ENDMARKER]:
            continue
        if not parens and t[0] in [tokenize.NEWLINE, tokenize.SEMI]:
            last_newline = True
            logical_end.append((t[3][0] - 1, t[2][1]))
            continue
        if last_newline and (not parens):
            logical_start.append((t[2][0] - 1, t[2][1]))
            last_newline = False
        if t[0] == tokenize.OP:
            if t[1] in '([{':
                parens += 1
            elif t[1] in '}])':
                parens -= 1
    return (logical_start, logical_end)

def _get_logical(source_lines, result, logical_start, logical_end):
    if False:
        while True:
            i = 10
    'Return the logical line corresponding to the result.\n\n    Assumes input is already E702-clean.\n\n    '
    row = result['line'] - 1
    col = result['column'] - 1
    ls = None
    le = None
    for i in range(0, len(logical_start), 1):
        assert logical_end
        x = logical_end[i]
        if x[0] > row or (x[0] == row and x[1] > col):
            le = x
            ls = logical_start[i]
            break
    if ls is None:
        return None
    original = source_lines[ls[0]:le[0] + 1]
    return (ls, le, original)

def get_item(items, index, default=None):
    if False:
        while True:
            i = 10
    if 0 <= index < len(items):
        return items[index]
    return default

def reindent(source, indent_size, leave_tabs=False):
    if False:
        while True:
            i = 10
    'Reindent all lines.'
    reindenter = Reindenter(source, leave_tabs)
    return reindenter.run(indent_size)

def code_almost_equal(a, b):
    if False:
        i = 10
        return i + 15
    'Return True if code is similar.\n\n    Ignore whitespace when comparing specific line.\n\n    '
    split_a = split_and_strip_non_empty_lines(a)
    split_b = split_and_strip_non_empty_lines(b)
    if len(split_a) != len(split_b):
        return False
    for (index, _) in enumerate(split_a):
        if ''.join(split_a[index].split()) != ''.join(split_b[index].split()):
            return False
    return True

def split_and_strip_non_empty_lines(text):
    if False:
        while True:
            i = 10
    'Return lines split by newline.\n\n    Ignore empty lines.\n\n    '
    return [line.strip() for line in text.splitlines() if line.strip()]

def refactor(source, fixer_names, ignore=None, filename=''):
    if False:
        while True:
            i = 10
    'Return refactored code using lib2to3.\n\n    Skip if ignore string is produced in the refactored code.\n\n    '
    not_found_end_of_file_newline = source and source.rstrip('\r\n') == source
    if not_found_end_of_file_newline:
        input_source = source + '\n'
    else:
        input_source = source
    from lib2to3 import pgen2
    try:
        new_text = refactor_with_2to3(input_source, fixer_names=fixer_names, filename=filename)
    except (pgen2.parse.ParseError, SyntaxError, UnicodeDecodeError, UnicodeEncodeError):
        return source
    if ignore:
        if ignore in new_text and ignore not in source:
            return source
    if not_found_end_of_file_newline:
        return new_text.rstrip('\r\n')
    return new_text

def code_to_2to3(select, ignore, where='', verbose=False):
    if False:
        return 10
    fixes = set()
    for (code, fix) in CODE_TO_2TO3.items():
        if code_match(code, select=select, ignore=ignore):
            if verbose:
                print('--->  Applying {} fix for {}'.format(where, code.upper()), file=sys.stderr)
            fixes |= set(fix)
    return fixes

def fix_2to3(source, aggressive=True, select=None, ignore=None, filename='', where='global', verbose=False):
    if False:
        return 10
    'Fix various deprecated code (via lib2to3).'
    if not aggressive:
        return source
    select = select or []
    ignore = ignore or []
    return refactor(source, code_to_2to3(select=select, ignore=ignore, where=where, verbose=verbose), filename=filename)

def find_newline(source):
    if False:
        print('Hello World!')
    'Return type of newline used in source.\n\n    Input is a list of lines.\n\n    '
    assert not isinstance(source, str)
    counter = collections.defaultdict(int)
    for line in source:
        if line.endswith(CRLF):
            counter[CRLF] += 1
        elif line.endswith(CR):
            counter[CR] += 1
        elif line.endswith(LF):
            counter[LF] += 1
    return (sorted(counter, key=counter.get, reverse=True) or [LF])[0]

def _get_indentword(source):
    if False:
        i = 10
        return i + 15
    'Return indentation type.'
    indent_word = '    '
    try:
        for t in generate_tokens(source):
            if t[0] == token.INDENT:
                indent_word = t[1]
                break
    except (SyntaxError, tokenize.TokenError):
        pass
    return indent_word

def _get_indentation(line):
    if False:
        i = 10
        return i + 15
    'Return leading whitespace.'
    if line.strip():
        non_whitespace_index = len(line) - len(line.lstrip())
        return line[:non_whitespace_index]
    return ''

def get_diff_text(old, new, filename):
    if False:
        i = 10
        return i + 15
    'Return text of unified diff between old and new.'
    newline = '\n'
    diff = difflib.unified_diff(old, new, 'original/' + filename, 'fixed/' + filename, lineterm=newline)
    text = ''
    for line in diff:
        text += line
        if text and (not line.endswith(newline)):
            text += newline + '\\ No newline at end of file' + newline
    return text

def _priority_key(pep8_result):
    if False:
        return 10
    'Key for sorting PEP8 results.\n\n    Global fixes should be done first. This is important for things like\n    indentation.\n\n    '
    priority = ['e701', 'e702', 'e225', 'e231', 'e201', 'e262']
    middle_index = 10000
    lowest_priority = ['e501']
    key = pep8_result['id'].lower()
    try:
        return priority.index(key)
    except ValueError:
        try:
            return middle_index + lowest_priority.index(key) + 1
        except ValueError:
            return middle_index

def shorten_line(tokens, source, indentation, indent_word, max_line_length, aggressive=False, experimental=False, previous_line=''):
    if False:
        return 10
    'Separate line at OPERATOR.\n\n    Multiple candidates will be yielded.\n\n    '
    for candidate in _shorten_line(tokens=tokens, source=source, indentation=indentation, indent_word=indent_word, aggressive=aggressive, previous_line=previous_line):
        yield candidate
    if aggressive:
        for key_token_strings in SHORTEN_OPERATOR_GROUPS:
            shortened = _shorten_line_at_tokens(tokens=tokens, source=source, indentation=indentation, indent_word=indent_word, key_token_strings=key_token_strings, aggressive=aggressive)
            if shortened is not None and shortened != source:
                yield shortened
    if experimental:
        for shortened in _shorten_line_at_tokens_new(tokens=tokens, source=source, indentation=indentation, max_line_length=max_line_length):
            yield shortened

def _shorten_line(tokens, source, indentation, indent_word, aggressive=False, previous_line=''):
    if False:
        for i in range(10):
            print('nop')
    'Separate line at OPERATOR.\n\n    The input is expected to be free of newlines except for inside multiline\n    strings and at the end.\n\n    Multiple candidates will be yielded.\n\n    '
    for (token_type, token_string, start_offset, end_offset) in token_offsets(tokens):
        if token_type == tokenize.COMMENT and (not is_probably_part_of_multiline(previous_line)) and (not is_probably_part_of_multiline(source)) and (not source[start_offset + 1:].strip().lower().startswith(('noqa', 'pragma:', 'pylint:'))):
            first = source[:start_offset]
            second = source[start_offset:]
            yield (indentation + second.strip() + '\n' + indentation + first.strip() + '\n')
        elif token_type == token.OP and token_string != '=':
            assert token_type != token.INDENT
            first = source[:end_offset]
            second_indent = indentation
            if first.rstrip().endswith('(') and source[end_offset:].lstrip().startswith(')'):
                pass
            elif first.rstrip().endswith('('):
                second_indent += indent_word
            elif '(' in first:
                second_indent += ' ' * (1 + first.find('('))
            else:
                second_indent += indent_word
            second = second_indent + source[end_offset:].lstrip()
            if not second.strip() or second.lstrip().startswith('#'):
                continue
            if second.lstrip().startswith(','):
                continue
            if first.rstrip().endswith('.'):
                continue
            if token_string in '+-*/':
                fixed = first + ' \\' + '\n' + second
            else:
                fixed = first + '\n' + second
            if check_syntax(normalize_multiline(fixed) if aggressive else fixed):
                yield (indentation + fixed)

def _is_binary_operator(token_type, text):
    if False:
        return 10
    return (token_type == tokenize.OP or text in ['and', 'or']) and text not in '()[]{},:.;@=%~'
Token = collections.namedtuple('Token', ['token_type', 'token_string', 'spos', 'epos', 'line'])

class ReformattedLines(object):
    """The reflowed lines of atoms.

    Each part of the line is represented as an "atom." They can be moved
    around when need be to get the optimal formatting.

    """

    class _Indent(object):
        """Represent an indentation in the atom stream."""

        def __init__(self, indent_amt):
            if False:
                return 10
            self._indent_amt = indent_amt

        def emit(self):
            if False:
                while True:
                    i = 10
            return ' ' * self._indent_amt

        @property
        def size(self):
            if False:
                return 10
            return self._indent_amt

    class _Space(object):
        """Represent a space in the atom stream."""

        def emit(self):
            if False:
                return 10
            return ' '

        @property
        def size(self):
            if False:
                print('Hello World!')
            return 1

    class _LineBreak(object):
        """Represent a line break in the atom stream."""

        def emit(self):
            if False:
                return 10
            return '\n'

        @property
        def size(self):
            if False:
                while True:
                    i = 10
            return 0

    def __init__(self, max_line_length):
        if False:
            i = 10
            return i + 15
        self._max_line_length = max_line_length
        self._lines = []
        self._bracket_depth = 0
        self._prev_item = None
        self._prev_prev_item = None

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.emit()

    def add(self, obj, indent_amt, break_after_open_bracket):
        if False:
            i = 10
            return i + 15
        if isinstance(obj, Atom):
            self._add_item(obj, indent_amt)
            return
        self._add_container(obj, indent_amt, break_after_open_bracket)

    def add_comment(self, item):
        if False:
            while True:
                i = 10
        num_spaces = 2
        if len(self._lines) > 1:
            if isinstance(self._lines[-1], self._Space):
                num_spaces -= 1
            if len(self._lines) > 2:
                if isinstance(self._lines[-2], self._Space):
                    num_spaces -= 1
        while num_spaces > 0:
            self._lines.append(self._Space())
            num_spaces -= 1
        self._lines.append(item)

    def add_indent(self, indent_amt):
        if False:
            i = 10
            return i + 15
        self._lines.append(self._Indent(indent_amt))

    def add_line_break(self, indent):
        if False:
            return 10
        self._lines.append(self._LineBreak())
        self.add_indent(len(indent))

    def add_line_break_at(self, index, indent_amt):
        if False:
            for i in range(10):
                print('nop')
        self._lines.insert(index, self._LineBreak())
        self._lines.insert(index + 1, self._Indent(indent_amt))

    def add_space_if_needed(self, curr_text, equal=False):
        if False:
            print('Hello World!')
        if not self._lines or isinstance(self._lines[-1], (self._LineBreak, self._Indent, self._Space)):
            return
        prev_text = str(self._prev_item)
        prev_prev_text = str(self._prev_prev_item) if self._prev_prev_item else ''
        if (self._prev_item.is_keyword or self._prev_item.is_string or self._prev_item.is_name or self._prev_item.is_number) and (curr_text[0] not in '([{.,:}])' or (curr_text[0] == '=' and equal)) or ((prev_prev_text != 'from' and prev_text[-1] != '.' and (curr_text != 'import')) and curr_text[0] != ':' and (prev_text[-1] in '}])' and curr_text[0] not in '.,}])' or prev_text[-1] in ':,' or (equal and prev_text == '=') or (self._prev_prev_item and (prev_text not in '+-' and (self._prev_prev_item.is_name or self._prev_prev_item.is_number or self._prev_prev_item.is_string)) and (prev_text in ('+', '-', '%', '*', '/', '//', '**', 'in'))))):
            self._lines.append(self._Space())

    def previous_item(self):
        if False:
            while True:
                i = 10
        'Return the previous non-whitespace item.'
        return self._prev_item

    def fits_on_current_line(self, item_extent):
        if False:
            print('Hello World!')
        return self.current_size() + item_extent <= self._max_line_length

    def current_size(self):
        if False:
            return 10
        'The size of the current line minus the indentation.'
        size = 0
        for item in reversed(self._lines):
            size += item.size
            if isinstance(item, self._LineBreak):
                break
        return size

    def line_empty(self):
        if False:
            while True:
                i = 10
        return self._lines and isinstance(self._lines[-1], (self._LineBreak, self._Indent))

    def emit(self):
        if False:
            print('Hello World!')
        string = ''
        for item in self._lines:
            if isinstance(item, self._LineBreak):
                string = string.rstrip()
            string += item.emit()
        return string.rstrip() + '\n'

    def _add_item(self, item, indent_amt):
        if False:
            for i in range(10):
                print('nop')
        'Add an item to the line.\n\n        Reflow the line to get the best formatting after the item is\n        inserted. The bracket depth indicates if the item is being\n        inserted inside of a container or not.\n\n        '
        if self._prev_item and self._prev_item.is_string and item.is_string:
            self._lines.append(self._LineBreak())
            self._lines.append(self._Indent(indent_amt))
        item_text = str(item)
        if self._lines and self._bracket_depth:
            self._prevent_default_initializer_splitting(item, indent_amt)
            if item_text in '.,)]}':
                self._split_after_delimiter(item, indent_amt)
        elif self._lines and (not self.line_empty()):
            if self.fits_on_current_line(len(item_text)):
                self._enforce_space(item)
            else:
                self._lines.append(self._LineBreak())
                self._lines.append(self._Indent(indent_amt))
        self._lines.append(item)
        (self._prev_item, self._prev_prev_item) = (item, self._prev_item)
        if item_text in '([{':
            self._bracket_depth += 1
        elif item_text in '}])':
            self._bracket_depth -= 1
            assert self._bracket_depth >= 0

    def _add_container(self, container, indent_amt, break_after_open_bracket):
        if False:
            for i in range(10):
                print('nop')
        actual_indent = indent_amt + 1
        if str(self._prev_item) != '=' and (not self.line_empty()) and (not self.fits_on_current_line(container.size + self._bracket_depth + 2)):
            if str(container)[0] == '(' and self._prev_item.is_name:
                break_after_open_bracket = True
                actual_indent = indent_amt + 4
            elif break_after_open_bracket or str(self._prev_item) not in '([{':
                self._lines.append(self._LineBreak())
                self._lines.append(self._Indent(indent_amt))
                break_after_open_bracket = False
        else:
            actual_indent = self.current_size() + 1
            break_after_open_bracket = False
        if isinstance(container, (ListComprehension, IfExpression)):
            actual_indent = indent_amt
        container.reflow(self, ' ' * actual_indent, break_after_open_bracket=break_after_open_bracket)

    def _prevent_default_initializer_splitting(self, item, indent_amt):
        if False:
            return 10
        "Prevent splitting between a default initializer.\n\n        When there is a default initializer, it's best to keep it all on\n        the same line. It's nicer and more readable, even if it goes\n        over the maximum allowable line length. This goes back along the\n        current line to determine if we have a default initializer, and,\n        if so, to remove extraneous whitespaces and add a line\n        break/indent before it if needed.\n\n        "
        if str(item) == '=':
            self._delete_whitespace()
            return
        if not self._prev_item or not self._prev_prev_item or str(self._prev_item) != '=':
            return
        self._delete_whitespace()
        prev_prev_index = self._lines.index(self._prev_prev_item)
        if isinstance(self._lines[prev_prev_index - 1], self._Indent) or self.fits_on_current_line(item.size + 1):
            return
        if isinstance(self._lines[prev_prev_index - 1], self._Space):
            del self._lines[prev_prev_index - 1]
        self.add_line_break_at(self._lines.index(self._prev_prev_item), indent_amt)

    def _split_after_delimiter(self, item, indent_amt):
        if False:
            i = 10
            return i + 15
        'Split the line only after a delimiter.'
        self._delete_whitespace()
        if self.fits_on_current_line(item.size):
            return
        last_space = None
        for current_item in reversed(self._lines):
            if last_space and (not isinstance(current_item, Atom) or not current_item.is_colon):
                break
            else:
                last_space = None
            if isinstance(current_item, self._Space):
                last_space = current_item
            if isinstance(current_item, (self._LineBreak, self._Indent)):
                return
        if not last_space:
            return
        self.add_line_break_at(self._lines.index(last_space), indent_amt)

    def _enforce_space(self, item):
        if False:
            return 10
        "Enforce a space in certain situations.\n\n        There are cases where we will want a space where normally we\n        wouldn't put one. This just enforces the addition of a space.\n\n        "
        if isinstance(self._lines[-1], (self._Space, self._LineBreak, self._Indent)):
            return
        if not self._prev_item:
            return
        item_text = str(item)
        prev_text = str(self._prev_item)
        if item_text == '.' and prev_text == 'from' or (item_text == 'import' and prev_text == '.') or (item_text == '(' and prev_text == 'import'):
            self._lines.append(self._Space())

    def _delete_whitespace(self):
        if False:
            i = 10
            return i + 15
        'Delete all whitespace from the end of the line.'
        while isinstance(self._lines[-1], (self._Space, self._LineBreak, self._Indent)):
            del self._lines[-1]

class Atom(object):
    """The smallest unbreakable unit that can be reflowed."""

    def __init__(self, atom):
        if False:
            for i in range(10):
                print('nop')
        self._atom = atom

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self._atom.token_string

    def __len__(self):
        if False:
            print('Hello World!')
        return self.size

    def reflow(self, reflowed_lines, continued_indent, extent, break_after_open_bracket=False, is_list_comp_or_if_expr=False, next_is_dot=False):
        if False:
            return 10
        if self._atom.token_type == tokenize.COMMENT:
            reflowed_lines.add_comment(self)
            return
        total_size = extent if extent else self.size
        if self._atom.token_string not in ',:([{}])':
            total_size += 1
        prev_item = reflowed_lines.previous_item()
        if not is_list_comp_or_if_expr and (not reflowed_lines.fits_on_current_line(total_size)) and (not (next_is_dot and reflowed_lines.fits_on_current_line(self.size + 1))) and (not reflowed_lines.line_empty()) and (not self.is_colon) and (not (prev_item and prev_item.is_name and (str(self) == '('))):
            reflowed_lines.add_line_break(continued_indent)
        else:
            reflowed_lines.add_space_if_needed(str(self))
        reflowed_lines.add(self, len(continued_indent), break_after_open_bracket)

    def emit(self):
        if False:
            while True:
                i = 10
        return self.__repr__()

    @property
    def is_keyword(self):
        if False:
            return 10
        return keyword.iskeyword(self._atom.token_string)

    @property
    def is_string(self):
        if False:
            print('Hello World!')
        return self._atom.token_type == tokenize.STRING

    @property
    def is_name(self):
        if False:
            while True:
                i = 10
        return self._atom.token_type == tokenize.NAME

    @property
    def is_number(self):
        if False:
            i = 10
            return i + 15
        return self._atom.token_type == tokenize.NUMBER

    @property
    def is_comma(self):
        if False:
            for i in range(10):
                print('nop')
        return self._atom.token_string == ','

    @property
    def is_colon(self):
        if False:
            return 10
        return self._atom.token_string == ':'

    @property
    def size(self):
        if False:
            print('Hello World!')
        return len(self._atom.token_string)

class Container(object):
    """Base class for all container types."""

    def __init__(self, items):
        if False:
            i = 10
            return i + 15
        self._items = items

    def __repr__(self):
        if False:
            return 10
        string = ''
        last_was_keyword = False
        for item in self._items:
            if item.is_comma:
                string += ', '
            elif item.is_colon:
                string += ': '
            else:
                item_string = str(item)
                if string and (last_was_keyword or (not string.endswith(tuple('([{,.:}]) ')) and (not item_string.startswith(tuple('([{,.:}])'))))):
                    string += ' '
                string += item_string
            last_was_keyword = item.is_keyword
        return string

    def __iter__(self):
        if False:
            print('Hello World!')
        for element in self._items:
            yield element

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        return self._items[idx]

    def reflow(self, reflowed_lines, continued_indent, break_after_open_bracket=False):
        if False:
            for i in range(10):
                print('nop')
        last_was_container = False
        for (index, item) in enumerate(self._items):
            next_item = get_item(self._items, index + 1)
            if isinstance(item, Atom):
                is_list_comp_or_if_expr = isinstance(self, (ListComprehension, IfExpression))
                item.reflow(reflowed_lines, continued_indent, self._get_extent(index), is_list_comp_or_if_expr=is_list_comp_or_if_expr, next_is_dot=next_item and str(next_item) == '.')
                if last_was_container and item.is_comma:
                    reflowed_lines.add_line_break(continued_indent)
                last_was_container = False
            else:
                reflowed_lines.add(item, len(continued_indent), break_after_open_bracket)
                last_was_container = not isinstance(item, (ListComprehension, IfExpression))
            if break_after_open_bracket and index == 0 and (str(item) == self.open_bracket) and (not next_item or str(next_item) != self.close_bracket) and (len(self._items) != 3 or not isinstance(next_item, Atom)):
                reflowed_lines.add_line_break(continued_indent)
                break_after_open_bracket = False
            else:
                next_next_item = get_item(self._items, index + 2)
                if str(item) not in ['.', '%', 'in'] and next_item and (not isinstance(next_item, Container)) and (str(next_item) != ':') and next_next_item and (not isinstance(next_next_item, Atom) or str(next_item) == 'not') and (not reflowed_lines.line_empty()) and (not reflowed_lines.fits_on_current_line(self._get_extent(index + 1) + 2)):
                    reflowed_lines.add_line_break(continued_indent)

    def _get_extent(self, index):
        if False:
            i = 10
            return i + 15
        'The extent of the full element.\n\n        E.g., the length of a function call or keyword.\n\n        '
        extent = 0
        prev_item = get_item(self._items, index - 1)
        seen_dot = prev_item and str(prev_item) == '.'
        while index < len(self._items):
            item = get_item(self._items, index)
            index += 1
            if isinstance(item, (ListComprehension, IfExpression)):
                break
            if isinstance(item, Container):
                if prev_item and prev_item.is_name:
                    if seen_dot:
                        extent += 1
                    else:
                        extent += item.size
                    prev_item = item
                    continue
            elif str(item) not in ['.', '=', ':', 'not'] and (not item.is_name) and (not item.is_string):
                break
            if str(item) == '.':
                seen_dot = True
            extent += item.size
            prev_item = item
        return extent

    @property
    def is_string(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    @property
    def size(self):
        if False:
            while True:
                i = 10
        return len(self.__repr__())

    @property
    def is_keyword(self):
        if False:
            return 10
        return False

    @property
    def is_name(self):
        if False:
            return 10
        return False

    @property
    def is_comma(self):
        if False:
            i = 10
            return i + 15
        return False

    @property
    def is_colon(self):
        if False:
            i = 10
            return i + 15
        return False

    @property
    def open_bracket(self):
        if False:
            return 10
        return None

    @property
    def close_bracket(self):
        if False:
            return 10
        return None

class Tuple(Container):
    """A high-level representation of a tuple."""

    @property
    def open_bracket(self):
        if False:
            for i in range(10):
                print('nop')
        return '('

    @property
    def close_bracket(self):
        if False:
            print('Hello World!')
        return ')'

class List(Container):
    """A high-level representation of a list."""

    @property
    def open_bracket(self):
        if False:
            return 10
        return '['

    @property
    def close_bracket(self):
        if False:
            for i in range(10):
                print('nop')
        return ']'

class DictOrSet(Container):
    """A high-level representation of a dictionary or set."""

    @property
    def open_bracket(self):
        if False:
            for i in range(10):
                print('nop')
        return '{'

    @property
    def close_bracket(self):
        if False:
            print('Hello World!')
        return '}'

class ListComprehension(Container):
    """A high-level representation of a list comprehension."""

    @property
    def size(self):
        if False:
            for i in range(10):
                print('nop')
        length = 0
        for item in self._items:
            if isinstance(item, IfExpression):
                break
            length += item.size
        return length

class IfExpression(Container):
    """A high-level representation of an if-expression."""

def _parse_container(tokens, index, for_or_if=None):
    if False:
        print('Hello World!')
    'Parse a high-level container, such as a list, tuple, etc.'
    items = [Atom(Token(*tokens[index]))]
    index += 1
    num_tokens = len(tokens)
    while index < num_tokens:
        tok = Token(*tokens[index])
        if tok.token_string in ',)]}':
            if for_or_if == 'for':
                return (ListComprehension(items), index - 1)
            elif for_or_if == 'if':
                return (IfExpression(items), index - 1)
            items.append(Atom(tok))
            if tok.token_string == ')':
                return (Tuple(items), index)
            elif tok.token_string == ']':
                return (List(items), index)
            elif tok.token_string == '}':
                return (DictOrSet(items), index)
        elif tok.token_string in '([{':
            (container, index) = _parse_container(tokens, index)
            items.append(container)
        elif tok.token_string == 'for':
            (container, index) = _parse_container(tokens, index, 'for')
            items.append(container)
        elif tok.token_string == 'if':
            (container, index) = _parse_container(tokens, index, 'if')
            items.append(container)
        else:
            items.append(Atom(tok))
        index += 1
    return (None, None)

def _parse_tokens(tokens):
    if False:
        i = 10
        return i + 15
    'Parse the tokens.\n\n    This converts the tokens into a form where we can manipulate them\n    more easily.\n\n    '
    index = 0
    parsed_tokens = []
    num_tokens = len(tokens)
    while index < num_tokens:
        tok = Token(*tokens[index])
        assert tok.token_type != token.INDENT
        if tok.token_type == tokenize.NEWLINE:
            break
        if tok.token_string in '([{':
            (container, index) = _parse_container(tokens, index)
            if not container:
                return None
            parsed_tokens.append(container)
        else:
            parsed_tokens.append(Atom(tok))
        index += 1
    return parsed_tokens

def _reflow_lines(parsed_tokens, indentation, max_line_length, start_on_prefix_line):
    if False:
        print('Hello World!')
    'Reflow the lines so that it looks nice.'
    if str(parsed_tokens[0]) == 'def':
        continued_indent = indentation + ' ' * 2 * DEFAULT_INDENT_SIZE
    else:
        continued_indent = indentation + ' ' * DEFAULT_INDENT_SIZE
    break_after_open_bracket = not start_on_prefix_line
    lines = ReformattedLines(max_line_length)
    lines.add_indent(len(indentation.lstrip('\r\n')))
    if not start_on_prefix_line:
        first_token = get_item(parsed_tokens, 0)
        second_token = get_item(parsed_tokens, 1)
        if first_token and second_token and (str(second_token)[0] == '(') and (len(indentation) + len(first_token) + 1 == len(continued_indent)):
            return None
    for item in parsed_tokens:
        lines.add_space_if_needed(str(item), equal=True)
        save_continued_indent = continued_indent
        if start_on_prefix_line and isinstance(item, Container):
            start_on_prefix_line = False
            continued_indent = ' ' * (lines.current_size() + 1)
        item.reflow(lines, continued_indent, break_after_open_bracket)
        continued_indent = save_continued_indent
    return lines.emit()

def _shorten_line_at_tokens_new(tokens, source, indentation, max_line_length):
    if False:
        for i in range(10):
            print('nop')
    'Shorten the line taking its length into account.\n\n    The input is expected to be free of newlines except for inside\n    multiline strings and at the end.\n\n    '
    yield (indentation + source)
    parsed_tokens = _parse_tokens(tokens)
    if parsed_tokens:
        fixed = _reflow_lines(parsed_tokens, indentation, max_line_length, start_on_prefix_line=True)
        if fixed and check_syntax(normalize_multiline(fixed.lstrip())):
            yield fixed
        fixed = _reflow_lines(parsed_tokens, indentation, max_line_length, start_on_prefix_line=False)
        if fixed and check_syntax(normalize_multiline(fixed.lstrip())):
            yield fixed

def _shorten_line_at_tokens(tokens, source, indentation, indent_word, key_token_strings, aggressive):
    if False:
        i = 10
        return i + 15
    'Separate line by breaking at tokens in key_token_strings.\n\n    The input is expected to be free of newlines except for inside\n    multiline strings and at the end.\n\n    '
    offsets = []
    for (index, _t) in enumerate(token_offsets(tokens)):
        (token_type, token_string, start_offset, end_offset) = _t
        assert token_type != token.INDENT
        if token_string in key_token_strings:
            unwanted_next_token = {'(': ')', '[': ']', '{': '}'}.get(token_string)
            if unwanted_next_token:
                if get_item(tokens, index + 1, default=[None, None])[1] == unwanted_next_token or get_item(tokens, index + 2, default=[None, None])[1] == unwanted_next_token:
                    continue
            if index > 2 and token_string == '(' and (tokens[index - 1][1] in ',(%['):
                continue
            if end_offset < len(source) - 1:
                offsets.append(end_offset)
        else:
            previous_token = get_item(tokens, index - 1)
            if token_type == tokenize.STRING and previous_token and (previous_token[0] == tokenize.STRING):
                offsets.append(start_offset)
    current_indent = None
    fixed = None
    for line in split_at_offsets(source, offsets):
        if fixed:
            fixed += '\n' + current_indent + line
            for symbol in '([{':
                if line.endswith(symbol):
                    current_indent += indent_word
        else:
            fixed = line
            assert not current_indent
            current_indent = indent_word
    assert fixed is not None
    if check_syntax(normalize_multiline(fixed) if aggressive > 1 else fixed):
        return indentation + fixed
    return None

def token_offsets(tokens):
    if False:
        i = 10
        return i + 15
    'Yield tokens and offsets.'
    end_offset = 0
    previous_end_row = 0
    previous_end_column = 0
    for t in tokens:
        token_type = t[0]
        token_string = t[1]
        (start_row, start_column) = t[2]
        (end_row, end_column) = t[3]
        end_offset += start_column
        if previous_end_row == start_row:
            end_offset -= previous_end_column
        start_offset = end_offset
        end_offset += len(token_string)
        yield (token_type, token_string, start_offset, end_offset)
        previous_end_row = end_row
        previous_end_column = end_column

def normalize_multiline(line):
    if False:
        for i in range(10):
            print('nop')
    'Normalize multiline-related code that will cause syntax error.\n\n    This is for purposes of checking syntax.\n\n    '
    if line.startswith('def ') and line.rstrip().endswith(':'):
        return line + ' pass'
    elif line.startswith('return '):
        return 'def _(): ' + line
    elif line.startswith('@'):
        return line + 'def _(): pass'
    elif line.startswith('class '):
        return line + ' pass'
    elif line.startswith(('if ', 'elif ', 'for ', 'while ')):
        return line + ' pass'
    return line

def fix_whitespace(line, offset, replacement):
    if False:
        i = 10
        return i + 15
    'Replace whitespace at offset and return fixed line.'
    left = line[:offset].rstrip('\n\r \t\\')
    right = line[offset:].lstrip('\n\r \t\\')
    if right.startswith('#'):
        return line
    return left + replacement + right

def _execute_pep8(pep8_options, source):
    if False:
        i = 10
        return i + 15
    'Execute pycodestyle via python method calls.'

    class QuietReport(pycodestyle.BaseReport):
        """Version of checker that does not print."""

        def __init__(self, options):
            if False:
                return 10
            super(QuietReport, self).__init__(options)
            self.__full_error_results = []

        def error(self, line_number, offset, text, check):
            if False:
                print('Hello World!')
            'Collect errors.'
            code = super(QuietReport, self).error(line_number, offset, text, check)
            if code:
                self.__full_error_results.append({'id': code, 'line': line_number, 'column': offset + 1, 'info': text})

        def full_error_results(self):
            if False:
                i = 10
                return i + 15
            "Return error results in detail.\n\n            Results are in the form of a list of dictionaries. Each\n            dictionary contains 'id', 'line', 'column', and 'info'.\n\n            "
            return self.__full_error_results
    checker = pycodestyle.Checker('', lines=source, reporter=QuietReport, **pep8_options)
    checker.check_all()
    return checker.report.full_error_results()

def _remove_leading_and_normalize(line, with_rstrip=True):
    if False:
        for i in range(10):
            print('nop')
    if with_rstrip:
        return line.lstrip(' \t\x0b').rstrip(CR + LF) + '\n'
    return line.lstrip(' \t\x0b')

class Reindenter(object):
    """Reindents badly-indented code to uniformly use four-space indentation.

    Released to the public domain, by Tim Peters, 03 October 2000.

    """

    def __init__(self, input_text, leave_tabs=False):
        if False:
            print('Hello World!')
        sio = io.StringIO(input_text)
        source_lines = sio.readlines()
        self.string_content_line_numbers = multiline_string_lines(input_text)
        self.lines = []
        for (line_number, line) in enumerate(source_lines, start=1):
            if line_number in self.string_content_line_numbers:
                self.lines.append(line)
            else:
                with_rstrip = line_number != len(source_lines)
                if leave_tabs:
                    self.lines.append(_get_indentation(line) + _remove_leading_and_normalize(line, with_rstrip))
                else:
                    self.lines.append(_get_indentation(line).expandtabs() + _remove_leading_and_normalize(line, with_rstrip))
        self.lines.insert(0, None)
        self.index = 1
        self.input_text = input_text

    def run(self, indent_size=DEFAULT_INDENT_SIZE):
        if False:
            print('Hello World!')
        'Fix indentation and return modified line numbers.\n\n        Line numbers are indexed at 1.\n\n        '
        if indent_size < 1:
            return self.input_text
        try:
            stats = _reindent_stats(tokenize.generate_tokens(self.getline))
        except (SyntaxError, tokenize.TokenError):
            return self.input_text
        lines = self.lines
        stats.append((len(lines), 0))
        have2want = {}
        after = []
        i = stats[0][0]
        after.extend(lines[1:i])
        for i in range(len(stats) - 1):
            (thisstmt, thislevel) = stats[i]
            nextstmt = stats[i + 1][0]
            have = _leading_space_count(lines[thisstmt])
            want = thislevel * indent_size
            if want < 0:
                if have:
                    want = have2want.get(have, -1)
                    if want < 0:
                        for j in range(i + 1, len(stats) - 1):
                            (jline, jlevel) = stats[j]
                            if jlevel >= 0:
                                if have == _leading_space_count(lines[jline]):
                                    want = jlevel * indent_size
                                break
                    if want < 0:
                        for j in range(i - 1, -1, -1):
                            (jline, jlevel) = stats[j]
                            if jlevel >= 0:
                                want = have + _leading_space_count(after[jline - 1]) - _leading_space_count(lines[jline])
                                break
                    if want < 0:
                        want = have
                else:
                    want = 0
            assert want >= 0
            have2want[have] = want
            diff = want - have
            if diff == 0 or have == 0:
                after.extend(lines[thisstmt:nextstmt])
            else:
                for (line_number, line) in enumerate(lines[thisstmt:nextstmt], start=thisstmt):
                    if line_number in self.string_content_line_numbers:
                        after.append(line)
                    elif diff > 0:
                        if line == '\n':
                            after.append(line)
                        else:
                            after.append(' ' * diff + line)
                    else:
                        remove = min(_leading_space_count(line), -diff)
                        after.append(line[remove:])
        return ''.join(after)

    def getline(self):
        if False:
            i = 10
            return i + 15
        'Line-getter for tokenize.'
        if self.index >= len(self.lines):
            line = ''
        else:
            line = self.lines[self.index]
            self.index += 1
        return line

def _reindent_stats(tokens):
    if False:
        i = 10
        return i + 15
    "Return list of (lineno, indentlevel) pairs.\n\n    One for each stmt and comment line. indentlevel is -1 for comment\n    lines, as a signal that tokenize doesn't know what to do about them;\n    indeed, they're our headache!\n\n    "
    find_stmt = 1
    level = 0
    stats = []
    for t in tokens:
        token_type = t[0]
        sline = t[2][0]
        line = t[4]
        if token_type == tokenize.NEWLINE:
            find_stmt = 1
        elif token_type == tokenize.INDENT:
            find_stmt = 1
            level += 1
        elif token_type == tokenize.DEDENT:
            find_stmt = 1
            level -= 1
        elif token_type == tokenize.COMMENT:
            if find_stmt:
                stats.append((sline, -1))
        elif token_type == tokenize.NL:
            pass
        elif find_stmt:
            find_stmt = 0
            if line:
                stats.append((sline, level))
    return stats

def _leading_space_count(line):
    if False:
        while True:
            i = 10
    'Return number of leading spaces in line.'
    i = 0
    while i < len(line) and line[i] == ' ':
        i += 1
    return i

def refactor_with_2to3(source_text, fixer_names, filename=''):
    if False:
        print('Hello World!')
    'Use lib2to3 to refactor the source.\n\n    Return the refactored source code.\n\n    '
    from lib2to3.refactor import RefactoringTool
    fixers = ['lib2to3.fixes.fix_' + name for name in fixer_names]
    tool = RefactoringTool(fixer_names=fixers, explicit=fixers)
    from lib2to3.pgen2 import tokenize as lib2to3_tokenize
    try:
        return str(tool.refactor_string(source_text, name=filename))
    except lib2to3_tokenize.TokenError:
        return source_text

def check_syntax(code):
    if False:
        while True:
            i = 10
    'Return True if syntax is okay.'
    try:
        return compile(code, '<string>', 'exec', dont_inherit=True)
    except (SyntaxError, TypeError, ValueError):
        return False

def find_with_line_numbers(pattern, contents):
    if False:
        i = 10
        return i + 15
    "A wrapper around 're.finditer' to find line numbers.\n\n    Returns a list of line numbers where pattern was found in contents.\n    "
    matches = list(re.finditer(pattern, contents))
    if not matches:
        return []
    end = matches[-1].start()
    newline_offsets = {-1: 0}
    for (line_num, m) in enumerate(re.finditer('\\n', contents), 1):
        offset = m.start()
        if offset > end:
            break
        newline_offsets[offset] = line_num

    def get_line_num(match, contents):
        if False:
            i = 10
            return i + 15
        'Get the line number of string in a files contents.\n\n        Failing to find the newline is OK, -1 maps to 0\n\n        '
        newline_offset = contents.rfind('\n', 0, match.start())
        return newline_offsets[newline_offset]
    return [get_line_num(match, contents) + 1 for match in matches]

def get_disabled_ranges(source):
    if False:
        return 10
    'Returns a list of tuples representing the disabled ranges.\n\n    If disabled and no re-enable will disable for rest of file.\n\n    '
    enable_line_nums = find_with_line_numbers(ENABLE_REGEX, source)
    disable_line_nums = find_with_line_numbers(DISABLE_REGEX, source)
    total_lines = len(re.findall('\n', source)) + 1
    enable_commands = {}
    for num in enable_line_nums:
        enable_commands[num] = True
    for num in disable_line_nums:
        enable_commands[num] = False
    disabled_ranges = []
    currently_enabled = True
    disabled_start = None
    for (line, commanded_enabled) in sorted(enable_commands.items()):
        if commanded_enabled is False and currently_enabled is True:
            disabled_start = line
            currently_enabled = False
        elif commanded_enabled is True and currently_enabled is False:
            disabled_ranges.append((disabled_start, line))
            currently_enabled = True
    if currently_enabled is False:
        disabled_ranges.append((disabled_start, total_lines))
    return disabled_ranges

def filter_disabled_results(result, disabled_ranges):
    if False:
        return 10
    'Filter out reports based on tuple of disabled ranges.\n\n    '
    line = result['line']
    for disabled_range in disabled_ranges:
        if disabled_range[0] <= line <= disabled_range[1]:
            return False
    return True

def filter_results(source, results, aggressive):
    if False:
        while True:
            i = 10
    'Filter out spurious reports from pycodestyle.\n\n    If aggressive is True, we allow possibly unsafe fixes (E711, E712).\n\n    '
    non_docstring_string_line_numbers = multiline_string_lines(source, include_docstrings=False)
    all_string_line_numbers = multiline_string_lines(source, include_docstrings=True)
    commented_out_code_line_numbers = commented_out_code_lines(source)
    disabled_ranges = get_disabled_ranges(source)
    if disabled_ranges:
        results = [result for result in results if filter_disabled_results(result, disabled_ranges)]
    has_e901 = any((result['id'].lower() == 'e901' for result in results))
    for r in results:
        issue_id = r['id'].lower()
        if r['line'] in non_docstring_string_line_numbers:
            if issue_id.startswith(('e1', 'e501', 'w191')):
                continue
        if r['line'] in all_string_line_numbers:
            if issue_id in ['e501']:
                continue
        if not aggressive and r['line'] + 1 in all_string_line_numbers:
            if issue_id.startswith(('w29', 'w39')):
                continue
        if aggressive <= 0:
            if issue_id.startswith(('e711', 'e72', 'w6')):
                continue
        if aggressive <= 1:
            if issue_id.startswith(('e712', 'e713', 'e714')):
                continue
        if aggressive <= 2:
            if issue_id.startswith('e704'):
                continue
        if r['line'] in commented_out_code_line_numbers:
            if issue_id.startswith(('e261', 'e262', 'e501')):
                continue
        if has_e901:
            if issue_id.startswith(('e1', 'e7')):
                continue
        yield r

def multiline_string_lines(source, include_docstrings=False):
    if False:
        return 10
    'Return line numbers that are within multiline strings.\n\n    The line numbers are indexed at 1.\n\n    Docstrings are ignored.\n\n    '
    line_numbers = set()
    previous_token_type = ''
    try:
        for t in generate_tokens(source):
            token_type = t[0]
            start_row = t[2][0]
            end_row = t[3][0]
            if token_type == tokenize.STRING and start_row != end_row:
                if include_docstrings or previous_token_type != tokenize.INDENT:
                    line_numbers |= set(range(1 + start_row, 1 + end_row))
            previous_token_type = token_type
    except (SyntaxError, tokenize.TokenError):
        pass
    return line_numbers

def commented_out_code_lines(source):
    if False:
        for i in range(10):
            print('nop')
    'Return line numbers of comments that are likely code.\n\n    Commented-out code is bad practice, but modifying it just adds even\n    more clutter.\n\n    '
    line_numbers = []
    try:
        for t in generate_tokens(source):
            token_type = t[0]
            token_string = t[1]
            start_row = t[2][0]
            line = t[4]
            if not line.lstrip().startswith('#'):
                continue
            if token_type == tokenize.COMMENT:
                stripped_line = token_string.lstrip('#').strip()
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=SyntaxWarning)
                    if ' ' in stripped_line and '#' not in stripped_line and check_syntax(stripped_line):
                        line_numbers.append(start_row)
    except (SyntaxError, tokenize.TokenError):
        pass
    return line_numbers

def shorten_comment(line, max_line_length, last_comment=False):
    if False:
        return 10
    'Return trimmed or split long comment line.\n\n    If there are no comments immediately following it, do a text wrap.\n    Doing this wrapping on all comments in general would lead to jagged\n    comment text.\n\n    '
    assert len(line) > max_line_length
    line = line.rstrip()
    indentation = _get_indentation(line) + '# '
    max_line_length = min(max_line_length, len(indentation) + 72)
    MIN_CHARACTER_REPEAT = 5
    if len(line) - len(line.rstrip(line[-1])) >= MIN_CHARACTER_REPEAT and (not line[-1].isalnum()):
        return line[:max_line_length] + '\n'
    elif last_comment and re.match('\\s*#+\\s*\\w+', line):
        split_lines = textwrap.wrap(line.lstrip(' \t#'), initial_indent=indentation, subsequent_indent=indentation, width=max_line_length, break_long_words=False, break_on_hyphens=False)
        return '\n'.join(split_lines) + '\n'
    return line + '\n'

def normalize_line_endings(lines, newline):
    if False:
        for i in range(10):
            print('nop')
    'Return fixed line endings.\n\n    All lines will be modified to use the most common line ending.\n    '
    line = [line.rstrip('\n\r') + newline for line in lines]
    if line and lines[-1] == lines[-1].rstrip('\n\r'):
        line[-1] = line[-1].rstrip('\n\r')
    return line

def mutual_startswith(a, b):
    if False:
        while True:
            i = 10
    return b.startswith(a) or a.startswith(b)

def code_match(code, select, ignore):
    if False:
        print('Hello World!')
    if ignore:
        assert not isinstance(ignore, str)
        for ignored_code in [c.strip() for c in ignore]:
            if mutual_startswith(code.lower(), ignored_code.lower()):
                return False
    if select:
        assert not isinstance(select, str)
        for selected_code in [c.strip() for c in select]:
            if mutual_startswith(code.lower(), selected_code.lower()):
                return True
        return False
    return True

def fix_code(source, options=None, encoding=None, apply_config=False):
    if False:
        i = 10
        return i + 15
    'Return fixed source code.\n\n    "encoding" will be used to decode "source" if it is a byte string.\n\n    '
    options = _get_options(options, apply_config)
    options.ignore = [opt.upper() for opt in options.ignore]
    options.select = [opt.upper() for opt in options.select]
    ignore_opt = options.ignore
    if not {'W50', 'W503', 'W504'} & set(ignore_opt):
        options.ignore.append('W50')
    if not isinstance(source, str):
        source = source.decode(encoding or get_encoding())
    sio = io.StringIO(source)
    return fix_lines(sio.readlines(), options=options)

def _get_options(raw_options, apply_config):
    if False:
        while True:
            i = 10
    'Return parsed options.'
    if not raw_options:
        return parse_args([''], apply_config=apply_config)
    if isinstance(raw_options, dict):
        options = parse_args([''], apply_config=apply_config)
        for (name, value) in raw_options.items():
            if not hasattr(options, name):
                raise ValueError("No such option '{}'".format(name))
            expected_type = type(getattr(options, name))
            if not isinstance(expected_type, (str,)):
                if isinstance(value, (str,)):
                    raise ValueError("Option '{}' should not be a string".format(name))
            setattr(options, name, value)
    else:
        options = raw_options
    return options

def fix_lines(source_lines, options, filename=''):
    if False:
        return 10
    'Return fixed source code.'
    original_newline = find_newline(source_lines)
    tmp_source = ''.join(normalize_line_endings(source_lines, '\n'))
    previous_hashes = set()
    if options.line_range:
        fixed_source = tmp_source
    else:
        fixed_source = apply_global_fixes(tmp_source, options, filename=filename)
    passes = 0
    long_line_ignore_cache = set()
    while hash(fixed_source) not in previous_hashes:
        if options.pep8_passes >= 0 and passes > options.pep8_passes:
            break
        passes += 1
        previous_hashes.add(hash(fixed_source))
        tmp_source = copy.copy(fixed_source)
        fix = FixPEP8(filename, options, contents=tmp_source, long_line_ignore_cache=long_line_ignore_cache)
        fixed_source = fix.fix()
    sio = io.StringIO(fixed_source)
    return ''.join(normalize_line_endings(sio.readlines(), original_newline))

def fix_file(filename, options=None, output=None, apply_config=False):
    if False:
        while True:
            i = 10
    if not options:
        options = parse_args([filename], apply_config=apply_config)
    original_source = readlines_from_file(filename)
    fixed_source = original_source
    if options.in_place or options.diff or output:
        encoding = detect_encoding(filename)
    if output:
        output = LineEndingWrapper(wrap_output(output, encoding=encoding))
    fixed_source = fix_lines(fixed_source, options, filename=filename)
    if options.diff:
        new = io.StringIO(fixed_source)
        new = new.readlines()
        diff = get_diff_text(original_source, new, filename)
        if output:
            output.write(diff)
            output.flush()
        elif options.jobs > 1:
            diff = diff.encode(encoding)
        return diff
    elif options.in_place:
        original = ''.join(original_source).splitlines()
        fixed = fixed_source.splitlines()
        original_source_last_line = original_source[-1].split('\n')[-1] if original_source else ''
        fixed_source_last_line = fixed_source.split('\n')[-1]
        if original != fixed or original_source_last_line != fixed_source_last_line:
            with open_with_encoding(filename, 'w', encoding=encoding) as fp:
                fp.write(fixed_source)
            return fixed_source
        return None
    elif output:
        output.write(fixed_source)
        output.flush()
    return fixed_source

def global_fixes():
    if False:
        while True:
            i = 10
    'Yield multiple (code, function) tuples.'
    for function in list(globals().values()):
        if inspect.isfunction(function):
            arguments = _get_parameters(function)
            if arguments[:1] != ['source']:
                continue
            code = extract_code_from_function(function)
            if code:
                yield (code, function)

def _get_parameters(function):
    if False:
        while True:
            i = 10
    if sys.version_info.major >= 3:
        if inspect.ismethod(function):
            function = function.__func__
        return list(inspect.signature(function).parameters)
    else:
        return inspect.getargspec(function)[0]

def apply_global_fixes(source, options, where='global', filename='', codes=None):
    if False:
        while True:
            i = 10
    'Run global fixes on source code.\n\n    These are fixes that only need be done once (unlike those in\n    FixPEP8, which are dependent on pycodestyle).\n\n    '
    if codes is None:
        codes = []
    if any((code_match(code, select=options.select, ignore=options.ignore) for code in ['E101', 'E111'])):
        source = reindent(source, indent_size=options.indent_size, leave_tabs=not code_match('W191', select=options.select, ignore=options.ignore))
    for (code, function) in global_fixes():
        if code_match(code, select=options.select, ignore=options.ignore):
            if options.verbose:
                print('--->  Applying {} fix for {}'.format(where, code.upper()), file=sys.stderr)
            source = function(source, aggressive=options.aggressive)
    source = fix_2to3(source, aggressive=options.aggressive, select=options.select, ignore=options.ignore, filename=filename, where=where, verbose=options.verbose)
    return source

def extract_code_from_function(function):
    if False:
        i = 10
        return i + 15
    'Return code handled by function.'
    if not function.__name__.startswith('fix_'):
        return None
    code = re.sub('^fix_', '', function.__name__)
    if not code:
        return None
    try:
        int(code[1:])
    except ValueError:
        return None
    return code

def _get_package_version():
    if False:
        while True:
            i = 10
    packages = ['pycodestyle: {}'.format(pycodestyle.__version__)]
    return ', '.join(packages)

def create_parser():
    if False:
        for i in range(10):
            print('nop')
    'Return command-line parser.'
    parser = argparse.ArgumentParser(description=docstring_summary(__doc__), prog='autopep8')
    parser.add_argument('--version', action='version', version='%(prog)s {} ({})'.format(__version__, _get_package_version()))
    parser.add_argument('-v', '--verbose', action='count', default=0, help='print verbose messages; multiple -v result in more verbose messages')
    parser.add_argument('-d', '--diff', action='store_true', help='print the diff for the fixed source')
    parser.add_argument('-i', '--in-place', action='store_true', help='make changes to files in place')
    parser.add_argument('--global-config', metavar='filename', default=DEFAULT_CONFIG, help='path to a global pep8 config file; if this file does not exist then this is ignored (default: {})'.format(DEFAULT_CONFIG))
    parser.add_argument('--ignore-local-config', action='store_true', help="don't look for and apply local config files; if not passed, defaults are updated with any config files in the project's root directory")
    parser.add_argument('-r', '--recursive', action='store_true', help='run recursively over directories; must be used with --in-place or --diff')
    parser.add_argument('-j', '--jobs', type=int, metavar='n', default=1, help='number of parallel jobs; match CPU count if value is less than 1')
    parser.add_argument('-p', '--pep8-passes', metavar='n', default=-1, type=int, help='maximum number of additional pep8 passes (default: infinite)')
    parser.add_argument('-a', '--aggressive', action='count', default=0, help='enable non-whitespace changes; multiple -a result in more aggressive changes')
    parser.add_argument('--experimental', action='store_true', help='enable experimental fixes')
    parser.add_argument('--exclude', metavar='globs', help='exclude file/directory names that match these comma-separated globs')
    parser.add_argument('--list-fixes', action='store_true', help='list codes for fixes; used by --ignore and --select')
    parser.add_argument('--ignore', metavar='errors', default='', help='do not fix these errors/warnings (default: {})'.format(DEFAULT_IGNORE))
    parser.add_argument('--select', metavar='errors', default='', help='fix only these errors/warnings (e.g. E4,W)')
    parser.add_argument('--max-line-length', metavar='n', default=79, type=int, help='set maximum allowed line length (default: %(default)s)')
    parser.add_argument('--line-range', '--range', metavar='line', default=None, type=int, nargs=2, help='only fix errors found within this inclusive range of line numbers (e.g. 1 99); line numbers are indexed at 1')
    parser.add_argument('--indent-size', default=DEFAULT_INDENT_SIZE, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--hang-closing', action='store_true', help='hang-closing option passed to pycodestyle')
    parser.add_argument('--exit-code', action='store_true', help='change to behavior of exit code. default behavior of return value, 0 is no differences, 1 is error exit. return 2 when add this option. 2 is exists differences.')
    parser.add_argument('files', nargs='*', help="files to format or '-' for standard in")
    return parser

def _expand_codes(codes, ignore_codes):
    if False:
        while True:
            i = 10
    'expand to individual E/W codes'
    ret = set()
    is_conflict = False
    if all((any((conflicting_code.startswith(code) for code in codes)) for conflicting_code in CONFLICTING_CODES)):
        is_conflict = True
    is_ignore_w503 = 'W503' in ignore_codes
    is_ignore_w504 = 'W504' in ignore_codes
    for code in codes:
        if code == 'W':
            if is_ignore_w503 and is_ignore_w504:
                ret.update({'W1', 'W2', 'W3', 'W505', 'W6'})
            elif is_ignore_w503:
                ret.update({'W1', 'W2', 'W3', 'W504', 'W505', 'W6'})
            else:
                ret.update({'W1', 'W2', 'W3', 'W503', 'W505', 'W6'})
        elif code in ('W5', 'W50'):
            if is_ignore_w503 and is_ignore_w504:
                ret.update({'W505'})
            elif is_ignore_w503:
                ret.update({'W504', 'W505'})
            else:
                ret.update({'W503', 'W505'})
        elif not (code in ('W503', 'W504') and is_conflict):
            ret.add(code)
    return ret

def parse_args(arguments, apply_config=False):
    if False:
        return 10
    'Parse command-line options.'
    parser = create_parser()
    args = parser.parse_args(arguments)
    if not args.files and (not args.list_fixes):
        parser.exit(EXIT_CODE_ARGPARSE_ERROR, 'incorrect number of arguments')
    args.files = [decode_filename(name) for name in args.files]
    if apply_config:
        parser = read_config(args, parser)
        try:
            parser_with_pyproject_toml = read_pyproject_toml(args, parser)
        except Exception:
            parser_with_pyproject_toml = None
        if parser_with_pyproject_toml:
            parser = parser_with_pyproject_toml
        args = parser.parse_args(arguments)
        args.files = [decode_filename(name) for name in args.files]
    if '-' in args.files:
        if len(args.files) > 1:
            parser.exit(EXIT_CODE_ARGPARSE_ERROR, 'cannot mix stdin and regular files')
        if args.diff:
            parser.exit(EXIT_CODE_ARGPARSE_ERROR, '--diff cannot be used with standard input')
        if args.in_place:
            parser.exit(EXIT_CODE_ARGPARSE_ERROR, '--in-place cannot be used with standard input')
        if args.recursive:
            parser.exit(EXIT_CODE_ARGPARSE_ERROR, '--recursive cannot be used with standard input')
    if len(args.files) > 1 and (not (args.in_place or args.diff)):
        parser.exit(EXIT_CODE_ARGPARSE_ERROR, 'autopep8 only takes one filename as argument unless the "--in-place" or "--diff" args are used')
    if args.recursive and (not (args.in_place or args.diff)):
        parser.exit(EXIT_CODE_ARGPARSE_ERROR, '--recursive must be used with --in-place or --diff')
    if args.in_place and args.diff:
        parser.exit(EXIT_CODE_ARGPARSE_ERROR, '--in-place and --diff are mutually exclusive')
    if args.max_line_length <= 0:
        parser.exit(EXIT_CODE_ARGPARSE_ERROR, '--max-line-length must be greater than 0')
    if args.indent_size <= 0:
        parser.exit(EXIT_CODE_ARGPARSE_ERROR, '--indent-size must be greater than 0')
    if args.select:
        args.select = _expand_codes(_split_comma_separated(args.select), _split_comma_separated(args.ignore) if args.ignore else [])
    if args.ignore:
        args.ignore = _split_comma_separated(args.ignore)
        if all((not any((conflicting_code.startswith(ignore_code) for ignore_code in args.ignore)) for conflicting_code in CONFLICTING_CODES)):
            args.ignore.update(CONFLICTING_CODES)
    elif not args.select:
        if args.aggressive:
            args.select = {'E', 'W1', 'W2', 'W3', 'W6'}
        else:
            args.ignore = _split_comma_separated(DEFAULT_IGNORE)
    if args.exclude:
        args.exclude = _split_comma_separated(args.exclude)
    else:
        args.exclude = {}
    if args.jobs < 1:
        import multiprocessing
        args.jobs = multiprocessing.cpu_count()
    if args.jobs > 1 and (not (args.in_place or args.diff)):
        parser.exit(EXIT_CODE_ARGPARSE_ERROR, 'parallel jobs requires --in-place')
    if args.line_range:
        if args.line_range[0] <= 0:
            parser.exit(EXIT_CODE_ARGPARSE_ERROR, '--range must be positive numbers')
        if args.line_range[0] > args.line_range[1]:
            parser.exit(EXIT_CODE_ARGPARSE_ERROR, 'First value of --range should be less than or equal to the second')
    return args

def _get_normalize_options(args, config, section, option_list):
    if False:
        while True:
            i = 10
    for (k, v) in config.items(section):
        norm_opt = k.lstrip('-').replace('-', '_')
        if not option_list.get(norm_opt):
            continue
        opt_type = option_list[norm_opt]
        if opt_type is int:
            if v.strip() == 'auto':
                if args.verbose:
                    print(f'ignore config: {k}={v}')
                continue
            value = config.getint(section, k)
        elif opt_type is bool:
            value = config.getboolean(section, k)
        else:
            value = config.get(section, k)
        yield (norm_opt, k, value)

def read_config(args, parser):
    if False:
        print('Hello World!')
    'Read both user configuration and local configuration.'
    config = SafeConfigParser()
    try:
        if args.verbose and os.path.exists(args.global_config):
            print('read config path: {}'.format(args.global_config))
        config.read(args.global_config)
        if not args.ignore_local_config:
            parent = tail = args.files and os.path.abspath(os.path.commonprefix(args.files))
            while tail:
                if config.read([os.path.join(parent, fn) for fn in PROJECT_CONFIG]):
                    if args.verbose:
                        for fn in PROJECT_CONFIG:
                            config_file = os.path.join(parent, fn)
                            if not os.path.exists(config_file):
                                continue
                            print('read config path: {}'.format(os.path.join(parent, fn)))
                    break
                (parent, tail) = os.path.split(parent)
        defaults = {}
        option_list = {o.dest: o.type or type(o.default) for o in parser._actions}
        for section in ['pep8', 'pycodestyle', 'flake8']:
            if not config.has_section(section):
                continue
            for (norm_opt, k, value) in _get_normalize_options(args, config, section, option_list):
                if args.verbose:
                    print('enable config: section={}, key={}, value={}'.format(section, k, value))
                defaults[norm_opt] = value
        parser.set_defaults(**defaults)
    except Error:
        pass
    return parser

def read_pyproject_toml(args, parser):
    if False:
        while True:
            i = 10
    'Read pyproject.toml and load configuration.'
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    config = None
    if os.path.exists(args.global_config):
        with open(args.global_config, 'rb') as fp:
            config = tomllib.load(fp)
    if not args.ignore_local_config:
        parent = tail = args.files and os.path.abspath(os.path.commonprefix(args.files))
        while tail:
            pyproject_toml = os.path.join(parent, 'pyproject.toml')
            if os.path.exists(pyproject_toml):
                with open(pyproject_toml, 'rb') as fp:
                    config = tomllib.load(fp)
                    break
            (parent, tail) = os.path.split(parent)
    if not config:
        return None
    if config.get('tool', {}).get('autopep8') is None:
        return None
    config = config.get('tool').get('autopep8')
    defaults = {}
    option_list = {o.dest: o.type or type(o.default) for o in parser._actions}
    TUPLED_OPTIONS = ('ignore', 'select')
    for (k, v) in config.items():
        norm_opt = k.lstrip('-').replace('-', '_')
        if not option_list.get(norm_opt):
            continue
        if type(v) in (list, tuple) and norm_opt in TUPLED_OPTIONS:
            value = ','.join(v)
        else:
            value = v
        if args.verbose:
            print('enable pyproject.toml config: key={}, value={}'.format(k, value))
        defaults[norm_opt] = value
    if defaults:
        parser.set_defaults(**defaults)
    return parser

def _split_comma_separated(string):
    if False:
        for i in range(10):
            print('nop')
    'Return a set of strings.'
    return {text.strip() for text in string.split(',') if text.strip()}

def decode_filename(filename):
    if False:
        print('Hello World!')
    'Return Unicode filename.'
    if isinstance(filename, str):
        return filename
    return filename.decode(sys.getfilesystemencoding())

def supported_fixes():
    if False:
        for i in range(10):
            print('nop')
    'Yield pep8 error codes that autopep8 fixes.\n\n    Each item we yield is a tuple of the code followed by its\n    description.\n\n    '
    yield ('E101', docstring_summary(reindent.__doc__))
    instance = FixPEP8(filename=None, options=None, contents='')
    for attribute in dir(instance):
        code = re.match('fix_([ew][0-9][0-9][0-9])', attribute)
        if code:
            yield (code.group(1).upper(), re.sub('\\s+', ' ', docstring_summary(getattr(instance, attribute).__doc__)))
    for (code, function) in sorted(global_fixes()):
        yield (code.upper() + (4 - len(code)) * ' ', re.sub('\\s+', ' ', docstring_summary(function.__doc__)))
    for code in sorted(CODE_TO_2TO3):
        yield (code.upper() + (4 - len(code)) * ' ', re.sub('\\s+', ' ', docstring_summary(fix_2to3.__doc__)))

def docstring_summary(docstring):
    if False:
        print('Hello World!')
    'Return summary of docstring.'
    return docstring.split('\n')[0] if docstring else ''

def line_shortening_rank(candidate, indent_word, max_line_length, experimental=False):
    if False:
        for i in range(10):
            print('nop')
    'Return rank of candidate.\n\n    This is for sorting candidates.\n\n    '
    if not candidate.strip():
        return 0
    rank = 0
    lines = candidate.rstrip().split('\n')
    offset = 0
    if not lines[0].lstrip().startswith('#') and lines[0].rstrip()[-1] not in '([{':
        for (opening, closing) in ('()', '[]', '{}'):
            opening_loc = lines[0].find(opening)
            closing_loc = lines[0].find(closing)
            if opening_loc >= 0:
                if closing_loc < 0 or closing_loc != opening_loc + 1:
                    offset = max(offset, 1 + opening_loc)
    current_longest = max((offset + len(x.strip()) for x in lines))
    rank += 4 * max(0, current_longest - max_line_length)
    rank += len(lines)
    rank += 2 * standard_deviation((len(line) for line in lines))
    bad_staring_symbol = {'(': ')', '[': ']', '{': '}'}.get(lines[0][-1])
    if len(lines) > 1:
        if bad_staring_symbol and lines[1].lstrip().startswith(bad_staring_symbol):
            rank += 20
    for (lineno, current_line) in enumerate(lines):
        current_line = current_line.strip()
        if current_line.startswith('#'):
            continue
        for bad_start in ['.', '%', '+', '-', '/']:
            if current_line.startswith(bad_start):
                rank += 100
            if current_line == bad_start:
                rank += 1000
        if current_line.endswith(('.', '%', '+', '-', '/')) and "': " in current_line:
            rank += 1000
        if current_line.endswith(('(', '[', '{', '.')):
            if len(current_line) <= len(indent_word):
                rank += 100
            if current_line.endswith('(') and current_line[:-1].rstrip().endswith(','):
                rank += 100
            if current_line.endswith('[') and len(current_line) > 1 and (current_line[-2].isalnum() or current_line[-2] in ']'):
                rank += 300
            if current_line.endswith('.'):
                rank += 100
            if has_arithmetic_operator(current_line):
                rank += 100
        if re.match('.*[(\\[{]\\s*[\\-\\+~]$', current_line.rstrip('\\ ')):
            rank += 1000
        if re.match('.*lambda\\s*\\*$', current_line.rstrip('\\ ')):
            rank += 1000
        if current_line.endswith(('%', '(', '[', '{')):
            rank -= 20
        if current_line.startswith('for '):
            rank -= 50
        if current_line.endswith('\\'):
            total_len = len(current_line)
            lineno += 1
            while lineno < len(lines):
                total_len += len(lines[lineno])
                if lines[lineno].lstrip().startswith('#'):
                    total_len = max_line_length
                    break
                if not lines[lineno].endswith('\\'):
                    break
                lineno += 1
            if total_len < max_line_length:
                rank += 10
            else:
                rank += 100 if experimental else 1
        if ',' in current_line and current_line.endswith(':'):
            rank += 10
        if current_line.endswith(':'):
            rank += 100
        rank += 10 * count_unbalanced_brackets(current_line)
    return max(0, rank)

def standard_deviation(numbers):
    if False:
        i = 10
        return i + 15
    'Return standard deviation.'
    numbers = list(numbers)
    if not numbers:
        return 0
    mean = sum(numbers) / len(numbers)
    return (sum(((n - mean) ** 2 for n in numbers)) / len(numbers)) ** 0.5

def has_arithmetic_operator(line):
    if False:
        i = 10
        return i + 15
    'Return True if line contains any arithmetic operators.'
    for operator in pycodestyle.ARITHMETIC_OP:
        if operator in line:
            return True
    return False

def count_unbalanced_brackets(line):
    if False:
        return 10
    'Return number of unmatched open/close brackets.'
    count = 0
    for (opening, closing) in ['()', '[]', '{}']:
        count += abs(line.count(opening) - line.count(closing))
    return count

def split_at_offsets(line, offsets):
    if False:
        return 10
    'Split line at offsets.\n\n    Return list of strings.\n\n    '
    result = []
    previous_offset = 0
    current_offset = 0
    for current_offset in sorted(offsets):
        if current_offset < len(line) and previous_offset != current_offset:
            result.append(line[previous_offset:current_offset].strip())
        previous_offset = current_offset
    result.append(line[current_offset:])
    return result

class LineEndingWrapper(object):
    """Replace line endings to work with sys.stdout.

    It seems that sys.stdout expects only '\\n' as the line ending, no matter
    the platform. Otherwise, we get repeated line endings.

    """

    def __init__(self, output):
        if False:
            i = 10
            return i + 15
        self.__output = output

    def write(self, s):
        if False:
            i = 10
            return i + 15
        self.__output.write(s.replace('\r\n', '\n').replace('\r', '\n'))

    def flush(self):
        if False:
            return 10
        self.__output.flush()

def match_file(filename, exclude):
    if False:
        print('Hello World!')
    'Return True if file is okay for modifying/recursing.'
    base_name = os.path.basename(filename)
    if base_name.startswith('.'):
        return False
    for pattern in exclude:
        if fnmatch.fnmatch(base_name, pattern):
            return False
        if fnmatch.fnmatch(filename, pattern):
            return False
    if not os.path.isdir(filename) and (not is_python_file(filename)):
        return False
    return True

def find_files(filenames, recursive, exclude):
    if False:
        while True:
            i = 10
    'Yield filenames.'
    while filenames:
        name = filenames.pop(0)
        if recursive and os.path.isdir(name):
            for (root, directories, children) in os.walk(name):
                filenames += [os.path.join(root, f) for f in children if match_file(os.path.join(root, f), exclude)]
                directories[:] = [d for d in directories if match_file(os.path.join(root, d), exclude)]
        else:
            is_exclude_match = False
            for pattern in exclude:
                if fnmatch.fnmatch(name, pattern):
                    is_exclude_match = True
                    break
            if not is_exclude_match:
                yield name

def _fix_file(parameters):
    if False:
        return 10
    'Helper function for optionally running fix_file() in parallel.'
    if parameters[1].verbose:
        print('[file:{}]'.format(parameters[0]), file=sys.stderr)
    try:
        return fix_file(*parameters)
    except IOError as error:
        print(str(error), file=sys.stderr)
        raise error

def fix_multiple_files(filenames, options, output=None):
    if False:
        while True:
            i = 10
    'Fix list of files.\n\n    Optionally fix files recursively.\n\n    '
    results = []
    filenames = find_files(filenames, options.recursive, options.exclude)
    if options.jobs > 1:
        import multiprocessing
        pool = multiprocessing.Pool(options.jobs)
        rets = []
        for name in filenames:
            ret = pool.apply_async(_fix_file, ((name, options),))
            rets.append(ret)
        pool.close()
        pool.join()
        if options.diff:
            for r in rets:
                sys.stdout.write(r.get().decode())
                sys.stdout.flush()
        results.extend([x.get() for x in rets if x is not None])
    else:
        for name in filenames:
            ret = _fix_file((name, options, output))
            if ret is None:
                continue
            if options.diff:
                if ret != '':
                    results.append(ret)
            elif options.in_place:
                results.append(ret)
            else:
                original_source = readlines_from_file(name)
                if ''.join(original_source).splitlines() != ret.splitlines():
                    results.append(ret)
    return results

def is_python_file(filename):
    if False:
        while True:
            i = 10
    'Return True if filename is Python file.'
    if filename.endswith('.py'):
        return True
    try:
        with open_with_encoding(filename, limit_byte_check=MAX_PYTHON_FILE_DETECTION_BYTES) as f:
            text = f.read(MAX_PYTHON_FILE_DETECTION_BYTES)
            if not text:
                return False
            first_line = text.splitlines()[0]
    except (IOError, IndexError):
        return False
    if not PYTHON_SHEBANG_REGEX.match(first_line):
        return False
    return True

def is_probably_part_of_multiline(line):
    if False:
        i = 10
        return i + 15
    "Return True if line is likely part of a multiline string.\n\n    When multiline strings are involved, pep8 reports the error as being\n    at the start of the multiline string, which doesn't work for us.\n\n    "
    return '"""' in line or "'''" in line or line.rstrip().endswith('\\')

def wrap_output(output, encoding):
    if False:
        return 10
    'Return output with specified encoding.'
    return codecs.getwriter(encoding)(output.buffer if hasattr(output, 'buffer') else output)

def get_encoding():
    if False:
        for i in range(10):
            print('nop')
    'Return preferred encoding.'
    return locale.getpreferredencoding() or sys.getdefaultencoding()

def main(argv=None, apply_config=True):
    if False:
        i = 10
        return i + 15
    'Command-line entry.'
    if argv is None:
        argv = sys.argv
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except AttributeError:
        pass
    try:
        args = parse_args(argv[1:], apply_config=apply_config)
        if args.list_fixes:
            for (code, description) in sorted(supported_fixes()):
                print('{code} - {description}'.format(code=code, description=description))
            return EXIT_CODE_OK
        if args.files == ['-']:
            assert not args.in_place
            encoding = sys.stdin.encoding or get_encoding()
            read_stdin = sys.stdin.read()
            fixed_stdin = fix_code(read_stdin, args, encoding=encoding)
            wrap_output(sys.stdout, encoding=encoding).write(fixed_stdin)
            if hash(read_stdin) != hash(fixed_stdin):
                if args.exit_code:
                    return EXIT_CODE_EXISTS_DIFF
        else:
            if args.in_place or args.diff:
                args.files = list(set(args.files))
            else:
                assert len(args.files) == 1
                assert not args.recursive
            results = fix_multiple_files(args.files, args, sys.stdout)
            if args.diff:
                ret = any([len(ret) != 0 for ret in results])
            else:
                ret = any([ret is not None for ret in results])
            if args.exit_code and ret:
                return EXIT_CODE_EXISTS_DIFF
    except IOError:
        return EXIT_CODE_ERROR
    except KeyboardInterrupt:
        return EXIT_CODE_ERROR

class CachedTokenizer(object):
    """A one-element cache around tokenize.generate_tokens().

    Original code written by Ned Batchelder, in coverage.py.

    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.last_text = None
        self.last_tokens = None

    def generate_tokens(self, text):
        if False:
            return 10
        'A stand-in for tokenize.generate_tokens().'
        if text != self.last_text:
            string_io = io.StringIO(text)
            self.last_tokens = list(tokenize.generate_tokens(string_io.readline))
            self.last_text = text
        return self.last_tokens
_cached_tokenizer = CachedTokenizer()
generate_tokens = _cached_tokenizer.generate_tokens
if __name__ == '__main__':
    sys.exit(main())