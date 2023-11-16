"""Test case for pyparser."""
import os
import re
import textwrap
import tokenize
import pyparser

def _make_tuple(op):
    if False:
        i = 10
        return i + 15
    return lambda x: (op, x)
NL = tokenize.NL
NEWLINE = tokenize.NEWLINE
NAME = _make_tuple(tokenize.NAME)
OP = _make_tuple(tokenize.OP)
INDENT = tokenize.INDENT
DEDENT = tokenize.DEDENT
COMMENT = tokenize.COMMENT
STRING = tokenize.STRING
NUMBER = tokenize.NUMBER
END = tokenize.ENDMARKER
token_names = {NL: 'NL', NEWLINE: 'NEWLINE', INDENT: 'INDENT', COMMENT: 'COMMENT', DEDENT: 'DEDENT', STRING: 'STRING', NUMBER: 'NUMBER', END: 'END', tokenize.OP: 'OP', tokenize.NAME: 'NAME'}
Ws = pyparser.Whitespace
Comment = pyparser.Comment
Comment_banner = (pyparser.Comment, 'banner')
Comment_code = (pyparser.Comment, 'code')
Docstring = pyparser.Docstring
Import_stdlib = (pyparser.ImportBlock, 'stdlib')
Import_3rdpty = (pyparser.ImportBlock, 'third-party')
Import_1stpty = (pyparser.ImportBlock, 'first-party')
Expression = pyparser.Expression
Function = (pyparser.Callable, 'def')
Class = (pyparser.Callable, 'class')

def assert_same_code(code1, code2):
    if False:
        while True:
            i = 10
    'Verify whether 2 code fragments are identical, and if not print an error message.'
    regex = re.compile('\\s+\\\\$', re.M)
    code1 = re.sub(regex, '\\\\', code1)
    code2 = re.sub(regex, '\\\\', code2)
    if code2 != code1:
        print()
        lines_code1 = code1.splitlines()
        lines_code2 = code2.splitlines()
        n_diffs = 0
        for i in range(len(lines_code1)):
            old_line = lines_code1[i]
            new_line = lines_code2[i] if i < len(lines_code2) else ''
            if old_line != new_line:
                print('%3d - %s' % (i + 1, old_line))
                print('%3d + %s' % (i + 1, new_line))
                n_diffs += 1
                if n_diffs == 5:
                    break
        raise AssertionError('Unparsed code1 does not match the original.')

def test_tokenization():
    if False:
        return 10
    '\n    Test function for ``pyparser._normalize_tokens()``.\n\n    Even though this function is private, it is extremely important to verify that it behaves correctly. In\n    particular, we want to check that it does not break the round-trip guarantee of the tokenizer, and that it\n    fixes all the problems that the original tokenizer has.\n    '

    def _parse_to_tokens(text):
        if False:
            for i in range(10):
                print('nop')
        'Parse text into tokens and then normalize them.'
        gen = iter(text.splitlines(True))
        readline = gen.next if hasattr(gen, 'next') else gen.__next__
        return pyparser._tokenize(readline)

    def _unparse_tokens(tokens):
        if False:
            i = 10
            return i + 15
        'Convert tokens back into the source code.'
        return tokenize.untokenize((t.token for t in tokens))

    def _assert_tokens(tokens, target):
        if False:
            while True:
                i = 10
        'Check that the tokens list corresponds to the target provided.'
        for i in range(len(tokens)):
            assert i < len(target), 'Token %d %r not expected' % (i, tokens[i])
            tok = tokens[i]
            trg = target[i]
            valid = False
            if isinstance(trg, int):
                if tok.op == trg:
                    valid = True
                name = token_names[trg]
            elif isinstance(trg, tuple) and len(trg) == 2:
                if tok.op == trg[0] and tok.str == trg[1]:
                    valid = True
                name = '%s(%s)' % (token_names[trg[0]], trg[1])
            else:
                assert False, 'Unknown target: %r' % trg
            if not valid:
                assert False, 'Mismatched token %d: found %r, should be %r' % (i, tok, name)
        assert len(target) == len(tokens), 'Expected too many tokens: %d vs %d' % (len(tokens), len(target))

    def check_code(code, expected_tokens=None, filename=None):
        if False:
            i = 10
            return i + 15
        'Test parsing of the given piece of code.'
        code = textwrap.dedent(code)
        if filename:
            print('Testing tokenization of %s:' % filename, end=' ')
        else:
            check_code.index = getattr(check_code, 'index', 0) + 1
            print('Testing tokenization %d:' % check_code.index, end=' ')
        tokens = _parse_to_tokens(code)
        try:
            try:
                unparsed = _unparse_tokens(tokens)
            except ValueError as e:
                raise AssertionError('Cannot unparse tokens: %s' % e)
            assert_same_code(code, unparsed)
            if expected_tokens:
                _assert_tokens(tokens, expected_tokens)
            print('ok')
        except AssertionError as e:
            print(u'Error: %s' % e)
            print(u'Original code fragment:\n' + code)
            print('Tokens:')
            for (i, tok) in enumerate(tokens):
                print('%3d %r' % (i, tok))
            raise
    check_code('\n        try:\n            while True:\n                pass\n                # comment\n        except: pass\n        ', [NL, NAME('try'), OP(':'), NEWLINE, INDENT, NAME('while'), NAME('True'), OP(':'), NEWLINE, INDENT, NAME('pass'), NEWLINE, COMMENT, NL, DEDENT, DEDENT, NAME('except'), OP(':'), NAME('pass'), NEWLINE, END])
    check_code('\n        try:\n            while True:\n                pass\n            # comment\n        except: pass\n        ', [NL, NAME('try'), OP(':'), NEWLINE, INDENT, NAME('while'), NAME('True'), OP(':'), NEWLINE, INDENT, NAME('pass'), NEWLINE, DEDENT, COMMENT, NL, DEDENT, NAME('except'), OP(':'), NAME('pass'), NEWLINE, END])
    check_code('\n        try:\n            while True:\n                pass\n        # comment\n        except: pass\n        ', [NL, NAME('try'), OP(':'), NEWLINE, INDENT, NAME('while'), NAME('True'), OP(':'), NEWLINE, INDENT, NAME('pass'), NEWLINE, DEDENT, DEDENT, COMMENT, NL, NAME('except'), OP(':'), NAME('pass'), NEWLINE, END])
    check_code('\n        def func():\n            # function\n            pass\n        ', [NL, NAME('def'), NAME('func'), OP('('), OP(')'), OP(':'), NEWLINE, INDENT, COMMENT, NL, NAME('pass'), NEWLINE, DEDENT, END])
    check_code('\n        def func():  # function\n                     # hanging comment\n            pass\n        ', [NL, NAME('def'), NAME('func'), OP('('), OP(')'), OP(':'), COMMENT, NEWLINE, INDENT, COMMENT, NL, NAME('pass'), NEWLINE, DEDENT, END])
    check_code('\n        def foo():\n            pass\n\n        #comment\n        def bar():\n            pass\n        ', [NL, NAME('def'), NAME('foo'), OP('('), OP(')'), OP(':'), NEWLINE, INDENT, NAME('pass'), NEWLINE, DEDENT, NL, COMMENT, NL, NAME('def'), NAME('bar'), OP('('), OP(')'), OP(':'), NEWLINE, INDENT, NAME('pass'), NEWLINE, DEDENT, END])
    check_code('\n        def hello():\n\n\n            print("hello")\n        ', [NL, NAME('def'), NAME('hello'), OP('('), OP(')'), OP(':'), NEWLINE, INDENT, NL, NL, NAME('print'), OP('('), STRING, OP(')'), NEWLINE, DEDENT, END])
    check_code('\n        class Foo:\n            def foo(self):\n                pass\n\n            def bar(self):\n                return\n        ', [NL, NAME('class'), NAME('Foo'), OP(':'), NEWLINE, INDENT, NAME('def'), NAME('foo'), OP('('), NAME('self'), OP(')'), OP(':'), NEWLINE, INDENT, NAME('pass'), NEWLINE, DEDENT, NL, NAME('def'), NAME('bar'), OP('('), NAME('self'), OP(')'), OP(':'), NEWLINE, INDENT, NAME('return'), NEWLINE, DEDENT, DEDENT, END])
    check_code('\n        def foo():\n            # Attempt to create the output directory\n            try:\n                os.makedirs(destdir)\n            except OSError as e:\n                raise\n        ', [NL, NAME('def'), NAME('foo'), OP('('), OP(')'), OP(':'), NEWLINE, INDENT, COMMENT, NL, NAME('try'), OP(':'), NEWLINE, INDENT, NAME('os'), OP('.'), NAME('makedirs'), OP('('), NAME('destdir'), OP(')'), NEWLINE, DEDENT, NAME('except'), NAME('OSError'), NAME('as'), NAME('e'), OP(':'), NEWLINE, INDENT, NAME('raise'), NEWLINE, DEDENT, DEDENT, END])
    check_code('\n        handler = lambda: None  # noop\n                                # (will redefine later)\n\n        ################################################################################\n\n        # comment 1\n        print("I\'m done.")\n        ', [NL, NAME('handler'), OP('='), NAME('lambda'), OP(':'), NAME('None'), COMMENT, NEWLINE, COMMENT, NL, NL, COMMENT, NL, NL, COMMENT, NL, NAME('print'), OP('('), STRING, OP(')'), NEWLINE, END])
    check_code('\n        def test3():\n            x = 1\n        # bad\n            print(x)\n        ', [NL, NAME('def'), NAME('test3'), OP('('), OP(')'), OP(':'), NEWLINE, INDENT, NAME('x'), OP('='), NUMBER, NEWLINE, COMMENT, NL, NAME('print'), OP('('), NAME('x'), OP(')'), NEWLINE, DEDENT, END])
    check_code("\n        class Foo(object):\n            #-------------\n            def bar(self):\n                if True:\n                    pass\n\n        # Originally the DEDENTs are all the way down near the decorator. Here we're testing how they'd travel\n        # all the way up across multiple comments.\n\n        # comment 3\n\n        # commmmmmmment 4\n        @decorator\n        ", [NL, NAME('class'), NAME('Foo'), OP('('), NAME('object'), OP(')'), OP(':'), NEWLINE, INDENT, COMMENT, NL, NAME('def'), NAME('bar'), OP('('), NAME('self'), OP(')'), OP(':'), NEWLINE, INDENT, NAME('if'), NAME('True'), OP(':'), NEWLINE, INDENT, NAME('pass'), NEWLINE, DEDENT, DEDENT, DEDENT, NL, COMMENT, NL, COMMENT, NL, NL, COMMENT, NL, NL, COMMENT, NL, OP('@'), NAME('decorator'), NEWLINE, END])
    check_code('\n        if True:\n            if False:\n                # INDENT will be inserted before this comment\n                raise\n                # DEDENT will be after this comment\n            else:\n                praise()\n        ', [NL, NAME('if'), NAME('True'), OP(':'), NEWLINE, INDENT, NAME('if'), NAME('False'), OP(':'), NEWLINE, INDENT, COMMENT, NL, NAME('raise'), NEWLINE, COMMENT, NL, DEDENT, NAME('else'), OP(':'), NEWLINE, INDENT, NAME('praise'), OP('('), OP(')'), NEWLINE, DEDENT, DEDENT, END])
    for directory in ['.', '../../h2o-py/h2o', '../../h2o-py/tests']:
        absdir = os.path.abspath(directory)
        for (dir_name, subdirs, files) in os.walk(absdir):
            for f in files:
                if f.endswith('.py'):
                    filename = os.path.join(dir_name, f)
                    with open(filename, 'rt', encoding='utf-8') as fff:
                        check_code(fff.read(), filename=filename)

def test_pyparser():
    if False:
        while True:
            i = 10
    'Test case: general parsing.'

    def _check_blocks(actual, expected):
        if False:
            return 10
        assert actual, 'No parse results'
        for i in range(len(actual)):
            assert i < len(expected), 'Unexpected block %d:\n%r' % (i, actual[i])
            valid = False
            if isinstance(expected[i], type):
                if isinstance(actual[i], expected[i]):
                    valid = True
            elif isinstance(expected[i], tuple):
                if isinstance(actual[i], expected[i][0]) and actual[i].type == expected[i][1]:
                    valid = True
            if not valid:
                assert False, 'Invalid block: expected %r, got %r' % (expected[i], actual[i])

    def check_code(code, blocks=None, filename=None):
        if False:
            print('Hello World!')
        code = textwrap.dedent(code)
        if not code.endswith('\n'):
            code += '\n'
        if filename:
            print('Testing file %s...' % filename, end=' ')
        else:
            check_code.index = getattr(check_code, 'index', 0) + 1
            print('Testing code fragment %d...' % check_code.index, end=' ')
        preparsed = None
        parsed = None
        unparsed = None
        try:
            preparsed = pyparser.parse_text(code)
            parsed = preparsed.parse(2)
            try:
                unparsed = parsed.unparse()
            except ValueError as e:
                for (i, tok) in enumerate(parsed.tokens):
                    print('%3d %r' % (i, tok))
                raise AssertionError('Cannot unparse code: %s' % e)
            assert_same_code(code, unparsed)
            if blocks:
                _check_blocks(parsed.parsed, blocks)
            print('ok')
        except AssertionError as e:
            print()
            print(u'Error: ' + str(e))
            print(u'Original code fragment:\n' + code)
            if unparsed:
                print(u'Unparsed code:\n' + unparsed)
            if parsed:
                print(parsed)
                for (i, tok) in enumerate(parsed.tokens):
                    print('%3d %r' % (i, tok))
            raise
        except Exception as e:
            print()
            print(u'Error: ' + str(e))
            if preparsed:
                print('Preparsed tokens:')
                for (i, tok) in enumerate(preparsed.tokens):
                    print('%4d %r' % (i, tok))
            else:
                print('Initial parsing has failed...')
            raise
    check_code('\n        # -*- encoding: utf-8 -*-\n        # copyright: 2016 h2o.ai\n        """\n        A code example.\n\n        It\'s not supposed to be functional, or even functionable.\n        """\n        # Standard library imports\n        import sys\n        import time\n        import this\n\n        import h2o\n        from h2o import H2OFrame, init\n        from . import *\n\n\n\n        # Do some initalization for legacy python versions\n        handler = lambda: None  # noop\n                                # (will redefine later)\n\n        ################################################################################\n\n        # comment 1\n        class Foo(object):\n            #------ Public -------------------------------------------------------------\n            def bar(self):\n                pass\n\n        # def foo():\n        #     print(1)\n        #\n        #     print(2)\n\n        # comment 2\n        @decorated(\n            1, 2, (3))\n        @dddd\n        def bar():\n            # be\n            # happy\n            print("bar!")\n        # bye', [Ws, Comment, Docstring, Ws, Import_stdlib, Ws, Import_1stpty, Ws, Expression, Ws, Expression, Ws, Comment_banner, Ws, Class, Ws, Comment_code, Ws, Function, Comment, Ws])
    for directory in ['.', '../../h2o-py', '../../py']:
        absdir = os.path.abspath(directory)
        for (dir_name, subdirs, files) in os.walk(absdir):
            for f in files:
                if f.endswith('.py'):
                    filename = os.path.join(dir_name, f)
                    with open(filename, 'rt', encoding='utf-8') as fff:
                        check_code(fff.read(), filename=filename)
test_pyparser()