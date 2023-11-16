from __future__ import absolute_import
import unittest
import logging
import os
import codecs
from lark import logger
from lark.tools.nearley import create_code_for_nearley_grammar, main as nearley_tool_main
logger.setLevel(logging.INFO)
TEST_PATH = os.path.abspath(os.path.dirname(__file__))
NEARLEY_PATH = os.path.join(TEST_PATH, 'nearley')
BUILTIN_PATH = os.path.join(NEARLEY_PATH, 'builtin')
if not os.path.exists(BUILTIN_PATH):
    logger.warning('Nearley not included. Skipping Nearley tests! (use git submodule to add)')
    raise ImportError('Skipping Nearley tests!')
try:
    import js2py
except RuntimeError as e:
    if 'python version' in str(e):
        raise ImportError('js2py does not support this python version')
    raise

class TestNearley(unittest.TestCase):

    def test_css(self):
        if False:
            while True:
                i = 10
        fn = os.path.join(NEARLEY_PATH, 'examples/csscolor.ne')
        with open(fn) as f:
            grammar = f.read()
        code = create_code_for_nearley_grammar(grammar, 'csscolor', BUILTIN_PATH, os.path.dirname(fn))
        d = {}
        exec(code, d)
        parse = d['parse']
        c = parse('#a199ff')
        assert c['r'] == 161
        assert c['g'] == 153
        assert c['b'] == 255
        c = parse('rgb(255, 70%, 3)')
        assert c['r'] == 255
        assert c['g'] == 178
        assert c['b'] == 3

    def test_include(self):
        if False:
            i = 10
            return i + 15
        fn = os.path.join(NEARLEY_PATH, 'test/grammars/folder-test.ne')
        with open(fn) as f:
            grammar = f.read()
        code = create_code_for_nearley_grammar(grammar, 'main', BUILTIN_PATH, os.path.dirname(fn))
        d = {}
        exec(code, d)
        parse = d['parse']
        parse('a')
        parse('b')

    def test_multi_include(self):
        if False:
            print('Hello World!')
        fn = os.path.join(NEARLEY_PATH, 'test/grammars/multi-include-test.ne')
        with open(fn) as f:
            grammar = f.read()
        code = create_code_for_nearley_grammar(grammar, 'main', BUILTIN_PATH, os.path.dirname(fn))
        d = {}
        exec(code, d)
        parse = d['parse']
        parse('a')
        parse('b')
        parse('c')

    def test_utf8(self):
        if False:
            i = 10
            return i + 15
        grammar = u'main -> "±a"'
        code = create_code_for_nearley_grammar(grammar, 'main', BUILTIN_PATH, './')
        d = {}
        exec(code, d)
        parse = d['parse']
        parse(u'±a')

    def test_backslash(self):
        if False:
            print('Hello World!')
        grammar = 'main -> "\\""'
        code = create_code_for_nearley_grammar(grammar, 'main', BUILTIN_PATH, './')
        d = {}
        exec(code, d)
        parse = d['parse']
        parse(u'"')

    def test_null(self):
        if False:
            return 10
        grammar = 'main -> "a" | null'
        code = create_code_for_nearley_grammar(grammar, 'main', BUILTIN_PATH, './')
        d = {}
        exec(code, d)
        parse = d['parse']
        parse('a')
        parse('')

    def test_utf8_2(self):
        if False:
            for i in range(10):
                print('nop')
        fn = os.path.join(TEST_PATH, 'grammars/unicode.ne')
        nearley_tool_main(fn, 'x', NEARLEY_PATH)

    def test_include_utf8(self):
        if False:
            print('Hello World!')
        fn = os.path.join(TEST_PATH, 'grammars/include_unicode.ne')
        nearley_tool_main(fn, 'main', NEARLEY_PATH)
if __name__ == '__main__':
    unittest.main()