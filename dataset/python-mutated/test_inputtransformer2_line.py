"""Tests for the line-based transformers in IPython.core.inputtransformer2

Line-based transformers are the simpler ones; token-based transformers are
more complex. See test_inputtransformer2 for tests for token-based transformers.
"""
from IPython.core import inputtransformer2 as ipt2
CELL_MAGIC = ('%%foo arg\nbody 1\nbody 2\n', "get_ipython().run_cell_magic('foo', 'arg', 'body 1\\nbody 2\\n')\n")

def test_cell_magic():
    if False:
        i = 10
        return i + 15
    for (sample, expected) in [CELL_MAGIC]:
        assert ipt2.cell_magic(sample.splitlines(keepends=True)) == expected.splitlines(keepends=True)
CLASSIC_PROMPT = ('>>> for a in range(5):\n...     print(a)\n', 'for a in range(5):\n    print(a)\n')
CLASSIC_PROMPT_L2 = ('for a in range(5):\n...     print(a)\n...     print(a ** 2)\n', 'for a in range(5):\n    print(a)\n    print(a ** 2)\n')

def test_classic_prompt():
    if False:
        print('Hello World!')
    for (sample, expected) in [CLASSIC_PROMPT, CLASSIC_PROMPT_L2]:
        assert ipt2.classic_prompt(sample.splitlines(keepends=True)) == expected.splitlines(keepends=True)
IPYTHON_PROMPT = ('In [1]: for a in range(5):\n   ...:     print(a)\n', 'for a in range(5):\n    print(a)\n')
IPYTHON_PROMPT_L2 = ('for a in range(5):\n   ...:     print(a)\n   ...:     print(a ** 2)\n', 'for a in range(5):\n    print(a)\n    print(a ** 2)\n')
IPYTHON_PROMPT_VI_INS = ('[ins] In [11]: def a():\n          ...:     123\n          ...:\n          ...: 123\n', 'def a():\n    123\n\n123\n')
IPYTHON_PROMPT_VI_NAV = ('[nav] In [11]: def a():\n          ...:     123\n          ...:\n          ...: 123\n', 'def a():\n    123\n\n123\n')

def test_ipython_prompt():
    if False:
        return 10
    for (sample, expected) in [IPYTHON_PROMPT, IPYTHON_PROMPT_L2, IPYTHON_PROMPT_VI_INS, IPYTHON_PROMPT_VI_NAV]:
        assert ipt2.ipython_prompt(sample.splitlines(keepends=True)) == expected.splitlines(keepends=True)
INDENT_SPACES = ('     if True:\n        a = 3\n', 'if True:\n   a = 3\n')
INDENT_TABS = ('\tif True:\n\t\tb = 4\n', 'if True:\n\tb = 4\n')

def test_leading_indent():
    if False:
        for i in range(10):
            print('nop')
    for (sample, expected) in [INDENT_SPACES, INDENT_TABS]:
        assert ipt2.leading_indent(sample.splitlines(keepends=True)) == expected.splitlines(keepends=True)
LEADING_EMPTY_LINES = ('    \t\n\nif True:\n    a = 3\n\nb = 4\n', 'if True:\n    a = 3\n\nb = 4\n')
ONLY_EMPTY_LINES = ('    \t\n\n', '    \t\n\n')

def test_leading_empty_lines():
    if False:
        for i in range(10):
            print('nop')
    for (sample, expected) in [LEADING_EMPTY_LINES, ONLY_EMPTY_LINES]:
        assert ipt2.leading_empty_lines(sample.splitlines(keepends=True)) == expected.splitlines(keepends=True)
CRLF_MAGIC = (['%%ls\r\n'], ["get_ipython().run_cell_magic('ls', '', '')\n"])

def test_crlf_magic():
    if False:
        while True:
            i = 10
    for (sample, expected) in [CRLF_MAGIC]:
        assert ipt2.cell_magic(sample) == expected