"""Tests for debugging machinery.
"""
import builtins
import os
import sys
import platform
from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch
from IPython.core import debugger
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
from IPython.testing.decorators import skip_win32
import pytest

class _FakeInput(object):
    """
    A fake input stream for pdb's interactive debugger.  Whenever a
    line is read, print it (to simulate the user typing it), and then
    return it.  The set of lines to return is specified in the
    constructor; they should not have trailing newlines.
    """

    def __init__(self, lines):
        if False:
            return 10
        self.lines = iter(lines)

    def readline(self):
        if False:
            while True:
                i = 10
        line = next(self.lines)
        print(line)
        return line + '\n'

class PdbTestInput(object):
    """Context manager that makes testing Pdb in doctests easier."""

    def __init__(self, input):
        if False:
            i = 10
            return i + 15
        self.input = input

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.real_stdin = sys.stdin
        sys.stdin = _FakeInput(self.input)

    def __exit__(self, *exc):
        if False:
            i = 10
            return i + 15
        sys.stdin = self.real_stdin

def test_ipdb_magics():
    if False:
        return 10
    'Test calling some IPython magics from ipdb.\n\n    First, set up some test functions and classes which we can inspect.\n\n    >>> class ExampleClass(object):\n    ...    """Docstring for ExampleClass."""\n    ...    def __init__(self):\n    ...        """Docstring for ExampleClass.__init__"""\n    ...        pass\n    ...    def __str__(self):\n    ...        return "ExampleClass()"\n\n    >>> def example_function(x, y, z="hello"):\n    ...     """Docstring for example_function."""\n    ...     pass\n\n    >>> old_trace = sys.gettrace()\n\n    Create a function which triggers ipdb.\n\n    >>> def trigger_ipdb():\n    ...    a = ExampleClass()\n    ...    debugger.Pdb().set_trace()\n\n    >>> with PdbTestInput([\n    ...    \'pdef example_function\',\n    ...    \'pdoc ExampleClass\',\n    ...    \'up\',\n    ...    \'down\',\n    ...    \'list\',\n    ...    \'pinfo a\',\n    ...    \'ll\',\n    ...    \'continue\',\n    ... ]):\n    ...     trigger_ipdb()\n    --Return--\n    None\n    > <doctest ...>(3)trigger_ipdb()\n          1 def trigger_ipdb():\n          2    a = ExampleClass()\n    ----> 3    debugger.Pdb().set_trace()\n    <BLANKLINE>\n    ipdb> pdef example_function\n     example_function(x, y, z=\'hello\')\n     ipdb> pdoc ExampleClass\n    Class docstring:\n        Docstring for ExampleClass.\n    Init docstring:\n        Docstring for ExampleClass.__init__\n    ipdb> up\n    > <doctest ...>(11)<module>()\n          7    \'pinfo a\',\n          8    \'ll\',\n          9    \'continue\',\n         10 ]):\n    ---> 11     trigger_ipdb()\n    <BLANKLINE>\n    ipdb> down\n    None\n    > <doctest ...>(3)trigger_ipdb()\n          1 def trigger_ipdb():\n          2    a = ExampleClass()\n    ----> 3    debugger.Pdb().set_trace()\n    <BLANKLINE>\n    ipdb> list\n          1 def trigger_ipdb():\n          2    a = ExampleClass()\n    ----> 3    debugger.Pdb().set_trace()\n    <BLANKLINE>\n    ipdb> pinfo a\n    Type:           ExampleClass\n    String form:    ExampleClass()\n    Namespace:      Local...\n    Docstring:      Docstring for ExampleClass.\n    Init docstring: Docstring for ExampleClass.__init__\n    ipdb> ll\n          1 def trigger_ipdb():\n          2    a = ExampleClass()\n    ----> 3    debugger.Pdb().set_trace()\n    <BLANKLINE>\n    ipdb> continue\n    \n    Restore previous trace function, e.g. for coverage.py    \n    \n    >>> sys.settrace(old_trace)\n    '

def test_ipdb_magics2():
    if False:
        i = 10
        return i + 15
    "Test ipdb with a very short function.\n    \n    >>> old_trace = sys.gettrace()\n\n    >>> def bar():\n    ...     pass\n\n    Run ipdb.\n\n    >>> with PdbTestInput([\n    ...    'continue',\n    ... ]):\n    ...     debugger.Pdb().runcall(bar)\n    > <doctest ...>(2)bar()\n          1 def bar():\n    ----> 2    pass\n    <BLANKLINE>\n    ipdb> continue\n    \n    Restore previous trace function, e.g. for coverage.py    \n    \n    >>> sys.settrace(old_trace)\n    "

def can_quit():
    if False:
        print('Hello World!')
    "Test that quit work in ipydb\n\n    >>> old_trace = sys.gettrace()\n\n    >>> def bar():\n    ...     pass\n\n    >>> with PdbTestInput([\n    ...    'quit',\n    ... ]):\n    ...     debugger.Pdb().runcall(bar)\n    > <doctest ...>(2)bar()\n            1 def bar():\n    ----> 2    pass\n    <BLANKLINE>\n    ipdb> quit\n\n    Restore previous trace function, e.g. for coverage.py\n\n    >>> sys.settrace(old_trace)\n    "

def can_exit():
    if False:
        i = 10
        return i + 15
    "Test that quit work in ipydb\n\n    >>> old_trace = sys.gettrace()\n\n    >>> def bar():\n    ...     pass\n\n    >>> with PdbTestInput([\n    ...    'exit',\n    ... ]):\n    ...     debugger.Pdb().runcall(bar)\n    > <doctest ...>(2)bar()\n            1 def bar():\n    ----> 2    pass\n    <BLANKLINE>\n    ipdb> exit\n\n    Restore previous trace function, e.g. for coverage.py\n\n    >>> sys.settrace(old_trace)\n    "

def test_interruptible_core_debugger():
    if False:
        return 10
    'The debugger can be interrupted.\n\n    The presumption is there is some mechanism that causes a KeyboardInterrupt\n    (this is implemented in ipykernel).  We want to ensure the\n    KeyboardInterrupt cause debugging to cease.\n    '

    def raising_input(msg='', called=[0]):
        if False:
            while True:
                i = 10
        called[0] += 1
        assert called[0] == 1, 'input() should only be called once!'
        raise KeyboardInterrupt()
    tracer_orig = sys.gettrace()
    try:
        with patch.object(builtins, 'input', raising_input):
            debugger.InterruptiblePdb().set_trace()
    finally:
        sys.settrace(tracer_orig)

@skip_win32
def test_xmode_skip():
    if False:
        print('Hello World!')
    'that xmode skip frames\n\n    Not as a doctest as pytest does not run doctests.\n    '
    import pexpect
    env = os.environ.copy()
    env['IPY_TEST_SIMPLE_PROMPT'] = '1'
    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor'], env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.expect('IPython')
    child.expect('\n')
    child.expect_exact('In [1]')
    block = dedent('\n    def f():\n        __tracebackhide__ = True\n        g()\n\n    def g():\n        raise ValueError\n\n    f()\n    ')
    for line in block.splitlines():
        child.sendline(line)
        child.expect_exact(line)
    child.expect_exact('skipping')
    block = dedent('\n    def f():\n        __tracebackhide__ = True\n        g()\n\n    def g():\n        from IPython.core.debugger import set_trace\n        set_trace()\n\n    f()\n    ')
    for line in block.splitlines():
        child.sendline(line)
        child.expect_exact(line)
    child.expect('ipdb>')
    child.sendline('w')
    child.expect('hidden')
    child.expect('ipdb>')
    child.sendline('skip_hidden false')
    child.sendline('w')
    child.expect('__traceba')
    child.expect('ipdb>')
    child.close()
skip_decorators_blocks = ('\n    def helpers_helper():\n        pass # should not stop here except breakpoint\n    ', '\n    def helper_1():\n        helpers_helper() # should not stop here\n    ', '\n    def helper_2():\n        pass # should not stop here\n    ', '\n    def pdb_skipped_decorator2(function):\n        def wrapped_fn(*args, **kwargs):\n            __debuggerskip__ = True\n            helper_2()\n            __debuggerskip__ = False\n            result = function(*args, **kwargs)\n            __debuggerskip__ = True\n            helper_2()\n            return result\n        return wrapped_fn\n    ', '\n    def pdb_skipped_decorator(function):\n        def wrapped_fn(*args, **kwargs):\n            __debuggerskip__ = True\n            helper_1()\n            __debuggerskip__ = False\n            result = function(*args, **kwargs)\n            __debuggerskip__ = True\n            helper_2()\n            return result\n        return wrapped_fn\n    ', '\n    @pdb_skipped_decorator\n    @pdb_skipped_decorator2\n    def bar(x, y):\n        return x * y\n    ', 'import IPython.terminal.debugger as ipdb', '\n    def f():\n        ipdb.set_trace()\n        bar(3, 4)\n    ', '\n    f()\n    ')

def _decorator_skip_setup():
    if False:
        while True:
            i = 10
    import pexpect
    env = os.environ.copy()
    env['IPY_TEST_SIMPLE_PROMPT'] = '1'
    env['PROMPT_TOOLKIT_NO_CPR'] = '1'
    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor'], env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.expect('IPython')
    child.expect('\n')
    child.timeout = 5 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.str_last_chars = 500
    dedented_blocks = [dedent(b).strip() for b in skip_decorators_blocks]
    in_prompt_number = 1
    for cblock in dedented_blocks:
        child.expect_exact(f'In [{in_prompt_number}]:')
        in_prompt_number += 1
        for line in cblock.splitlines():
            child.sendline(line)
            child.expect_exact(line)
        child.sendline('')
    return child

@pytest.mark.skip(reason='recently fail for unknown reason on CI')
@skip_win32
def test_decorator_skip():
    if False:
        return 10
    'test that decorator frames can be skipped.'
    child = _decorator_skip_setup()
    child.expect_exact('ipython-input-8')
    child.expect_exact('3     bar(3, 4)')
    child.expect('ipdb>')
    child.expect('ipdb>')
    child.sendline('step')
    child.expect_exact('step')
    child.expect_exact('--Call--')
    child.expect_exact('ipython-input-6')
    child.expect_exact('1 @pdb_skipped_decorator')
    child.sendline('s')
    child.expect_exact('return x * y')
    child.close()

@pytest.mark.skip(reason='recently fail for unknown reason on CI')
@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='issues on PyPy')
@skip_win32
def test_decorator_skip_disabled():
    if False:
        return 10
    'test that decorator frame skipping can be disabled'
    child = _decorator_skip_setup()
    child.expect_exact('3     bar(3, 4)')
    for (input_, expected) in [('skip_predicates debuggerskip False', ''), ('skip_predicates', 'debuggerskip : False'), ('step', '---> 2     def wrapped_fn'), ('step', '----> 3         __debuggerskip__'), ('step', '----> 4         helper_1()'), ('step', '---> 1 def helper_1():'), ('next', '----> 2     helpers_helper()'), ('next', '--Return--'), ('next', '----> 5         __debuggerskip__ = False')]:
        child.expect('ipdb>')
        child.sendline(input_)
        child.expect_exact(input_)
        child.expect_exact(expected)
    child.close()

@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='issues on PyPy')
@skip_win32
def test_decorator_skip_with_breakpoint():
    if False:
        for i in range(10):
            print('nop')
    'test that decorator frame skipping can be disabled'
    import pexpect
    env = os.environ.copy()
    env['IPY_TEST_SIMPLE_PROMPT'] = '1'
    env['PROMPT_TOOLKIT_NO_CPR'] = '1'
    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor'], env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.str_last_chars = 500
    child.expect('IPython')
    child.expect('\n')
    child.timeout = 5 * IPYTHON_TESTING_TIMEOUT_SCALE
    with NamedTemporaryFile(suffix='.py', dir='.', delete=True) as tf:
        name = tf.name[:-3].split('/')[-1]
        tf.write('\n'.join([dedent(x) for x in skip_decorators_blocks[:-1]]).encode())
        tf.flush()
        codeblock = f'from {name} import f'
        dedented_blocks = [codeblock, 'f()']
        in_prompt_number = 1
        for cblock in dedented_blocks:
            child.expect_exact(f'In [{in_prompt_number}]:')
            in_prompt_number += 1
            for line in cblock.splitlines():
                child.sendline(line)
                child.expect_exact(line)
            child.sendline('')
        child.expect_exact('47     bar(3, 4)')
        for (input_, expected) in [(f'b {name}.py:3', ''), ('step', '1---> 3     pass # should not stop here except'), ('step', '---> 38 @pdb_skipped_decorator'), ('continue', '')]:
            child.expect('ipdb>')
            child.sendline(input_)
            child.expect_exact(input_)
            child.expect_exact(expected)
    child.close()

@skip_win32
def test_where_erase_value():
    if False:
        for i in range(10):
            print('nop')
    'Test that `where` does not access f_locals and erase values.'
    import pexpect
    env = os.environ.copy()
    env['IPY_TEST_SIMPLE_PROMPT'] = '1'
    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor'], env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.expect('IPython')
    child.expect('\n')
    child.expect_exact('In [1]')
    block = dedent('\n    def simple_f():\n         myvar = 1\n         print(myvar)\n         1/0\n         print(myvar)\n    simple_f()    ')
    for line in block.splitlines():
        child.sendline(line)
        child.expect_exact(line)
    child.expect_exact('ZeroDivisionError')
    child.expect_exact('In [2]:')
    child.sendline('%debug')
    child.expect('ipdb>')
    child.sendline('myvar')
    child.expect('1')
    child.expect('ipdb>')
    child.sendline('myvar = 2')
    child.expect_exact('ipdb>')
    child.sendline('myvar')
    child.expect_exact('2')
    child.expect('ipdb>')
    child.sendline('where')
    child.expect('ipdb>')
    child.sendline('myvar')
    child.expect_exact('2')
    child.expect('ipdb>')
    child.close()