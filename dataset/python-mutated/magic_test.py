"""Magic unit test."""
import ast
import unittest
import streamlit.runtime.scriptrunner.magic as magic
from tests.testutil import patch_config_options

class MagicTest(unittest.TestCase):
    """Test for Magic
    The test counts the number of substitutions that magic.add_code do for
    a few code snippets. The test passes if the expected number of
    substitutions have been made.
    """

    def _testCode(self, code: str, expected_count: int) -> None:
        if False:
            return 10
        tree = magic.add_magic(code, './')
        count = 0
        for node in ast.walk(tree):
            if type(node) is ast.Call and magic.MAGIC_MODULE_NAME in ast.dump(node.func):
                count += 1
        self.assertEqual(expected_count, count, f'There must be exactly {expected_count} {magic.MAGIC_MODULE_NAME} nodes, but found {count}')

    def test_simple_statement(self):
        if False:
            print('Hello World!')
        'Test simple statements'
        CODE_SIMPLE_STATEMENTS = '\na = 1\nb = 10\na\nb\n'
        self._testCode(CODE_SIMPLE_STATEMENTS, 2)

    def test_if_statement(self):
        if False:
            print('Hello World!')
        'Test if statements'
        CODE_IF_STATEMENT = '\na = 1\nif True:\n    a\n    if False:\n        a\n    elif False:\n        a\n    else:\n        a\nelse:\n    a\n'
        self._testCode(CODE_IF_STATEMENT, 5)

    def test_for_statement(self):
        if False:
            while True:
                i = 10
        'Test for statements'
        CODE_FOR_STATEMENT = '\na = 1\nfor i in range(10):\n    for j in range(2):\n        a\n\n'
        self._testCode(CODE_FOR_STATEMENT, 1)

    def test_try_statement(self):
        if False:
            while True:
                i = 10
        'Test try statements'
        CODE_TRY_STATEMENT = '\ntry:\n    a = 10\n    a\nexcept Exception:\n    try:\n        a\n    finally:\n        a\nfinally:\n    a\n'
        self._testCode(CODE_TRY_STATEMENT, 4)

    def test_function_call_statement(self):
        if False:
            for i in range(10):
                print('nop')
        'Test with function calls'
        CODE_FUNCTION_CALL = '\ndef myfunc(a):\n    a\na =10\nmyfunc(a)\n'
        self._testCode(CODE_FUNCTION_CALL, 1)

    def test_with_statement(self):
        if False:
            print('Hello World!')
        "Test 'with' statements"
        CODE_WITH_STATEMENT = '\na = 10\nwith None:\n    a\n'
        self._testCode(CODE_WITH_STATEMENT, 1)

    def test_while_statement(self):
        if False:
            while True:
                i = 10
        "Test 'while' statements"
        CODE_WHILE_STATEMENT = '\na = 10\nwhile True:\n    a\n'
        self._testCode(CODE_WHILE_STATEMENT, 1)

    def test_yield_statement(self):
        if False:
            while True:
                i = 10
        "Test that 'yield' expressions do not get magicked"
        CODE_YIELD_STATEMENT = '\ndef yield_func():\n    yield\n'
        self._testCode(CODE_YIELD_STATEMENT, 0)

    def test_yield_from_statement(self):
        if False:
            print('Hello World!')
        "Test that 'yield from' expressions do not get magicked"
        CODE_YIELD_FROM_STATEMENT = '\ndef yield_func():\n    yield from None\n'
        self._testCode(CODE_YIELD_FROM_STATEMENT, 0)

    def test_await_expression(self):
        if False:
            while True:
                i = 10
        "Test that 'await' expressions do not get magicked"
        CODE_AWAIT_EXPRESSION = '\nasync def await_func(a):\n    await coro()\n'
        self._testCode(CODE_AWAIT_EXPRESSION, 0)

    def test_async_function_statement(self):
        if False:
            return 10
        'Test async function definitions'
        CODE_ASYNC_FUNCTION = '\nasync def myfunc(a):\n    a\n'
        self._testCode(CODE_ASYNC_FUNCTION, 1)

    def test_async_with_statement(self):
        if False:
            return 10
        "Test 'async with' statements"
        CODE_ASYNC_WITH = '\nasync def myfunc(a):\n    async with None:\n        a\n'
        self._testCode(CODE_ASYNC_WITH, 1)

    def test_async_for_statement(self):
        if False:
            while True:
                i = 10
        "Test 'async for' statements"
        CODE_ASYNC_FOR = '\nasync def myfunc(a):\n    async for _ in None:\n        a\n'
        self._testCode(CODE_ASYNC_FOR, 1)

    def test_docstring_is_ignored_func(self):
        if False:
            i = 10
            return i + 15
        "Test that docstrings don't print in the app"
        CODE = "\ndef myfunc(a):\n    '''This is the docstring'''\n    return 42\n"
        self._testCode(CODE, 0)

    def test_docstring_is_ignored_async_func(self):
        if False:
            print('Hello World!')
        "Test that async function docstrings don't print in the app by default"
        CODE = "\nasync def myfunc(a):\n    '''This is the docstring for async func'''\n    return 43\n"
        self._testCode(CODE, 0)

    def test_display_root_docstring_config_option(self):
        if False:
            print('Hello World!')
        'Test that magic.displayRootDocString skips/includes docstrings when True/False.'
        CODE = "\n'''This is a top-level docstring'''\n\n'this is a string that should always be magicked'\n\ndef my_func():\n    '''This is a function docstring'''\n\n    'this is a string that should always be magicked'\n\nclass MyClass:\n    '''This is a class docstring'''\n\n    'this is a string that should never be magicked'\n\n    def __init__(self):\n        '''This is a method docstring'''\n\n        'this is a string that should always be magicked'\n"
        self._testCode(CODE, 3)
        with patch_config_options({'magic.displayRootDocString': True}):
            self._testCode(CODE, 4)
        with patch_config_options({'magic.displayRootDocString': False}):
            self._testCode(CODE, 3)

    def test_display_last_expr_config_option(self):
        if False:
            print('Hello World!')
        'Test that magic.displayLastExprIfNoSemicolon causes the last function ast.Expr\n        node in a file to be wrapped in st.write().'
        CODE_WITHOUT_SEMICOLON = '\nthis_should_not_be_magicked()\n\ndef my_func():\n    this_should_not_be_magicked()\n\nclass MyClass:\n    this_should_not_be_magicked()\n\n    def __init__(self):\n        this_should_not_be_magicked()\n\nthis_is_the_last_expr()\n\n# Some newlines for good measure\n\n\n'
        self._testCode(CODE_WITHOUT_SEMICOLON, 0)
        with patch_config_options({'magic.displayLastExprIfNoSemicolon': True}):
            self._testCode(CODE_WITHOUT_SEMICOLON, 1)
        with patch_config_options({'magic.displayLastExprIfNoSemicolon': False}):
            self._testCode(CODE_WITHOUT_SEMICOLON, 0)
        CODE_WITH_SEMICOLON = '\nthis_should_not_be_magicked()\n\ndef my_func():\n    this_should_not_be_magicked()\n\nclass MyClass:\n    this_should_not_be_magicked()\n\n    def __init__(self):\n        this_should_not_be_magicked()\n\nthis_is_the_last_expr();\n\n# Some newlines for good measure\n\n\n'
        self._testCode(CODE_WITH_SEMICOLON, 0)
        with patch_config_options({'magic.displayLastExprIfNoSemicolon': True}):
            self._testCode(CODE_WITH_SEMICOLON, 0)
        with patch_config_options({'magic.displayLastExprIfNoSemicolon': False}):
            self._testCode(CODE_WITH_SEMICOLON, 0)