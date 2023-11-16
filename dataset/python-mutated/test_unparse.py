"""Tests for the unparse.py script in the Tools/parser directory."""
import unittest
import test.support
import pathlib
import random
import tokenize
import ast

def read_pyfile(filename):
    if False:
        for i in range(10):
            print('nop')
    'Read and return the contents of a Python source file (as a\n    string), taking into account the file encoding.'
    with tokenize.open(filename) as stream:
        return stream.read()
for_else = 'def f():\n    for x in range(10):\n        break\n    else:\n        y = 2\n    z = 3\n'
while_else = 'def g():\n    while True:\n        break\n    else:\n        y = 2\n    z = 3\n'
relative_import = 'from . import fred\nfrom .. import barney\nfrom .australia import shrimp as prawns\n'
nonlocal_ex = 'def f():\n    x = 1\n    def g():\n        nonlocal x\n        x = 2\n        y = 7\n        def h():\n            nonlocal x, y\n'
raise_from = 'try:\n    1 / 0\nexcept ZeroDivisionError as e:\n    raise ArithmeticError from e\n'
class_decorator = '@f1(arg)\n@f2\nclass Foo: pass\n'
elif1 = 'if cond1:\n    suite1\nelif cond2:\n    suite2\nelse:\n    suite3\n'
elif2 = 'if cond1:\n    suite1\nelif cond2:\n    suite2\n'
try_except_finally = 'try:\n    suite1\nexcept ex1:\n    suite2\nexcept ex2:\n    suite3\nelse:\n    suite4\nfinally:\n    suite5\n'
with_simple = 'with f():\n    suite1\n'
with_as = 'with f() as x:\n    suite1\n'
with_two_items = 'with f() as x, g() as y:\n    suite1\n'
docstring_prefixes = ('', 'class foo:\n    ', 'def foo():\n    ', 'async def foo():\n    ')

class ASTTestCase(unittest.TestCase):

    def assertASTEqual(self, ast1, ast2):
        if False:
            print('Hello World!')
        self.assertEqual(ast.dump(ast1), ast.dump(ast2))

    def check_ast_roundtrip(self, code1, **kwargs):
        if False:
            i = 10
            return i + 15
        with self.subTest(code1=code1, ast_parse_kwargs=kwargs):
            ast1 = ast.parse(code1, **kwargs)
            code2 = ast.unparse(ast1)
            ast2 = ast.parse(code2, **kwargs)
            self.assertASTEqual(ast1, ast2)

    def check_invalid(self, node, raises=ValueError):
        if False:
            print('Hello World!')
        with self.subTest(node=node):
            self.assertRaises(raises, ast.unparse, node)

    def get_source(self, code1, code2=None):
        if False:
            i = 10
            return i + 15
        code2 = code2 or code1
        code1 = ast.unparse(ast.parse(code1))
        return (code1, code2)

    def check_src_roundtrip(self, code1, code2=None):
        if False:
            i = 10
            return i + 15
        (code1, code2) = self.get_source(code1, code2)
        with self.subTest(code1=code1, code2=code2):
            self.assertEqual(code2, code1)

    def check_src_dont_roundtrip(self, code1, code2=None):
        if False:
            print('Hello World!')
        (code1, code2) = self.get_source(code1, code2)
        with self.subTest(code1=code1, code2=code2):
            self.assertNotEqual(code2, code1)

class UnparseTestCase(ASTTestCase):

    def test_fstrings(self):
        if False:
            print('Hello World!')
        self.check_ast_roundtrip('f\'{f"{0}"*3}\'')
        self.check_ast_roundtrip('f\'{f"{y}"*3}\'')
        self.check_ast_roundtrip("f''")
        self.check_ast_roundtrip('f"""\'end\' "quote\\""""')

    def test_fstrings_complicated(self):
        if False:
            while True:
                i = 10
        self.check_ast_roundtrip('f\'\'\'{"\'"}\'\'\'')
        self.check_ast_roundtrip('f\'\'\'-{f"""*{f"+{f\'.{x}.\'}+"}*"""}-\'\'\'')
        self.check_ast_roundtrip('f\'\'\'-{f"""*{f"+{f\'.{x}.\'}+"}*"""}-\'single quote\\\'\'\'\'')
        self.check_ast_roundtrip('f"""{\'\'\'\n\'\'\'}"""')
        self.check_ast_roundtrip('f"""{g(\'\'\'\n\'\'\')}"""')
        self.check_ast_roundtrip('f"a\\r\\nb"')
        self.check_ast_roundtrip('f"\\u2028{\'x\'}"')

    def test_strings(self):
        if False:
            i = 10
            return i + 15
        self.check_ast_roundtrip("u'foo'")
        self.check_ast_roundtrip("r'foo'")
        self.check_ast_roundtrip("b'foo'")

    def test_del_statement(self):
        if False:
            i = 10
            return i + 15
        self.check_ast_roundtrip('del x, y, z')

    def test_shifts(self):
        if False:
            print('Hello World!')
        self.check_ast_roundtrip('45 << 2')
        self.check_ast_roundtrip('13 >> 7')

    def test_for_else(self):
        if False:
            print('Hello World!')
        self.check_ast_roundtrip(for_else)

    def test_while_else(self):
        if False:
            return 10
        self.check_ast_roundtrip(while_else)

    def test_unary_parens(self):
        if False:
            return 10
        self.check_ast_roundtrip('(-1)**7')
        self.check_ast_roundtrip('(-1.)**8')
        self.check_ast_roundtrip('(-1j)**6')
        self.check_ast_roundtrip('not True or False')
        self.check_ast_roundtrip('True or not False')

    def test_integer_parens(self):
        if False:
            print('Hello World!')
        self.check_ast_roundtrip('3 .__abs__()')

    def test_huge_float(self):
        if False:
            while True:
                i = 10
        self.check_ast_roundtrip('1e1000')
        self.check_ast_roundtrip('-1e1000')
        self.check_ast_roundtrip('1e1000j')
        self.check_ast_roundtrip('-1e1000j')

    def test_nan(self):
        if False:
            while True:
                i = 10
        self.assertASTEqual(ast.parse(ast.unparse(ast.Constant(value=float('nan')))), ast.parse('1e1000 - 1e1000'))

    def test_min_int(self):
        if False:
            print('Hello World!')
        self.check_ast_roundtrip(str(-2 ** 31))
        self.check_ast_roundtrip(str(-2 ** 63))

    def test_imaginary_literals(self):
        if False:
            return 10
        self.check_ast_roundtrip('7j')
        self.check_ast_roundtrip('-7j')
        self.check_ast_roundtrip('0j')
        self.check_ast_roundtrip('-0j')

    def test_lambda_parentheses(self):
        if False:
            i = 10
            return i + 15
        self.check_ast_roundtrip('(lambda: int)()')

    def test_chained_comparisons(self):
        if False:
            while True:
                i = 10
        self.check_ast_roundtrip('1 < 4 <= 5')
        self.check_ast_roundtrip('a is b is c is not d')

    def test_function_arguments(self):
        if False:
            return 10
        self.check_ast_roundtrip('def f(): pass')
        self.check_ast_roundtrip('def f(a): pass')
        self.check_ast_roundtrip('def f(b = 2): pass')
        self.check_ast_roundtrip('def f(a, b): pass')
        self.check_ast_roundtrip('def f(a, b = 2): pass')
        self.check_ast_roundtrip('def f(a = 5, b = 2): pass')
        self.check_ast_roundtrip('def f(*, a = 1, b = 2): pass')
        self.check_ast_roundtrip('def f(*, a = 1, b): pass')
        self.check_ast_roundtrip('def f(*, a, b = 2): pass')
        self.check_ast_roundtrip('def f(a, b = None, *, c, **kwds): pass')
        self.check_ast_roundtrip('def f(a=2, *args, c=5, d, **kwds): pass')
        self.check_ast_roundtrip('def f(*args, **kwargs): pass')

    def test_relative_import(self):
        if False:
            print('Hello World!')
        self.check_ast_roundtrip(relative_import)

    def test_nonlocal(self):
        if False:
            print('Hello World!')
        self.check_ast_roundtrip(nonlocal_ex)

    def test_raise_from(self):
        if False:
            while True:
                i = 10
        self.check_ast_roundtrip(raise_from)

    def test_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_ast_roundtrip("b'123'")

    def test_annotations(self):
        if False:
            i = 10
            return i + 15
        self.check_ast_roundtrip('def f(a : int): pass')
        self.check_ast_roundtrip('def f(a: int = 5): pass')
        self.check_ast_roundtrip('def f(*args: [int]): pass')
        self.check_ast_roundtrip('def f(**kwargs: dict): pass')
        self.check_ast_roundtrip('def f() -> None: pass')

    def test_set_literal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_ast_roundtrip("{'a', 'b', 'c'}")

    def test_empty_set(self):
        if False:
            return 10
        self.assertASTEqual(ast.parse(ast.unparse(ast.Set(elts=[]))), ast.parse('{*()}'))

    def test_set_comprehension(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_ast_roundtrip('{x for x in range(5)}')

    def test_dict_comprehension(self):
        if False:
            print('Hello World!')
        self.check_ast_roundtrip('{x: x*x for x in range(10)}')

    def test_class_decorators(self):
        if False:
            return 10
        self.check_ast_roundtrip(class_decorator)

    def test_class_definition(self):
        if False:
            print('Hello World!')
        self.check_ast_roundtrip('class A(metaclass=type, *[], **{}): pass')

    def test_elifs(self):
        if False:
            while True:
                i = 10
        self.check_ast_roundtrip(elif1)
        self.check_ast_roundtrip(elif2)

    def test_try_except_finally(self):
        if False:
            i = 10
            return i + 15
        self.check_ast_roundtrip(try_except_finally)

    def test_starred_assignment(self):
        if False:
            i = 10
            return i + 15
        self.check_ast_roundtrip('a, *b, c = seq')
        self.check_ast_roundtrip('a, (*b, c) = seq')
        self.check_ast_roundtrip('a, *b[0], c = seq')
        self.check_ast_roundtrip('a, *(b, c) = seq')

    def test_with_simple(self):
        if False:
            while True:
                i = 10
        self.check_ast_roundtrip(with_simple)

    def test_with_as(self):
        if False:
            while True:
                i = 10
        self.check_ast_roundtrip(with_as)

    def test_with_two_items(self):
        if False:
            while True:
                i = 10
        self.check_ast_roundtrip(with_two_items)

    def test_dict_unpacking_in_dict(self):
        if False:
            i = 10
            return i + 15
        self.check_ast_roundtrip("{**{'y': 2}, 'x': 1}")
        self.check_ast_roundtrip("{**{'y': 2}, **{'x': 1}}")

    def test_slices(self):
        if False:
            print('Hello World!')
        self.check_ast_roundtrip('a[i]')
        self.check_ast_roundtrip('a[i,]')
        self.check_ast_roundtrip('a[i, j]')
        self.check_ast_roundtrip('a[(*a,)]')
        self.check_ast_roundtrip('a[(a:=b)]')
        self.check_ast_roundtrip('a[(a:=b,c)]')
        self.check_ast_roundtrip('a[()]')
        self.check_ast_roundtrip('a[i:j]')
        self.check_ast_roundtrip('a[:j]')
        self.check_ast_roundtrip('a[i:]')
        self.check_ast_roundtrip('a[i:j:k]')
        self.check_ast_roundtrip('a[:j:k]')
        self.check_ast_roundtrip('a[i::k]')
        self.check_ast_roundtrip('a[i:j,]')
        self.check_ast_roundtrip('a[i:j, k]')

    def test_invalid_raise(self):
        if False:
            print('Hello World!')
        self.check_invalid(ast.Raise(exc=None, cause=ast.Name(id='X')))

    def test_invalid_fstring_constant(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_invalid(ast.JoinedStr(values=[ast.Constant(value=100)]))

    def test_invalid_fstring_conversion(self):
        if False:
            i = 10
            return i + 15
        self.check_invalid(ast.FormattedValue(value=ast.Constant(value='a', kind=None), conversion=ord('Y'), format_spec=None))

    def test_invalid_fstring_backslash(self):
        if False:
            print('Hello World!')
        self.check_invalid(ast.FormattedValue(value=ast.Constant(value='\\\\')))

    def test_invalid_yield_from(self):
        if False:
            while True:
                i = 10
        self.check_invalid(ast.YieldFrom(value=None))

    def test_docstrings(self):
        if False:
            i = 10
            return i + 15
        docstrings = ('this ends with double quote"', 'this includes a """triple quote"""', '\r', '\\r', '\t', '\\t', '\n', '\\n', '\r\\r\t\\t\n\\n', '""">>> content = """blabla""" <<<"""', 'foo\\n\\x00', '\' \\\'\\\'\\\'""" ""\\\'\\\' \\\'', '🐍⛎𩸽üéş^\\\\X\\\\BB⟿')
        for docstring in docstrings:
            self.check_ast_roundtrip(f"'''{docstring}'''")

    def test_constant_tuples(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_src_roundtrip(ast.Constant(value=(1,), kind=None), '(1,)')
        self.check_src_roundtrip(ast.Constant(value=(1, 2, 3), kind=None), '(1, 2, 3)')

    def test_function_type(self):
        if False:
            while True:
                i = 10
        for function_type in ('() -> int', '(int, int) -> int', '(Callable[complex], More[Complex(call.to_typevar())]) -> None'):
            self.check_ast_roundtrip(function_type, mode='func_type')

    def test_type_comments(self):
        if False:
            i = 10
            return i + 15
        for statement in ('a = 5 # type:', 'a = 5 # type: int', 'a = 5 # type: int and more', 'def x(): # type: () -> None\n\tpass', 'def x(y): # type: (int) -> None and more\n\tpass', 'async def x(): # type: () -> None\n\tpass', 'async def x(y): # type: (int) -> None and more\n\tpass', 'for x in y: # type: int\n\tpass', 'async for x in y: # type: int\n\tpass', 'with x(): # type: int\n\tpass', 'async with x(): # type: int\n\tpass'):
            self.check_ast_roundtrip(statement, type_comments=True)

    def test_type_ignore(self):
        if False:
            for i in range(10):
                print('nop')
        for statement in ('a = 5 # type: ignore', 'a = 5 # type: ignore and more', 'def x(): # type: ignore\n\tpass', 'def x(y): # type: ignore and more\n\tpass', 'async def x(): # type: ignore\n\tpass', 'async def x(y): # type: ignore and more\n\tpass', 'for x in y: # type: ignore\n\tpass', 'async for x in y: # type: ignore\n\tpass', 'with x(): # type: ignore\n\tpass', 'async with x(): # type: ignore\n\tpass'):
            self.check_ast_roundtrip(statement, type_comments=True)

class CosmeticTestCase(ASTTestCase):
    """Test if there are cosmetic issues caused by unnecessary additions"""

    def test_simple_expressions_parens(self):
        if False:
            print('Hello World!')
        self.check_src_roundtrip('(a := b)')
        self.check_src_roundtrip('await x')
        self.check_src_roundtrip('x if x else y')
        self.check_src_roundtrip('lambda x: x')
        self.check_src_roundtrip('1 + 1')
        self.check_src_roundtrip('1 + 2 / 3')
        self.check_src_roundtrip('(1 + 2) / 3')
        self.check_src_roundtrip('(1 + 2) * 3 + 4 * (5 + 2)')
        self.check_src_roundtrip('(1 + 2) * 3 + 4 * (5 + 2) ** 2')
        self.check_src_roundtrip('~x')
        self.check_src_roundtrip('x and y')
        self.check_src_roundtrip('x and y and z')
        self.check_src_roundtrip('x and (y and x)')
        self.check_src_roundtrip('(x and y) and z')
        self.check_src_roundtrip('(x ** y) ** z ** q')
        self.check_src_roundtrip('x >> y')
        self.check_src_roundtrip('x << y')
        self.check_src_roundtrip('x >> y and x >> z')
        self.check_src_roundtrip('x + y - z * q ^ t ** k')
        self.check_src_roundtrip('P * V if P and V else n * R * T')
        self.check_src_roundtrip('lambda P, V, n: P * V == n * R * T')
        self.check_src_roundtrip('flag & (other | foo)')
        self.check_src_roundtrip('not x == y')
        self.check_src_roundtrip('x == (not y)')
        self.check_src_roundtrip('yield x')
        self.check_src_roundtrip('yield from x')
        self.check_src_roundtrip('call((yield x))')
        self.check_src_roundtrip('return x + (yield x)')

    def test_class_bases_and_keywords(self):
        if False:
            i = 10
            return i + 15
        self.check_src_roundtrip('class X:\n    pass')
        self.check_src_roundtrip('class X(A):\n    pass')
        self.check_src_roundtrip('class X(A, B, C, D):\n    pass')
        self.check_src_roundtrip('class X(x=y):\n    pass')
        self.check_src_roundtrip('class X(metaclass=z):\n    pass')
        self.check_src_roundtrip('class X(x=y, z=d):\n    pass')
        self.check_src_roundtrip('class X(A, x=y):\n    pass')
        self.check_src_roundtrip('class X(A, **kw):\n    pass')
        self.check_src_roundtrip('class X(*args):\n    pass')
        self.check_src_roundtrip('class X(*args, **kwargs):\n    pass')

    def test_fstrings(self):
        if False:
            return 10
        self.check_src_roundtrip('f\'\'\'-{f"""*{f"+{f\'.{x}.\'}+"}*"""}-\'\'\'')
        self.check_src_roundtrip('f"\\u2028{\'x\'}"')
        self.check_src_roundtrip("f'{x}\\n'")
        self.check_src_roundtrip('f\'\'\'{"""\n"""}\\n\'\'\'')
        self.check_src_roundtrip('f\'\'\'{f"""{x}\n"""}\\n\'\'\'')

    def test_docstrings(self):
        if False:
            while True:
                i = 10
        docstrings = ('"""simple doc string"""', '"""A more complex one\n            with some newlines"""', '"""Foo bar baz\n\n            empty newline"""', '"""With some \t"""', '"""Foo "bar" baz """', '"""\\r"""', '""""""', '"""\'\'\'"""', '"""\'\'\'\'\'\'"""', '"""🐍⛎𩸽üéş^\\\\X\\\\BB⟿"""', '"""end in single \'quote\'"""', '\'\'\'end in double "quote"\'\'\'', '"""almost end in double "quote"."""')
        for prefix in docstring_prefixes:
            for docstring in docstrings:
                self.check_src_roundtrip(f'{prefix}{docstring}')

    def test_docstrings_negative_cases(self):
        if False:
            print('Hello World!')
        docstrings_negative = ('a = """false"""', '"""false""" + """unless its optimized"""', '1 + 1\n"""false"""', 'f"""no, top level but f-fstring"""')
        for prefix in docstring_prefixes:
            for negative in docstrings_negative:
                src = f'{prefix}{negative}'
                self.check_ast_roundtrip(src)
                self.check_src_dont_roundtrip(src)

    def test_unary_op_factor(self):
        if False:
            print('Hello World!')
        for prefix in ('+', '-', '~'):
            self.check_src_roundtrip(f'{prefix}1')
        for prefix in ('not',):
            self.check_src_roundtrip(f'{prefix} 1')

    def test_slices(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_src_roundtrip('a[1]')
        self.check_src_roundtrip('a[1, 2]')
        self.check_src_roundtrip('a[(1, *a)]')

class DirectoryTestCase(ASTTestCase):
    """Test roundtrip behaviour on all files in Lib and Lib/test."""
    lib_dir = pathlib.Path(__file__).parent / '..'
    test_directories = (lib_dir, lib_dir / 'test')
    run_always_files = {'test_grammar.py', 'test_syntax.py', 'test_compile.py', 'test_ast.py', 'test_asdl_parser.py', 'test_fstring.py', 'test_patma.py'}
    _files_to_test = None

    @classmethod
    def files_to_test(cls):
        if False:
            while True:
                i = 10
        if cls._files_to_test is not None:
            return cls._files_to_test
        items = [item.resolve() for directory in cls.test_directories for item in directory.glob('*.py') if not item.name.startswith('bad')]
        if not test.support.is_resource_enabled('cpu'):
            tests_to_run_always = {item for item in items if item.name in cls.run_always_files}
            items = set(random.sample(items, 10))
            items = list(items | tests_to_run_always)
        cls._files_to_test = items
        return items

    def test_files(self):
        if False:
            for i in range(10):
                print('nop')
        for item in self.files_to_test():
            if test.support.verbose:
                print(f'Testing {item.absolute()}')
            with self.subTest(filename=item):
                source = read_pyfile(item)
                self.check_ast_roundtrip(source)
if __name__ == '__main__':
    unittest.main()