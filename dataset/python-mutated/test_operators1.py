"""Test operators (basic tests)."""
from pytype.tests import test_base
from pytype.tests import test_utils

class ConcreteTest(test_base.BaseTest, test_utils.OperatorsTestMixin):
    """Tests for operators on concrete values (no unknowns)."""

    def test_add(self):
        if False:
            while True:
                i = 10
        self.check_expr('x + y', ['x=1', 'y=2'], 'int')
        self.check_expr('x + y', ['x=1.0', 'y=2'], 'float')
        self.check_expr('x + y', ['x=1', 'y=2.0'], 'float')
        self.check_expr('x + y', ['x=1.1', 'y=2.1'], 'float')

    def test_add2(self):
        if False:
            return 10
        self.check_expr('x + y', ['x=1', 'y=2j'], 'complex')
        self.check_expr('x + y', ['x=1.0', 'y=2j'], 'complex')
        self.check_expr('x + y', ['x=2j', 'y=1'], 'complex')
        self.check_expr('x + y', ['x=3+2j', 'y=1.0'], 'complex')
        self.check_expr('x + y', ['x=1j', 'y=2j'], 'complex')

    def test_add3(self):
        if False:
            while True:
                i = 10
        self.check_expr('x + y', ["x='1'", "y='2'"], 'str')
        self.check_expr('x + y', ['x=[1]', 'y=[2]'], 'list[int]')
        self.check_expr('x + y', ['a=1', 'x=[a,a,a]', 'y=[a,a,a]'], 'list[int]')
        self.check_expr('x + y', ['a=1', 'x=[a,a,a]', 'y=[]'], 'list[int]')
        self.check_expr('x + y', ['a=1', 'x=[]', 'y=[a,a,a]'], 'list[int]')

    def test_add4(self):
        if False:
            while True:
                i = 10
        self.check_expr('x + y', ['x=[]', 'y=[]'], 'list[nothing]')
        self.check_expr('x + y', ['x=[1]', "y=['abc']"], 'list[int | str]')
        self.check_expr('x + y', ['x=(1,)', 'y=(2,)'], 'tuple[int, int]')
        self.check_expr('x + y', ['x=(1,)', 'y=(2.0,)'], 'tuple[int, float]')

    def test_and(self):
        if False:
            return 10
        self.check_expr('x & y', ['x=3', 'y=5'], 'int')
        self.check_expr('x & y', ['x={1}', 'y={1, 2}'], 'set[int]')
        self.check_expr('x & y', ['x={1}', 'y={1.2}'], 'set[int]')
        self.check_expr('x & y', ['x={1, 2}', 'y=set([1])'], 'set[int]')
        self.check_expr('x & y', ['x=1', 'y=2'], 'int')

    def test_frozenset_ops(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_expr('x & y', ['x=frozenset()', 'y=frozenset()'], 'frozenset[nothing]')
        self.check_expr('x - y', ['x=frozenset()', 'y=frozenset()'], 'frozenset[nothing]')
        self.check_expr('x | y', ['x=frozenset([1.0])', 'y=frozenset([2.2])'], 'frozenset[float]')

    def test_contains(self):
        if False:
            print('Hello World!')
        self.check_expr('x in y', ['x=[1]', 'y=[1, 2]'], 'bool')
        self.check_expr('x in y', ["x='ab'", "y='abcd'"], 'bool')
        self.check_expr('x in y', ["x='ab'", "y=['abcd']"], 'bool')

    def test_div(self):
        if False:
            i = 10
            return i + 15
        self.check_expr('x / y', ['x=1.0', 'y=2'], 'float')
        self.check_expr('x / y', ['x=1', 'y=2.0'], 'float')
        self.check_expr('x / y', ['x=1.1', 'y=2.1'], 'float')
        self.check_expr('x / y', ['x=1j', 'y=2j'], 'complex')

    def test_div2(self):
        if False:
            return 10
        self.check_expr('x / y', ['x=1', 'y=2j'], 'complex')
        self.check_expr('x / y', ['x=1.0', 'y=2j'], 'complex')
        self.check_expr('x / y', ['x=2j', 'y=1j'], 'complex')
        self.check_expr('x / y', ['x=2j', 'y=1'], 'complex')
        self.check_expr('x / y', ['x=3+2j', 'y=1.0'], 'complex')

    def test_floordiv(self):
        if False:
            print('Hello World!')
        self.check_expr('x // y', ['x=1', 'y=2'], 'int')
        self.check_expr('x // y', ['x=1.0', 'y=2'], 'float')
        self.check_expr('x // y', ['x=1', 'y=2.0'], 'float')
        self.check_expr('x // y', ['x=1.1', 'y=2.1'], 'float')
        self.check_expr('x // y', ['x=1j', 'y=2j'], 'complex')

    def test_floordiv2(self):
        if False:
            i = 10
            return i + 15
        self.check_expr('x // y', ['x=1', 'y=2j'], 'complex')
        self.check_expr('x // y', ['x=1.0', 'y=2j'], 'complex')
        self.check_expr('x // y', ['x=2j', 'y=1j'], 'complex')
        self.check_expr('x // y', ['x=2j', 'y=1'], 'complex')
        self.check_expr('x // y', ['x=3+2j', 'y=1.0'], 'complex')

    def test_invert(self):
        if False:
            i = 10
            return i + 15
        self.check_expr('~x', ['x=3'], 'int')
        self.check_expr('~x', ['x=False'], 'int')

    def test_lshift(self):
        if False:
            return 10
        self.check_expr('x << y', ['x=1', 'y=2'], 'int')

    def test_rshift(self):
        if False:
            print('Hello World!')
        self.check_expr('x >> y', ['x=1', 'y=2'], 'int')

    def test_sub(self):
        if False:
            print('Hello World!')
        self.check_expr('x - y', ['x=1', 'y=2'], 'int')
        self.check_expr('x - y', ['x=1.0', 'y=2'], 'float')
        self.check_expr('x - y', ['x=1', 'y=2.0'], 'float')
        self.check_expr('x - y', ['x=1.1', 'y=2.1'], 'float')

    def test_sub2(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_expr('x - y', ['x=1j', 'y=2j'], 'complex')
        self.check_expr('x - y', ['x={1}', 'y={1, 2}'], 'set[int]')
        self.check_expr('x - y', ['x={1}', 'y={1.2}'], 'set[int]')
        self.check_expr('x - y', ['x={1, 2}', 'y=set([1])'], 'set[int]')

    def test_sub_frozenset(self):
        if False:
            return 10
        self.check_expr('x - y', ['x={1, 2}', 'y=frozenset([1.0])'], 'set[int]')

    def test_mod(self):
        if False:
            return 10
        self.check_expr('x % y', ['x=1', 'y=2'], 'int')
        self.check_expr('x % y', ['x=1.5', 'y=2.5'], 'float')
        self.check_expr('x % y', ["x='%r'", 'y=set()'], 'str')

    def test_mul(self):
        if False:
            while True:
                i = 10
        self.check_expr('x * y', ['x=1', 'y=2'], 'int')
        self.check_expr('x * y', ['x=1', 'y=2.1'], 'float')
        self.check_expr('x * y', ['x=1+2j', 'y=2.1+3.4j'], 'complex')
        self.check_expr('x * y', ["x='x'", 'y=3'], 'str')
        self.check_expr('x * y', ['x=3', "y='x'"], 'str')

    def test_mul2(self):
        if False:
            print('Hello World!')
        self.check_expr('x * y', ['x=[1, 2]', 'y=3'], 'list[int]')
        self.check_expr('x * y', ['x=99', 'y=[1.0, 2]'], 'list[int | float]')
        self.check_expr('x * y', ['x=(1, 2)', 'y=3'], 'tuple[int, ...]')
        self.check_expr('x * y', ['x=0', 'y=(1, 2.0)'], 'tuple[int | float, ...]')

    def test_neg(self):
        if False:
            while True:
                i = 10
        self.check_expr('-x', ['x=1'], 'int')
        self.check_expr('-x', ['x=1.5'], 'float')
        self.check_expr('-x', ['x=1j'], 'complex')

    def test_or(self):
        if False:
            return 10
        self.check_expr('x | y', ['x=1', 'y=2'], 'int')
        self.check_expr('x | y', ['x={1}', 'y={2}'], 'set[int]')

    def test_pos(self):
        if False:
            print('Hello World!')
        self.check_expr('+x', ['x=1'], 'int')
        self.check_expr('+x', ['x=1.5'], 'float')
        self.check_expr('+x', ['x=2 + 3.1j'], 'complex')

    def test_pow(self):
        if False:
            while True:
                i = 10
        self.check_expr('x ** y', ['x=1', 'y=2'], 'int | float')
        self.check_expr('x ** y', ['x=1', 'y=-2'], 'int | float')
        self.check_expr('x ** y', ['x=1.0', 'y=2'], 'float')
        self.check_expr('x ** y', ['x=1', 'y=2.0'], 'float')
        self.check_expr('x ** y', ['x=1.1', 'y=2.1'], 'float')
        self.check_expr('x ** y', ['x=1j', 'y=2j'], 'complex')

    def test_xor(self):
        if False:
            print('Hello World!')
        self.check_expr('x ^ y', ['x=1', 'y=2'], 'int')
        self.check_expr('x ^ y', ['x={1}', 'y={2}'], 'set[int]')

    def test_add_type_parameter_instance(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Union\n      v = None  # type: Union[str]\n      d = {v: 42}\n      for k, _ in sorted(d.items()):\n        k + " as "\n    ')

class OverloadTest(test_base.BaseTest, test_utils.OperatorsTestMixin):
    """Tests for overloading operators."""

    def test_add(self):
        if False:
            return 10
        self.check_binary('__add__', '+')

    def test_and(self):
        if False:
            print('Hello World!')
        self.check_binary('__and__', '&')

    def test_or(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_binary('__or__', '|')

    def test_sub(self):
        if False:
            while True:
                i = 10
        self.check_binary('__sub__', '-')

    def test_floordiv(self):
        if False:
            i = 10
            return i + 15
        self.check_binary('__floordiv__', '//')

    def test_mod(self):
        if False:
            while True:
                i = 10
        self.check_binary('__mod__', '%')

    def test_mul(self):
        if False:
            i = 10
            return i + 15
        self.check_binary('__mul__', '*')

    def test_pow(self):
        if False:
            return 10
        self.check_binary('__pow__', '**')

    def test_lshift(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_binary('__lshift__', '<<')

    def test_rshift(self):
        if False:
            print('Hello World!')
        self.check_binary('__rshift__', '>>')

    def test_invert(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_unary('__invert__', '~')

    def test_neg(self):
        if False:
            return 10
        self.check_unary('__neg__', '-')

    def test_pos(self):
        if False:
            print('Hello World!')
        self.check_unary('__pos__', '+')

    def test_nonzero(self):
        if False:
            i = 10
            return i + 15
        self.check_unary('__nonzero__', 'not', 'bool')

class ReverseTest(test_base.BaseTest, test_utils.OperatorsTestMixin):
    """Tests for reverse operators."""

    def test_add(self):
        if False:
            print('Hello World!')
        self.check_reverse('add', '+')

    def test_and(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_reverse('and', '&')

    def test_floordiv(self):
        if False:
            print('Hello World!')
        self.check_reverse('floordiv', '//')

    def test_lshift(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_reverse('lshift', '<<')

    def test_rshift(self):
        if False:
            i = 10
            return i + 15
        self.check_reverse('rshift', '>>')

    def test_mod(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_reverse('mod', '%')

    def test_mul(self):
        if False:
            while True:
                i = 10
        self.check_reverse('mul', '*')

    def test_or(self):
        if False:
            return 10
        self.check_reverse('or', '|')

    def test_pow(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_reverse('pow', '**')

    def test_sub(self):
        if False:
            print('Hello World!')
        self.check_reverse('sub', '-')

    def test_custom(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('test.pyi', '\n        from typing import Tuple\n        class Test():\n          def __or__(self, other: Tuple[int, ...]) -> bool: ...\n          def __ror__(self, other: Tuple[int, ...]) -> bool: ...\n      ')
            ty = self.Infer('\n        import test\n        x = test.Test() | (1, 2)\n        y = (1, 2) | test.Test()\n        def f(t):\n          return t | (1, 2)\n        def g(t):\n          return (1, 2) | t\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import test\n        from typing import Any\n        x = ...  # type: bool\n        y = ...  # type: bool\n        def f(t) -> Any: ...\n        def g(t) -> Any: ...\n      ')

    def test_custom_reverse_unused(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def __sub__(self, other):\n          return 42\n        def __rsub__(self, other):\n          return ""\n      (Foo() - Foo()).real\n    ')

    def test_inherited_custom_reverse_unused(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo:\n        def __sub__(self, other):\n          return 42\n        def __rsub__(self, other):\n          return ""\n      class Bar(Foo):\n        pass\n      (Foo() - Bar()).real\n    ')

    def test_custom_reverse_only(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        def __sub__(self, other):\n          return ""\n      class Bar(Foo):\n        def __rsub__(self, other):\n          return 42\n      (Foo() - Bar()).real\n    ')

    def test_unknown_left(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def __rsub__(self, other):\n          return ""\n      (__any_object__ - Foo()).real\n    ')

    def test_unknown_right(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        def __sub__(self, other):\n          return ""\n      (Foo() - __any_object__).real  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'real.*str'})

class InplaceTest(test_base.BaseTest, test_utils.OperatorsTestMixin):
    """Tests for in-place operators."""

    def test_add(self):
        if False:
            print('Hello World!')
        self.check_inplace('iadd', '+=')

    def test_and(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_inplace('iand', '&=')

    def test_floordiv(self):
        if False:
            i = 10
            return i + 15
        self.check_inplace('ifloordiv', '//=')

    def test_lshift(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_inplace('ilshift', '<<=')

    def test_rshift(self):
        if False:
            i = 10
            return i + 15
        self.check_inplace('irshift', '>>=')

    def test_mod(self):
        if False:
            i = 10
            return i + 15
        self.check_inplace('imod', '%=')

    def test_mul(self):
        if False:
            return 10
        self.check_inplace('imul', '*=')

    def test_or(self):
        if False:
            return 10
        self.check_inplace('ior', '|=')

    def test_pow(self):
        if False:
            i = 10
            return i + 15
        self.check_inplace('ipow', '**=')

    def test_sub(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_inplace('isub', '-=')

    def test_list_add(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      class A: pass\n      v = []\n      v += A()  # unsupported-operands[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'A.*Iterable'})

class BindingsTest(test_base.BaseTest):
    """Tests that we correctly handle results without bindings."""

    def test_subscr(self):
        if False:
            while True:
                i = 10
        self.options.tweak(report_errors=False)
        self.InferWithErrors("\n      { 'path': __path__[0] }\n    ")
if __name__ == '__main__':
    test_base.main()