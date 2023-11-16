"""Test operators, using __any_object__."""
from pytype.tests import test_base

class OperatorsWithAnyTests(test_base.BaseTest):
    """Operator tests."""

    @test_base.skip('Needs __radd__ on all builtins')
    def test_add1(self):
        if False:
            i = 10
            return i + 15
        'Test that __add__, __radd__ are working.'
        ty = self.Infer('\n      def t_testAdd1(x):\n        return x + 2.0\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def t_testAdd1(x: Union[int, float, complex, bool]) -> Union[float, complex]: ...\n    ')

    @test_base.skip('Needs __radd__ on all builtins')
    def test_add2(self):
        if False:
            while True:
                i = 10
        'Test that __add__, __radd__ are working.'
        ty = self.Infer('\n      def t_testAdd2(x):\n        return 2.0 + x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def t_testAdd2(x: Union[int, float, complex, bool]) -> Union[float, complex]: ...\n    ')

    def test_add3(self):
        if False:
            i = 10
            return i + 15
        'Test that __add__, __radd__ are working.'
        ty = self.Infer('\n      def t_testAdd3(x):\n        return x + "abc"\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def t_testAdd3(x) -> Any: ...\n    ')

    def test_str_mul(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that __mul__, __rmul__ are working.'
        ty = self.Infer('\n      def t_testAdd4(x):\n        return "abc" * x\n    ')
        self.assertTypesMatchPytd(ty, '\n      def t_testAdd4(x) -> str: ...\n    ')

    def test_pow1(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def t_testPow1(x, y):\n        return x ** y\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def t_testPow1(x, y) -> Any: ...\n    ')

    def test_isinstance1(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def t_testIsinstance1(x):\n        return isinstance(x, int)\n    ')
        self.assertTypesMatchPytd(ty, '\n      def t_testIsinstance1(x) -> bool: ...\n    ')

    def test_call_any(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      t_testCallAny = __any_object__\n      t_testCallAny()  # error because there\'s no "def f()..."\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      t_testCallAny = ...  # type: Any\n    ')

    @test_base.skip('Needs NameError support')
    def test_undefined_module(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def t_testSys():\n        return sys\n      t_testSys()\n      ', deep=False)
        self.assertEqual(ty.Lookup('t_testSys').signatures[0].exceptions, self.nameerror)

    def test_subscr(self):
        if False:
            print('Hello World!')
        self.Check('\n      x = "foo" if __random__ else __any_object__\n      d = {"foo": 42}\n      d[x]  # BINARY_SUBSCR\n      "foo" + x  # BINARY_ADD\n      "%s" % x  # BINARY_MODULO\n    ')

    def test_bad_add(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n       x = "foo" if __random__ else None\n       "foo" + x  # unsupported-operands[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['unsupported operand type(s) for +: str and None']})

    def test_object_and_any(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Any\n      foo: object\n      bar: Any\n      print(foo + bar)\n    ')
if __name__ == '__main__':
    test_base.main()