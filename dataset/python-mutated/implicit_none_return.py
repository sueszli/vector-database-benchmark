from .common import StaticTestBase

class ImplicitNoneReturnTests(StaticTestBase):

    def test_implicit_none_return_good(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            def f() -> int | None:\n                pass\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.f(), None)

    def test_implicit_none_return_error(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n            def f() -> int:\n                pass\n        '
        self.type_error(codestr, "Function has declared return type 'int' but can implicitly return None", 'def f() -> int')