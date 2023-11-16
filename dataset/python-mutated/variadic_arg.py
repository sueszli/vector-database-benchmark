from compiler.errors import TypedSyntaxError
from unittest import skip, skipIf
from .common import StaticTestBase
try:
    import cinderjit
except ImportError:
    cinderjit = None

class VariadicArgTests(StaticTestBase):

    def test_load_iterable_arg(self):
        if False:
            return 10
        codestr = '\n        def x(a: int, b: int, c: str, d: float, e: float) -> int:\n            return 7\n\n        def y() -> int:\n            p = ("hi", 0.1, 0.2)\n            return x(1, 3, *p)\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 0)
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 1)
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 2)
        self.assertNotInBytecode(y, 'LOAD_ITERABLE_ARG', 3)
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertEqual(y_callable(), 7)

    def test_load_iterable_arg_default_overridden(self):
        if False:
            return 10
        codestr = '\n            def x(a: int, b: int, c: str, d: float = 10.1, e: float = 20.1) -> bool:\n                return bool(\n                    a == 1\n                    and b == 3\n                    and c == "hi"\n                    and d == 0.1\n                    and e == 0.2\n                )\n\n            def y() -> bool:\n                p = ("hi", 0.1, 0.2)\n                return x(1, 3, *p)\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertNotInBytecode(y, 'LOAD_ITERABLE_ARG', 3)
        self.assertNotInBytecode(y, 'LOAD_MAPPING_ARG', 3)
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertTrue(y_callable())

    def test_load_iterable_arg_multi_star(self):
        if False:
            print('Hello World!')
        codestr = '\n        def x(a: int, b: int, c: str, d: float, e: float) -> int:\n            return 7\n\n        def y() -> int:\n            p = (1, 3)\n            q = ("hi", 0.1, 0.2)\n            return x(*p, *q)\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertNotInBytecode(y, 'LOAD_ITERABLE_ARG')
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertEqual(y_callable(), 7)

    def test_load_iterable_arg_star_not_last(self):
        if False:
            print('Hello World!')
        codestr = "\n        def x(a: int, b: int, c: str, d: float, e: float) -> int:\n            return 7\n\n        def y() -> int:\n            p = (1, 3, 'abc', 0.1)\n            return x(*p, 1.0)\n        "
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertNotInBytecode(y, 'LOAD_ITERABLE_ARG')
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertEqual(y_callable(), 7)

    def test_load_iterable_arg_failure(self):
        if False:
            while True:
                i = 10
        codestr = '\n        def x(a: int, b: int, c: str, d: float, e: float) -> int:\n            return 7\n\n        def y() -> int:\n            p = ("hi", 0.1)\n            return x(1, 3, *p)\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 0)
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 1)
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 2)
        self.assertNotInBytecode(y, 'LOAD_ITERABLE_ARG', 3)
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            with self.assertRaises(IndexError):
                y_callable()

    def test_load_iterable_arg_sequence(self):
        if False:
            while True:
                i = 10
        codestr = '\n        def x(a: int, b: int, c: str, d: float, e: float) -> int:\n            return 7\n\n        def y() -> int:\n            p = ["hi", 0.1, 0.2]\n            return x(1, 3, *p)\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 0)
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 1)
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 2)
        self.assertNotInBytecode(y, 'LOAD_ITERABLE_ARG', 3)
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertEqual(y_callable(), 7)

    def test_load_iterable_arg_sequence_1(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        def x(a: int, b: int, c: str, d: float, e: float) -> int:\n            return 7\n\n        def gen():\n            for i in ["hi", 0.05, 0.2]:\n                yield i\n\n        def y() -> int:\n            g = gen()\n            return x(1, 3, *g)\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 0)
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 1)
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 2)
        self.assertNotInBytecode(y, 'LOAD_ITERABLE_ARG', 3)
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertEqual(y_callable(), 7)

    def test_load_iterable_arg_sequence_failure(self):
        if False:
            return 10
        codestr = '\n        def x(a: int, b: int, c: str, d: float, e: float) -> int:\n            return 7\n\n        def y() -> int:\n            p = ["hi", 0.1]\n            return x(1, 3, *p)\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 0)
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 1)
        self.assertInBytecode(y, 'LOAD_ITERABLE_ARG', 2)
        self.assertNotInBytecode(y, 'LOAD_ITERABLE_ARG', 3)
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            with self.assertRaises(IndexError):
                y_callable()

    def test_load_mapping_arg(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        def x(a: int, b: int, c: str, d: float=-0.1, e: float=1.1, f: str="something") -> bool:\n            return bool(f == "yo" and d == 1.0 and e == 1.1)\n\n        def y() -> bool:\n            d = {"d": 1.0}\n            return x(1, 3, "hi", f="yo", **d)\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertInBytecode(y, 'LOAD_MAPPING_ARG', 3)
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertTrue(y_callable())

    def test_load_mapping_and_iterable_args_failure_1(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Fails because we don't supply enough positional args\n        "
        codestr = '\n        def x(a: int, b: int, c: str, d: float=2.2, e: float=1.1, f: str="something") -> bool:\n            return bool(a == 1 and b == 3 and f == "yo" and d == 2.2 and e == 1.1)\n\n        def y() -> bool:\n            return x(1, 3, f="yo")\n        '
        with self.assertRaisesRegex(SyntaxError, 'Function foo.x expects a value for argument c'):
            self.compile(codestr, modname='foo')

    def test_load_mapping_arg_failure(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fails because we supply an extra kwarg\n        '
        codestr = '\n        def x(a: int, b: int, c: str, d: float=2.2, e: float=1.1, f: str="something") -> bool:\n            return bool(a == 1 and b == 3 and f == "yo" and d == 2.2 and e == 1.1)\n\n        def y() -> bool:\n            return x(1, 3, "hi", f="yo", g="lol")\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'Given argument g does not exist in the definition of foo.x'):
            self.compile(codestr, modname='foo')

    def test_load_mapping_arg_custom_class(self):
        if False:
            while True:
                i = 10
        '\n        Fails because we supply a custom class for the mapped args, instead of a dict\n        '
        codestr = '\n        def x(a: int, b: int, c: str="hello") -> bool:\n            return bool(a == 1 and b == 3 and c == "hello")\n\n        class C:\n            def __getitem__(self, key: str) -> str | None:\n                if key == "c":\n                    return "hi"\n\n            def keys(self):\n                return ["c"]\n\n        def y() -> bool:\n            return x(1, 3, **C())\n        '
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            with self.assertRaisesRegex(TypeError, 'argument after \\*\\* must be a dict, not C'):
                self.assertTrue(y_callable())

    def test_load_mapping_arg_use_defaults(self):
        if False:
            return 10
        codestr = '\n        def x(a: int, b: int, c: str, d: float=-0.1, e: float=1.1, f: str="something") -> bool:\n            return bool(f == "yo" and d == -0.1 and e == 1.1)\n\n        def y() -> bool:\n            d = {"d": 1.0}\n            return x(1, 3, "hi", f="yo")\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertInBytecode(y, 'LOAD_CONST', 1.1)
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertTrue(y_callable())

    def test_default_arg_non_const(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        class C: pass\n        def x(val=C()) -> C:\n            return val\n\n        def f() -> C:\n            return x()\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertInBytecode(f, 'CALL_FUNCTION')

    def test_default_arg_non_const_kw_provided(self):
        if False:
            print('Hello World!')
        codestr = '\n        class C: pass\n        def x(val:object=C()):\n            return val\n\n        def f():\n            return x(val=42)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(), 42)

    def test_load_mapping_arg_order(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        def x(a: int, b: int, c: str, d: float=-0.1, e: float=1.1, f: str="something") -> bool:\n            return bool(\n                a == 1\n                and b == 3\n                and c == "hi"\n                and d == 1.1\n                and e == 3.3\n                and f == "hmm"\n            )\n\n        stuff = []\n        def q() -> float:\n            stuff.append("q")\n            return 1.1\n\n        def r() -> float:\n            stuff.append("r")\n            return 3.3\n\n        def s() -> str:\n            stuff.append("s")\n            return "hmm"\n\n        def y() -> bool:\n            return x(1, 3, "hi", f=s(), d=q(), e=r())\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertInBytecode(y, 'STORE_FAST', '_pystatic_.0._tmp__d')
        self.assertInBytecode(y, 'LOAD_FAST', '_pystatic_.0._tmp__d')
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertTrue(y_callable())
            self.assertEqual(['s', 'q', 'r'], mod.stuff)

    def test_load_mapping_arg_order_with_variadic_kw_args(self):
        if False:
            while True:
                i = 10
        codestr = '\n        def x(a: int, b: int, c: str, d: float=-0.1, e: float=1.1, f: str="something", g: str="look-here") -> bool:\n            return bool(\n                a == 1\n                and b == 3\n                and c == "hi"\n                and d == 1.1\n                and e == 3.3\n                and f == "hmm"\n                and g == "overridden"\n            )\n\n        stuff = []\n        def q() -> float:\n            stuff.append("q")\n            return 1.1\n\n        def r() -> float:\n            stuff.append("r")\n            return 3.3\n\n        def s() -> str:\n            stuff.append("s")\n            return "hmm"\n\n        def y() -> bool:\n            kw = {"g": "overridden"}\n            return x(1, 3, "hi", f=s(), **kw, d=q(), e=r())\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertInBytecode(y, 'STORE_FAST', '_pystatic_.0._tmp__d')
        self.assertInBytecode(y, 'LOAD_FAST', '_pystatic_.0._tmp__d')
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertTrue(y_callable())
            self.assertEqual(['s', 'q', 'r'], mod.stuff)

    def test_load_mapping_arg_order_with_variadic_kw_args_one_positional(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        def x(a: int, b: int, c: str, d: float=-0.1, e: float=1.1, f: str="something", g: str="look-here") -> bool:\n            return bool(\n                a == 1\n                and b == 3\n                and c == "hi"\n                and d == 1.1\n                and e == 3.3\n                and f == "hmm"\n                and g == "overridden"\n            )\n\n        stuff = []\n        def q() -> float:\n            stuff.append("q")\n            return 1.1\n\n        def r() -> float:\n            stuff.append("r")\n            return 3.3\n\n        def s() -> str:\n            stuff.append("s")\n            return "hmm"\n\n\n        def y() -> bool:\n            kw = {"g": "overridden"}\n            return x(1, 3, "hi", 1.1, f=s(), **kw, e=r())\n        '
        y = self.find_code(self.compile(codestr, modname='foo'), name='y')
        self.assertNotInBytecode(y, 'STORE_FAST', '_pystatic_.0._tmp__d')
        self.assertNotInBytecode(y, 'LOAD_FAST', '_pystatic_.0._tmp__d')
        with self.in_module(codestr) as mod:
            y_callable = mod.y
            self.assertTrue(y_callable())
            self.assertEqual(['s', 'r'], mod.stuff)

    def test_load_mapping_arg_stack_effect(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        def g(x=None) -> None:\n            pass\n\n        def f():\n            return [\n                g(**{})\n                for i in ()\n            ]\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            if self._inline_comprehensions:
                self.assertInBytecode(f, 'LOAD_MAPPING_ARG', 3)
            else:
                self.assertNotInBytecode(f, 'LOAD_MAPPING_ARG', 3)
            self.assertEqual(f(), [])