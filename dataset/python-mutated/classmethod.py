import asyncio
from unittest import skip, skipIf
from unittest.mock import patch
from .common import StaticTestBase
try:
    import cinderjit
except ImportError:
    cinderjit = None

class ClassMethodTests(StaticTestBase):

    def test_classmethod_from_non_final_class_calls_invoke_function(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                 @classmethod\n                 def foo(cls):\n                     return cls\n            def f():\n                return C.foo()\n        '
        with self.in_module(codestr, name='mymod') as mod:
            f = mod.f
            C = mod.C
            self.assertInBytecode(f, 'INVOKE_FUNCTION', (('mymod', 'C', 'foo'), 1))
            self.assertEqual(f(), C)

    def test_classmethod_from_final_class_calls_invoke_function(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from typing import final\n            @final\n            class C:\n                 @classmethod\n                 def foo(cls):\n                     return cls\n            def f():\n                return C.foo()\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            C = mod.C
            self.assertInBytecode(f, 'INVOKE_FUNCTION')
            self.assertEqual(f(), C)

    def test_classmethod_from_instance_calls_invoke_method(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            class C:\n                 @classmethod\n                 def foo(cls):\n                     return cls\n            def f(c: C):\n                return c.foo()\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            C = mod.C
            c = C()
            self.assertInBytecode(f, 'INVOKE_METHOD')
            self.assertEqual(f(c), C)

    def test_classmethod_override_from_instance_calls_override(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C:\n                 @classmethod\n                 def foo(cls, x: int) -> int:\n                     return x\n            class D(C):\n                 @classmethod\n                 def foo(cls, x: int) -> int:\n                     return x + 2\n\n            def f(c: C):\n                return c.foo(0)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            D = mod.D
            d = D()
            self.assertInBytecode(f, 'INVOKE_METHOD')
            self.assertEqual(f(d), 2)

    def test_classmethod_override_from_non_static_instance_calls_override(self):
        if False:
            print('Hello World!')
        codestr = '\n            class C:\n                 @classmethod\n                 def foo(cls, x: int) -> int:\n                     return x\n\n            def f(c: C) -> int:\n                return c.foo(42)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            C = mod.C

            class D(C):

                @classmethod
                def foo(cls, x: int) -> int:
                    if False:
                        print('Hello World!')
                    return x + 30
            d = D()
            self.assertInBytecode(f, 'INVOKE_METHOD')
            self.assertEqual(f(d), 72)

    def test_classmethod_non_class_method_override(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            class C:\n                 @classmethod\n                 def foo(cls, x: int) -> int:\n                     return x\n            class D(C):\n                 def foo(cls, x: int) -> int:\n                     return x + 2\n\n            def f(c: C):\n                return c.foo(0)\n        '
        self.type_error(codestr, 'class cannot hide inherited member')

    def test_classmethod_dynamic_call(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            class C:\n                def __init__(self, x: int) -> None:\n                    self.x = x\n\n                @classmethod\n                def foo(cls, *, x: int) -> int:\n                    return x\n\n            d = C.foo(x=1)\n        '
        with self.in_module(codestr) as mod:
            d = mod.d
            self.assertEqual(d, 1)

    def test_final_classmethod_calls_another(self):
        if False:
            return 10
        codestr = '\n            from typing import final\n            @final\n            class C:\n                @classmethod\n                def foo(cls) -> int:\n                    return 3\n\n                @classmethod\n                def bar(cls, i: int) -> int:\n                    return cls.foo() + i\n        '
        with self.in_module(codestr, name='mymod') as mod:
            C = mod.C
            self.assertInBytecode(C.bar, 'INVOKE_FUNCTION', (('mymod', 'C', 'foo'), 1))
            self.assertEqual(C.bar(6), 9)

    def test_classmethod_calls_another(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            class C:\n                @classmethod\n                def foo(cls) -> int:\n                    return 3\n\n                @classmethod\n                def bar(cls, i: int) -> int:\n                    return cls.foo() + i\n        '
        with self.in_module(codestr, name='mymod') as mod:
            C = mod.C
            self.assertNotInBytecode(C.bar, 'INVOKE_FUNCTION')
            self.assertInBytecode(C.bar, 'INVOKE_METHOD')
            self.assertEqual(C.bar(6), 9)

    def test_classmethod_calls_another_from_static_subclass(self):
        if False:
            return 10
        codestr = '\n            class C:\n                @classmethod\n                def foo(cls) -> int:\n                    return 3\n\n                @classmethod\n                def bar(cls, i: int) -> int:\n                    return cls.foo() + i\n            class D(C):\n                @classmethod\n                def foo(cls) -> int:\n                    return 42\n        '
        with self.in_module(codestr, name='mymod') as mod:
            D = mod.D
            self.assertInBytecode(D.bar, 'INVOKE_METHOD')
            self.assertEqual(D.bar(6), 48)

    def test_classmethod_calls_another_from_nonstatic_subclass(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            class C:\n                @classmethod\n                def foo(cls) -> int:\n                    return 3\n\n                @classmethod\n                def bar(cls, i: int) -> int:\n                    return cls.foo() + i\n        '
        with self.in_module(codestr, name='mymod') as mod:
            C = mod.C

            class D(C):

                @classmethod
                def foo(cls) -> int:
                    if False:
                        i = 10
                        return i + 15
                    return 42
            self.assertInBytecode(D.bar, 'INVOKE_METHOD')
            self.assertEqual(D.bar(6), 48)

    def test_classmethod_dynamic_subclass(self):
        if False:
            print('Hello World!')
        codestr = '\n            class C:\n                @classmethod\n                async def foo(cls) -> int:\n                    return 3\n\n                async def bar(self) -> int:\n                    return await self.foo()\n\n                def return_foo_typ(self):\n                    return self.foo()\n        '
        with self.in_module(codestr, name='mymod') as mod:
            C = mod.C

            class D(C):
                pass
            d = D()
            asyncio.run(d.bar())

    def test_patch(self):
        if False:
            print('Hello World!')
        codestr = '\n            class C:\n                @classmethod\n                def caller(cls):\n                    if cls.is_testing():\n                        return True\n                    return False\n\n                @classmethod\n                def is_testing(cls):\n                    return True\n\n            class Child(C):\n                pass\n        '
        with self.in_module(codestr) as mod:
            with patch(f'{mod.__name__}.C.is_testing', return_value=False) as p:
                c = mod.Child()
                self.assertEqual(c.caller(), False)
                self.assertEqual(p.call_args[0], (mod.Child,))

    def test_classmethod_on_type(self):
        if False:
            print('Hello World!')
        codestr = '\n            class C(type):\n                @classmethod\n                def x(cls):\n                    return cls\n\n            def f(c: C):\n                return c.x()\n\n            def f1(c: type[C]):\n                return c.x()\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.f(mod.C('foo', (object,), {})), mod.C)
            self.assertEqual(mod.f1(mod.C), mod.C)

    def test_classmethod_dynamic_subclass_override_async(self):
        if False:
            return 10
        codestr = '\n            class C:\n                @classmethod\n                async def foo(cls) -> int:\n                    return 3\n\n                async def bar(self) -> int:\n                    return await self.foo()\n\n                def return_foo_typ(self):\n                    return self.foo()\n        '
        with self.in_module(codestr, name='mymod') as mod:
            C = mod.C

            class D(C):

                async def foo(self) -> int:
                    return 42
            d = D()
            asyncio.run(d.bar())

    def test_classmethod_dynamic_subclass_override_nondesc_async(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            class C:\n                @classmethod\n                async def foo(cls) -> int:\n                    return 3\n\n                async def bar(self) -> int:\n                    return await self.foo()\n\n                def return_foo_typ(self):\n                    return self.foo()\n        '
        with self.in_module(codestr, name='mymod') as mod:
            C = mod.C

            class Callable:

                async def __call__(self):
                    return 42

            class D(C):
                foo = Callable()
            d = D()
            asyncio.run(d.bar())

    def test_classmethod_dynamic_subclass_override(self):
        if False:
            return 10
        codestr = '\n            class C:\n                @classmethod\n                def foo(cls) -> int:\n                    return 3\n\n                def bar(self) -> int:\n                    return self.foo()\n\n                def return_foo_typ(self):\n                    return self.foo()\n        '
        with self.in_module(codestr, name='mymod') as mod:
            C = mod.C

            class D(C):

                def foo(self) -> int:
                    if False:
                        return 10
                    return 42
            d = D()
            self.assertEqual(d.bar(), 42)

    def test_classmethod_other_dec(self):
        if False:
            print('Hello World!')
        codestr = '\n            from typing import final\n\n            def mydec(f):\n                return f\n            @final\n            class C:\n                @classmethod\n                @mydec\n                def foo(cls) -> int:\n                    return 3\n\n                def f(self):\n                    return self.foo()\n        '
        with self.in_module(codestr, name='mymod') as mod:
            C = mod.C
            self.assertEqual(C().f(), 3)

    def test_invoke_non_static_subtype_async_classmethod(self):
        if False:
            print('Hello World!')
        codestr = '\n            class C:\n                x = 3\n\n                @classmethod\n                async def f(cls) -> int:\n                    return cls.x\n\n                async def g(self) -> int:\n                    return await self.f()\n        '
        with self.in_module(codestr) as mod:

            class D(mod.C):
                pass
            d = D()
            self.assertEqual(asyncio.run(d.g()), 3)

    def test_classmethod_invoke_method_cached(self):
        if False:
            return 10
        cases = [True, False]
        for should_make_hot in cases:
            with self.subTest(should_make_hot=should_make_hot):
                codestr = '\n                    class C:\n                        @classmethod\n                        def foo(cls) -> int:\n                            return 3\n\n                    def f(c: C):\n                        return c.foo()\n                '
                with self.in_module(codestr, name='mymod') as mod:
                    C = mod.C
                    f = mod.f
                    c = C()
                    if should_make_hot:
                        for i in range(50):
                            f(c)
                    self.assertInBytecode(f, 'INVOKE_METHOD')
                    self.assertEqual(f(c), 3)

    def test_classmethod_async_invoke_method_cached(self):
        if False:
            while True:
                i = 10
        cases = [True, False]
        for should_make_hot in cases:
            with self.subTest(should_make_hot=should_make_hot):
                codestr = '\n                class C:\n                    async def instance_method(self) -> int:\n                        return (await self.foo())\n\n                    @classmethod\n                    async def foo(cls) -> int:\n                        return 3\n\n                async def f(c: C):\n                    return await c.instance_method()\n                '
                with self.in_module(codestr, name='mymod') as mod:
                    C = mod.C
                    f = mod.f

                    async def make_hot():
                        c = C()
                        for i in range(50):
                            await f(c)
                    if should_make_hot:
                        asyncio.run(make_hot())
                    self.assertInBytecode(C.instance_method, 'INVOKE_METHOD')
                    self.assertEqual(asyncio.run(f(C())), 3)

    def test_invoke_starargs(self):
        if False:
            while True:
                i = 10
        codestr = '\n\n            class C:\n                @classmethod\n                def foo(self, x: int) -> int:\n                    return 3\n\n                def f(self, *args):\n                    return self.foo(*args)\n        '
        with self.in_module(codestr, name='mymod') as mod:
            C = mod.C
            self.assertEqual(C().f(42), 3)

    def test_invoke_starargs_starkwargs(self):
        if False:
            print('Hello World!')
        codestr = '\n\n            class C:\n                @classmethod\n                def foo(self, x: int) -> int:\n                    return 3\n\n                def f(self, *args, **kwargs):\n                    return self.foo(*args, **kwargs)\n        '
        with self.in_module(codestr, name='mymod') as mod:
            C = mod.C
            self.assertNotInBytecode(C.f, 'INVOKE_METHOD')
            self.assertEqual(C().f(42), 3)