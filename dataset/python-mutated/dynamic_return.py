from unittest import skip
from .common import StaticTestBase

class DynamicReturnTests(StaticTestBase):

    def test_dynamic_return(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __future__ import annotations\n            from __static__ import allow_weakrefs, dynamic_return\n            import weakref\n\n            singletons = []\n\n            @allow_weakrefs\n            class C:\n                @dynamic_return\n                @staticmethod\n                def make() -> C:\n                    return weakref.proxy(singletons[0])\n\n                def g(self) -> int:\n                    return 1\n\n            singletons.append(C())\n\n            def f() -> int:\n                c = C.make()\n                return c.g()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.C.make, 'CAST')
            self.assertInBytecode(mod.f, 'INVOKE_FUNCTION', ((mod.__name__, 'C', 'make'), 0))
            self.assertNotInBytecode(mod.f, 'INVOKE_METHOD')
            self.assertEqual(mod.f(), 1)
            self.assertEqual(mod.C.make.__annotations__, {'return': 'C'})

    def test_dynamic_return_known_type(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from __future__ import annotations\n            from __static__ import allow_weakrefs, dynamic_return\n            import weakref\n\n            singletons = []\n\n            @allow_weakrefs\n            class C:\n                @dynamic_return\n                @staticmethod\n                def make() -> C:\n                    return 1\n\n            singletons.append(C())\n\n            def f() -> int:\n                return C.make()\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.C.make, 'CAST')
            self.assertInBytecode(mod.f, 'INVOKE_FUNCTION', ((mod.__name__, 'C', 'make'), 0))
            self.assertNotInBytecode(mod.f, 'INVOKE_METHOD')
            self.assertEqual(mod.f(), 1)
            self.assertEqual(mod.C.make.__annotations__, {'return': 'C'})

    def test_dynamic_return_async_fn(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        from __static__ import dynamic_return\n\n        class C:\n            @dynamic_return\n            def fn(self) -> int:\n                return 3\n\n        def f() -> int:\n            return C().fn()\n        '
        with self.in_strict_module(codestr) as mod:
            f = mod.f
            self.assertInBytecode(f, 'CAST', ('builtins', 'int'))
            self.assertEqual(f(), 3)