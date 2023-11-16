from .common import StaticTestBase

class AugAssignTests(StaticTestBase):

    def test_aug_assign(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        def f(l):\n            l[0] += 1\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            l = [1]
            f(l)
            self.assertEqual(l[0], 2)

    def test_field(self):
        if False:
            while True:
                i = 10
        codestr = '\n        class C:\n            def __init__(self):\n                self.x = 1\n\n        def f(a: C):\n            a.x += 1\n        '
        code = self.compile(codestr, modname='foo')
        code = self.find_code(code, name='f')
        self.assertInBytecode(code, 'LOAD_FIELD', ('foo', 'C', 'x'))
        self.assertInBytecode(code, 'STORE_FIELD', ('foo', 'C', 'x'))

    def test_primitive_int(self):
        if False:
            while True:
                i = 10
        codestr = '\n        from __static__ import int8, box, unbox\n\n        def a(i: int) -> int:\n            j: int8 = unbox(i)\n            j += 2\n            return box(j)\n        '
        with self.in_module(codestr) as mod:
            a = mod.a
            self.assertInBytecode(a, 'PRIMITIVE_BINARY_OP', 0)
            self.assertEqual(a(3), 5)

    def test_inexact(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        def something():\n            return 3\n\n        def t():\n            a: int = something()\n\n            b = 0\n            b += a\n            return b\n        '
        with self.in_module(codestr) as mod:
            t = mod.t
            self.assertInBytecode(t, 'INPLACE_ADD')
            self.assertEqual(t(), 3)

    def test_list(self):
        if False:
            while True:
                i = 10
        for prim_idx in [True, False]:
            with self.subTest(prim_idx=prim_idx):
                codestr = f"\n                    from __static__ import int32\n\n                    def f(x: int):\n                        l = [x]\n                        i: {('int32' if prim_idx else 'int')} = 0\n                        l[i] += 1\n                        return l[i]\n                "
                with self.in_module(codestr) as mod:
                    self.assertEqual(mod.f(3), 4)

    def test_checked_list(self):
        if False:
            print('Hello World!')
        for prim_idx in [True, False]:
            with self.subTest(prim_idx=prim_idx):
                codestr = f"\n                    from __static__ import CheckedList, int32\n\n                    def f(x: int):\n                        l: CheckedList[int] = [x]\n                        i: {('int32' if prim_idx else 'int')} = 0\n                        l[i] += 1\n                        return l[i]\n                "
                with self.in_module(codestr) as mod:
                    self.assertEqual(mod.f(3), 4)

    def test_checked_dict(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __static__ import CheckedDict\n\n            def f(x: int):\n                d: CheckedDict[int, int] = {0: x}\n                d[0] += 1\n                return d[0]\n        '
        with self.in_module(codestr) as mod:
            self.assertEqual(mod.f(3), 4)