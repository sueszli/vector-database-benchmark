from compiler.static.types import TypedSyntaxError
from .common import StaticTestBase, type_mismatch

class CRangeTests(StaticTestBase):

    def test_crange_only_limit(self):
        if False:
            while True:
                i = 10
        codestr = '\n        from __static__ import crange, int64, box\n\n        def sum_until(n: int) -> int:\n            x: int64 = 0\n            for j in crange(int64(n)):\n                x += j\n            return box(x)\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.sum_until, 'FOR_ITER')
            self.assertEqual(mod.sum_until(6), 15)

    def test_crange_start_and_limit(self):
        if False:
            return 10
        codestr = '\n        from __static__ import crange, int64, box\n\n        def sum_between(m: int, n: int) -> int:\n            x: int64 = 0\n            for j in crange(int64(m), int64(n)):\n                x += j\n            return box(x)\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertNotInBytecode(mod.sum_between, 'FOR_ITER')
            self.assertEqual(mod.sum_between(3, 6), 12)

    def test_crange_incorrect_arg_count(self):
        if False:
            print('Hello World!')
        codestr = '\n        from __static__ import crange, int64, box\n\n        for j in crange():\n            pass\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'crange\\(\\) accepts only 1 or 2 parameters'):
            self.compile(codestr)
        other_codestr = '\n        from __static__ import crange, int64, box\n\n        for j in crange(int64(0), int64(1), int64(2)):\n            pass\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'crange\\(\\) accepts only 1 or 2 parameters'):
            self.compile(other_codestr)

    def test_crange_break_start_and_limit(self):
        if False:
            print('Hello World!')
        codestr = '\n        from __static__ import crange, int64, box\n\n        def sum_between(m: int, n: int) -> int:\n            x: int64 = 0\n            for j in crange(int64(m), int64(n)):\n                x += j\n                if x == 7:\n                    break\n            return box(x)\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertEqual(mod.sum_between(3, 6), 7)

    def test_crange_break_only_limit(self):
        if False:
            return 10
        codestr = '\n        from __static__ import crange, int64, box\n\n        def sum_until(n: int) -> int:\n            x: int64 = 0\n            for j in crange(int64(n)):\n                x += j\n                if x == 6:\n                    break\n            return box(x)\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertEqual(mod.sum_until(6), 6)

    def test_crange_orelse_iterator_exhausted(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        from __static__ import crange, int64, box\n\n        def sum_until(n: int) -> int:\n            x: int64 = 0\n            for j in crange(int64(n)):\n                x += j\n            else:\n                return 666\n            return box(x)\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertEqual(mod.sum_until(6), 666)

    def test_crange_orelse_iterator_not_exhausted(self):
        if False:
            return 10
        codestr = '\n        from __static__ import crange, int64, box\n\n        def sum_until(n: int) -> int:\n            x: int64 = 0\n            for j in crange(int64(n)):\n                x += j\n                if x == 6:\n                    break\n            else:\n                return 666\n            return box(x)\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertEqual(mod.sum_until(6), 6)

    def test_crange_without_for_loop(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        from __static__ import crange, int64, box\n\n        def bad_fn():\n            x: int64 = 1\n            y: int64 = 4\n\n            z = crange(1, 4)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'crange\\(\\) must be used as an iterator in a for loop'):
            self.compile(codestr)

    def test_crange_in_loop_body(self):
        if False:
            while True:
                i = 10
        codestr = '\n        from __static__ import crange, int64, box\n\n        def sum_until() -> None:\n            x: int64 = 0\n            for j in range(4):\n                p = crange(int64(14))\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'crange\\(\\) must be used as an iterator in a for loop'):
            self.compile(codestr)

    def test_crange_incompatible_arg_types(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n        from __static__ import crange, int64, box\n\n        for j in crange(12):\n            pass\n        '
        with self.assertRaisesRegex(TypedSyntaxError, "can't use crange with arg: Literal\\[12\\]"):
            self.compile(codestr)
        codestr = '\n        from __static__ import crange, int64, box\n\n        for j in crange(object()):\n            pass\n        '
        with self.assertRaisesRegex(TypedSyntaxError, "can't use crange with arg: object"):
            self.compile(codestr)

    def test_crange_continue(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        from __static__ import crange, int64, box\n\n        def run_loop() -> int:\n            n: int64 = 7\n            c = 0\n            for i in crange(n):\n                c += 1\n                continue\n            return c\n        '
        with self.in_strict_module(codestr) as mod:
            self.assertEqual(mod.run_loop(), 7)