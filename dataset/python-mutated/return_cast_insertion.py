from .common import StaticTestBase

class ReturnCastInsertionTests(StaticTestBase):

    def test_no_cast_to_object(self) -> None:
        if False:
            while True:
                i = 10
        'We never cast to object, for object or dynamic or no annotation.'
        for is_async in [True, False]:
            for ann in ['object', 'open', None]:
                with self.subTest(ann=ann, is_async=is_async):
                    prefix = 'async ' if is_async else ''
                    full_ann = f' -> {ann}' if ann else ''
                    codestr = f'\n                        {prefix}def f(x){full_ann}:\n                            return x\n                    '
                    f_code = self.find_code(self.compile(codestr), 'f')
                    self.assertNotInBytecode(f_code, 'CAST')

    def test_annotated_method_does_not_cast_lower(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = f"\n            def f() -> str:\n                return 'abc'.lower()\n        "
        f_code = self.find_code(self.compile(codestr), 'f')
        self.assertNotInBytecode(f_code, 'CAST')
        self.assertInBytecode(f_code, 'REFINE_TYPE')

    def test_annotated_method_does_not_cast_upper(self) -> None:
        if False:
            return 10
        codestr = f"\n            def f() -> str:\n                return 'abc'.upper()\n        "
        f_code = self.find_code(self.compile(codestr), 'f')
        self.assertNotInBytecode(f_code, 'CAST')
        self.assertInBytecode(f_code, 'REFINE_TYPE')

    def test_annotated_method_does_not_cast_isdigit(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = f"\n            def f() -> bool:\n                return 'abc'.isdigit()\n        "
        f_code = self.find_code(self.compile(codestr), 'f')
        self.assertNotInBytecode(f_code, 'CAST')
        self.assertInBytecode(f_code, 'REFINE_TYPE')

    def test_annotated_method_does_not_cast_known_subclass(self) -> None:
        if False:
            return 10
        codestr = f"\n            class C(str):\n                pass\n\n            def f() -> bool:\n                return C('abc').isdigit()\n        "
        f_code = self.find_code(self.compile(codestr), 'f')
        self.assertNotInBytecode(f_code, 'CAST')
        self.assertInBytecode(f_code, 'REFINE_TYPE')

    def test_annotated_method_casts_arbitrary_subclass(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = f'\n            def f(x: str) -> bool:\n                return x.isdigit()\n        '
        f_code = self.find_code(self.compile(codestr), 'f')
        self.assertInBytecode(f_code, 'CAST')
        self.assertNotInBytecode(f_code, 'REFINE_TYPE')

    def test_annotated_method_does_not_cast_if_valid_on_subclasses(self) -> None:
        if False:
            while True:
                i = 10
        codestr = f'\n            from __static__ import ContextDecorator\n            class C(ContextDecorator):\n                pass\n\n            def f() -> ContextDecorator:\n                return C()._recreate_cm()\n        '
        f_code = self.find_code(self.compile(codestr), 'f')
        self.assertNotInBytecode(f_code, 'CAST')