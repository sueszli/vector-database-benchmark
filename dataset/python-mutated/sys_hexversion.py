import re
import sys
from compiler.static.types import TypedSyntaxError
from itertools import product
from .common import StaticTestBase

class SysHexVersionTests(StaticTestBase):

    def test_sys_hexversion(self):
        if False:
            while True:
                i = 10
        current_version = sys.hexversion
        for (version_check_should_pass, is_on_left) in product((False, True), (False, True)):
            with self.subTest(msg=f'version_check_should_pass={version_check_should_pass!r}, is_on_left={is_on_left!r}'):
                if version_check_should_pass:
                    checked_version = current_version - 1
                    expected_attribute = 'a'
                else:
                    checked_version = current_version + 1
                    expected_attribute = 'b'
                if is_on_left:
                    condition_str = f'sys.hexversion >= {hex(checked_version)}'
                else:
                    condition_str = f'{hex(checked_version)} <= sys.hexversion'
                codestr = f'\n                import sys\n\n                if {condition_str}:\n                    class A:\n                        def a(self):\n                            pass\n                else:\n                    class A:\n                        def b(self):\n                            pass\n                '
                with self.in_strict_module(codestr) as mod:
                    self.assertTrue(hasattr(mod.A, expected_attribute))

    def test_sys_hexversion_unsupported_operator(self):
        if False:
            while True:
                i = 10
        op_to_err = {'in': 'in', 'is': 'is', 'is not': 'is'}
        for (op, is_on_left) in product(op_to_err.keys(), (True, False)):
            if is_on_left:
                condition_str = f'sys.hexversion {op} 50988528'
            else:
                condition_str = f'50988528 {op} sys.hexversion'
            with self.subTest(msg=f'op={op!r}, is_on_left={is_on_left!r}'):
                codestr = f'\n                import sys\n\n                if {condition_str}:\n                    class A:\n                        def a(self):\n                            pass\n                else:\n                    class A:\n                        def b(self):\n                            pass\n                '
                with self.assertRaisesRegex(TypedSyntaxError, 'Cannot redefine local variable A'):
                    self.compile(codestr)

    def test_sys_hexversion_dynamic_compare(self):
        if False:
            print('Hello World!')
        codestr = f'\n        import sys\n        from something import X\n\n        if sys.hexversion >= X:\n            class A:\n                def a(self):\n                    pass\n        else:\n            class A:\n                def b(self):\n                    pass\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'Cannot redefine local variable A'):
            self.compile(codestr)

    def test_sys_hexversion_compare_double(self):
        if False:
            return 10
        codestr = f'\n        import sys\n\n        if sys.hexversion >= 3.12:\n            class A:\n                def a(self):\n                    pass\n        else:\n            class A:\n                def b(self):\n                    pass\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'Cannot redefine local variable A'):
            self.compile(codestr)