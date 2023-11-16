import ast
from compiler.strict.common import DEFAULT_STUB_PATH
from compiler.strict.flag_extractor import BadFlagException, FlagExtractor, Flags
from textwrap import dedent
from typing import final, Optional, Sequence
from cinderx.strictmodule import StrictAnalysisResult, StrictModuleLoader
from .common import StrictTestBase
from .sandbox import sandbox

@final
class FlagExtractorTest(StrictTestBase):

    def _get_flags(self, code: str) -> Flags:
        if False:
            print('Hello World!')
        code = dedent(code)
        pyast = ast.parse(code)
        flags = FlagExtractor().get_flags(pyast)
        return flags

    def test_strict_import(self):
        if False:
            for i in range(10):
                print('nop')
        code = '\n        import __strict__\n        x = 1\n        '
        flags = self._get_flags(code)
        self.assertEqual(Flags(is_strict=True), flags)

    def test_static_import(self):
        if False:
            while True:
                i = 10
        code = '\n        import __static__\n        x = 1\n        '
        flags = self._get_flags(code)
        self.assertEqual(Flags(is_static=True), flags)

    def test_both_static_and_strict_import(self):
        if False:
            while True:
                i = 10
        code = '\n        import __static__\n        import __strict__\n        x = 1\n        '
        flags = self._get_flags(code)
        self.assertEqual(Flags(is_static=True, is_strict=True), flags)
        code = '\n        import __strict__\n        import __static__\n        x = 1\n        '
        flags = self._get_flags(code)
        self.assertEqual(Flags(is_static=True, is_strict=True), flags)

    def test_import_in_class(self):
        if False:
            for i in range(10):
                print('nop')
        code = '\n        class A:\n            import __strict__\n            x = 1\n        '
        self.assertRaisesRegex(BadFlagException, '__strict__ must be a globally namespaced import', lambda : self._get_flags(code))

    def test_import_in_function(self):
        if False:
            while True:
                i = 10
        code = '\n        def foo():\n            import __strict__\n            x = 1\n        '
        self.assertRaisesRegex(BadFlagException, '__strict__ must be a globally namespaced import', lambda : self._get_flags(code))

    def test_import_after_other_import(self):
        if False:
            for i in range(10):
                print('nop')
        code = '\n        import foo\n        import __strict__\n        x = 1\n        '
        self.assertRaisesRegex(BadFlagException, 'Cinder flag __strict__ must be at the top of a file', lambda : self._get_flags(code))

    def test_import_after_docstring(self):
        if False:
            i = 10
            return i + 15
        code = "\n        '''\n        here is a docstring\n        '''\n        import __strict__\n        x = 1\n        "
        self.assertEqual(Flags(is_strict=True), self._get_flags(code))

    def test_import_after_two_docstrings(self):
        if False:
            for i in range(10):
                print('nop')
        code = "\n        '''\n        here is a docstring\n        '''\n        '''\n        here is another docstring\n        '''\n        import __strict__\n        x = 1\n        "
        self.assertRaisesRegex(BadFlagException, 'Cinder flag __strict__ must be at the top of a file', lambda : self._get_flags(code))

    def test_import_after_constant(self):
        if False:
            while True:
                i = 10
        code = '\n        42\n        import __strict__\n        x = 1\n        '
        self.assertRaisesRegex(BadFlagException, 'Cinder flag __strict__ must be at the top of a file', lambda : self._get_flags(code))

    def test_import_after_docstring_and_constant(self):
        if False:
            print('Hello World!')
        code = "\n        '''\n        here is a docstring\n        '''\n        42\n        import __strict__\n        x = 1\n        "
        self.assertRaisesRegex(BadFlagException, 'Cinder flag __strict__ must be at the top of a file', lambda : self._get_flags(code))

    def test_import_after_class(self):
        if False:
            for i in range(10):
                print('nop')
        code = '\n        class Foo:\n            pass\n        import __strict__\n        x = 1\n        '
        self.assertRaisesRegex(BadFlagException, 'Cinder flag __strict__ must be at the top of a file', lambda : self._get_flags(code))

    def test_import_alias(self):
        if False:
            while True:
                i = 10
        code = '\n        import __strict__ as strict\n        x = 1\n        '
        self.assertRaisesRegex(BadFlagException, '__strict__ flag may not be aliased', lambda : self._get_flags(code))

    def test_flag_after_future_import(self):
        if False:
            print('Hello World!')
        code = '\n        from __future__ import annotations\n        import __strict__\n        '
        flags = self._get_flags(code)
        self.assertEqual(Flags(is_strict=True), flags)