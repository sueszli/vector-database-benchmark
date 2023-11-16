import tempfile
import textwrap
from pathlib import Path
from typing import List
import libcst as cst
import testslide
from ...tests import setup
from .. import coverage

class CoverageTest(testslide.TestCase):

    def assert_coverage_equal(self, file_content: str, expected_covered: List[int], expected_uncovered: List[int]) -> None:
        if False:
            i = 10
            return i + 15
        module = cst.MetadataWrapper(cst.parse_module(textwrap.dedent(file_content).strip()))
        actual_coverage = coverage.collect_coverage_for_module('test.py', module, strict_default=False)
        self.assertEqual(expected_covered, actual_coverage.covered_lines, 'Covered mismatch')
        self.assertEqual(expected_uncovered, actual_coverage.uncovered_lines, 'Not covered mismatch')

    def test_coverage_covered(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_coverage_equal('\n            def foo() -> int:\n                return 5\n            ', expected_covered=[0, 1], expected_uncovered=[])

    def test_coverage_uncovered(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_coverage_equal('\n            def foo():\n                return 5\n            ', expected_covered=[], expected_uncovered=[0, 1])

    def test_coverage_mixed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assert_coverage_equal('\n            import os\n\n            X = 5\n\n            def foo():\n                return 5\n\n            class Bar():\n\n                def baz(self, y) -> int:\n                    return y + 5\n            ', expected_covered=[0, 1, 2, 3, 6, 7, 8, 9, 10], expected_uncovered=[4, 5])

    def test_coverage_nested(self) -> None:
        if False:
            print('Hello World!')
        self.assert_coverage_equal('\n            def f():\n\n                def bar(x: int) -> None:\n                    return x\n\n                return 5\n            ', expected_covered=[2, 3], expected_uncovered=[0, 1, 4, 5])
        self.assert_coverage_equal('\n            level0: None = None\n            def level1():\n                def level2() -> None:\n                    def level3():\n                        def level4() -> None:\n                            def level5(): ...\n            ', expected_covered=[0, 2, 4], expected_uncovered=[1, 3, 5])

    def contains_uncovered_lines(self, file_content: str, strict_default: bool) -> bool:
        if False:
            print('Hello World!')
        module = cst.MetadataWrapper(cst.parse_module(textwrap.dedent(file_content).strip()))
        actual_coverage = coverage.collect_coverage_for_module('test.py', module, strict_default)
        return len(actual_coverage.uncovered_lines) > 0

    def test_coverage_strict(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.contains_uncovered_lines('\n                # No file specific comment\n                def foo(): ...\n                ', strict_default=False))
        self.assertFalse(self.contains_uncovered_lines('\n                # pyre-strict\n                def foo(): ...\n                ', strict_default=False))
        self.assertFalse(self.contains_uncovered_lines('\n                # No file specific comment\n                def foo(): ...\n                ', strict_default=True))
        self.assertTrue(self.contains_uncovered_lines('\n                # pyre-unsafe\n                def foo(): ...\n                ', strict_default=True))

    def test_find_root(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(coverage.find_root_path(local_root=Path('/root/local'), working_directory=Path('/irrelevant')), Path('/root/local'))
        self.assertEqual(coverage.find_root_path(local_root=None, working_directory=Path('/working/dir')), Path('/working/dir'))

    def test_collect_coverage(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as root:
            root_path: Path = Path(root)
            setup.ensure_files_exist(root_path, ['foo.py', 'bar.py'])
            foo_path = root_path / 'foo.py'
            bar_path = root_path / 'bar.py'
            baz_path = root_path / 'baz.py'
            data: List[coverage.FileCoverage] = coverage.collect_coverage_for_paths([foo_path, bar_path, baz_path], working_directory=root, strict_default=False)

            def is_collected(path: Path) -> bool:
                if False:
                    for i in range(10):
                        print('nop')
                return any((str(path.relative_to(root_path)) == coverage.filepath for coverage in data))
            self.assertTrue(is_collected(foo_path))
            self.assertTrue(is_collected(bar_path))
            self.assertFalse(is_collected(baz_path))