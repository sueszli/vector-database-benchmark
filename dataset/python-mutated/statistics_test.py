import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List
import libcst
import testslide
from ... import coverage_data
from ...tests import setup
from .. import statistics

def parse_code(code: str) -> libcst.MetadataWrapper:
    if False:
        i = 10
        return i + 15
    module = coverage_data.module_from_code(textwrap.dedent(code.rstrip()))
    if module is None:
        raise RuntimeError(f'Failed to parse code {code}')
    return module

class FixmeCountCollectorTest(testslide.TestCase):

    def assert_counts(self, source: str, expected_codes: Dict[int, List[int]], expected_no_codes: List[int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        source_module = parse_code(source.replace('FIXME', 'pyre-fixme'))
        result = statistics.FixmeCountCollector().collect(source_module)
        self.assertEqual(expected_codes, result.code)
        self.assertEqual(expected_no_codes, result.no_code)

    def test_count_fixmes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assert_counts('\n            # FIXME\n            # FIXME[8,]\n            ', {}, [2, 3])
        self.assert_counts('\n            # FIXME[3]: Example Error Message\n            # FIXME[3, 4]: Another Message\n\n            # FIXME[34]: Example\n            ', {3: [2, 3], 4: [3], 34: [5]}, [])
        self.assert_counts('\n            def foo(x: str) -> int:\n                return x  # FIXME[7]\n            ', {7: [3]}, [])
        self.assert_counts('\n            def foo(x: str) -> int:\n                # FIXME[7]: comments\n                return x\n            ', {7: [3]}, [])
        self.assert_counts('\n            def foo(x: str) -> int:\n                return x  # FIXME\n            ', {}, [3])
        self.assert_counts('\n            def foo(x: str) -> int:\n                return x  # FIXME: comments\n            ', {}, [3])
        self.assert_counts('\n            def foo(x: str) -> int:\n                return x # unrelated # FIXME[7]\n            ', {7: [3]}, [])
        self.assert_counts('\n            def foo(x: str) -> int:\n                return x # unrelated   #  FIXME[7] comments\n            ', {7: [3]}, [])
        self.assert_counts('\n            def foo(x: str) -> int:\n                return x # FIXME[7, 8]\n            ', {7: [3], 8: [3]}, [])

class IgnoreCountCollectorTest(testslide.TestCase):
    maxDiff = 2000

    def assert_counts(self, source: str, expected_codes: Dict[int, List[int]], expected_no_codes: List[int]) -> None:
        if False:
            while True:
                i = 10
        source_module = parse_code(source.replace('IGNORE', 'pyre-ignore'))
        result = statistics.IgnoreCountCollector().collect(source_module)
        self.assertEqual(expected_codes, result.code)
        self.assertEqual(expected_no_codes, result.no_code)

    def test_count_ignores(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assert_counts('# IGNORE[2]: Example Error Message', {2: [1]}, [])
        self.assert_counts('\n            # IGNORE[3]: Example Error Message\n\n            # IGNORE[34]: Example\n            ', {3: [2], 34: [4]}, [])
        self.assert_counts('\n            # IGNORE[2]: Example Error Message\n\n            # IGNORE[2]: message\n            ', {2: [2, 4]}, [])

class AnnotationCountCollectorTest(testslide.TestCase):

    def assert_counts(self, source: str, expected: Dict[str, int]) -> None:
        if False:
            i = 10
            return i + 15
        source_module = parse_code(source)
        result = statistics.AnnotationCountCollector().collect(source_module)
        self.assertDictEqual(expected, result.to_count_dict())

    def test_count_annotations(self) -> None:
        if False:
            print('Hello World!')
        self.assert_counts('\n            def foo(x) -> int:\n                pass\n            ', {'annotated_return_count': 1, 'annotated_globals_count': 0, 'annotated_parameter_count': 0, 'return_count': 1, 'globals_count': 0, 'parameter_count': 1, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 1, 'fully_annotated_function_count': 0, 'line_count': 3})
        self.assert_counts('\n            def bar(x: int, y):\n                pass\n            ', {'annotated_return_count': 0, 'annotated_globals_count': 0, 'annotated_parameter_count': 1, 'return_count': 1, 'globals_count': 0, 'parameter_count': 2, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 1, 'fully_annotated_function_count': 0, 'line_count': 3})
        self.assert_counts('\n            a = foo()\n            b: int = bar()\n            ', {'annotated_return_count': 0, 'annotated_globals_count': 2, 'annotated_parameter_count': 0, 'return_count': 0, 'globals_count': 2, 'parameter_count': 0, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 0, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 0, 'line_count': 3})
        self.assert_counts('\n            class A:\n                a: int = 100\n                b = ""\n            ', {'annotated_return_count': 0, 'annotated_globals_count': 0, 'annotated_parameter_count': 0, 'return_count': 0, 'globals_count': 0, 'parameter_count': 0, 'attribute_count': 2, 'annotated_attribute_count': 2, 'function_count': 0, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 0, 'line_count': 4})
        self.assert_counts('\n            def foo():\n                a: int = 100\n            ', {'annotated_return_count': 0, 'annotated_globals_count': 0, 'annotated_parameter_count': 0, 'return_count': 1, 'globals_count': 0, 'parameter_count': 0, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 0, 'line_count': 3})
        self.assert_counts('\n            def foo():\n                def bar(x: int) -> int:\n                    pass\n            ', {'annotated_return_count': 1, 'annotated_globals_count': 0, 'annotated_parameter_count': 1, 'return_count': 2, 'globals_count': 0, 'parameter_count': 1, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 2, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 1, 'line_count': 4})
        self.assert_counts('\n            class A:\n                def bar(self, x: int):\n                    pass\n            ', {'annotated_return_count': 0, 'annotated_globals_count': 0, 'annotated_parameter_count': 1, 'return_count': 1, 'globals_count': 0, 'parameter_count': 1, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 1, 'fully_annotated_function_count': 0, 'line_count': 4})
        self.assert_counts('\n            class A:\n                def bar(this, x: int) -> None:\n                    pass\n            ', {'annotated_return_count': 1, 'annotated_globals_count': 0, 'annotated_parameter_count': 1, 'return_count': 1, 'globals_count': 0, 'parameter_count': 1, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 1, 'line_count': 4})
        self.assert_counts('\n            class A:\n                @classmethod\n                def bar(cls, x: int):\n                    pass\n            ', {'annotated_return_count': 0, 'annotated_globals_count': 0, 'annotated_parameter_count': 1, 'return_count': 1, 'globals_count': 0, 'parameter_count': 1, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 1, 'fully_annotated_function_count': 0, 'line_count': 5})
        self.assert_counts('\n            def bar(self, x: int):\n                pass\n            ', {'annotated_return_count': 0, 'annotated_globals_count': 0, 'annotated_parameter_count': 1, 'return_count': 1, 'globals_count': 0, 'parameter_count': 2, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 1, 'fully_annotated_function_count': 0, 'line_count': 3})
        self.assert_counts('\n            class A:\n                @staticmethod\n                def bar(self, x: int) -> None:\n                    pass\n            ', {'annotated_return_count': 1, 'annotated_globals_count': 0, 'annotated_parameter_count': 1, 'return_count': 1, 'globals_count': 0, 'parameter_count': 2, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 1, 'fully_annotated_function_count': 0, 'line_count': 5})
        self.assert_counts('\n            def foo(x: str) -> str:\n                return x\n            ', {'return_count': 1, 'annotated_return_count': 1, 'globals_count': 0, 'annotated_globals_count': 0, 'parameter_count': 1, 'annotated_parameter_count': 1, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 1, 'line_count': 3})
        self.assert_counts('\n            class Test:\n                def foo(self, input: str) -> None:\n                    class Foo:\n                        pass\n\n                    pass\n\n                def bar(self, input: str) -> None:\n                    pass\n            ', {'return_count': 2, 'annotated_return_count': 2, 'globals_count': 0, 'annotated_globals_count': 0, 'parameter_count': 2, 'annotated_parameter_count': 2, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 2, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 2, 'line_count': 10})
        self.assert_counts('\n            x: int = 1\n            y = 2\n            z = foo\n\n            class Foo:\n                x = 1\n                y = foo\n            ', {'return_count': 0, 'annotated_return_count': 0, 'globals_count': 3, 'annotated_globals_count': 3, 'parameter_count': 0, 'annotated_parameter_count': 0, 'attribute_count': 2, 'annotated_attribute_count': 2, 'function_count': 0, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 0, 'line_count': 8})

    def test_count_annotations__partially_annotated_methods(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assert_counts('\n            class A:\n                def bar(self): ...\n            ', {'return_count': 1, 'annotated_return_count': 0, 'globals_count': 0, 'annotated_globals_count': 0, 'parameter_count': 0, 'annotated_parameter_count': 0, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 0, 'line_count': 3})
        self.assert_counts('\n            class A:\n                def bar(self) -> None: ...\n            ', {'return_count': 1, 'annotated_return_count': 1, 'globals_count': 0, 'annotated_globals_count': 0, 'parameter_count': 0, 'annotated_parameter_count': 0, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 1, 'line_count': 3})
        self.assert_counts('\n            class A:\n                def baz(self, x): ...\n            ', {'return_count': 1, 'annotated_return_count': 0, 'globals_count': 0, 'annotated_globals_count': 0, 'parameter_count': 1, 'annotated_parameter_count': 0, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 0, 'line_count': 3})
        self.assert_counts('\n            class A:\n                def baz(self, x) -> None: ...\n            ', {'return_count': 1, 'annotated_return_count': 1, 'globals_count': 0, 'annotated_globals_count': 0, 'parameter_count': 1, 'annotated_parameter_count': 0, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 1, 'fully_annotated_function_count': 0, 'line_count': 3})
        self.assert_counts('\n            class A:\n                def baz(self: Foo): ...\n            ', {'return_count': 1, 'annotated_return_count': 0, 'globals_count': 0, 'annotated_globals_count': 0, 'parameter_count': 0, 'annotated_parameter_count': 0, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 1, 'fully_annotated_function_count': 0, 'line_count': 3})

class StatisticsTest(testslide.TestCase):

    def test_collect_statistics(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            setup.ensure_files_exist(root_path, ['foo.py', 'bar.py'])
            foo_path = root_path / 'foo.py'
            bar_path = root_path / 'bar.py'
            data = statistics.collect_statistics([foo_path, bar_path], strict_default=False)
            self.assertIn(str(foo_path), data)
            self.assertIn(str(bar_path), data)

    def test_aggregate_statistics__single_file(self) -> None:
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            a_path = root_path / 'a.py'
            a_path.write_text(textwrap.dedent('\n                    # pyre-unsafe\n\n                    def foo():\n                        return 1\n                    '.rstrip()))
            self.assertEqual(statistics.aggregate_statistics(statistics.collect_statistics([a_path], strict_default=False)), statistics.AggregatedStatisticsData(annotations={'return_count': 1, 'annotated_return_count': 0, 'globals_count': 0, 'annotated_globals_count': 0, 'parameter_count': 0, 'annotated_parameter_count': 0, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 1, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 0, 'line_count': 5}, fixmes=0, ignores=0, strict=0, unsafe=1))

    def test_aggregate_statistics__multiple_files(self) -> None:
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            a_path = root_path / 'a.py'
            b_path = root_path / 'b.py'
            a_path.write_text(textwrap.dedent('\n                    # pyre-unsafe\n\n                    def foo():\n                        return 1\n                    '.rstrip()))
            b_path.write_text(textwrap.dedent('\n                    # pyre-strict\n\n                    def foo(x: int) -> int:\n                        return 1\n                    '.rstrip()))
            self.assertEqual(statistics.aggregate_statistics(statistics.collect_statistics([a_path, b_path], strict_default=False)), statistics.AggregatedStatisticsData(annotations={'return_count': 2, 'annotated_return_count': 1, 'globals_count': 0, 'annotated_globals_count': 0, 'parameter_count': 1, 'annotated_parameter_count': 1, 'attribute_count': 0, 'annotated_attribute_count': 0, 'function_count': 2, 'partially_annotated_function_count': 0, 'fully_annotated_function_count': 1, 'line_count': 10}, fixmes=0, ignores=0, strict=1, unsafe=1))