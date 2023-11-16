import tempfile
import textwrap
from pathlib import Path
from typing import Optional, Sequence
import libcst as cst
import testslide
from libcst.metadata import MetadataWrapper
from .. import coverage_data
from ..coverage_data import AnnotationCollector, find_module_paths, FunctionAnnotationInfo, FunctionAnnotationStatus, FunctionIdentifier, Location, module_from_code, module_from_path, ModuleMode, ParameterAnnotationInfo, ReturnAnnotationInfo, SuppressionKind, TypeErrorSuppression
from ..tests import setup

def parse_code(code: str) -> MetadataWrapper:
    if False:
        i = 10
        return i + 15
    module = module_from_code(textwrap.dedent(code.rstrip()))
    if module is None:
        raise RuntimeError(f'Failed to parse code {code}')
    return module

class ParsingHelpersTest(testslide.TestCase):

    def test_module_from_path(self) -> None:
        if False:
            return 10
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            source_path = root_path / 'source.py'
            source_path.write_text('reveal_type(42)')
            self.assertIsNotNone(module_from_path(source_path))
            self.assertIsNone(module_from_path(root_path / 'nonexistent.py'))

    def test_module_from_code(self) -> None:
        if False:
            return 10
        self.assertIsNotNone(module_from_code(textwrap.dedent('\n                    def foo() -> int:\n                        pass\n                    ')))
        self.assertIsNone(module_from_code(textwrap.dedent('\n                    def foo() ->\n                    ')))

class AnnotationCollectorTest(testslide.TestCase):
    maxDiff = 2000

    def _build_and_visit_annotation_collector(self, source: str) -> AnnotationCollector:
        if False:
            return 10
        source_module = parse_code(source)
        collector = AnnotationCollector()
        source_module.visit(collector)
        return collector

    def test_return_location(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        collector = self._build_and_visit_annotation_collector('\n            def foobar():\n                pass\n            ')
        returns = list(collector.returns())
        self.assertEqual(len(returns), 1)
        self.assertEqual(returns[0].location, Location(start_line=2, start_column=4, end_line=2, end_column=10))

    def test_line_count(self) -> None:
        if False:
            while True:
                i = 10
        source_module = MetadataWrapper(cst.parse_module('# No trailing newline'))
        collector = AnnotationCollector()
        source_module.visit(collector)
        self.assertEqual(collector.line_count, 1)
        source_module = MetadataWrapper(cst.parse_module('# With trailing newline\n'))
        collector = AnnotationCollector()
        source_module.visit(collector)
        self.assertEqual(collector.line_count, 2)

    def _assert_function_annotations(self, code: str, expected: Sequence[FunctionAnnotationInfo]) -> None:
        if False:
            for i in range(10):
                print('nop')
        module = parse_code(code)
        actual = coverage_data.collect_functions(module)
        self.assertEqual(actual, expected)

    def test_function_annotations__standalone_no_annotations(self) -> None:
        if False:
            i = 10
            return i + 15
        self._assert_function_annotations('\n            def f(x):\n                pass\n            ', [FunctionAnnotationInfo(identifier=FunctionIdentifier(parent=None, name='f'), location=Location(start_line=2, start_column=0, end_line=3, end_column=8), annotation_status=FunctionAnnotationStatus.NOT_ANNOTATED, returns=ReturnAnnotationInfo(is_annotated=False, location=Location(start_line=2, start_column=4, end_line=2, end_column=5)), parameters=[ParameterAnnotationInfo(name='x', is_annotated=False, location=Location(start_line=2, start_column=6, end_line=2, end_column=7))], is_method_or_classmethod=False)])

    def test_function_annotations__standalone_partially_annotated(self) -> None:
        if False:
            print('Hello World!')
        self._assert_function_annotations('\n            def f(x) -> None:\n                pass\n\n            def g(x: int):\n                pass\n            ', [FunctionAnnotationInfo(identifier=FunctionIdentifier(parent=None, name='f'), location=Location(start_line=2, start_column=0, end_line=3, end_column=8), annotation_status=FunctionAnnotationStatus.PARTIALLY_ANNOTATED, returns=ReturnAnnotationInfo(is_annotated=True, location=Location(start_line=2, start_column=4, end_line=2, end_column=5)), parameters=[ParameterAnnotationInfo(name='x', is_annotated=False, location=Location(start_line=2, start_column=6, end_line=2, end_column=7))], is_method_or_classmethod=False), FunctionAnnotationInfo(identifier=FunctionIdentifier(parent=None, name='g'), location=Location(start_line=5, start_column=0, end_line=6, end_column=8), annotation_status=FunctionAnnotationStatus.PARTIALLY_ANNOTATED, returns=ReturnAnnotationInfo(is_annotated=False, location=Location(start_line=5, start_column=4, end_line=5, end_column=5)), parameters=[ParameterAnnotationInfo(name='x', is_annotated=True, location=Location(start_line=5, start_column=6, end_line=5, end_column=7))], is_method_or_classmethod=False)])

    def test_function_annotations__standalone_fully_annotated(self) -> None:
        if False:
            i = 10
            return i + 15
        self._assert_function_annotations('\n            def f(x: int) -> None:\n                pass\n            ', [FunctionAnnotationInfo(identifier=FunctionIdentifier(parent=None, name='f'), location=Location(start_line=2, start_column=0, end_line=3, end_column=8), annotation_status=FunctionAnnotationStatus.FULLY_ANNOTATED, returns=ReturnAnnotationInfo(is_annotated=True, location=Location(start_line=2, start_column=4, end_line=2, end_column=5)), parameters=[ParameterAnnotationInfo(name='x', is_annotated=True, location=Location(start_line=2, start_column=6, end_line=2, end_column=7))], is_method_or_classmethod=False)])

    def test_function_annotations__annotated_method(self) -> None:
        if False:
            return 10
        self._assert_function_annotations('\n            class A:\n                def f(self, x: int) -> None:\n                    pass\n            ', [FunctionAnnotationInfo(identifier=FunctionIdentifier(parent='A', name='f'), location=Location(start_line=3, start_column=4, end_line=4, end_column=12), annotation_status=FunctionAnnotationStatus.FULLY_ANNOTATED, returns=ReturnAnnotationInfo(is_annotated=True, location=Location(start_line=3, start_column=8, end_line=3, end_column=9)), parameters=[ParameterAnnotationInfo(name='self', is_annotated=False, location=Location(start_line=3, start_column=10, end_line=3, end_column=14)), ParameterAnnotationInfo(name='x', is_annotated=True, location=Location(start_line=3, start_column=16, end_line=3, end_column=17))], is_method_or_classmethod=True)])

    def test_function_annotations__partially_annotated_static_method(self) -> None:
        if False:
            print('Hello World!')
        self._assert_function_annotations('\n            class A:\n                class Inner:\n                    @staticmethod\n                    def f(self, x: int) -> None:\n                        pass\n            ', [FunctionAnnotationInfo(identifier=FunctionIdentifier(parent='A.Inner', name='f'), location=Location(start_line=5, start_column=8, end_line=6, end_column=16), annotation_status=FunctionAnnotationStatus.PARTIALLY_ANNOTATED, returns=ReturnAnnotationInfo(is_annotated=True, location=Location(start_line=5, start_column=12, end_line=5, end_column=13)), parameters=[ParameterAnnotationInfo(name='self', is_annotated=False, location=Location(start_line=5, start_column=14, end_line=5, end_column=18)), ParameterAnnotationInfo(name='x', is_annotated=True, location=Location(start_line=5, start_column=20, end_line=5, end_column=21))], is_method_or_classmethod=False)])

class FunctionAnnotationStatusTest(testslide.TestCase):
    ANNOTATION = cst.Annotation(cst.Name('Foo'))

    def _parameter(self, name: str, annotated: bool) -> cst.Param:
        if False:
            print('Hello World!')
        return cst.Param(name=cst.Name(name), annotation=self.ANNOTATION if annotated else None)

    def test_from_function_data(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(FunctionAnnotationStatus.from_function_data(is_return_annotated=True, is_non_static_method=False, parameters=[self._parameter('x0', annotated=True), self._parameter('x1', annotated=True), self._parameter('x2', annotated=True)]), FunctionAnnotationStatus.FULLY_ANNOTATED)
        self.assertEqual(FunctionAnnotationStatus.from_function_data(is_return_annotated=True, is_non_static_method=False, parameters=[self._parameter('x0', annotated=False), self._parameter('x1', annotated=False), self._parameter('x2', annotated=False)]), FunctionAnnotationStatus.PARTIALLY_ANNOTATED)
        self.assertEqual(FunctionAnnotationStatus.from_function_data(is_return_annotated=False, is_non_static_method=False, parameters=[self._parameter('x0', annotated=False), self._parameter('x1', annotated=False), self._parameter('x2', annotated=False)]), FunctionAnnotationStatus.NOT_ANNOTATED)
        self.assertEqual(FunctionAnnotationStatus.from_function_data(is_return_annotated=False, is_non_static_method=False, parameters=[self._parameter('x0', annotated=True), self._parameter('x1', annotated=False), self._parameter('x2', annotated=False)]), FunctionAnnotationStatus.PARTIALLY_ANNOTATED)
        self.assertEqual(FunctionAnnotationStatus.from_function_data(is_return_annotated=True, is_non_static_method=True, parameters=[self._parameter('self', annotated=False), self._parameter('x1', annotated=False)]), FunctionAnnotationStatus.PARTIALLY_ANNOTATED)
        self.assertEqual(FunctionAnnotationStatus.from_function_data(is_return_annotated=True, is_non_static_method=True, parameters=[self._parameter('self', annotated=True), self._parameter('x1', annotated=False)]), FunctionAnnotationStatus.PARTIALLY_ANNOTATED)
        self.assertEqual(FunctionAnnotationStatus.from_function_data(is_return_annotated=True, is_non_static_method=True, parameters=[self._parameter('self', annotated=False)]), FunctionAnnotationStatus.FULLY_ANNOTATED)
        self.assertEqual(FunctionAnnotationStatus.from_function_data(is_return_annotated=False, is_non_static_method=True, parameters=[self._parameter('self', annotated=True)]), FunctionAnnotationStatus.PARTIALLY_ANNOTATED)

class SuppressionCollectorTest(testslide.TestCase):
    maxDiff = 2000

    def _assert_suppressions(self, source: str, expected: Sequence[TypeErrorSuppression]) -> None:
        if False:
            return 10
        source_module = parse_code(source.replace('PYRE_FIXME', 'pyre-fixme').replace('PYRE_IGNORE', 'pyre-ignore').replace('TYPE_IGNORE', 'type: ignore'))
        actual = coverage_data.collect_suppressions(source_module)
        self.assertEqual(actual, expected)

    def test_find_fixmes__simple(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._assert_suppressions('\n            # PYRE_FIXME\n            # PYRE_FIXME with message\n            # PYRE_FIXME[1]\n            # PYRE_FIXME[10, 11] with message\n            # PYRE_FIXME[10,]  (trailing comma is illegal, codes are ignored)\n            ', [TypeErrorSuppression(kind=SuppressionKind.PYRE_FIXME, location=Location(start_line=2, start_column=0, end_line=2, end_column=12), error_codes=None), TypeErrorSuppression(kind=SuppressionKind.PYRE_FIXME, location=Location(start_line=3, start_column=0, end_line=3, end_column=25), error_codes=None), TypeErrorSuppression(kind=SuppressionKind.PYRE_FIXME, location=Location(start_line=4, start_column=0, end_line=4, end_column=15), error_codes=[1]), TypeErrorSuppression(kind=SuppressionKind.PYRE_FIXME, location=Location(start_line=5, start_column=0, end_line=5, end_column=33), error_codes=[10, 11]), TypeErrorSuppression(kind=SuppressionKind.PYRE_FIXME, location=Location(start_line=6, start_column=0, end_line=6, end_column=65), error_codes=[])])

    def test_find_ignores__simple(self) -> None:
        if False:
            print('Hello World!')
        self._assert_suppressions('\n            # PYRE_IGNORE\n            # PYRE_IGNORE with message\n            # PYRE_IGNORE[1]\n            # PYRE_IGNORE[10, 11]\n            # PYRE_IGNORE[10, 11] with message\n            # PYRE_IGNORE[10,]  (trailing comma is illegal, codes are ignored)\n            ', [TypeErrorSuppression(kind=SuppressionKind.PYRE_IGNORE, location=Location(start_line=2, start_column=0, end_line=2, end_column=13), error_codes=None), TypeErrorSuppression(kind=SuppressionKind.PYRE_IGNORE, location=Location(start_line=3, start_column=0, end_line=3, end_column=26), error_codes=None), TypeErrorSuppression(kind=SuppressionKind.PYRE_IGNORE, location=Location(start_line=4, start_column=0, end_line=4, end_column=16), error_codes=[1]), TypeErrorSuppression(kind=SuppressionKind.PYRE_IGNORE, location=Location(start_line=5, start_column=0, end_line=5, end_column=21), error_codes=[10, 11]), TypeErrorSuppression(kind=SuppressionKind.PYRE_IGNORE, location=Location(start_line=6, start_column=0, end_line=6, end_column=34), error_codes=[10, 11]), TypeErrorSuppression(kind=SuppressionKind.PYRE_IGNORE, location=Location(start_line=7, start_column=0, end_line=7, end_column=66), error_codes=[])])

    def test_find_type_ignores(self) -> None:
        if False:
            return 10
        self._assert_suppressions("\n            # TYPE_IGNORE\n            # TYPE_IGNORE[1]  (codes won't be parsed)\n            ", [TypeErrorSuppression(kind=SuppressionKind.TYPE_IGNORE, location=Location(start_line=2, start_column=0, end_line=2, end_column=14), error_codes=None), TypeErrorSuppression(kind=SuppressionKind.TYPE_IGNORE, location=Location(start_line=3, start_column=0, end_line=3, end_column=42), error_codes=None)])

    def test_find_suppressions__trailing_comments(self) -> None:
        if False:
            return 10
        self._assert_suppressions('\n            a: int = 42.0 # PYRE_FIXME\n            b: int = 42.0 # leading comment # PYRE_FIXME[3, 4]\n            c: int = 42.0 # leading comment # PYRE_IGNORE[5]\n            f: int = 42.0 # leading comment # TYPE_IGNORE\n            ', [TypeErrorSuppression(kind=SuppressionKind.PYRE_FIXME, location=Location(start_line=2, start_column=14, end_line=2, end_column=26), error_codes=None), TypeErrorSuppression(kind=SuppressionKind.PYRE_FIXME, location=Location(start_line=3, start_column=14, end_line=3, end_column=50), error_codes=[3, 4]), TypeErrorSuppression(kind=SuppressionKind.PYRE_IGNORE, location=Location(start_line=4, start_column=14, end_line=4, end_column=48), error_codes=[5]), TypeErrorSuppression(kind=SuppressionKind.TYPE_IGNORE, location=Location(start_line=5, start_column=14, end_line=5, end_column=46), error_codes=None)])

    def test_find_suppressions__multiline_string(self) -> None:
        if False:
            i = 10
            return i + 15
        self._assert_suppressions("\n            '''\n            # PYRE_IGNORE\n            '''\n            ", [])

    def test_find_suppressions__nested_suppressions(self) -> None:
        if False:
            while True:
                i = 10
        self._assert_suppressions('\n            # # PYRE_IGNORE # TYPE_IGNORE\n            ', [TypeErrorSuppression(kind=SuppressionKind.PYRE_IGNORE, location=Location(start_line=2, start_column=0, end_line=2, end_column=30), error_codes=None), TypeErrorSuppression(kind=SuppressionKind.TYPE_IGNORE, location=Location(start_line=2, start_column=0, end_line=2, end_column=30), error_codes=None)])

class ModuleModecollectorTest(testslide.TestCase):

    def assert_counts(self, source: str, default_strict: bool, mode: ModuleMode, explicit_comment_line: Optional[int]) -> None:
        if False:
            print('Hello World!')
        source_module = parse_code(source)
        result = coverage_data.collect_mode(source_module, default_strict)
        self.assertEqual(mode, result.mode)
        self.assertEqual(explicit_comment_line, result.explicit_comment_line)

    def test_strict_files(self) -> None:
        if False:
            return 10
        self.assert_counts('\n            # pyre-unsafe\n\n            def foo():\n                return 1\n            ', default_strict=True, mode=ModuleMode.UNSAFE, explicit_comment_line=2)
        self.assert_counts('\n            # pyre-strict\n            def foo():\n                return 1\n            ', default_strict=False, mode=ModuleMode.STRICT, explicit_comment_line=2)
        self.assert_counts('\n            def foo():\n                return 1\n            ', default_strict=False, mode=ModuleMode.UNSAFE, explicit_comment_line=None)
        self.assert_counts('\n            def foo():\n                return 1\n            ', default_strict=True, mode=ModuleMode.STRICT, explicit_comment_line=None)
        self.assert_counts('\n            # pyre-ignore-all-errors\n            def foo():\n                return 1\n            ', default_strict=True, mode=ModuleMode.IGNORE_ALL, explicit_comment_line=2)
        self.assert_counts('\n            def foo(x: str) -> int:\n                return x\n            ', default_strict=False, mode=ModuleMode.UNSAFE, explicit_comment_line=None)
        self.assert_counts('\n            #  pyre-strict\n            def foo(x: str) -> int:\n                return x\n            ', default_strict=False, mode=ModuleMode.STRICT, explicit_comment_line=2)
        self.assert_counts('\n            #  pyre-ignore-all-errors[56]\n            def foo(x: str) -> int:\n                return x\n            ', default_strict=True, mode=ModuleMode.STRICT, explicit_comment_line=None)

class ModuleFindingHelpersTest(testslide.TestCase):

    def test_find_module_paths__basic(self) -> None:
        if False:
            return 10
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            setup.ensure_files_exist(root_path, ['s0.py', 'a/s1.py', 'b/s2.py', 'b/c/s3.py', 'b/s4.txt', 'b/__s5.py'])
            setup.ensure_directories_exists(root_path, ['b/d'])
            self.assertCountEqual(find_module_paths([root_path / 'a/s1.py', root_path / 'b/s2.py', root_path / 'b/s4.txt'], excludes=[]), [root_path / 'a/s1.py', root_path / 'b/s2.py'])
            self.assertCountEqual(find_module_paths([root_path], excludes=[]), [root_path / 's0.py', root_path / 'a/s1.py', root_path / 'b/s2.py', root_path / 'b/c/s3.py'])

    def test_find_module_paths__with_exclude(self) -> None:
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            setup.ensure_files_exist(root_path, ['s0.py', 'a/s1.py', 'b/s2.py', 'b/c/s3.py', 'b/s4.txt', 'b/__s5.py'])
            setup.ensure_directories_exists(root_path, ['b/d'])
            self.assertCountEqual(find_module_paths([root_path / 'a/s1.py', root_path / 'b/s2.py', root_path / 'b/s4.txt'], excludes=['.*2\\.py']), [root_path / 'a/s1.py'])
            self.assertCountEqual(find_module_paths([root_path], excludes=['.*2\\.py']), [root_path / 's0.py', root_path / 'a/s1.py', root_path / 'b/c/s3.py'])

    def test_find_module_paths__with_duplicates(self) -> None:
        if False:
            return 10
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            setup.ensure_files_exist(root_path, ['a/s1.py', 'a/s2.py'])
            self.assertCountEqual(find_module_paths([root_path / 'a/s1.py', root_path / 'a'], excludes=[]), [root_path / 'a/s1.py', root_path / 'a/s2.py'])