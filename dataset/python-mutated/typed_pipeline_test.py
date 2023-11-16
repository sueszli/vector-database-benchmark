"""Unit tests for the type-hint objects and decorators."""
import sys
import typing
import unittest
import apache_beam as beam
from apache_beam import pvalue
from apache_beam import typehints
from apache_beam.options.pipeline_options import OptionsContext
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.typehints import WithTypeHints
from apache_beam.typehints.decorators import get_signature

class MainInputTest(unittest.TestCase):

    def assertStartswith(self, msg, prefix):
        if False:
            print('Hello World!')
        self.assertTrue(msg.startswith(prefix), '"%s" does not start with "%s"' % (msg, prefix))

    def test_bad_main_input(self):
        if False:
            i = 10
            return i + 15

        @typehints.with_input_types(str, int)
        def repeat(s, times):
            if False:
                i = 10
                return i + 15
            return s * times
        with self.assertRaises(typehints.TypeCheckError):
            [1, 2, 3] | beam.Map(repeat, 3)

    def test_non_function(self):
        if False:
            i = 10
            return i + 15
        result = ['a', 'bb', 'c'] | beam.Map(str.upper)
        self.assertEqual(['A', 'BB', 'C'], sorted(result))
        result = ['xa', 'bbx', 'xcx'] | beam.Map(str.strip, 'x')
        self.assertEqual(['a', 'bb', 'c'], sorted(result))
        result = ['1', '10', '100'] | beam.Map(int)
        self.assertEqual([1, 10, 100], sorted(result))
        result = ['1', '10', '100'] | beam.Map(int, 16)
        self.assertEqual([1, 16, 256], sorted(result))

    def test_non_function_fails(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(typehints.TypeCheckError):
            [1, 2, 3] | beam.Map(str.upper)

    def test_loose_bounds(self):
        if False:
            while True:
                i = 10

        @typehints.with_input_types(typing.Union[int, float])
        @typehints.with_output_types(str)
        def format_number(x):
            if False:
                i = 10
                return i + 15
            return '%g' % x
        result = [1, 2, 3] | beam.Map(format_number)
        self.assertEqual(['1', '2', '3'], sorted(result))

    def test_typed_dofn_class(self):
        if False:
            return 10

        @typehints.with_input_types(int)
        @typehints.with_output_types(str)
        class MyDoFn(beam.DoFn):

            def process(self, element):
                if False:
                    return 10
                return [str(element)]
        result = [1, 2, 3] | beam.ParDo(MyDoFn())
        self.assertEqual(['1', '2', '3'], sorted(result))
        with self.assertRaisesRegex(typehints.TypeCheckError, 'requires.*int.*got.*str'):
            ['a', 'b', 'c'] | beam.ParDo(MyDoFn())
        with self.assertRaisesRegex(typehints.TypeCheckError, 'requires.*int.*got.*str'):
            [1, 2, 3] | (beam.ParDo(MyDoFn()) | 'again' >> beam.ParDo(MyDoFn()))

    def test_typed_dofn_method(self):
        if False:
            return 10

        class MyDoFn(beam.DoFn):

            def process(self, element: int) -> typehints.Tuple[str]:
                if False:
                    return 10
                return tuple(str(element))
        result = [1, 2, 3] | beam.ParDo(MyDoFn())
        self.assertEqual(['1', '2', '3'], sorted(result))
        with self.assertRaisesRegex(typehints.TypeCheckError, 'requires.*int.*got.*str'):
            _ = ['a', 'b', 'c'] | beam.ParDo(MyDoFn())
        with self.assertRaisesRegex(typehints.TypeCheckError, 'requires.*int.*got.*str'):
            _ = [1, 2, 3] | (beam.ParDo(MyDoFn()) | 'again' >> beam.ParDo(MyDoFn()))

    def test_typed_dofn_method_with_class_decorators(self):
        if False:
            while True:
                i = 10

        @typehints.with_input_types(typehints.Tuple[int, int])
        @typehints.with_output_types(int)
        class MyDoFn(beam.DoFn):

            def process(self, element: int) -> typehints.Tuple[str]:
                if False:
                    print('Hello World!')
                yield element[0]
        result = [(1, 2)] | beam.ParDo(MyDoFn())
        self.assertEqual([1], sorted(result))
        with self.assertRaisesRegex(typehints.TypeCheckError, "requires.*Tuple\\[<class \\'int\\'>, <class \\'int\\'>\\].*got.*str"):
            _ = ['a', 'b', 'c'] | beam.ParDo(MyDoFn())
        with self.assertRaisesRegex(typehints.TypeCheckError, "requires.*Tuple\\[<class \\'int\\'>, <class \\'int\\'>\\].*got.*int"):
            _ = [1, 2, 3] | (beam.ParDo(MyDoFn()) | 'again' >> beam.ParDo(MyDoFn()))

    def test_typed_callable_iterable_output(self):
        if False:
            while True:
                i = 10

        def do_fn(element: int) -> typehints.Iterable[typehints.Iterable[str]]:
            if False:
                print('Hello World!')
            return [[str(element)] * 2]
        result = [1, 2] | beam.ParDo(do_fn)
        self.assertEqual([['1', '1'], ['2', '2']], sorted(result))

    def test_typed_dofn_instance(self):
        if False:
            i = 10
            return i + 15

        @typehints.with_input_types(typehints.Tuple[int, int])
        @typehints.with_output_types(int)
        class MyDoFn(beam.DoFn):

            def process(self, element: typehints.Tuple[int, int]) -> typehints.List[int]:
                if False:
                    for i in range(10):
                        print('nop')
                return [str(element)]
        my_do_fn = MyDoFn().with_input_types(int).with_output_types(str)
        result = [1, 2, 3] | beam.ParDo(my_do_fn)
        self.assertEqual(['1', '2', '3'], sorted(result))
        with self.assertRaisesRegex(typehints.TypeCheckError, 'requires.*int.*got.*str'):
            _ = ['a', 'b', 'c'] | beam.ParDo(my_do_fn)
        with self.assertRaisesRegex(typehints.TypeCheckError, 'requires.*int.*got.*str'):
            _ = [1, 2, 3] | (beam.ParDo(my_do_fn) | 'again' >> beam.ParDo(my_do_fn))

    def test_typed_callable_instance(self):
        if False:
            for i in range(10):
                print('nop')

        @typehints.with_input_types(typehints.Tuple[int, int])
        @typehints.with_output_types(typehints.Generator[int])
        def do_fn(element: typehints.Tuple[int, int]) -> typehints.Generator[str]:
            if False:
                return 10
            yield str(element)
        pardo = beam.ParDo(do_fn).with_input_types(int).with_output_types(str)
        result = [1, 2, 3] | pardo
        self.assertEqual(['1', '2', '3'], sorted(result))
        with self.assertRaisesRegex(typehints.TypeCheckError, 'requires.*int.*got.*str'):
            _ = ['a', 'b', 'c'] | pardo
        with self.assertRaisesRegex(typehints.TypeCheckError, 'requires.*int.*got.*str'):
            _ = [1, 2, 3] | (pardo | 'again' >> pardo)

    def test_filter_type_hint(self):
        if False:
            return 10

        @typehints.with_input_types(int)
        def filter_fn(data):
            if False:
                i = 10
                return i + 15
            return data % 2
        self.assertEqual([1, 3], [1, 2, 3] | beam.Filter(filter_fn))

    def test_partition(self):
        if False:
            print('Hello World!')
        with TestPipeline() as p:
            (even, odd) = p | beam.Create([1, 2, 3]) | 'even_odd' >> beam.Partition(lambda e, _: e % 2, 2)
            self.assertIsNotNone(even.element_type)
            self.assertIsNotNone(odd.element_type)
            res_even = even | 'IdEven' >> beam.ParDo(lambda e: [e]).with_input_types(int)
            res_odd = odd | 'IdOdd' >> beam.ParDo(lambda e: [e]).with_input_types(int)
            assert_that(res_even, equal_to([2]), label='even_check')
            assert_that(res_odd, equal_to([1, 3]), label='odd_check')

    def test_typed_dofn_multi_output(self):
        if False:
            while True:
                i = 10

        class MyDoFn(beam.DoFn):

            def process(self, element):
                if False:
                    for i in range(10):
                        print('nop')
                if element % 2:
                    yield beam.pvalue.TaggedOutput('odd', element)
                else:
                    yield beam.pvalue.TaggedOutput('even', element)
        with TestPipeline() as p:
            res = p | beam.Create([1, 2, 3]) | beam.ParDo(MyDoFn()).with_outputs('odd', 'even')
            self.assertIsNotNone(res[None].element_type)
            self.assertIsNotNone(res['even'].element_type)
            self.assertIsNotNone(res['odd'].element_type)
            res_main = res[None] | 'id_none' >> beam.ParDo(lambda e: [e]).with_input_types(int)
            res_even = res['even'] | 'id_even' >> beam.ParDo(lambda e: [e]).with_input_types(int)
            res_odd = res['odd'] | 'id_odd' >> beam.ParDo(lambda e: [e]).with_input_types(int)
            assert_that(res_main, equal_to([]), label='none_check')
            assert_that(res_even, equal_to([2]), label='even_check')
            assert_that(res_odd, equal_to([1, 3]), label='odd_check')
        with self.assertRaises(ValueError):
            _ = res['undeclared tag']

    def test_typed_dofn_multi_output_no_tags(self):
        if False:
            return 10

        class MyDoFn(beam.DoFn):

            def process(self, element):
                if False:
                    print('Hello World!')
                if element % 2:
                    yield beam.pvalue.TaggedOutput('odd', element)
                else:
                    yield beam.pvalue.TaggedOutput('even', element)
        with TestPipeline() as p:
            res = p | beam.Create([1, 2, 3]) | beam.ParDo(MyDoFn()).with_outputs()
            self.assertIsNotNone(res[None].element_type)
            self.assertIsNotNone(res['even'].element_type)
            self.assertIsNotNone(res['odd'].element_type)
            res_main = res[None] | 'id_none' >> beam.ParDo(lambda e: [e]).with_input_types(int)
            res_even = res['even'] | 'id_even' >> beam.ParDo(lambda e: [e]).with_input_types(int)
            res_odd = res['odd'] | 'id_odd' >> beam.ParDo(lambda e: [e]).with_input_types(int)
            assert_that(res_main, equal_to([]), label='none_check')
            assert_that(res_even, equal_to([2]), label='even_check')
            assert_that(res_odd, equal_to([1, 3]), label='odd_check')

    def test_typed_ptransform_fn_pre_hints(self):
        if False:
            print('Hello World!')

        @beam.ptransform_fn
        @typehints.with_input_types(int)
        def MyMap(pcoll):
            if False:
                return 10
            return pcoll | beam.ParDo(lambda x: [x])
        self.assertListEqual([1, 2, 3], [1, 2, 3] | MyMap())
        with self.assertRaises(typehints.TypeCheckError):
            _ = ['a'] | MyMap()

    def test_typed_ptransform_fn_post_hints(self):
        if False:
            i = 10
            return i + 15

        @typehints.with_input_types(int)
        @beam.ptransform_fn
        def MyMap(pcoll):
            if False:
                while True:
                    i = 10
            return pcoll | beam.ParDo(lambda x: [x])
        self.assertListEqual([1, 2, 3], [1, 2, 3] | MyMap())
        with self.assertRaises(typehints.TypeCheckError):
            _ = ['a'] | MyMap()

    def test_typed_ptransform_fn_multi_input_types_pos(self):
        if False:
            print('Hello World!')

        @beam.ptransform_fn
        @beam.typehints.with_input_types(str, int)
        def multi_input(pcoll_tuple, additional_arg):
            if False:
                return 10
            (_, _) = pcoll_tuple
            assert additional_arg == 'additional_arg'
        with TestPipeline() as p:
            pcoll1 = p | 'c1' >> beam.Create(['a'])
            pcoll2 = p | 'c2' >> beam.Create([1])
            _ = (pcoll1, pcoll2) | multi_input('additional_arg')
            with self.assertRaises(typehints.TypeCheckError):
                _ = (pcoll2, pcoll1) | 'fails' >> multi_input('additional_arg')

    def test_typed_ptransform_fn_multi_input_types_kw(self):
        if False:
            for i in range(10):
                print('nop')

        @beam.ptransform_fn
        @beam.typehints.with_input_types(strings=str, integers=int)
        def multi_input(pcoll_dict, additional_arg):
            if False:
                for i in range(10):
                    print('nop')
            _ = pcoll_dict['strings']
            _ = pcoll_dict['integers']
            assert additional_arg == 'additional_arg'
        with TestPipeline() as p:
            pcoll1 = p | 'c1' >> beam.Create(['a'])
            pcoll2 = p | 'c2' >> beam.Create([1])
            _ = {'strings': pcoll1, 'integers': pcoll2} | multi_input('additional_arg')
            with self.assertRaises(typehints.TypeCheckError):
                _ = {'strings': pcoll2, 'integers': pcoll1} | 'fails' >> multi_input('additional_arg')

    def test_typed_dofn_method_not_iterable(self):
        if False:
            while True:
                i = 10

        class MyDoFn(beam.DoFn):

            def process(self, element: int) -> str:
                if False:
                    while True:
                        i = 10
                return str(element)
        with self.assertRaisesRegex(ValueError, 'str.*is not iterable'):
            _ = [1, 2, 3] | beam.ParDo(MyDoFn())

    def test_typed_dofn_method_return_none(self):
        if False:
            for i in range(10):
                print('nop')

        class MyDoFn(beam.DoFn):

            def process(self, unused_element: int) -> None:
                if False:
                    return 10
                pass
        result = [1, 2, 3] | beam.ParDo(MyDoFn())
        self.assertListEqual([], result)

    def test_typed_dofn_method_return_optional(self):
        if False:
            for i in range(10):
                print('nop')

        class MyDoFn(beam.DoFn):

            def process(self, unused_element: int) -> typehints.Optional[typehints.Iterable[int]]:
                if False:
                    return 10
                pass
        result = [1, 2, 3] | beam.ParDo(MyDoFn())
        self.assertListEqual([], result)

    def test_typed_dofn_method_return_optional_not_iterable(self):
        if False:
            i = 10
            return i + 15

        class MyDoFn(beam.DoFn):

            def process(self, unused_element: int) -> typehints.Optional[int]:
                if False:
                    print('Hello World!')
                pass
        with self.assertRaisesRegex(ValueError, 'int.*is not iterable'):
            _ = [1, 2, 3] | beam.ParDo(MyDoFn())

    def test_typed_callable_not_iterable(self):
        if False:
            return 10

        def do_fn(element: int) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return element
        with self.assertRaisesRegex(typehints.TypeCheckError, 'int.*is not iterable'):
            _ = [1, 2, 3] | beam.ParDo(do_fn)

    def test_typed_dofn_kwonly(self):
        if False:
            i = 10
            return i + 15

        class MyDoFn(beam.DoFn):

            def process(self, element: int, *, side_input: str) -> typehints.Generator[typehints.Optional[str]]:
                if False:
                    while True:
                        i = 10
                yield (str(element) if side_input else None)
        my_do_fn = MyDoFn()
        result = [1, 2, 3] | beam.ParDo(my_do_fn, side_input='abc')
        self.assertEqual(['1', '2', '3'], sorted(result))
        with self.assertRaisesRegex(typehints.TypeCheckError, 'requires.*str.*got.*int.*side_input'):
            _ = [1, 2, 3] | beam.ParDo(my_do_fn, side_input=1)

    def test_typed_dofn_var_kwargs(self):
        if False:
            while True:
                i = 10

        class MyDoFn(beam.DoFn):

            def process(self, element: int, **side_inputs: typehints.Dict[str, str]) -> typehints.Generator[typehints.Optional[str]]:
                if False:
                    return 10
                yield (str(element) if side_inputs else None)
        my_do_fn = MyDoFn()
        result = [1, 2, 3] | beam.ParDo(my_do_fn, foo='abc', bar='def')
        self.assertEqual(['1', '2', '3'], sorted(result))
        with self.assertRaisesRegex(typehints.TypeCheckError, 'requires.*str.*got.*int.*side_inputs'):
            _ = [1, 2, 3] | beam.ParDo(my_do_fn, a=1)

    def test_typed_callable_string_literals(self):
        if False:
            while True:
                i = 10

        def do_fn(element: 'int') -> 'typehints.List[str]':
            if False:
                return 10
            return [[str(element)] * 2]
        result = [1, 2] | beam.ParDo(do_fn)
        self.assertEqual([['1', '1'], ['2', '2']], sorted(result))

    def test_typed_ptransform_fn(self):
        if False:
            for i in range(10):
                print('nop')

        @beam.ptransform_fn
        @typehints.with_input_types(int)
        def MyMap(pcoll):
            if False:
                i = 10
                return i + 15

            def fn(element: int):
                if False:
                    return 10
                yield element
            return pcoll | beam.ParDo(fn)
        self.assertListEqual([1, 2, 3], [1, 2, 3] | MyMap())
        with self.assertRaisesRegex(typehints.TypeCheckError, 'int.*got.*str'):
            _ = ['a'] | MyMap()

    def test_typed_ptransform_fn_conflicting_hints(self):
        if False:
            while True:
                i = 10

        @beam.ptransform_fn
        @typehints.with_input_types(int)
        def MyMap(pcoll):
            if False:
                for i in range(10):
                    print('nop')

            def fn(element: float):
                if False:
                    i = 10
                    return i + 15
                yield element
            return pcoll | beam.ParDo(fn)
        with self.assertRaisesRegex(typehints.TypeCheckError, 'ParDo.*requires.*float.*got.*int'):
            _ = [1, 2, 3] | MyMap()
        with self.assertRaisesRegex(typehints.TypeCheckError, 'MyMap.*expected.*int.*got.*str'):
            _ = ['a'] | MyMap()

    def test_typed_dofn_string_literals(self):
        if False:
            return 10

        class MyDoFn(beam.DoFn):

            def process(self, element: 'int') -> 'typehints.List[str]':
                if False:
                    print('Hello World!')
                return [[str(element)] * 2]
        result = [1, 2] | beam.ParDo(MyDoFn())
        self.assertEqual([['1', '1'], ['2', '2']], sorted(result))

    def test_typed_map(self):
        if False:
            while True:
                i = 10

        def fn(element: int) -> int:
            if False:
                return 10
            return element * 2
        result = [1, 2, 3] | beam.Map(fn)
        self.assertEqual([2, 4, 6], sorted(result))

    def test_typed_map_return_optional(self):
        if False:
            print('Hello World!')

        def fn(element: int) -> typehints.Optional[int]:
            if False:
                i = 10
                return i + 15
            if element > 1:
                return element
        result = [1, 2, 3] | beam.Map(fn)
        self.assertCountEqual([None, 2, 3], result)

    def test_typed_flatmap(self):
        if False:
            print('Hello World!')

        def fn(element: int) -> typehints.Iterable[int]:
            if False:
                for i in range(10):
                    print('nop')
            yield (element * 2)
        result = [1, 2, 3] | beam.FlatMap(fn)
        self.assertCountEqual([2, 4, 6], result)

    def test_typed_flatmap_output_hint_not_iterable(self):
        if False:
            print('Hello World!')

        def fn(element: int) -> int:
            if False:
                i = 10
                return i + 15
            return element * 2
        with self.assertRaisesRegex(typehints.TypeCheckError, 'int.*is not iterable'):
            _ = [1, 2, 3] | beam.FlatMap(fn)

    def test_typed_flatmap_output_value_not_iterable(self):
        if False:
            return 10

        def fn(element: int) -> typehints.Iterable[int]:
            if False:
                i = 10
                return i + 15
            return element * 2
        with self.assertRaisesRegex(TypeError, 'int.*is not iterable'):
            _ = [1, 2, 3] | beam.FlatMap(fn)

    def test_typed_flatmap_optional(self):
        if False:
            i = 10
            return i + 15

        def fn(element: int) -> typehints.Optional[typehints.Iterable[int]]:
            if False:
                i = 10
                return i + 15
            if element > 1:
                yield (element * 2)

        def fn2(element: int) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return element
        result = [1, 2, 3] | beam.FlatMap(fn) | beam.Map(fn2)
        self.assertCountEqual([4, 6], result)

    def test_typed_ptransform_with_no_error(self):
        if False:
            return 10

        class StrToInt(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[str]) -> beam.pvalue.PCollection[int]:
                if False:
                    while True:
                        i = 10
                return pcoll | beam.Map(lambda x: int(x))

        class IntToStr(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[int]) -> beam.pvalue.PCollection[str]:
                if False:
                    print('Hello World!')
                return pcoll | beam.Map(lambda x: str(x))
        _ = ['1', '2', '3'] | StrToInt() | IntToStr()

    def test_typed_ptransform_with_bad_typehints(self):
        if False:
            while True:
                i = 10

        class StrToInt(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[str]) -> beam.pvalue.PCollection[int]:
                if False:
                    i = 10
                    return i + 15
                return pcoll | beam.Map(lambda x: int(x))

        class IntToStr(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[str]) -> beam.pvalue.PCollection[str]:
                if False:
                    print('Hello World!')
                return pcoll | beam.Map(lambda x: str(x))
        with self.assertRaisesRegex(typehints.TypeCheckError, "Input type hint violation at IntToStr: expected <class 'str'>, got <class 'int'>"):
            _ = ['1', '2', '3'] | StrToInt() | IntToStr()

    def test_typed_ptransform_with_bad_input(self):
        if False:
            for i in range(10):
                print('nop')

        class StrToInt(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[str]) -> beam.pvalue.PCollection[int]:
                if False:
                    i = 10
                    return i + 15
                return pcoll | beam.Map(lambda x: int(x))

        class IntToStr(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[int]) -> beam.pvalue.PCollection[str]:
                if False:
                    while True:
                        i = 10
                return pcoll | beam.Map(lambda x: str(x))
        with self.assertRaisesRegex(typehints.TypeCheckError, "Input type hint violation at StrToInt: expected <class 'str'>, got <class 'int'>"):
            _ = [1, 2, 3] | StrToInt() | IntToStr()

    def test_typed_ptransform_with_partial_typehints(self):
        if False:
            while True:
                i = 10

        class StrToInt(beam.PTransform):

            def expand(self, pcoll) -> beam.pvalue.PCollection[int]:
                if False:
                    while True:
                        i = 10
                return pcoll | beam.Map(lambda x: int(x))

        class IntToStr(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[int]) -> beam.pvalue.PCollection[str]:
                if False:
                    return 10
                return pcoll | beam.Map(lambda x: str(x))
        _ = [1, 2, 3] | StrToInt() | IntToStr()

    def test_typed_ptransform_with_bare_wrappers(self):
        if False:
            return 10

        class StrToInt(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
                if False:
                    for i in range(10):
                        print('nop')
                return pcoll | beam.Map(lambda x: int(x))

        class IntToStr(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[int]) -> beam.pvalue.PCollection[str]:
                if False:
                    while True:
                        i = 10
                return pcoll | beam.Map(lambda x: str(x))
        _ = [1, 2, 3] | StrToInt() | IntToStr()

    def test_typed_ptransform_with_no_typehints(self):
        if False:
            print('Hello World!')

        class StrToInt(beam.PTransform):

            def expand(self, pcoll):
                if False:
                    i = 10
                    return i + 15
                return pcoll | beam.Map(lambda x: int(x))

        class IntToStr(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[int]) -> beam.pvalue.PCollection[str]:
                if False:
                    print('Hello World!')
                return pcoll | beam.Map(lambda x: str(x))
        _ = [1, 2, 3] | StrToInt() | IntToStr()

    def test_typed_ptransform_with_generic_annotations(self):
        if False:
            return 10
        T = typing.TypeVar('T')

        class IntToInt(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[T]) -> beam.pvalue.PCollection[T]:
                if False:
                    return 10
                return pcoll | beam.Map(lambda x: x)

        class IntToStr(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[T]) -> beam.pvalue.PCollection[str]:
                if False:
                    while True:
                        i = 10
                return pcoll | beam.Map(lambda x: str(x))
        _ = [1, 2, 3] | IntToInt() | IntToStr()

    def test_typed_ptransform_with_do_outputs_tuple_compiles(self):
        if False:
            return 10

        class MyDoFn(beam.DoFn):

            def process(self, element: int, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                if element % 2:
                    yield beam.pvalue.TaggedOutput('odd', 1)
                else:
                    yield beam.pvalue.TaggedOutput('even', 1)

        class MyPTransform(beam.PTransform):

            def expand(self, pcoll: beam.pvalue.PCollection[int]):
                if False:
                    i = 10
                    return i + 15
                return pcoll | beam.ParDo(MyDoFn()).with_outputs('odd', 'even')
        _ = [1, 2, 3] | MyPTransform()

    def test_typed_ptransform_with_unknown_type_vars_tuple_compiles(self):
        if False:
            i = 10
            return i + 15

        @typehints.with_input_types(typing.TypeVar('T'))
        @typehints.with_output_types(typing.TypeVar('U'))
        def produces_unkown(e):
            if False:
                print('Hello World!')
            return e

        @typehints.with_input_types(int)
        def requires_int(e):
            if False:
                i = 10
                return i + 15
            return e

        class MyPTransform(beam.PTransform):

            def expand(self, pcoll):
                if False:
                    return 10
                unknowns = pcoll | beam.Map(produces_unkown)
                ints = pcoll | beam.Map(int)
                return (unknowns, ints) | beam.Flatten() | beam.Map(requires_int)
        _ = [1, 2, 3] | MyPTransform()

class NativeTypesTest(unittest.TestCase):

    def test_good_main_input(self):
        if False:
            print('Hello World!')

        @typehints.with_input_types(typing.Tuple[str, int])
        def munge(s_i):
            if False:
                return 10
            (s, i) = s_i
            return (s + 's', i * 2)
        result = [('apple', 5), ('pear', 3)] | beam.Map(munge)
        self.assertEqual([('apples', 10), ('pears', 6)], sorted(result))

    def test_bad_main_input(self):
        if False:
            while True:
                i = 10

        @typehints.with_input_types(typing.Tuple[str, str])
        def munge(s_i):
            if False:
                for i in range(10):
                    print('nop')
            (s, i) = s_i
            return (s + 's', i * 2)
        with self.assertRaises(typehints.TypeCheckError):
            [('apple', 5), ('pear', 3)] | beam.Map(munge)

    def test_bad_main_output(self):
        if False:
            while True:
                i = 10

        @typehints.with_input_types(typing.Tuple[int, int])
        @typehints.with_output_types(typing.Tuple[str, str])
        def munge(a_b):
            if False:
                while True:
                    i = 10
            (a, b) = a_b
            return (str(a), str(b))
        with self.assertRaises(typehints.TypeCheckError):
            [(5, 4), (3, 2)] | beam.Map(munge) | 'Again' >> beam.Map(munge)

class SideInputTest(unittest.TestCase):

    def _run_repeat_test(self, repeat):
        if False:
            while True:
                i = 10
        self._run_repeat_test_good(repeat)
        self._run_repeat_test_bad(repeat)

    @OptionsContext(pipeline_type_check=True)
    def _run_repeat_test_good(self, repeat):
        if False:
            return 10
        result = ['a', 'bb', 'c'] | beam.Map(repeat, 3)
        self.assertEqual(['aaa', 'bbbbbb', 'ccc'], sorted(result))
        result = ['a', 'bb', 'c'] | beam.Map(repeat, times=3)
        self.assertEqual(['aaa', 'bbbbbb', 'ccc'], sorted(result))

    def _run_repeat_test_bad(self, repeat):
        if False:
            return 10
        with self.assertRaises(typehints.TypeCheckError):
            ['a', 'bb', 'c'] | beam.Map(repeat, 'z')
        with self.assertRaises(typehints.TypeCheckError):
            ['a', 'bb', 'c'] | beam.Map(repeat, times='z')
        with self.assertRaises(typehints.TypeCheckError):
            ['a', 'bb', 'c'] | beam.Map(repeat, 3, 4)
        if all((param.default == param.empty for param in get_signature(repeat).parameters.values())):
            with self.assertRaisesRegex(typehints.TypeCheckError, '(takes exactly|missing a required)'):
                ['a', 'bb', 'c'] | beam.Map(repeat)

    def test_basic_side_input_hint(self):
        if False:
            while True:
                i = 10

        @typehints.with_input_types(str, int)
        def repeat(s, times):
            if False:
                return 10
            return s * times
        self._run_repeat_test(repeat)

    def test_keyword_side_input_hint(self):
        if False:
            return 10

        @typehints.with_input_types(str, times=int)
        def repeat(s, times):
            if False:
                for i in range(10):
                    print('nop')
            return s * times
        self._run_repeat_test(repeat)

    def test_default_typed_hint(self):
        if False:
            return 10

        @typehints.with_input_types(str, int)
        def repeat(s, times=3):
            if False:
                print('Hello World!')
            return s * times
        self._run_repeat_test(repeat)

    def test_default_untyped_hint(self):
        if False:
            i = 10
            return i + 15

        @typehints.with_input_types(str)
        def repeat(s, times=3):
            if False:
                print('Hello World!')
            return s * times
        self._run_repeat_test_good(repeat)

    @OptionsContext(pipeline_type_check=True)
    def test_varargs_side_input_hint(self):
        if False:
            print('Hello World!')

        @typehints.with_input_types(str, int)
        def repeat(s, *times):
            if False:
                i = 10
                return i + 15
            return s * times[0]
        result = ['a', 'bb', 'c'] | beam.Map(repeat, 3)
        self.assertEqual(['aaa', 'bbbbbb', 'ccc'], sorted(result))
        with self.assertRaisesRegex(typehints.TypeCheckError, "requires Tuple\\[<class \\'int\\'>, ...\\] but got Tuple\\[<class \\'str\\'>, ...\\]"):
            ['a', 'bb', 'c'] | beam.Map(repeat, 'z')

    def test_var_positional_only_side_input_hint(self):
        if False:
            while True:
                i = 10
        result = ['a', 'b', 'c'] | beam.Map(lambda *args: args, 5).with_input_types(str, int).with_output_types(typehints.Tuple[str, int])
        self.assertEqual([('a', 5), ('b', 5), ('c', 5)], sorted(result))
        with self.assertRaisesRegex(typehints.TypeCheckError, "requires Tuple\\[Union\\[<class \\'int\\'>, <class \\'str\\'>\\], ...\\] but got Tuple\\[Union\\[<class \\'float\\'>, <class \\'int\\'>\\], ...\\]"):
            _ = [1.2] | beam.Map(lambda *_: 'a', 5).with_input_types(int, str)

    def test_var_keyword_side_input_hint(self):
        if False:
            print('Hello World!')
        result = ['a', 'b', 'c'] | beam.Map(lambda e, **kwargs: (e, kwargs), kw=5).with_input_types(str, ignored=int)
        self.assertEqual([('a', {'kw': 5}), ('b', {'kw': 5}), ('c', {'kw': 5})], sorted(result))
        with self.assertRaisesRegex(typehints.TypeCheckError, "requires Dict\\[<class \\'str\\'>, <class \\'str\\'>\\] but got Dict\\[<class \\'str\\'>, <class \\'int\\'>\\]"):
            _ = ['a', 'b', 'c'] | beam.Map(lambda e, **_: 'a', kw=5).with_input_types(str, ignored=str)

    def test_deferred_side_inputs(self):
        if False:
            for i in range(10):
                print('nop')

        @typehints.with_input_types(str, int)
        def repeat(s, times):
            if False:
                print('Hello World!')
            return s * times
        with TestPipeline() as p:
            main_input = p | beam.Create(['a', 'bb', 'c'])
            side_input = p | 'side' >> beam.Create([3])
            result = main_input | beam.Map(repeat, pvalue.AsSingleton(side_input))
            assert_that(result, equal_to(['aaa', 'bbbbbb', 'ccc']))
        bad_side_input = p | 'bad_side' >> beam.Create(['z'])
        with self.assertRaises(typehints.TypeCheckError):
            main_input | 'bis' >> beam.Map(repeat, pvalue.AsSingleton(bad_side_input))

    def test_deferred_side_input_iterable(self):
        if False:
            print('Hello World!')

        @typehints.with_input_types(str, typing.Iterable[str])
        def concat(glue, items):
            if False:
                print('Hello World!')
            return glue.join(sorted(items))
        with TestPipeline() as p:
            main_input = p | beam.Create(['a', 'bb', 'c'])
            side_input = p | 'side' >> beam.Create(['x', 'y', 'z'])
            result = main_input | beam.Map(concat, pvalue.AsIter(side_input))
            assert_that(result, equal_to(['xayaz', 'xbbybbz', 'xcycz']))
        bad_side_input = p | 'bad_side' >> beam.Create([1, 2, 3])
        with self.assertRaises(typehints.TypeCheckError):
            main_input | 'fail' >> beam.Map(concat, pvalue.AsIter(bad_side_input))

class CustomTransformTest(unittest.TestCase):

    class CustomTransform(beam.PTransform):

        def _extract_input_pvalues(self, pvalueish):
            if False:
                return 10
            return (pvalueish, (pvalueish['in0'], pvalueish['in1']))

        def expand(self, pvalueish):
            if False:
                for i in range(10):
                    print('nop')
            return {'out0': pvalueish['in0'], 'out1': pvalueish['in1']}

        def with_input_types(self, *args, **kwargs):
            if False:
                return 10
            return WithTypeHints.with_input_types(self, *args, **kwargs)

        def with_output_types(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return WithTypeHints.with_output_types(self, *args, **kwargs)
    test_input = {'in0': ['a', 'b', 'c'], 'in1': [1, 2, 3]}

    def check_output(self, result):
        if False:
            return 10
        self.assertEqual(['a', 'b', 'c'], sorted(result['out0']))
        self.assertEqual([1, 2, 3], sorted(result['out1']))

    def test_custom_transform(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(self.test_input | self.CustomTransform())

    def test_keyword_type_hints(self):
        if False:
            i = 10
            return i + 15
        self.check_output(self.test_input | self.CustomTransform().with_input_types(in0=str, in1=int))
        self.check_output(self.test_input | self.CustomTransform().with_input_types(in0=str))
        self.check_output(self.test_input | self.CustomTransform().with_output_types(out0=str, out1=int))
        with self.assertRaises(typehints.TypeCheckError):
            self.test_input | self.CustomTransform().with_input_types(in0=int)
        with self.assertRaises(typehints.TypeCheckError):
            self.test_input | self.CustomTransform().with_output_types(out0=int)

    def test_flat_type_hint(self):
        if False:
            print('Hello World!')
        {'in0': ['a', 'b', 'c'], 'in1': ['x', 'y', 'z']} | self.CustomTransform().with_input_types(str)
        with self.assertRaises(typehints.TypeCheckError):
            self.test_input | self.CustomTransform().with_input_types(str)
        with self.assertRaises(typehints.TypeCheckError):
            self.test_input | self.CustomTransform().with_input_types(int)
        with self.assertRaises(typehints.TypeCheckError):
            self.test_input | self.CustomTransform().with_output_types(int)

class AnnotationsTest(unittest.TestCase):

    def test_pardo_wrapper_builtin_method(self):
        if False:
            print('Hello World!')
        th = beam.ParDo(str.strip).get_type_hints()
        if sys.version_info < (3, 7):
            self.assertEqual(th.input_types, ((str,), {}))
        else:
            self.assertEqual(th.input_types, ((str, typehints.Any), {}))
        self.assertEqual(th.output_types, ((typehints.Any,), {}))

    def test_pardo_wrapper_builtin_type(self):
        if False:
            i = 10
            return i + 15
        th = beam.ParDo(list).get_type_hints()
        self.assertEqual(th.input_types, ((typehints.Any,), {}))
        self.assertEqual(th.output_types, ((typehints.Any,), {}))

    def test_pardo_wrapper_builtin_func(self):
        if False:
            i = 10
            return i + 15
        th = beam.ParDo(len).get_type_hints()
        self.assertIsNone(th.input_types)
        self.assertIsNone(th.output_types)

    def test_pardo_dofn(self):
        if False:
            while True:
                i = 10

        class MyDoFn(beam.DoFn):

            def process(self, element: int) -> typehints.Generator[str]:
                if False:
                    while True:
                        i = 10
                yield str(element)
        th = beam.ParDo(MyDoFn()).get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((str,), {}))

    def test_pardo_dofn_not_iterable(self):
        if False:
            for i in range(10):
                print('nop')

        class MyDoFn(beam.DoFn):

            def process(self, element: int) -> str:
                if False:
                    i = 10
                    return i + 15
                return str(element)
        with self.assertRaisesRegex(ValueError, 'str.*is not iterable'):
            _ = beam.ParDo(MyDoFn()).get_type_hints()

    def test_pardo_wrapper(self):
        if False:
            i = 10
            return i + 15

        def do_fn(element: int) -> typehints.Iterable[str]:
            if False:
                print('Hello World!')
            return [str(element)]
        th = beam.ParDo(do_fn).get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((str,), {}))

    def test_pardo_wrapper_tuple(self):
        if False:
            while True:
                i = 10

        def do_fn(element: int) -> typehints.Iterable[typehints.Tuple[str, int]]:
            if False:
                return 10
            return [(str(element), element)]
        th = beam.ParDo(do_fn).get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((typehints.Tuple[str, int],), {}))

    def test_pardo_wrapper_not_iterable(self):
        if False:
            while True:
                i = 10

        def do_fn(element: int) -> str:
            if False:
                while True:
                    i = 10
            return str(element)
        with self.assertRaisesRegex(typehints.TypeCheckError, 'str.*is not iterable'):
            _ = beam.ParDo(do_fn).get_type_hints()

    def test_flat_map_wrapper(self):
        if False:
            for i in range(10):
                print('nop')

        def map_fn(element: int) -> typehints.Iterable[int]:
            if False:
                for i in range(10):
                    print('nop')
            return [element, element + 1]
        th = beam.FlatMap(map_fn).get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((int,), {}))

    def test_flat_map_wrapper_optional_output(self):
        if False:
            return 10

        def map_fn(element: int) -> typehints.Optional[typehints.Iterable[int]]:
            if False:
                i = 10
                return i + 15
            return [element, element + 1]
        th = beam.FlatMap(map_fn).get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((int,), {}))

    @unittest.skip('https://github.com/apache/beam/issues/19961: Py3 annotations not yet supported for MapTuple')
    def test_flat_map_tuple_wrapper(self):
        if False:
            return 10

        def tuple_map_fn(a: str, b: str, c: str) -> typehints.Iterable[str]:
            if False:
                print('Hello World!')
            return [a, b, c]
        th = beam.FlatMapTuple(tuple_map_fn).get_type_hints()
        self.assertEqual(th.input_types, ((str, str, str), {}))
        self.assertEqual(th.output_types, ((str,), {}))

    def test_map_wrapper(self):
        if False:
            while True:
                i = 10

        def map_fn(unused_element: int) -> int:
            if False:
                while True:
                    i = 10
            return 1
        th = beam.Map(map_fn).get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((int,), {}))

    def test_map_wrapper_optional_output(self):
        if False:
            i = 10
            return i + 15

        def map_fn(unused_element: int) -> typehints.Optional[int]:
            if False:
                return 10
            return 1
        th = beam.Map(map_fn).get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((typehints.Optional[int],), {}))

    @unittest.skip('https://github.com/apache/beam/issues/19961: Py3 annotations not yet supported for MapTuple')
    def test_map_tuple(self):
        if False:
            while True:
                i = 10

        def tuple_map_fn(a: str, b: str, c: str) -> str:
            if False:
                while True:
                    i = 10
            return a + b + c
        th = beam.MapTuple(tuple_map_fn).get_type_hints()
        self.assertEqual(th.input_types, ((str, str, str), {}))
        self.assertEqual(th.output_types, ((str,), {}))

    def test_filter_wrapper(self):
        if False:
            for i in range(10):
                print('nop')

        def filter_fn(element: int) -> bool:
            if False:
                while True:
                    i = 10
            return bool(element % 2)
        th = beam.Filter(filter_fn).get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((int,), {}))
if __name__ == '__main__':
    unittest.main()