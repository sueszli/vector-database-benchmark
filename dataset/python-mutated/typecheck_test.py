"""
Unit tests for typecheck.

See additional runtime_type_check=True tests in ptransform_test.py.
"""
import os
import tempfile
import unittest
from typing import Iterable
from typing import Tuple
import apache_beam as beam
from apache_beam import Pipeline
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import TypeOptions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.typehints import TypeCheckError
from apache_beam.typehints import decorators
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types
decorators._enable_from_callable = True

class MyDoFn(beam.DoFn):

    def __init__(self, output_filename):
        if False:
            return 10
        super().__init__()
        self.output_filename = output_filename

    def _output(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a file used to record function calls.'
        if not hasattr(self, 'output_file'):
            self._output_file = open(self.output_filename, 'at', buffering=1)
        return self._output_file

    def start_bundle(self):
        if False:
            while True:
                i = 10
        self._output().write('start_bundle\n')

    def finish_bundle(self):
        if False:
            print('Hello World!')
        self._output().write('finish_bundle\n')

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self._output().write('setup\n')

    def teardown(self):
        if False:
            for i in range(10):
                print('nop')
        self._output().write('teardown\n')
        self._output().close()

    def process(self, element: int, *args, **kwargs) -> Iterable[int]:
        if False:
            i = 10
            return i + 15
        self._output().write('process\n')
        yield element

class MyDoFnBadAnnotation(MyDoFn):

    def process(self, element: int, *args, **kwargs) -> int:
        if False:
            i = 10
            return i + 15
        return super().process()

class RuntimeTypeCheckTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.p = TestPipeline(options=PipelineOptions(runtime_type_check=True, performance_runtime_type_check=False))

    def test_setup(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(e: int) -> int:
            if False:
                i = 10
                return i + 15
            return str(e)
        with self.assertRaisesRegex(TypeCheckError, 'output should be.*int.*received.*str'):
            _ = self.p | beam.Create([1, 2, 3]) | beam.Map(fn)
            self.p.run()

    def test_wrapper_pass_through(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmp_dirname:
            path = os.path.join(tmp_dirname + 'tmp_filename')
            dofn = MyDoFn(path)
            result = self.p | beam.Create([1, 2, 3]) | beam.ParDo(dofn)
            assert_that(result, equal_to([1, 2, 3]))
            self.p.run()
            with open(path, mode='r') as ft:
                lines = [line.strip() for line in ft]
                self.assertListEqual(['setup', 'start_bundle', 'process', 'process', 'process', 'finish_bundle', 'teardown'], lines)

    def test_wrapper_pipeline_type_check(self):
        if False:
            i = 10
            return i + 15
        with tempfile.NamedTemporaryFile(mode='w+t') as f:
            dofn = MyDoFnBadAnnotation(f.name)
            with self.assertRaisesRegex(ValueError, 'int.*is not iterable'):
                _ = self.p | beam.Create([1, 2, 3]) | beam.ParDo(dofn)

class PerformanceRuntimeTypeCheckTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.p = Pipeline(options=PipelineOptions(performance_runtime_type_check=True, pipeline_type_check=False))

    def assertStartswith(self, msg, prefix):
        if False:
            return 10
        self.assertTrue(msg.startswith(prefix), '"%s" does not start with "%s"' % (msg, prefix))

    def test_simple_input_error(self):
        if False:
            return 10
        with self.assertRaises(TypeCheckError) as e:
            self.p | beam.Create([1, 1]) | beam.FlatMap(lambda x: [int(x)]).with_input_types(str).with_output_types(int)
            self.p.run()
        self.assertIn("Type-hint for argument: 'x' violated. Expected an instance of {}, instead found 1, an instance of {}".format(str, int), e.exception.args[0])

    def test_simple_output_error(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeCheckError) as e:
            self.p | beam.Create(['1', '1']) | beam.FlatMap(lambda x: [int(x)]).with_input_types(int).with_output_types(int)
            self.p.run()
        self.assertIn("Type-hint for argument: 'x' violated. Expected an instance of {}, instead found 1, an instance of {}.".format(int, str), e.exception.args[0])

    def test_simple_input_error_with_kwarg_typehints(self):
        if False:
            for i in range(10):
                print('nop')

        @with_input_types(element=int)
        @with_output_types(int)
        class ToInt(beam.DoFn):

            def process(self, element, *args, **kwargs):
                if False:
                    print('Hello World!')
                yield int(element)
        with self.assertRaises(TypeCheckError) as e:
            self.p | beam.Create(['1', '1']) | beam.ParDo(ToInt())
            self.p.run()
        self.assertStartswith(e.exception.args[0], "Runtime type violation detected within ParDo(ToInt): Type-hint for argument: 'element' violated. Expected an instance of {}, instead found 1, an instance of {}.".format(int, str))

    def test_do_fn_returning_non_iterable_throws_error(self):
        if False:
            i = 10
            return i + 15

        def incorrect_par_do_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 5
        with self.assertRaises(TypeError) as cm:
            self.p | beam.Create([1, 1]) | beam.FlatMap(incorrect_par_do_fn)
            self.p.run()
        self.assertStartswith(cm.exception.args[0], "'int' object is not iterable ")

    def test_simple_type_satisfied(self):
        if False:
            i = 10
            return i + 15

        @with_input_types(int, int)
        @with_output_types(int)
        class AddWithNum(beam.DoFn):

            def process(self, element, num):
                if False:
                    return 10
                return [element + num]
        results = self.p | 'T' >> beam.Create([1, 2, 3]).with_output_types(int) | 'Add' >> beam.ParDo(AddWithNum(), 1)
        assert_that(results, equal_to([2, 3, 4]))
        self.p.run()

    def test_simple_type_violation(self):
        if False:
            print('Hello World!')
        self.p._options.view_as(TypeOptions).pipeline_type_check = False

        @with_output_types(str)
        @with_input_types(x=int)
        def int_to_string(x):
            if False:
                while True:
                    i = 10
            return str(x)
        self.p | 'Create' >> beam.Create(['some_string']) | 'ToStr' >> beam.Map(int_to_string)
        with self.assertRaises(TypeCheckError) as e:
            self.p.run()
        self.assertStartswith(e.exception.args[0], "Runtime type violation detected within ParDo(ToStr): Type-hint for argument: 'x' violated. Expected an instance of {}, instead found some_string, an instance of {}.".format(int, str))

    def test_pipeline_checking_satisfied_but_run_time_types_violate(self):
        if False:
            print('Hello World!')
        self.p._options.view_as(TypeOptions).pipeline_type_check = False

        @with_output_types(Tuple[bool, int])
        @with_input_types(a=int)
        def is_even_as_key(a):
            if False:
                while True:
                    i = 10
            return (a % 2, a)
        self.p | 'Nums' >> beam.Create(range(1)).with_output_types(int) | 'IsEven' >> beam.Map(is_even_as_key) | 'Parity' >> beam.GroupByKey()
        with self.assertRaises(TypeCheckError) as e:
            self.p.run()
        self.assertStartswith(e.exception.args[0], "Runtime type violation detected within ParDo(IsEven): Type-hint for return type violated: Tuple[<class 'bool'>, <class 'int'>] hint type-constraint violated. The type of element #0 in the passed tuple is incorrect. Expected an instance of type <class 'bool'>, instead received an instance of type int. ")

    def test_pipeline_runtime_checking_violation_composite_type_output(self):
        if False:
            i = 10
            return i + 15
        self.p._options.view_as(TypeOptions).pipeline_type_check = False
        with self.assertRaises(TypeCheckError) as e:
            self.p | beam.Create([(1, 3.0)]) | 'Swap' >> beam.FlatMap(lambda x_y1: [x_y1[0] + x_y1[1]]).with_input_types(Tuple[int, float]).with_output_types(int)
            self.p.run()
        self.assertStartswith(e.exception.args[0], 'Runtime type violation detected within ParDo(Swap): Type-hint for return type violated. Expected an instance of {}, instead found 4.0, an instance of {}.'.format(int, float))

    def test_downstream_input_type_hint_error_has_descriptive_error_msg(self):
        if False:
            return 10

        @with_input_types(int)
        @with_output_types(int)
        class IntToInt(beam.DoFn):

            def process(self, element, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                yield element

        @with_input_types(str)
        @with_output_types(int)
        class StrToInt(beam.DoFn):

            def process(self, element, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                yield int(element)
        with self.assertRaises(TypeCheckError) as e:
            self.p | beam.Create([9]) | beam.ParDo(IntToInt()) | beam.ParDo(StrToInt())
            self.p.run()
        self.assertStartswith(e.exception.args[0], "Runtime type violation detected within ParDo(StrToInt): Type-hint for argument: 'element' violated. Expected an instance of {}, instead found 9, an instance of {}. [while running 'ParDo(IntToInt)']".format(str, int))
if __name__ == '__main__':
    unittest.main()