"""Unit tests for side inputs."""
import logging
import unittest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class KeywordOnlyArgsTests(unittest.TestCase):

    def test_side_input_keyword_only_args(self):
        if False:
            print('Hello World!')
        with TestPipeline() as pipeline:

            def sort_with_side_inputs(x, *s, reverse=False):
                if False:
                    return 10
                for y in s:
                    yield sorted([x] + y, reverse=reverse)

            def sort_with_side_inputs_without_default_values(x, *s, reverse):
                if False:
                    i = 10
                    return i + 15
                for y in s:
                    yield sorted([x] + y, reverse=reverse)
            pcol = pipeline | 'start' >> beam.Create([1, 2])
            side = pipeline | 'side' >> beam.Create([3, 4])
            result1 = pcol | 'compute1' >> beam.FlatMap(sort_with_side_inputs, beam.pvalue.AsList(side), reverse=True)
            assert_that(result1, equal_to([[4, 3, 1], [4, 3, 2]]), label='assert1')
            result2 = pcol | 'compute2' >> beam.FlatMap(sort_with_side_inputs, beam.pvalue.AsList(side))
            assert_that(result2, equal_to([[1, 3, 4], [2, 3, 4]]), label='assert2')
            result3 = pcol | 'compute3' >> beam.FlatMap(sort_with_side_inputs)
            assert_that(result3, equal_to([]), label='assert3')
            result4 = pcol | 'compute4' >> beam.FlatMap(sort_with_side_inputs, reverse=True)
            assert_that(result4, equal_to([]), label='assert4')
            result5 = pcol | 'compute5' >> beam.FlatMap(sort_with_side_inputs_without_default_values, beam.pvalue.AsList(side), reverse=True)
            assert_that(result5, equal_to([[4, 3, 1], [4, 3, 2]]), label='assert5')
            result6 = pcol | 'compute6' >> beam.FlatMap(sort_with_side_inputs_without_default_values, beam.pvalue.AsList(side), reverse=False)
            assert_that(result6, equal_to([[1, 3, 4], [2, 3, 4]]), label='assert6')
            result7 = pcol | 'compute7' >> beam.FlatMap(sort_with_side_inputs_without_default_values, reverse=False)
            assert_that(result7, equal_to([]), label='assert7')
            result8 = pcol | 'compute8' >> beam.FlatMap(sort_with_side_inputs_without_default_values, reverse=True)
            assert_that(result8, equal_to([]), label='assert8')

    def test_combine_keyword_only_args(self):
        if False:
            i = 10
            return i + 15
        with TestPipeline() as pipeline:

            def bounded_sum(values, *s, bound=500):
                if False:
                    for i in range(10):
                        print('nop')
                return min(sum(values) + sum(s), bound)

            def bounded_sum_without_default_values(values, *s, bound):
                if False:
                    print('Hello World!')
                return min(sum(values) + sum(s), bound)
            pcoll = pipeline | 'start' >> beam.Create([6, 3, 1])
            result1 = pcoll | 'sum1' >> beam.CombineGlobally(bounded_sum, 5, 8, bound=20)
            result2 = pcoll | 'sum2' >> beam.CombineGlobally(bounded_sum, 0, 0)
            result3 = pcoll | 'sum3' >> beam.CombineGlobally(bounded_sum)
            result4 = pcoll | 'sum4' >> beam.CombineGlobally(bounded_sum, bound=5)
            result5 = pcoll | 'sum5' >> beam.CombineGlobally(bounded_sum_without_default_values, 5, 8, bound=20)
            result6 = pcoll | 'sum6' >> beam.CombineGlobally(bounded_sum_without_default_values, 0, 0, bound=500)
            result7 = pcoll | 'sum7' >> beam.CombineGlobally(bounded_sum_without_default_values, bound=500)
            result8 = pcoll | 'sum8' >> beam.CombineGlobally(bounded_sum_without_default_values, bound=5)
            assert_that(result1, equal_to([20]), label='assert1')
            assert_that(result2, equal_to([10]), label='assert2')
            assert_that(result3, equal_to([10]), label='assert3')
            assert_that(result4, equal_to([5]), label='assert4')
            assert_that(result5, equal_to([20]), label='assert5')
            assert_that(result6, equal_to([10]), label='assert6')
            assert_that(result7, equal_to([10]), label='assert7')
            assert_that(result8, equal_to([5]), label='assert8')

    def test_do_fn_keyword_only_args(self):
        if False:
            return 10
        with TestPipeline() as pipeline:

            class MyDoFn(beam.DoFn):

                def process(self, element, *s, bound=500):
                    if False:
                        return 10
                    return [min(sum(s) + element, bound)]
            pcoll = pipeline | 'start' >> beam.Create([6, 3, 1])
            result1 = pcoll | 'sum1' >> beam.ParDo(MyDoFn(), 5, 8, bound=15)
            result2 = pcoll | 'sum2' >> beam.ParDo(MyDoFn(), 5, 8)
            result3 = pcoll | 'sum3' >> beam.ParDo(MyDoFn())
            result4 = pcoll | 'sum4' >> beam.ParDo(MyDoFn(), bound=5)
            assert_that(result1, equal_to([15, 15, 14]), label='assert1')
            assert_that(result2, equal_to([19, 16, 14]), label='assert2')
            assert_that(result3, equal_to([6, 3, 1]), label='assert3')
            assert_that(result4, equal_to([5, 3, 1]), label='assert4')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()