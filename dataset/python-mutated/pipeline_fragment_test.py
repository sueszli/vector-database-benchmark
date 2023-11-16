"""Tests for apache_beam.runners.interactive.pipeline_fragment."""
import unittest
from unittest.mock import patch
import apache_beam as beam
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive import interactive_runner as ir
from apache_beam.runners.interactive import pipeline_fragment as pf
from apache_beam.runners.interactive.testing.mock_ipython import mock_get_ipython
from apache_beam.runners.interactive.testing.pipeline_assertion import assert_pipeline_equal
from apache_beam.runners.interactive.testing.pipeline_assertion import assert_pipeline_proto_equal
from apache_beam.testing.test_stream import TestStream

@unittest.skipIf(not ie.current_env().is_interactive_ready, '[interactive] dependency is not installed.')
class PipelineFragmentTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        ie.new_env()
        ie.current_env()._is_in_ipython = True
        ie.current_env()._is_in_notebook = True

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_build_pipeline_fragment(self, cell):
        if False:
            for i in range(10):
                print('nop')
        with cell:
            p = beam.Pipeline(ir.InteractiveRunner())
            p_expected = beam.Pipeline(ir.InteractiveRunner())
            ib.watch(locals())
        with cell:
            init = p | 'Init' >> beam.Create(range(10))
            init_expected = p_expected | 'Init' >> beam.Create(range(10))
        with cell:
            square = init | 'Square' >> beam.Map(lambda x: x * x)
            _ = init | 'Cube' >> beam.Map(lambda x: x ** 3)
            _ = init_expected | 'Square' >> beam.Map(lambda x: x * x)
        ib.watch(locals())
        fragment = pf.PipelineFragment([square]).deduce_fragment()
        assert_pipeline_equal(self, p_expected, fragment)

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_user_pipeline_intact_after_deducing_pipeline_fragment(self, cell):
        if False:
            for i in range(10):
                print('nop')
        with cell:
            p = beam.Pipeline(ir.InteractiveRunner())
            ib.watch({'p': p})
        with cell:
            init = p | 'Init' >> beam.Create(range(10))
        with cell:
            square = init | 'Square' >> beam.Map(lambda x: x * x)
        with cell:
            cube = init | 'Cube' >> beam.Map(lambda x: x ** 3)
        ib.watch({'init': init, 'square': square, 'cube': cube})
        user_pipeline_proto_before_deducing_fragment = p.to_runner_api(return_context=False)
        _ = pf.PipelineFragment([square]).deduce_fragment()
        user_pipeline_proto_after_deducing_fragment = p.to_runner_api(return_context=False)
        assert_pipeline_proto_equal(self, user_pipeline_proto_before_deducing_fragment, user_pipeline_proto_after_deducing_fragment)

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_pipeline_fragment_produces_correct_data(self, cell):
        if False:
            print('Hello World!')
        with cell:
            p = beam.Pipeline(ir.InteractiveRunner())
            ib.watch({'p': p})
        with cell:
            init = p | 'Init' >> beam.Create(range(5))
        with cell:
            square = init | 'Square' >> beam.Map(lambda x: x * x)
            _ = init | 'Cube' >> beam.Map(lambda x: x ** 3)
        ib.watch(locals())
        result = pf.PipelineFragment([square]).run()
        self.assertEqual([0, 1, 4, 9, 16], list(result.get(square)))

    def test_fragment_does_not_prune_teststream(self):
        if False:
            i = 10
            return i + 15
        'Tests that the fragment does not prune the TestStream composite parts.\n    '
        options = StandardOptions(streaming=True)
        p = beam.Pipeline(ir.InteractiveRunner(), options)
        test_stream = p | TestStream(output_tags=['a', 'b'])
        a = test_stream['a'] | 'a' >> beam.Map(lambda _: _)
        b = test_stream['b'] | 'b' >> beam.Map(lambda _: _)
        fragment = pf.PipelineFragment([b]).deduce_fragment()
        fragment.to_runner_api()

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_pipeline_composites(self, cell):
        if False:
            i = 10
            return i + 15
        'Tests that composites are supported.\n    '
        with cell:
            p = beam.Pipeline(ir.InteractiveRunner())
            ib.watch({'p': p})
        with cell:
            init = p | 'Init' >> beam.Create(range(5))
        with cell:

            @beam.ptransform_fn
            def Bar(pcoll):
                if False:
                    return 10
                return pcoll | beam.Map(lambda n: 2 * n)

            @beam.ptransform_fn
            def Foo(pcoll):
                if False:
                    i = 10
                    return i + 15
                p1 = pcoll | beam.Map(lambda n: 3 * n)
                p2 = pcoll | beam.Map(str)
                bar = p1 | Bar()
                return {'pc1': p1, 'pc2': p2, 'bar': bar}
            res = init | Foo()
            ib.watch(res)
        pc = res['bar']
        result = pf.PipelineFragment([pc]).run()
        self.assertEqual([0, 6, 12, 18, 24], list(result.get(pc)))

    def test_ib_show_without_using_ir(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that ib.show is called when ir is not specified.\n    '
        p = beam.Pipeline()
        print_words = p | beam.Create(['this is a test']) | beam.Map(print)
        with self.assertRaises(RuntimeError):
            ib.show(print_words)
if __name__ == '__main__':
    unittest.main()