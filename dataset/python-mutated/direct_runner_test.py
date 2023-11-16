import threading
import unittest
from collections import defaultdict
import hamcrest as hc
import apache_beam as beam
from apache_beam.metrics.cells import DistributionData
from apache_beam.metrics.cells import DistributionResult
from apache_beam.metrics.execution import MetricKey
from apache_beam.metrics.execution import MetricResult
from apache_beam.metrics.metric import Metrics
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.metrics.metricbase import MetricName
from apache_beam.pipeline import Pipeline
from apache_beam.runners import DirectRunner
from apache_beam.runners import TestDirectRunner
from apache_beam.runners import create_runner
from apache_beam.runners.direct.evaluation_context import _ExecutionContext
from apache_beam.runners.direct.transform_evaluator import _GroupByKeyOnlyEvaluator
from apache_beam.runners.direct.transform_evaluator import _TransformEvaluator
from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class DirectPipelineResultTest(unittest.TestCase):

    def test_waiting_on_result_stops_executor_threads(self):
        if False:
            while True:
                i = 10
        pre_test_threads = set((t.ident for t in threading.enumerate()))
        for runner in ['DirectRunner', 'BundleBasedDirectRunner', 'SwitchingDirectRunner']:
            pipeline = test_pipeline.TestPipeline(runner=runner)
            _ = pipeline | beam.Create([{'foo': 'bar'}])
            result = pipeline.run()
            result.wait_until_finish()
            post_test_threads = set((t.ident for t in threading.enumerate()))
            new_threads = post_test_threads - pre_test_threads
            self.assertEqual(len(new_threads), 0)

    def test_direct_runner_metrics(self):
        if False:
            return 10

        class MyDoFn(beam.DoFn):

            def start_bundle(self):
                if False:
                    print('Hello World!')
                count = Metrics.counter(self.__class__, 'bundles')
                count.inc()

            def finish_bundle(self):
                if False:
                    for i in range(10):
                        print('nop')
                count = Metrics.counter(self.__class__, 'finished_bundles')
                count.inc()

            def process(self, element):
                if False:
                    while True:
                        i = 10
                gauge = Metrics.gauge(self.__class__, 'latest_element')
                gauge.set(element)
                count = Metrics.counter(self.__class__, 'elements')
                count.inc()
                distro = Metrics.distribution(self.__class__, 'element_dist')
                distro.update(element)
                return [element]
        p = Pipeline(DirectRunner())
        pcoll = p | beam.Create([1, 2, 3, 4, 5], reshuffle=False) | 'Do' >> beam.ParDo(MyDoFn())
        assert_that(pcoll, equal_to([1, 2, 3, 4, 5]))
        result = p.run()
        result.wait_until_finish()
        metrics = result.metrics().query(MetricsFilter().with_step('Do'))
        namespace = '{}.{}'.format(MyDoFn.__module__, MyDoFn.__name__)
        hc.assert_that(metrics['counters'], hc.contains_inanyorder(MetricResult(MetricKey('Do', MetricName(namespace, 'elements')), 5, 5), MetricResult(MetricKey('Do', MetricName(namespace, 'bundles')), 1, 1), MetricResult(MetricKey('Do', MetricName(namespace, 'finished_bundles')), 1, 1)))
        hc.assert_that(metrics['distributions'], hc.contains_inanyorder(MetricResult(MetricKey('Do', MetricName(namespace, 'element_dist')), DistributionResult(DistributionData(15, 5, 1, 5)), DistributionResult(DistributionData(15, 5, 1, 5)))))
        gauge_result = metrics['gauges'][0]
        hc.assert_that(gauge_result.key, hc.equal_to(MetricKey('Do', MetricName(namespace, 'latest_element'))))
        hc.assert_that(gauge_result.committed.value, hc.equal_to(5))
        hc.assert_that(gauge_result.attempted.value, hc.equal_to(5))

    def test_create_runner(self):
        if False:
            while True:
                i = 10
        self.assertTrue(isinstance(create_runner('DirectRunner'), DirectRunner))
        self.assertTrue(isinstance(create_runner('TestDirectRunner'), TestDirectRunner))

class BundleBasedRunnerTest(unittest.TestCase):

    def test_type_hints(self):
        if False:
            print('Hello World!')
        with test_pipeline.TestPipeline(runner='BundleBasedDirectRunner') as p:
            _ = p | beam.Create([[]]).with_output_types(beam.typehints.List[int]) | beam.combiners.Count.Globally()

    def test_impulse(self):
        if False:
            print('Hello World!')
        with test_pipeline.TestPipeline(runner='BundleBasedDirectRunner') as p:
            assert_that(p | beam.Impulse(), equal_to([b'']))

class DirectRunnerRetryTests(unittest.TestCase):

    def test_retry_fork_graph(self):
        if False:
            while True:
                i = 10
        p = beam.Pipeline(runner='BundleBasedDirectRunner')
        global count_b, count_c
        (count_b, count_c) = (0, 0)

        def f_b(x):
            if False:
                i = 10
                return i + 15
            global count_b
            count_b += 1
            raise Exception('exception in f_b')

        def f_c(x):
            if False:
                print('Hello World!')
            global count_c
            count_c += 1
            raise Exception('exception in f_c')
        names = p | 'CreateNodeA' >> beam.Create(['Ann', 'Joe'])
        fork_b = names | 'SendToB' >> beam.Map(f_b)
        fork_c = names | 'SendToC' >> beam.Map(f_c)
        with self.assertRaises(Exception):
            p.run().wait_until_finish()
        assert count_b == count_c == 4

    def test_no_partial_writeouts(self):
        if False:
            while True:
                i = 10

        class TestTransformEvaluator(_TransformEvaluator):

            def __init__(self):
                if False:
                    print('Hello World!')
                self._execution_context = _ExecutionContext(None, {})

            def start_bundle(self):
                if False:
                    while True:
                        i = 10
                self.step_context = self._execution_context.get_step_context()

            def process_element(self, element):
                if False:
                    while True:
                        i = 10
                (k, v) = element
                state = self.step_context.get_keyed_state(k)
                state.add_state(None, _GroupByKeyOnlyEvaluator.ELEMENTS_TAG, v)
        evaluator = TestTransformEvaluator()
        evaluator.start_bundle()
        self.assertIsNone(evaluator.step_context.existing_keyed_state.get('key'))
        self.assertIsNone(evaluator.step_context.partial_keyed_state.get('key'))
        evaluator.process_element(['key', 'value'])
        self.assertEqual(evaluator.step_context.existing_keyed_state['key'].state, defaultdict(lambda : defaultdict(list)))
        self.assertEqual(evaluator.step_context.partial_keyed_state['key'].state, {None: {'elements': ['value']}})
        evaluator.process_element(['key', 'value2'])
        self.assertEqual(evaluator.step_context.existing_keyed_state['key'].state, defaultdict(lambda : defaultdict(list)))
        self.assertEqual(evaluator.step_context.partial_keyed_state['key'].state, {None: {'elements': ['value', 'value2']}})
        evaluator._execution_context.reset()
        evaluator.start_bundle()
        evaluator.process_element(['key', 'value'])
        self.assertEqual(evaluator.step_context.existing_keyed_state['key'].state, defaultdict(lambda : defaultdict(list)))
        self.assertEqual(evaluator.step_context.partial_keyed_state['key'].state, {None: {'elements': ['value']}})
if __name__ == '__main__':
    unittest.main()