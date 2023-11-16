"""ValidatesRunner tests for CombineFn lifecycle and bundle methods."""
import unittest
import pytest
from parameterized import parameterized_class
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.runners.direct import direct_runner
from apache_beam.runners.portability import fn_api_runner
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.transforms.combinefn_lifecycle_pipeline import CallSequenceEnforcingCombineFn
from apache_beam.transforms.combinefn_lifecycle_pipeline import run_combine
from apache_beam.transforms.combinefn_lifecycle_pipeline import run_pardo

@pytest.mark.it_validatesrunner
class CombineFnLifecycleTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.pipeline = TestPipeline(is_integration_test=True)

    def test_combine(self):
        if False:
            return 10
        run_combine(self.pipeline)

    def test_non_liftable_combine(self):
        if False:
            i = 10
            return i + 15
        run_combine(self.pipeline, lift_combiners=False)

    def test_combining_value_state(self):
        if False:
            print('Hello World!')
        if 'DataflowRunner' in self.pipeline.get_pipeline_options().view_as(StandardOptions).runner:
            self.skipTest('https://github.com/apache/beam/issues/20722')
        run_pardo(self.pipeline)

@parameterized_class([{'runner': direct_runner.BundleBasedDirectRunner}, {'runner': fn_api_runner.FnApiRunner}])
class LocalCombineFnLifecycleTest(unittest.TestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        CallSequenceEnforcingCombineFn.instances.clear()

    def test_combine(self):
        if False:
            return 10
        run_combine(TestPipeline(runner=self.runner()))
        self._assert_teardown_called()

    def test_non_liftable_combine(self):
        if False:
            print('Hello World!')
        test_options = PipelineOptions(flags=['--allow_unsafe_triggers'])
        run_combine(TestPipeline(runner=self.runner(), options=test_options), lift_combiners=False)
        self._assert_teardown_called()

    def test_combining_value_state(self):
        if False:
            for i in range(10):
                print('nop')
        run_pardo(TestPipeline(runner=self.runner()))
        self._assert_teardown_called()

    def _assert_teardown_called(self):
        if False:
            while True:
                i = 10
        'Ensures that teardown has been invoked for all CombineFns.'
        for instance in CallSequenceEnforcingCombineFn.instances:
            self.assertTrue(instance._teardown_called)
if __name__ == '__main__':
    unittest.main()