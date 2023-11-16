"""
This is a load test that runs a set of basic microbenchmarks for the Python SDK
and the DirectRunner.

This test does not need any additional options passed to run, besides the
dataset information.

Example test run:

python -m apache_beam.testing.load_tests.microbenchmarks_test     --test-pipeline-options="
    --project=big-query-project
    --input_options='{}'
    --region=...
    --publish_to_big_query=true
    --metrics_dataset=python_load_tests
    --metrics_table=microbenchmarks"

or:

./gradlew -PloadTest.args="
    --publish_to_big_query=true
    --project=...
    --region=...
    --input_options='{}'
    --metrics_dataset=python_load_tests
    --metrics_table=microbenchmarks
    --runner=DirectRunner" -PloadTest.mainClass=apache_beam.testing.load_tests.microbenchmarks_test -Prunner=DirectRunner :sdks:python:apache_beam:testing:load_tests:run
"""
import logging
import time
from apache_beam.testing.load_tests.load_test import LoadTest
from apache_beam.tools import fn_api_runner_microbenchmark
from apache_beam.tools import teststream_microbenchmark
from apache_beam.transforms.util import _BatchSizeEstimator

class MicroBenchmarksLoadTest(LoadTest):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.extra_metrics.update(self._run_fn_api_runner_microbenchmark())
        self.extra_metrics.update(self._run_teststream_microbenchmark())

    def _run_teststream_microbenchmark(self):
        if False:
            i = 10
            return i + 15
        start = time.perf_counter()
        result = teststream_microbenchmark.run_benchmark(verbose=False)
        sizes = list(result[0].values())[0]
        costs = list(result[1].values())[0]
        (a, b) = _BatchSizeEstimator.linear_regression_no_numpy(sizes, costs)
        return {'teststream_microbenchmark_runtime_sec': time.perf_counter() - start, 'teststream_microbenchmark_fixed_cost_ms': a * 1000, 'teststream_microbenchmark_per_element_cost_ms': b * 1000}

    def _run_fn_api_runner_microbenchmark(self):
        if False:
            for i in range(10):
                print('nop')
        start = time.perf_counter()
        result = fn_api_runner_microbenchmark.run_benchmark(verbose=False)
        sizes = list(result[0].values())[0]
        costs = list(result[1].values())[0]
        (a, b) = _BatchSizeEstimator.linear_regression_no_numpy(sizes, costs)
        return {'fn_api_runner_microbenchmark_runtime_sec': time.perf_counter() - start, 'fn_api_runner_microbenchmark_fixed_cost_ms': a * 1000, 'fn_api_runner_microbenchmark_per_element_cost_ms': b * 1000}
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    MicroBenchmarksLoadTest().run()