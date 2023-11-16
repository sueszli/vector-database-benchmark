"""A word-counting workflow."""
import time
from hamcrest.library.number.ordering_comparison import greater_than
import apache_beam as beam
from apache_beam.metrics import Metrics
from apache_beam.testing.metric_result_matchers import DistributionMatcher
from apache_beam.testing.metric_result_matchers import MetricResultMatcher
SLEEP_TIME_SECS = 1
INPUT = [0, 0, 0, 100]
METRIC_NAMESPACE = 'apache_beam.runners.dataflow.dataflow_exercise_metrics_pipeline.UserMetricsDoFn'

def metric_matchers():
    if False:
        while True:
            i = 10
    'MetricResult matchers common to all tests.'
    matchers = [MetricResultMatcher(name='total_values', namespace=METRIC_NAMESPACE, step='metrics', attempted=sum(INPUT), committed=sum(INPUT)), MetricResultMatcher(name='ExecutionTime_StartBundle', step='metrics', attempted=greater_than(0), committed=greater_than(0)), MetricResultMatcher(name='ExecutionTime_ProcessElement', step='metrics', attempted=greater_than(0), committed=greater_than(0)), MetricResultMatcher(name='ExecutionTime_FinishBundle', step='metrics', attempted=greater_than(0), committed=greater_than(0)), MetricResultMatcher(name='distribution_values', namespace=METRIC_NAMESPACE, step='metrics', attempted=DistributionMatcher(sum_value=sum(INPUT), count_value=len(INPUT), min_value=min(INPUT), max_value=max(INPUT)), committed=DistributionMatcher(sum_value=sum(INPUT), count_value=len(INPUT), min_value=min(INPUT), max_value=max(INPUT))), MetricResultMatcher(name='ElementCount', labels={'output_user_name': 'metrics-out0', 'original_name': 'metrics-out0-ElementCount'}, attempted=greater_than(0), committed=greater_than(0)), MetricResultMatcher(name='MeanByteCount', labels={'output_user_name': 'metrics-out0', 'original_name': 'metrics-out0-MeanByteCount'}, attempted=greater_than(0), committed=greater_than(0))]
    pcoll_names = ['GroupByKey/Reify-out0', 'GroupByKey/Read-out0', 'map_to_common_key-out0', 'GroupByKey/GroupByWindow-out0', 'GroupByKey/Read-out0', 'GroupByKey/Reify-out0']
    for name in pcoll_names:
        matchers.extend([MetricResultMatcher(name='ElementCount', labels={'output_user_name': name, 'original_name': '%s-ElementCount' % name}, attempted=greater_than(0), committed=greater_than(0)), MetricResultMatcher(name='MeanByteCount', labels={'output_user_name': name, 'original_name': '%s-MeanByteCount' % name}, attempted=greater_than(0), committed=greater_than(0))])
    return matchers

class UserMetricsDoFn(beam.DoFn):
    """Parse each line of input text into words."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.total_metric = Metrics.counter(self.__class__, 'total_values')
        self.dist_metric = Metrics.distribution(self.__class__, 'distribution_values')
        self.latest_metric = Metrics.gauge(self.__class__, 'latest_value')

    def start_bundle(self):
        if False:
            i = 10
            return i + 15
        time.sleep(SLEEP_TIME_SECS)

    def process(self, element):
        if False:
            return 10
        'Returns the processed element and increments the metrics.'
        elem_int = int(element)
        self.total_metric.inc(elem_int)
        self.dist_metric.update(elem_int)
        self.latest_metric.set(elem_int)
        time.sleep(SLEEP_TIME_SECS)
        return [elem_int]

    def finish_bundle(self):
        if False:
            while True:
                i = 10
        time.sleep(SLEEP_TIME_SECS)

def apply_and_run(pipeline):
    if False:
        while True:
            i = 10
    'Given an initialized Pipeline applies transforms and runs it.'
    _ = pipeline | beam.Create(INPUT) | 'metrics' >> beam.ParDo(UserMetricsDoFn()) | 'map_to_common_key' >> beam.Map(lambda x: ('key', x)) | beam.GroupByKey() | 'm_out' >> beam.FlatMap(lambda x: [1, 2, 3, 4, 5, beam.pvalue.TaggedOutput('once', x), beam.pvalue.TaggedOutput('twice', x), beam.pvalue.TaggedOutput('twice', x)])
    result = pipeline.run()
    result.wait_until_finish()
    return result