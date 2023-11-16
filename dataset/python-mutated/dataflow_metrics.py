"""
DataflowRunner implementation of MetricResults. It is in charge of
responding to queries of current metrics by going to the dataflow
service.
"""
import argparse
import logging
import numbers
import sys
from collections import defaultdict
from apache_beam.metrics.cells import DistributionData
from apache_beam.metrics.cells import DistributionResult
from apache_beam.metrics.execution import MetricKey
from apache_beam.metrics.execution import MetricResult
from apache_beam.metrics.metric import MetricResults
from apache_beam.metrics.metricbase import MetricName
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
_LOGGER = logging.getLogger(__name__)

def _get_match(proto, filter_fn):
    if False:
        return 10
    'Finds and returns the first element that matches a query.\n\n  If no element matches the query, it throws ValueError.\n  If more than one element matches the query, it returns only the first.\n  '
    query = [elm for elm in proto if filter_fn(elm)]
    if len(query) == 0:
        raise ValueError('Could not find element')
    elif len(query) > 1:
        raise ValueError('Too many matches')
    return query[0]
STEP_LABEL = 'step'
STRUCTURED_NAME_LABELS = set(['execution_step', 'original_name', 'output_user_name'])

class DataflowMetrics(MetricResults):
    """Implementation of MetricResults class for the Dataflow runner."""

    def __init__(self, dataflow_client=None, job_result=None, job_graph=None):
        if False:
            while True:
                i = 10
        'Initialize the Dataflow metrics object.\n\n    Args:\n      dataflow_client: apiclient.DataflowApplicationClient to interact with the\n        dataflow service.\n      job_result: DataflowPipelineResult with the state and id information of\n        the job.\n      job_graph: apiclient.Job instance to be able to translate between internal\n        step names (e.g. "s2"), and user step names (e.g. "split").\n    '
        super().__init__()
        self._dataflow_client = dataflow_client
        self.job_result = job_result
        self._queried_after_termination = False
        self._cached_metrics = None
        self._job_graph = job_graph

    @staticmethod
    def _is_counter(metric_result):
        if False:
            i = 10
            return i + 15
        return isinstance(metric_result.attempted, numbers.Number)

    @staticmethod
    def _is_distribution(metric_result):
        if False:
            return 10
        return isinstance(metric_result.attempted, DistributionResult)

    def _translate_step_name(self, internal_name):
        if False:
            for i in range(10):
                print('nop')
        'Translate between internal step names (e.g. "s1") and user step names.'
        if not self._job_graph:
            raise ValueError('Could not translate the internal step name %r since job graph is not available.' % internal_name)
        user_step_name = None
        if self._job_graph and internal_name in self._job_graph.proto_pipeline.components.transforms.keys():
            user_step_name = self._job_graph.proto_pipeline.components.transforms[internal_name].unique_name
        else:
            try:
                step = _get_match(self._job_graph.proto.steps, lambda x: x.name == internal_name)
                user_step_name = _get_match(step.properties.additionalProperties, lambda x: x.key == 'user_name').value.string_value
            except ValueError:
                pass
        if not user_step_name:
            raise ValueError('Could not translate the internal step name %r.' % internal_name)
        return user_step_name

    def _get_metric_key(self, metric):
        if False:
            for i in range(10):
                print('nop')
        'Populate the MetricKey object for a queried metric result.'
        step = ''
        name = metric.name.name
        labels = {}
        try:
            step = _get_match(metric.name.context.additionalProperties, lambda x: x.key == STEP_LABEL).value
            step = self._translate_step_name(step)
        except ValueError:
            pass
        namespace = 'dataflow/v1b3'
        try:
            namespace = _get_match(metric.name.context.additionalProperties, lambda x: x.key == 'namespace').value
        except ValueError:
            pass
        for kv in metric.name.context.additionalProperties:
            if kv.key in STRUCTURED_NAME_LABELS:
                labels[kv.key] = kv.value
        return MetricKey(step, MetricName(namespace, name), labels=labels)

    def _populate_metrics(self, response, result, user_metrics=False):
        if False:
            i = 10
            return i + 15
        'Move metrics from response to results as MetricResults.'
        if user_metrics:
            metrics = [metric for metric in response.metrics if metric.name.origin == 'user']
        else:
            metrics = [metric for metric in response.metrics if metric.name.origin == 'dataflow/v1b3']
        metrics_by_name = defaultdict(lambda : {})
        for metric in metrics:
            if metric.name.name.endswith('_MIN') or metric.name.name.endswith('_MAX') or metric.name.name.endswith('_MEAN') or metric.name.name.endswith('_COUNT'):
                continue
            is_tentative = [prop for prop in metric.name.context.additionalProperties if prop.key == 'tentative' and prop.value == 'true']
            tentative_or_committed = 'tentative' if is_tentative else 'committed'
            metric_key = self._get_metric_key(metric)
            if metric_key is None:
                continue
            metrics_by_name[metric_key][tentative_or_committed] = metric
        for (metric_key, metric) in metrics_by_name.items():
            attempted = self._get_metric_value(metric['tentative'])
            committed = self._get_metric_value(metric['committed'])
            result.append(MetricResult(metric_key, attempted=attempted, committed=committed))

    def _get_metric_value(self, metric):
        if False:
            while True:
                i = 10
        'Get a metric result object from a MetricUpdate from Dataflow API.'
        if metric is None:
            return None
        if metric.scalar is not None:
            return metric.scalar.integer_value
        elif metric.distribution is not None:
            dist_count = _get_match(metric.distribution.object_value.properties, lambda x: x.key == 'count').value.integer_value
            dist_min = _get_match(metric.distribution.object_value.properties, lambda x: x.key == 'min').value.integer_value
            dist_max = _get_match(metric.distribution.object_value.properties, lambda x: x.key == 'max').value.integer_value
            dist_sum = _get_match(metric.distribution.object_value.properties, lambda x: x.key == 'sum').value.integer_value
            if dist_sum is None:
                _LOGGER.info('Distribution metric sum value seems to have overflowed integer_value range, the correctness of sum or mean value may not be guaranteed: %s' % metric.distribution)
                dist_sum = int(_get_match(metric.distribution.object_value.properties, lambda x: x.key == 'sum').value.double_value)
            return DistributionResult(DistributionData(dist_sum, dist_count, dist_min, dist_max))
        else:
            return None

    def _get_metrics_from_dataflow(self, job_id=None):
        if False:
            i = 10
            return i + 15
        'Return cached metrics or query the dataflow service.'
        if not job_id:
            try:
                job_id = self.job_result.job_id()
            except AttributeError:
                job_id = None
        if not job_id:
            raise ValueError('Can not query metrics. Job id is unknown.')
        if self._cached_metrics:
            return self._cached_metrics
        job_metrics = self._dataflow_client.get_job_metrics(job_id)
        if self.job_result and self.job_result.is_in_terminal_state():
            self._cached_metrics = job_metrics
        return job_metrics

    def all_metrics(self, job_id=None):
        if False:
            return 10
        'Return all user and system metrics from the dataflow service.'
        metric_results = []
        response = self._get_metrics_from_dataflow(job_id=job_id)
        self._populate_metrics(response, metric_results, user_metrics=True)
        self._populate_metrics(response, metric_results, user_metrics=False)
        return metric_results

    def query(self, filter=None):
        if False:
            for i in range(10):
                print('nop')
        metric_results = []
        response = self._get_metrics_from_dataflow()
        self._populate_metrics(response, metric_results, user_metrics=True)
        return {self.COUNTERS: [elm for elm in metric_results if self.matches(filter, elm.key) and DataflowMetrics._is_counter(elm)], self.DISTRIBUTIONS: [elm for elm in metric_results if self.matches(filter, elm.key) and DataflowMetrics._is_distribution(elm)], self.GAUGES: []}

def main(argv):
    if False:
        print('Hello World!')
    'Print the metric results for the dataflow --job_id and --project.\n\n  Instead of running an entire pipeline which takes several minutes, use this\n  main method to display MetricResults for a specific --job_id and --project\n  which takes only a few seconds.\n  '
    try:
        from apache_beam.runners.dataflow.internal import apiclient
    except ImportError:
        raise ImportError('Google Cloud Dataflow runner not available, please install apache_beam[gcp]')
    if argv[0] == __file__:
        argv = argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id', type=str, help='The job id to query metrics for.')
    parser.add_argument('-p', '--project', type=str, help='The project name to query metrics for.')
    flags = parser.parse_args(argv)
    options = PipelineOptions()
    gcloud_options = options.view_as(GoogleCloudOptions)
    gcloud_options.project = flags.project
    dataflow_client = apiclient.DataflowApplicationClient(options)
    df_metrics = DataflowMetrics(dataflow_client)
    all_metrics = df_metrics.all_metrics(job_id=flags.job_id)
    _LOGGER.info('Printing all MetricResults for %s in %s', flags.job_id, flags.project)
    for metric_result in all_metrics:
        _LOGGER.info(metric_result)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main(sys.argv)