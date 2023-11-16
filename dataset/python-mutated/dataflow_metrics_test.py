"""
Tests corresponding to the DataflowRunner implementation of MetricsResult,
the DataflowMetrics class.
"""
import types
import unittest
import mock
from apache_beam import DoFn
from apache_beam import ParDo
from apache_beam.metrics.cells import DistributionData
from apache_beam.metrics.cells import DistributionResult
from apache_beam.metrics.execution import MetricKey
from apache_beam.metrics.execution import MetricResult
from apache_beam.metrics.metricbase import MetricName
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.pipeline import Pipeline
from apache_beam.runners.dataflow import dataflow_metrics
from apache_beam.testing import metric_result_matchers
from apache_beam.testing.metric_result_matchers import MetricResultMatcher
from apache_beam.transforms import Create
from apache_beam.transforms.environments import DockerEnvironment
try:
    from apache_beam.runners.dataflow.internal import apiclient
except ImportError:
    apiclient = None

class DictToObject(object):
    """Translate from a dict(list()) structure to an object structure"""

    def __init__(self, data):
        if False:
            i = 10
            return i + 15
        for (name, value) in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if False:
            i = 10
            return i + 15
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        return DictToObject(value) if isinstance(value, dict) else value

class TestDataflowMetrics(unittest.TestCase):
    ONLY_COUNTERS_LIST = {'metrics': [{'name': {'context': {'additionalProperties': [{'key': 'namespace', 'value': '__main__.WordExtractingDoFn'}, {'key': 'step', 'value': 's2'}, {'key': 'tentative', 'value': 'true'}]}, 'name': 'words', 'origin': 'user'}, 'scalar': {'integer_value': 26185}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'namespace', 'value': '__main__.WordExtractingDoFn'}, {'key': 'step', 'value': 's2'}]}, 'name': 'words', 'origin': 'user'}, 'scalar': {'integer_value': 26181}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'namespace', 'value': '__main__.WordExtractingDoFn'}, {'key': 'step', 'value': 's2'}, {'key': 'tentative', 'value': 'true'}]}, 'name': 'empty_lines', 'origin': 'user'}, 'scalar': {'integer_value': 1080}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'namespace', 'value': '__main__.WordExtractingDoFn'}, {'key': 'step', 'value': 's2'}]}, 'name': 'empty_lines', 'origin': 'user'}, 'scalar': {'integer_value': 1080}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}]}
    STRUCTURED_COUNTER_LIST = {'metrics': [{'name': {'context': {'additionalProperties': [{'key': 'namespace', 'value': '__main__.WordExtractingDoFn'}, {'key': 'step', 'value': 's2'}, {'key': 'tentative', 'value': 'true'}]}, 'name': 'word_lengths', 'origin': 'user'}, 'scalar': {'integer_value': 109475}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'namespace', 'value': '__main__.WordExtractingDoFn'}, {'key': 'step', 'value': 's2'}]}, 'name': 'word_lengths', 'origin': 'user'}, 'scalar': {'integer_value': 109475}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'namespace', 'value': '__main__.WordExtractingDoFn'}, {'key': 'step', 'value': 's2'}, {'key': 'tentative', 'value': 'true'}]}, 'name': 'word_length_dist', 'origin': 'user'}, 'scalar': None, 'distribution': {'object_value': {'properties': [{'key': 'min', 'value': {'integer_value': 2}}, {'key': 'max', 'value': {'integer_value': 16}}, {'key': 'count', 'value': {'integer_value': 2}}, {'key': 'mean', 'value': {'integer_value': 9}}, {'key': 'sum', 'value': {'integer_value': 18}}]}}, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'namespace', 'value': '__main__.WordExtractingDoFn'}, {'key': 'step', 'value': 's2'}]}, 'name': 'word_length_dist', 'origin': 'user'}, 'scalar': None, 'distribution': {'object_value': {'properties': [{'key': 'min', 'value': {'integer_value': 2}}, {'key': 'max', 'value': {'integer_value': 16}}, {'key': 'count', 'value': {'integer_value': 2}}, {'key': 'mean', 'value': {'integer_value': 9}}, {'key': 'sum', 'value': {'integer_value': 18}}]}}, 'updateTime': '2017-03-22T18:47:06.402Z'}]}
    SYSTEM_COUNTERS_LIST = {'metrics': [{'name': {'context': {'additionalProperties': [{'key': 'original_name', 'value': 'ToIsmRecordForMultimap-out0-ElementCount'}, {'key': 'output_user_name', 'value': 'ToIsmRecordForMultimap-out0'}]}, 'name': 'ElementCount', 'origin': 'dataflow/v1b3'}, 'scalar': {'integer_value': 42}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'original_name', 'value': 'ToIsmRecordForMultimap-out0-ElementCount'}, {'key': 'output_user_name', 'value': 'ToIsmRecordForMultimap-out0'}, {'key': 'tentative', 'value': 'true'}]}, 'name': 'ElementCount', 'origin': 'dataflow/v1b3'}, 'scalar': {'integer_value': 42}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'original_name', 'value': 'Read-out0-MeanByteCount'}, {'key': 'output_user_name', 'value': 'GroupByKey/Read-out0'}]}, 'name': 'MeanByteCount', 'origin': 'dataflow/v1b3'}, 'scalar': {'integer_value': 31}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'original_name', 'value': 'Read-out0-MeanByteCount'}, {'key': 'output_user_name', 'value': 'GroupByKey/Read-out0'}, {'key': 'tentative', 'value': 'true'}]}, 'name': 'MeanByteCount', 'origin': 'dataflow/v1b3'}, 'scalar': {'integer_value': 31}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'step', 'value': 'write/Write/Write'}]}, 'name': 'ExecutionTime_ProcessElement', 'origin': 'dataflow/v1b3'}, 'scalar': {'integer_value': 1000}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}, {'name': {'context': {'additionalProperties': [{'key': 'step', 'value': 'write/Write/Write'}, {'key': 'tentative', 'value': 'true'}]}, 'name': 'ExecutionTime_ProcessElement', 'origin': 'dataflow/v1b3'}, 'scalar': {'integer_value': 1000}, 'distribution': None, 'updateTime': '2017-03-22T18:47:06.402Z'}]}

    def setup_mock_client_result(self, counter_list=None):
        if False:
            while True:
                i = 10
        mock_client = mock.Mock()
        mock_query_result = DictToObject(counter_list)
        mock_client.get_job_metrics.return_value = mock_query_result
        mock_job_result = mock.Mock()
        mock_job_result.job_id.return_value = 1
        mock_job_result.is_in_terminal_state.return_value = False
        return (mock_client, mock_job_result)

    def test_cache_functions(self):
        if False:
            i = 10
            return i + 15
        (mock_client, mock_job_result) = self.setup_mock_client_result(self.STRUCTURED_COUNTER_LIST)
        dm = dataflow_metrics.DataflowMetrics(mock_client, mock_job_result)
        self.assertTrue(dm._cached_metrics is None)
        dm.query()
        self.assertTrue(dm._cached_metrics is None)
        mock_job_result.is_in_terminal_state.return_value = True
        dm.query()
        self.assertTrue(dm._cached_metrics)

    def test_query_structured_metrics(self):
        if False:
            for i in range(10):
                print('nop')
        (mock_client, mock_job_result) = self.setup_mock_client_result(self.STRUCTURED_COUNTER_LIST)
        dm = dataflow_metrics.DataflowMetrics(mock_client, mock_job_result)
        dm._translate_step_name = types.MethodType(lambda self, x: 'split', dm)
        query_result = dm.query()
        expected_counters = [MetricResult(MetricKey('split', MetricName('__main__.WordExtractingDoFn', 'word_lengths')), 109475, 109475)]
        self.assertEqual(query_result['counters'], expected_counters)
        expected_distributions = [MetricResult(MetricKey('split', MetricName('__main__.WordExtractingDoFn', 'word_length_dist')), DistributionResult(DistributionData(18, 2, 2, 16)), DistributionResult(DistributionData(18, 2, 2, 16)))]
        self.assertEqual(query_result['distributions'], expected_distributions)

    @unittest.skipIf(apiclient is None, 'GCP dependencies are not installed')
    def test_translate_portable_job_step_name(self):
        if False:
            for i in range(10):
                print('nop')
        (mock_client, mock_job_result) = self.setup_mock_client_result(self.ONLY_COUNTERS_LIST)
        pipeline_options = PipelineOptions(['--experiments=use_portable_job_submission', '--temp_location=gs://any-location/temp', '--project=dummy_project'])
        pipeline = Pipeline(options=pipeline_options)
        pipeline | Create([1, 2, 3]) | 'MyTestParDo' >> ParDo(DoFn())
        test_environment = DockerEnvironment(container_image='test_default_image')
        (proto_pipeline, _) = pipeline.to_runner_api(return_context=True, default_environment=test_environment)
        job = apiclient.Job(pipeline_options, proto_pipeline)
        dm = dataflow_metrics.DataflowMetrics(mock_client, mock_job_result, job)
        self.assertEqual('MyTestParDo', dm._translate_step_name('ref_AppliedPTransform_MyTestParDo_14'))

    def test_query_counters(self):
        if False:
            while True:
                i = 10
        (mock_client, mock_job_result) = self.setup_mock_client_result(self.ONLY_COUNTERS_LIST)
        dm = dataflow_metrics.DataflowMetrics(mock_client, mock_job_result)
        dm._translate_step_name = types.MethodType(lambda self, x: 'split', dm)
        query_result = dm.query()
        expected_counters = [MetricResult(MetricKey('split', MetricName('__main__.WordExtractingDoFn', 'empty_lines')), 1080, 1080), MetricResult(MetricKey('split', MetricName('__main__.WordExtractingDoFn', 'words')), 26181, 26185)]
        self.assertEqual(sorted(query_result['counters'], key=lambda x: x.key.metric.name), sorted(expected_counters, key=lambda x: x.key.metric.name))

    def test_system_counters_set_labels_and_step_name(self):
        if False:
            print('Hello World!')
        (mock_client, mock_job_result) = self.setup_mock_client_result(self.SYSTEM_COUNTERS_LIST)
        test_object = dataflow_metrics.DataflowMetrics(mock_client, mock_job_result)
        all_metrics = test_object.all_metrics()
        matchers = [MetricResultMatcher(name='ElementCount', labels={'original_name': 'ToIsmRecordForMultimap-out0-ElementCount', 'output_user_name': 'ToIsmRecordForMultimap-out0'}, attempted=42, committed=42), MetricResultMatcher(name='MeanByteCount', labels={'original_name': 'Read-out0-MeanByteCount', 'output_user_name': 'GroupByKey/Read-out0'}, attempted=31, committed=31), MetricResultMatcher(name='ExecutionTime_ProcessElement', step='write/Write/Write', attempted=1000, committed=1000)]
        errors = metric_result_matchers.verify_all(all_metrics, matchers)
        self.assertFalse(errors, errors)
if __name__ == '__main__':
    unittest.main()