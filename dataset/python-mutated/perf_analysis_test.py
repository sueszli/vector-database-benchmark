import logging
import os
import re
import unittest
import mock
import numpy as np
import pandas as pd
try:
    import apache_beam.testing.analyzers.perf_analysis as analysis
    from apache_beam.io.filesystems import FileSystems
    from apache_beam.testing.analyzers import constants
    from apache_beam.testing.analyzers import github_issues_utils
    from apache_beam.testing.analyzers.perf_analysis_utils import BigQueryMetricsFetcher
    from apache_beam.testing.analyzers.perf_analysis_utils import MetricContainer
    from apache_beam.testing.analyzers.perf_analysis_utils import TestConfigContainer
    from apache_beam.testing.analyzers.perf_analysis_utils import is_change_point_in_valid_window
    from apache_beam.testing.analyzers.perf_analysis_utils import is_edge_change_point
    from apache_beam.testing.analyzers.perf_analysis_utils import is_sibling_change_point
    from apache_beam.testing.analyzers.perf_analysis_utils import e_divisive
    from apache_beam.testing.analyzers.perf_analysis_utils import filter_change_points_by_median_threshold
    from apache_beam.testing.analyzers.perf_analysis_utils import find_change_points
    from apache_beam.testing.analyzers.perf_analysis_utils import find_latest_change_point_index
    from apache_beam.testing.analyzers.perf_analysis_utils import validate_config
    from apache_beam.testing.load_tests import load_test_metrics_utils
except ImportError as e:
    raise unittest.SkipTest('Missing dependencies to run perf analysis tests.')

def get_fake_data_with_no_change_point(*args, **kwargs):
    if False:
        while True:
            i = 10
    num_samples = 20
    metric_values = [1] * num_samples
    timestamps = [pd.Timestamp(i) for i in range(num_samples)]
    return MetricContainer(metric_values, timestamps)

def get_fake_data_with_change_point(*args, **kwargs):
    if False:
        while True:
            i = 10
    num_samples = 20
    metric_values = [0] * 12 + [3] + [4] * 7
    timestamps = [pd.Timestamp(i) for i in range(num_samples)]
    return MetricContainer(metric_values, timestamps)

def get_existing_issue_data(**kwargs):
    if False:
        print('Hello World!')
    return pd.DataFrame([{constants._CHANGE_POINT_TIMESTAMP_LABEL: pd.Timestamp(13), constants._ISSUE_NUMBER: np.array([0])}])

class TestChangePointAnalysis(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.single_change_point_series = [0] * 10 + [1] * 10
        self.multiple_change_point_series = self.single_change_point_series + [2] * 20
        self.timestamps = [pd.Timestamp(i) for i in range(5)]
        self.params = {'test_description': 'fake_description', 'metrics_dataset': 'fake_dataset', 'metrics_table': 'fake_table', 'project': 'fake_project', 'metric_name': 'fake_metric_name'}
        self.test_id = 'fake_id'

    def test_edivisive_means(self):
        if False:
            i = 10
            return i + 15
        change_point_indexes = e_divisive(self.single_change_point_series)
        self.assertEqual(change_point_indexes, [10])
        change_point_indexes = e_divisive(self.multiple_change_point_series)
        self.assertEqual(sorted(change_point_indexes), [10, 20])

    def test_is_changepoint_in_valid_window(self):
        if False:
            for i in range(10):
                print('nop')
        changepoint_to_recent_run_window = 19
        change_point_index = 14
        is_valid = is_change_point_in_valid_window(changepoint_to_recent_run_window, change_point_index)
        self.assertEqual(is_valid, True)

    def test_change_point_outside_inspection_window_is_not_a_valid_alert(self):
        if False:
            for i in range(10):
                print('nop')
        changepoint_to_recent_run_window = 12
        change_point_index = 14
        is_valid = is_change_point_in_valid_window(changepoint_to_recent_run_window, change_point_index)
        self.assertEqual(is_valid, False)

    def test_validate_config(self):
        if False:
            i = 10
            return i + 15
        test_keys = {'test_description', 'metrics_dataset', 'metrics_table', 'project', 'metric_name'}
        self.assertEqual(test_keys, constants._PERF_TEST_KEYS)
        self.assertTrue(validate_config(test_keys))

    def test_duplicate_change_point(self):
        if False:
            i = 10
            return i + 15
        change_point_index = 2
        min_runs_between_change_points = 1
        is_alert = is_sibling_change_point(previous_change_point_timestamps=[self.timestamps[0]], timestamps=self.timestamps, change_point_index=change_point_index, min_runs_between_change_points=min_runs_between_change_points, test_id=self.test_id)
        self.assertTrue(is_alert)

    def test_duplicate_change_points_are_not_valid_alerts(self):
        if False:
            i = 10
            return i + 15
        change_point_index = 2
        min_runs_between_change_points = 1
        is_alert = is_sibling_change_point(previous_change_point_timestamps=[self.timestamps[3]], timestamps=self.timestamps, change_point_index=change_point_index, min_runs_between_change_points=min_runs_between_change_points, test_id=self.test_id)
        self.assertFalse(is_alert)
        is_alert = is_sibling_change_point(previous_change_point_timestamps=[self.timestamps[0], self.timestamps[3]], timestamps=self.timestamps, change_point_index=change_point_index, min_runs_between_change_points=min_runs_between_change_points, test_id=self.test_id)
        self.assertFalse(is_alert)

    @mock.patch.object(BigQueryMetricsFetcher, 'fetch_metric_data', get_fake_data_with_no_change_point)
    def test_no_alerts_when_no_change_points(self):
        if False:
            print('Hello World!')
        test_config_container = analysis.get_test_config_container(params=self.params, test_id=self.test_id, metric_name=self.params['metric_name'])
        is_alert = analysis.run_change_point_analysis(test_config_container=test_config_container, big_query_metrics_fetcher=BigQueryMetricsFetcher())
        self.assertFalse(is_alert)

    @mock.patch.object(BigQueryMetricsFetcher, 'fetch_metric_data', get_fake_data_with_change_point)
    @mock.patch('apache_beam.testing.analyzers.perf_analysis.get_existing_issues_data', return_value=None)
    @mock.patch('apache_beam.testing.analyzers.perf_analysis.publish_issue_metadata_to_big_query', return_value=None)
    @mock.patch('apache_beam.testing.analyzers.perf_analysis.create_performance_alert', return_value=(0, ''))
    def test_alert_on_data_with_change_point(self, *args):
        if False:
            i = 10
            return i + 15
        test_config_container = analysis.get_test_config_container(params=self.params, test_id=self.test_id, metric_name=self.params['metric_name'])
        is_alert = analysis.run_change_point_analysis(test_config_container=test_config_container, big_query_metrics_fetcher=BigQueryMetricsFetcher())
        self.assertTrue(is_alert)

    @mock.patch.object(BigQueryMetricsFetcher, 'fetch_metric_data', get_fake_data_with_change_point)
    @mock.patch('apache_beam.testing.analyzers.perf_analysis.get_existing_issues_data', get_existing_issue_data)
    @mock.patch('apache_beam.testing.analyzers.perf_analysis.publish_issue_metadata_to_big_query', return_value=None)
    @mock.patch('apache_beam.testing.analyzers.perf_analysis.create_performance_alert', return_value=(0, ''))
    def test_alert_on_data_with_reported_change_point(self, *args):
        if False:
            i = 10
            return i + 15
        test_config_container = analysis.get_test_config_container(params=self.params, test_id=self.test_id, metric_name=self.params['metric_name'])
        is_alert = analysis.run_change_point_analysis(test_config_container=test_config_container, big_query_metrics_fetcher=BigQueryMetricsFetcher())
        self.assertFalse(is_alert)

    def test_change_point_has_anomaly_marker_in_gh_description(self):
        if False:
            return 10
        metric_container = get_fake_data_with_change_point()
        metric_values = metric_container.values
        change_point_index = find_latest_change_point_index(metric_values)
        test_config_container = TestConfigContainer(project=self.params['project'], metrics_dataset=self.params['metrics_dataset'], metrics_table=self.params['metrics_table'], metric_name=self.params['metric_name'], test_id=self.test_id, test_description=self.params['test_description'], test_name=self.params.get('test_name', None), labels=self.params.get('labels', None))
        description = github_issues_utils.get_issue_description(test_config_container=test_config_container, metric_container=metric_container, change_point_index=change_point_index, max_results_to_display=constants._NUM_RESULTS_TO_DISPLAY_ON_ISSUE_DESCRIPTION)
        runs_info = next((line for line in description.split(2 * os.linesep) if re.match('timestamp: .*, metric_value: .*', line.strip())), '')
        pattern = 'timestamp: .+ (\\d{4}), metric_value: (\\d+.\\d+) <---- Anomaly'
        match = re.search(pattern, runs_info)
        self.assertTrue(match)

    def test_change_point_on_noisy_data(self):
        if False:
            return 10

        def read_csv(path):
            if False:
                return 10
            with FileSystems.open(path) as fp:
                return pd.read_csv(fp)
        metric_data = read_csv('gs://apache-beam-ml/testing/inputs/test_data_with_noise.csv')
        metric_values = metric_data[load_test_metrics_utils.VALUE_LABEL].tolist()
        change_points = find_change_points(metric_values)
        self.assertEqual(change_points[0], 20)
        valid_points = filter_change_points_by_median_threshold(metric_values, change_points)
        self.assertEqual(len(valid_points), 0)

    def test_change_point_on_edge_segment(self):
        if False:
            i = 10
            return i + 15
        data = [1] * 50 + [100]
        change_points = find_change_points(data)
        self.assertEqual(change_points, [50])
        self.assertEqual(is_edge_change_point(change_points[0], len(data)), True)
        self.assertEqual(find_latest_change_point_index(data), None)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    os.environ['GITHUB_TOKEN'] = 'fake_token'
    unittest.main()