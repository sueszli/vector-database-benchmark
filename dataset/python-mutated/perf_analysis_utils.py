import abc
import logging
from dataclasses import asdict
from dataclasses import dataclass
from statistics import median
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import pandas as pd
import yaml
from google.api_core import exceptions
from apache_beam.testing.analyzers import constants
from apache_beam.testing.load_tests import load_test_metrics_utils
from apache_beam.testing.load_tests.load_test_metrics_utils import BigQueryMetricsPublisher
from signal_processing_algorithms.energy_statistics.energy_statistics import e_divisive
try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None

@dataclass(frozen=True)
class GitHubIssueMetaData:
    """
  This class holds metadata that needs to be published to the
  BigQuery when a GitHub issue is created on a performance
  alert.
  """
    issue_timestamp: pd.Timestamp
    change_point_timestamp: pd.Timestamp
    test_name: str
    metric_name: str
    issue_number: int
    issue_url: str
    test_id: str
    change_point: float

@dataclass
class ChangePointConfig:
    """
  This class holds the change point configuration parameters.
  """
    min_runs_between_change_points: int = constants._DEFAULT_MIN_RUNS_BETWEEN_CHANGE_POINTS
    num_runs_in_change_point_window: int = constants._DEFAULT_NUM_RUMS_IN_CHANGE_POINT_WINDOW

@dataclass
class TestConfigContainer:
    metric_name: str
    project: str
    metrics_dataset: str
    metrics_table: str
    test_id: str
    test_description: str
    test_name: Optional[str] = None
    labels: Optional[List[str]] = None

@dataclass
class MetricContainer:
    """
  This class holds the metric values and timestamps for a given metric.
  Args:
    metric_values: List of metric values.
    timestamps: List of pandas timestamps corresponding to the metric values.
  """
    values: List[Union[int, float]]
    timestamps: List[pd.Timestamp]

    def sort_by_timestamp(self, in_place=True):
        if False:
            i = 10
            return i + 15
        '\n    Sorts the metric values and timestamps in ascending order wrt timestamps.\n    Args:\n      in_place: If True, sort the metric values and timestamps in place.\n    '
        (timestamps, values) = zip(*sorted(zip(self.timestamps, self.values)))
        if not in_place:
            return MetricContainer(values=values, timestamps=timestamps)
        (self.timestamps, self.values) = zip(*sorted(zip(self.timestamps, self.values)))

def is_change_point_in_valid_window(num_runs_in_change_point_window: int, latest_change_point_run: int) -> bool:
    if False:
        i = 10
        return i + 15
    return num_runs_in_change_point_window > latest_change_point_run

def get_existing_issues_data(table_name: str) -> Optional[pd.DataFrame]:
    if False:
        for i in range(10):
            print('nop')
    '\n  Finds the most recent GitHub issue created for the test_name.\n  If no table found with name=test_name, return (None, None)\n  else return latest created issue_number along with\n  '
    query = f'\n  SELECT * FROM {constants._BQ_PROJECT_NAME}.{constants._BQ_DATASET}.{table_name}\n  ORDER BY {constants._ISSUE_CREATION_TIMESTAMP_LABEL} DESC\n  LIMIT 10\n  '
    try:
        if bigquery is None:
            raise ImportError('Bigquery dependencies are not installed.')
        client = bigquery.Client()
        query_job = client.query(query=query)
        existing_issue_data = query_job.result().to_dataframe()
    except exceptions.NotFound:
        return None
    return existing_issue_data

def is_sibling_change_point(previous_change_point_timestamps: List[pd.Timestamp], change_point_index: int, timestamps: List[pd.Timestamp], min_runs_between_change_points: int, test_id: str) -> bool:
    if False:
        return 10
    '\n  Sibling change points are the change points that are close to each other.\n\n  Search the previous_change_point_timestamps with current observed\n  change point sibling window and determine if it is a duplicate\n  change point or not.\n  timestamps are expected to be in ascending order.\n\n  Return False if the current observed change point is a duplicate of\n  already reported change points else return True.\n  '
    sibling_change_point_min_timestamp = timestamps[max(0, change_point_index - min_runs_between_change_points)]
    sibling_change_point_max_timestamp = timestamps[min(change_point_index + min_runs_between_change_points, len(timestamps) - 1)]
    for previous_change_point_timestamp in previous_change_point_timestamps:
        if sibling_change_point_min_timestamp <= previous_change_point_timestamp <= sibling_change_point_max_timestamp:
            logging.info('Performance regression/improvement found for the test ID: %s. Since the change point timestamp %s lies within the sibling change point window: %s, alert is not raised.' % (test_id, previous_change_point_timestamp.strftime('%Y-%m-%d %H:%M:%S'), (sibling_change_point_min_timestamp.strftime('%Y-%m-%d %H:%M:%S'), sibling_change_point_max_timestamp.strftime('%Y-%m-%d %H:%M:%S'))))
            return False
    return True

def read_test_config(config_file_path: str) -> Dict:
    if False:
        while True:
            i = 10
    '\n  Reads the config file in which the data required to\n  run the change point analysis is specified.\n  '
    with open(config_file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def validate_config(keys):
    if False:
        i = 10
        return i + 15
    return constants._PERF_TEST_KEYS.issubset(keys)

def find_change_points(metric_values: List[Union[float, int]]):
    if False:
        while True:
            i = 10
    return e_divisive(metric_values)

def find_latest_change_point_index(metric_values: List[Union[float, int]]):
    if False:
        for i in range(10):
            print('nop')
    '\n  Args:\n   metric_values: Metric values used to run change point analysis.\n  Returns:\n   int: Right most change point index observed on metric_values.\n  '
    change_points_indices = find_change_points(metric_values)
    change_points_indices = filter_change_points_by_median_threshold(metric_values, change_points_indices)
    if not change_points_indices:
        return None
    change_points_indices.sort()
    change_point_index = change_points_indices[-1]
    if is_edge_change_point(change_point_index, len(metric_values), constants._EDGE_SEGMENT_SIZE):
        logging.info('The change point %s is located at the edge of the data with an edge segment size of %s. This change point will be ignored for now, awaiting additional data. Should the change point persist after gathering more data, an alert will be raised.' % (change_point_index, constants._EDGE_SEGMENT_SIZE))
        return None
    return change_point_index

def publish_issue_metadata_to_big_query(issue_metadata, table_name, project=constants._BQ_PROJECT_NAME):
    if False:
        return 10
    '\n  Published issue_metadata to BigQuery with table name.\n  '
    bq_metrics_publisher = BigQueryMetricsPublisher(project_name=project, dataset=constants._BQ_DATASET, table=table_name, bq_schema=constants._SCHEMA)
    bq_metrics_publisher.publish([asdict(issue_metadata)])
    logging.info('GitHub metadata is published to Big Query Dataset %s, table %s' % (constants._BQ_DATASET, table_name))

def create_performance_alert(test_config_container: TestConfigContainer, metric_container: MetricContainer, change_point_index: int, existing_issue_number: Optional[int]) -> Tuple[int, str]:
    if False:
        for i in range(10):
            print('nop')
    '\n  Creates performance alert on GitHub issues and returns GitHub issue\n  number and issue URL.\n  '
    from apache_beam.testing.analyzers import github_issues_utils
    description = github_issues_utils.get_issue_description(test_config_container=test_config_container, metric_container=metric_container, change_point_index=change_point_index, max_results_to_display=constants._NUM_RESULTS_TO_DISPLAY_ON_ISSUE_DESCRIPTION)
    (issue_number, issue_url) = github_issues_utils.report_change_point_on_issues(title=github_issues_utils._ISSUE_TITLE_TEMPLATE.format(test_config_container.test_id, test_config_container.metric_name), description=description, labels=test_config_container.labels, existing_issue_number=existing_issue_number)
    logging.info('Performance regression/improvement is alerted on issue #%s. Link : %s' % (issue_number, issue_url))
    return (issue_number, issue_url)

def filter_change_points_by_median_threshold(data: List[Union[int, float]], change_points: List[int], threshold: float=0.05):
    if False:
        print('Hello World!')
    '\n  Reduces the number of change points by filtering out the ones that are\n  not significant enough based on the relative median threshold. Default\n  value of threshold is 0.05.\n  '
    valid_change_points = []
    epsilon = 1e-10
    for idx in change_points:
        if idx == 0 or idx == len(data):
            continue
        left_segment = data[:idx]
        right_segment = data[idx:]
        left_value = median(left_segment)
        right_value = median(right_segment)
        relative_change = abs(right_value - left_value) / (left_value + epsilon)
        if relative_change > threshold:
            valid_change_points.append(idx)
    return valid_change_points

def is_edge_change_point(change_point_index, data_size, edge_segment_size=constants._EDGE_SEGMENT_SIZE):
    if False:
        return 10
    '\n  Removes the change points that are at the edges of the data.\n  Args:\n    change_point_index: Index of the change point.\n    data_size: Size of the data.\n    edge_segment_size: Size of the edge segment.\n  '
    return change_point_index > data_size - edge_segment_size

class MetricsFetcher(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fetch_metric_data(self, *, test_config: TestConfigContainer) -> MetricContainer:
        if False:
            while True:
                i = 10
        '\n    Define SQL query and fetch the timestamp values and metric values\n    from BigQuery tables.\n    '
        raise NotImplementedError

class BigQueryMetricsFetcher(MetricsFetcher):

    def fetch_metric_data(self, *, test_config: TestConfigContainer) -> MetricContainer:
        if False:
            return 10
        '\n    Args:\n      test_config: TestConfigContainer containing metadata required to fetch\n        metric data from BigQuery.\n    Returns:\n      MetricContainer containing metric values and timestamps.\n    '
        project = test_config.project
        metrics_dataset = test_config.metrics_dataset
        metrics_table = test_config.metrics_table
        metric_name = test_config.metric_name
        query = f"\n          SELECT *\n          FROM {project}.{metrics_dataset}.{metrics_table}\n          WHERE CONTAINS_SUBSTR(({load_test_metrics_utils.METRICS_TYPE_LABEL}), '{metric_name}')\n          ORDER BY {load_test_metrics_utils.SUBMIT_TIMESTAMP_LABEL} DESC\n          LIMIT {constants._NUM_DATA_POINTS_TO_RUN_CHANGE_POINT_ANALYSIS}\n        "
        if bigquery is None:
            raise ImportError('Bigquery dependencies are not installed.')
        client = bigquery.Client()
        query_job = client.query(query=query)
        metric_data = query_job.result().to_dataframe()
        return MetricContainer(values=metric_data[load_test_metrics_utils.VALUE_LABEL].tolist(), timestamps=metric_data[load_test_metrics_utils.SUBMIT_TIMESTAMP_LABEL].tolist())