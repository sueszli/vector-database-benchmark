import argparse
import logging
from apache_beam.testing.analyzers import constants
from apache_beam.testing.analyzers import perf_analysis
from apache_beam.testing.analyzers import perf_analysis_utils
from apache_beam.testing.analyzers.perf_analysis_utils import MetricContainer
from apache_beam.testing.analyzers.perf_analysis_utils import TestConfigContainer
try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None

class LoadTestMetricsFetcher(perf_analysis_utils.MetricsFetcher):
    """
    Metrics fetcher used to get metric data from a BigQuery table. The metrics
    are fetched and returned as a dataclass containing lists of timestamps and
    metric_values.
    """

    def fetch_metric_data(self, *, test_config: TestConfigContainer) -> MetricContainer:
        if False:
            for i in range(10):
                print('nop')
        if test_config.test_name:
            (test_name, pipeline_name) = test_config.test_name.split(',')
        else:
            raise Exception('test_name not provided in config.')
        query = f'\n      SELECT timestamp, metric.value\n      FROM {test_config.project}.{test_config.metrics_dataset}.{test_config.metrics_table}\n      CROSS JOIN UNNEST(metrics) AS metric\n      WHERE test_name = "{test_name}" AND pipeline_name = "{pipeline_name}" AND metric.name = "{test_config.metric_name}"\n      ORDER BY timestamp DESC\n      LIMIT {constants._NUM_DATA_POINTS_TO_RUN_CHANGE_POINT_ANALYSIS}\n    '
        logging.debug('Running query: %s' % query)
        if bigquery is None:
            raise ImportError('Bigquery dependencies are not installed.')
        client = bigquery.Client()
        query_job = client.query(query=query)
        metric_data = query_job.result().to_dataframe()
        if metric_data.empty:
            logging.error('No results returned from BigQuery. Please check the query.')
        return MetricContainer(values=metric_data['value'].tolist(), timestamps=metric_data['timestamp'].tolist())
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    load_test_metrics_fetcher = LoadTestMetricsFetcher()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', required=True, type=str, help='Path to the config file that contains data to run the Change Point Analysis.The default file will used will be apache_beam/testing/analyzers/tests.config.yml. If you would like to use the Change Point Analysis for finding performance regression in the tests, please provide an .yml file in the same structure as the above mentioned file. ')
    parser.add_argument('--save_alert_metadata', action='store_true', default=False, help='Save perf alert/ GH Issue metadata to BigQuery table.')
    (known_args, unknown_args) = parser.parse_known_args()
    if unknown_args:
        logging.warning('Discarding unknown arguments : %s ' % unknown_args)
    perf_analysis.run(big_query_metrics_fetcher=load_test_metrics_fetcher, config_file_path=known_args.config_file_path, save_alert_metadata=known_args.save_alert_metadata)