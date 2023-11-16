"""
Utility functions used for integrating Metrics API into load tests pipelines.

Metrics are send to BigQuery in following format:
test_id | submit_timestamp | metric_type | value

The 'test_id' is common for all metrics for one run.
Currently it is possible to have following metrics types:
* runtime
* total_bytes_count
"""
import json
import logging
import time
import uuid
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union
import requests
from requests.auth import HTTPBasicAuth
import apache_beam as beam
from apache_beam.metrics import Metrics
from apache_beam.metrics.metric import MetricResults
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.runners.runner import PipelineResult
from apache_beam.transforms.window import TimestampedValue
from apache_beam.utils.timestamp import Timestamp
try:
    from google.cloud import bigquery
    from google.cloud.bigquery.schema import SchemaField
    from google.cloud.exceptions import NotFound
except ImportError:
    bigquery = None
    SchemaField = None
    NotFound = None
RUNTIME_METRIC = 'runtime'
COUNTER_LABEL = 'total_bytes_count'
ID_LABEL = 'test_id'
SUBMIT_TIMESTAMP_LABEL = 'timestamp'
METRICS_TYPE_LABEL = 'metric'
VALUE_LABEL = 'value'
SCHEMA = [{'name': ID_LABEL, 'field_type': 'STRING', 'mode': 'REQUIRED'}, {'name': SUBMIT_TIMESTAMP_LABEL, 'field_type': 'TIMESTAMP', 'mode': 'REQUIRED'}, {'name': METRICS_TYPE_LABEL, 'field_type': 'STRING', 'mode': 'REQUIRED'}, {'name': VALUE_LABEL, 'field_type': 'FLOAT', 'mode': 'REQUIRED'}]
_LOGGER = logging.getLogger(__name__)

def parse_step(step_name):
    if False:
        return 10
    "Replaces white spaces and removes 'Step:' label\n\n  Args:\n    step_name(str): step name passed in metric ParDo\n\n  Returns:\n    lower case step name without namespace and step label\n  "
    prefix = 'step'
    step_name = step_name.lower().replace(' ', '_')
    step_name = step_name[len(prefix):] if prefix and step_name.startswith(prefix) else step_name
    return step_name.strip(':_')

def split_metrics_by_namespace_and_name(metrics, namespace, name):
    if False:
        for i in range(10):
            print('nop')
    'Splits metrics list namespace and name.\n\n  Args:\n    metrics: list of metrics from pipeline result\n    namespace(str): filter metrics by namespace\n    name(str): filter metrics by name\n\n  Returns:\n    two lists - one of metrics which are matching filters\n    and second of not matching\n  '
    matching_metrics = []
    not_matching_metrics = []
    for dist in metrics:
        if dist.key.metric.namespace == namespace and dist.key.metric.name == name:
            matching_metrics.append(dist)
        else:
            not_matching_metrics.append(dist)
    return (matching_metrics, not_matching_metrics)

def get_generic_distributions(generic_dists, metric_id):
    if False:
        i = 10
        return i + 15
    'Creates flatten list of distributions per its value type.\n  A generic distribution is the one which is not processed but saved in\n  the most raw version.\n\n  Args:\n    generic_dists: list of distributions to be saved\n    metric_id(uuid): id of the current test run\n\n  Returns:\n    list of dictionaries made from :class:`DistributionMetric`\n  '
    return sum((get_all_distributions_by_type(dist, metric_id) for dist in generic_dists), [])

def get_all_distributions_by_type(dist, metric_id):
    if False:
        while True:
            i = 10
    'Creates new list of objects with type of each distribution\n  metric value.\n\n  Args:\n    dist(object): DistributionMetric object to be parsed\n    metric_id(uuid): id of the current test run\n  Returns:\n    list of :class:`DistributionMetric` objects\n  '
    submit_timestamp = time.time()
    dist_types = ['count', 'max', 'min', 'sum', 'mean']
    distribution_dicts = []
    for dist_type in dist_types:
        try:
            distribution_dicts.append(get_distribution_dict(dist_type, submit_timestamp, dist, metric_id))
        except ValueError:
            continue
    return distribution_dicts

def get_distribution_dict(metric_type, submit_timestamp, dist, metric_id):
    if False:
        while True:
            i = 10
    'Function creates :class:`DistributionMetric`\n\n  Args:\n    metric_type(str): type of value from distribution metric which will\n      be saved (ex. max, min, mean, sum)\n    submit_timestamp: timestamp when metric is saved\n    dist(object) distribution object from pipeline result\n    metric_id(uuid): id of the current test run\n\n  Returns:\n    dictionary prepared for saving according to schema\n  '
    return DistributionMetric(dist, submit_timestamp, metric_id, metric_type).as_dict()

class MetricsReader(object):
    """
  A :class:`MetricsReader` retrieves metrics from pipeline result,
  prepares it for publishers and setup publishers.
  """

    def __init__(self, project_name=None, bq_table=None, bq_dataset=None, publish_to_bq=False, influxdb_options=None, namespace=None, filters=None):
        if False:
            print('Hello World!')
        'Initializes :class:`MetricsReader` .\n\n    Args:\n      project_name (str): project with BigQuery where metrics will be saved\n      bq_table (str): BigQuery table where metrics will be saved\n      bq_dataset (str): BigQuery dataset where metrics will be saved\n      namespace (str): Namespace of the metrics\n      filters: MetricFilter to query only filtered metrics\n    '
        self._namespace = namespace
        self.publishers: List[MetricsPublisher] = []
        self.publishers.append(ConsoleMetricsPublisher())
        bq_check = project_name and bq_table and bq_dataset and publish_to_bq
        if bq_check:
            bq_publisher = BigQueryMetricsPublisher(project_name, bq_table, bq_dataset)
            self.publishers.append(bq_publisher)
        if influxdb_options and influxdb_options.validate():
            self.publishers.append(InfluxDBMetricsPublisher(influxdb_options))
        else:
            _LOGGER.info('Missing InfluxDB options. Metrics will not be published to InfluxDB')
        self.filters = filters

    def get_counter_metric(self, result: PipelineResult, name: str) -> int:
        if False:
            print('Hello World!')
        "\n    Return the current value for a long counter, or -1 if can't be retrieved.\n    Note this uses only attempted metrics because some runners don't support\n    committed metrics.\n    "
        filters = MetricsFilter().with_namespace(self._namespace).with_name(name)
        counters = result.metrics().query(filters)[MetricResults.COUNTERS]
        num_results = len(counters)
        if num_results > 1:
            raise ValueError(f'More than one metric result matches name: {name} in namespace {self._namespace}. Metric results count: {num_results}')
        elif num_results == 0:
            return -1
        else:
            return counters[0].attempted

    def publish_metrics(self, result: PipelineResult, extra_metrics: Optional[dict]=None):
        if False:
            for i in range(10):
                print('nop')
        'Publish metrics from pipeline result to registered publishers.'
        metric_id = uuid.uuid4().hex
        metrics = result.metrics().query(self.filters)
        insert_dicts = self._prepare_all_metrics(metrics, metric_id)
        insert_dicts += self._prepare_extra_metrics(metric_id, extra_metrics)
        if len(insert_dicts) > 0:
            for publisher in self.publishers:
                publisher.publish(insert_dicts)

    def _prepare_extra_metrics(self, metric_id: str, extra_metrics: Optional[dict]=None):
        if False:
            return 10
        ts = time.time()
        if not extra_metrics:
            extra_metrics = {}
        return [Metric(ts, metric_id, v, label=k).as_dict() for (k, v) in extra_metrics.items()]

    def publish_values(self, labeled_values):
        if False:
            i = 10
            return i + 15
        'The method to publish simple labeled values.\n\n    Args:\n      labeled_values (List[Tuple(str, int)]): list of (label, value)\n    '
        metric_dicts = [Metric(time.time(), uuid.uuid4().hex, value, label=label).as_dict() for (label, value) in labeled_values]
        for publisher in self.publishers:
            publisher.publish(metric_dicts)

    def _prepare_all_metrics(self, metrics, metric_id):
        if False:
            print('Hello World!')
        insert_rows = self._get_counters(metrics['counters'], metric_id)
        insert_rows += self._get_distributions(metrics['distributions'], metric_id)
        return insert_rows

    def _get_counters(self, counters, metric_id):
        if False:
            return 10
        submit_timestamp = time.time()
        return [CounterMetric(counter, submit_timestamp, metric_id).as_dict() for counter in counters]

    def _get_distributions(self, distributions, metric_id):
        if False:
            for i in range(10):
                print('nop')
        rows = []
        (matching_namsespace, not_matching_namespace) = split_metrics_by_namespace_and_name(distributions, self._namespace, RUNTIME_METRIC)
        if len(matching_namsespace) > 0:
            runtime_metric = RuntimeMetric(matching_namsespace, metric_id)
            rows.append(runtime_metric.as_dict())
        if len(not_matching_namespace) > 0:
            rows += get_generic_distributions(not_matching_namespace, metric_id)
        return rows

class Metric(object):
    """Metric base class in ready-to-save format."""

    def __init__(self, submit_timestamp, metric_id, value, metric=None, label=None):
        if False:
            return 10
        'Initializes :class:`Metric`\n\n    Args:\n      metric (object): object of metric result\n      submit_timestamp (float): date-time of saving metric to database\n      metric_id (uuid): unique id to identify test run\n      value: value of metric\n      label: custom metric name to be saved in database\n    '
        self.submit_timestamp = submit_timestamp
        self.metric_id = metric_id
        self.label = label or metric.key.metric.namespace + '_' + parse_step(metric.key.step) + '_' + metric.key.metric.name
        self.value = value

    def as_dict(self):
        if False:
            i = 10
            return i + 15
        return {SUBMIT_TIMESTAMP_LABEL: self.submit_timestamp, ID_LABEL: self.metric_id, VALUE_LABEL: self.value, METRICS_TYPE_LABEL: self.label}

class CounterMetric(Metric):
    """The Counter Metric in ready-to-publish format.

  Args:
    counter_metric (object): counter metric object from MetricResult
    submit_timestamp (float): date-time of saving metric to database
    metric_id (uuid): unique id to identify test run
  """

    def __init__(self, counter_metric, submit_timestamp, metric_id):
        if False:
            i = 10
            return i + 15
        value = counter_metric.result
        super().__init__(submit_timestamp, metric_id, value, counter_metric)

class DistributionMetric(Metric):
    """The Distribution Metric in ready-to-publish format.

  Args:
    dist_metric (object): distribution metric object from MetricResult
    submit_timestamp (float): date-time of saving metric to database
    metric_id (uuid): unique id to identify test run
  """

    def __init__(self, dist_metric, submit_timestamp, metric_id, metric_type):
        if False:
            return 10
        custom_label = dist_metric.key.metric.namespace + '_' + parse_step(dist_metric.key.step) + '_' + metric_type + '_' + dist_metric.key.metric.name
        value = getattr(dist_metric.result, metric_type)
        if value is None:
            msg = '%s: the result is expected to be an integer, not None.' % custom_label
            _LOGGER.debug(msg)
            raise ValueError(msg)
        super().__init__(submit_timestamp, metric_id, value, dist_metric, custom_label)

class RuntimeMetric(Metric):
    """The Distribution Metric in ready-to-publish format.

  Args:
    runtime_list: list of distributions metrics from MetricResult
      with runtime name
    metric_id(uuid): unique id to identify test run
  """

    def __init__(self, runtime_list, metric_id):
        if False:
            while True:
                i = 10
        value = self._prepare_runtime_metrics(runtime_list)
        submit_timestamp = time.time()
        label = runtime_list[0].key.metric.namespace + '_' + RUNTIME_METRIC
        super().__init__(submit_timestamp, metric_id, value, None, label)

    def _prepare_runtime_metrics(self, distributions):
        if False:
            for i in range(10):
                print('nop')
        min_values = []
        max_values = []
        for dist in distributions:
            min_values.append(dist.result.min)
            max_values.append(dist.result.max)
        min_value = min(min_values)
        max_value = max(max_values)
        runtime_in_s = float(max_value - min_value)
        return runtime_in_s

class MetricsPublisher:
    """Base class for metrics publishers."""

    def publish(self, results):
        if False:
            return 10
        raise NotImplementedError

class ConsoleMetricsPublisher(MetricsPublisher):
    """A :class:`ConsoleMetricsPublisher` publishes collected metrics
  to console output."""

    def publish(self, results):
        if False:
            while True:
                i = 10
        if len(results) > 0:
            log = 'Load test results for test: %s and timestamp: %s:' % (results[0][ID_LABEL], results[0][SUBMIT_TIMESTAMP_LABEL])
            _LOGGER.info(log)
            for result in results:
                log = 'Metric: %s Value: %d' % (result[METRICS_TYPE_LABEL], result[VALUE_LABEL])
                _LOGGER.info(log)
        else:
            _LOGGER.info('No test results were collected.')

class BigQueryMetricsPublisher(MetricsPublisher):
    """A :class:`BigQueryMetricsPublisher` publishes collected metrics
  to BigQuery output."""

    def __init__(self, project_name, table, dataset, bq_schema=None):
        if False:
            print('Hello World!')
        if not bq_schema:
            bq_schema = SCHEMA
        self.bq = BigQueryClient(project_name, table, dataset, bq_schema)

    def publish(self, results):
        if False:
            i = 10
            return i + 15
        outputs = self.bq.save(results)
        if len(outputs) > 0:
            for output in outputs:
                if output['errors']:
                    _LOGGER.error(output)
                    raise ValueError('Unable save rows in BigQuery: {}'.format(output['errors']))

class BigQueryClient(object):
    """A :class:`BigQueryClient` publishes collected metrics to
  BigQuery output."""

    def __init__(self, project_name, table, dataset, bq_schema=None):
        if False:
            for i in range(10):
                print('nop')
        self.schema = bq_schema
        self._namespace = table
        self._client = bigquery.Client(project=project_name)
        self._schema_names = self._get_schema_names()
        schema = self._prepare_schema()
        self._get_or_create_table(schema, dataset)

    def _get_schema_names(self):
        if False:
            i = 10
            return i + 15
        return [schema['name'] for schema in self.schema]

    def _prepare_schema(self):
        if False:
            return 10
        return [SchemaField(**row) for row in self.schema]

    def _get_or_create_table(self, bq_schemas, dataset):
        if False:
            return 10
        if self._namespace == '':
            raise ValueError('Namespace cannot be empty.')
        dataset = self._get_dataset(dataset)
        table_ref = dataset.table(self._namespace)
        try:
            self._bq_table = self._client.get_table(table_ref)
        except NotFound:
            table = bigquery.Table(table_ref, schema=bq_schemas)
            self._bq_table = self._client.create_table(table)

    def _get_dataset(self, dataset_name):
        if False:
            for i in range(10):
                print('nop')
        bq_dataset_ref = self._client.dataset(dataset_name)
        try:
            bq_dataset = self._client.get_dataset(bq_dataset_ref)
        except NotFound:
            raise ValueError('Dataset {} does not exist in your project. You have to create table first.'.format(dataset_name))
        return bq_dataset

    def save(self, results):
        if False:
            for i in range(10):
                print('nop')
        return self._client.insert_rows(self._bq_table, results)

class InfluxDBMetricsPublisherOptions(object):

    def __init__(self, measurement, db_name, hostname, user=None, password=None):
        if False:
            print('Hello World!')
        self.measurement = measurement
        self.db_name = db_name
        self.hostname = hostname
        self.user = user
        self.password = password

    def validate(self):
        if False:
            i = 10
            return i + 15
        return bool(self.measurement) and bool(self.db_name)

    def http_auth_enabled(self):
        if False:
            return 10
        return self.user is not None and self.password is not None

class InfluxDBMetricsPublisher(MetricsPublisher):
    """Publishes collected metrics to InfluxDB database."""

    def __init__(self, options):
        if False:
            for i in range(10):
                print('nop')
        self.options = options

    def publish(self, results):
        if False:
            print('Hello World!')
        url = '{}/write'.format(self.options.hostname)
        payload = self._build_payload(results)
        query_str = {'db': self.options.db_name, 'precision': 's'}
        auth = HTTPBasicAuth(self.options.user, self.options.password) if self.options.http_auth_enabled() else None
        try:
            response = requests.post(url, params=query_str, data=payload, auth=auth, timeout=60)
        except requests.exceptions.RequestException as e:
            _LOGGER.warning('Failed to publish metrics to InfluxDB: ' + str(e))
        else:
            if response.status_code != 204:
                content = json.loads(response.content)
                _LOGGER.warning('Failed to publish metrics to InfluxDB. Received status code %s with an error message: %s' % (response.status_code, content['error']))

    def _build_payload(self, results):
        if False:
            for i in range(10):
                print('nop')

        def build_kv(mapping, key):
            if False:
                return 10
            return '{}={}'.format(key, mapping[key])
        points = []
        for result in results:
            comma_separated = [self.options.measurement, build_kv(result, METRICS_TYPE_LABEL), build_kv(result, ID_LABEL)]
            point = ','.join(comma_separated) + ' ' + build_kv(result, VALUE_LABEL) + ' ' + str(int(result[SUBMIT_TIMESTAMP_LABEL]))
            points.append(point)
        return '\n'.join(points)

class MeasureTime(beam.DoFn):
    """A distribution metric prepared to be added to pipeline as ParDo
   to measure runtime."""

    def __init__(self, namespace):
        if False:
            print('Hello World!')
        'Initializes :class:`MeasureTime`.\n\n      namespace(str): namespace of  metric\n    '
        self.namespace = namespace
        self.runtime = Metrics.distribution(self.namespace, RUNTIME_METRIC)

    def start_bundle(self):
        if False:
            return 10
        self.runtime.update(time.time())

    def finish_bundle(self):
        if False:
            return 10
        self.runtime.update(time.time())

    def process(self, element):
        if False:
            print('Hello World!')
        yield element

class MeasureBytes(beam.DoFn):
    """Metric to measure how many bytes was observed in pipeline."""
    LABEL = 'total_bytes'

    def __init__(self, namespace, extractor=None):
        if False:
            return 10
        'Initializes :class:`MeasureBytes`.\n\n    Args:\n      namespace(str): metric namespace\n      extractor: function to extract elements to be count\n    '
        self.namespace = namespace
        self.counter = Metrics.counter(self.namespace, self.LABEL)
        self.extractor = extractor if extractor else lambda x: (yield x)

    def process(self, element, *args):
        if False:
            print('Hello World!')
        for value in self.extractor(element, *args):
            self.counter.inc(len(value))
        yield element

class CountMessages(beam.DoFn):
    LABEL = 'total_messages'

    def __init__(self, namespace):
        if False:
            return 10
        self.namespace = namespace
        self.counter = Metrics.counter(self.namespace, self.LABEL)

    def process(self, element):
        if False:
            i = 10
            return i + 15
        self.counter.inc(1)
        yield element

class MeasureLatency(beam.DoFn):
    """A distribution metric which captures the latency based on the timestamps
  of the processed elements."""
    LABEL = 'latency'

    def __init__(self, namespace):
        if False:
            while True:
                i = 10
        'Initializes :class:`MeasureLatency`.\n\n      namespace(str): namespace of  metric\n    '
        self.namespace = namespace
        self.latency_ms = Metrics.distribution(self.namespace, self.LABEL)
        self.time_fn = time.time

    def process(self, element, timestamp=beam.DoFn.TimestampParam):
        if False:
            print('Hello World!')
        self.latency_ms.update(int(self.time_fn() * 1000) - timestamp.micros // 1000)
        yield element

class AssignTimestamps(beam.DoFn):
    """DoFn to assigned timestamps to elements."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.time_fn = time.time
        self.timestamp_val_fn = TimestampedValue
        self.timestamp_fn = Timestamp

    def process(self, element):
        if False:
            return 10
        yield self.timestamp_val_fn(element, self.timestamp_fn(micros=int(self.time_fn() * 1000000)))