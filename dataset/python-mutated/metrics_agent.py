import json
import logging
import os
import re
import threading
import time
import traceback
from collections import namedtuple
from typing import List, Tuple, Any, Dict
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily
from opencensus.metrics.export.value import ValueDouble
from opencensus.stats import aggregation
from opencensus.stats import measure as measure_module
from opencensus.stats.view_manager import ViewManager
from opencensus.stats.stats_recorder import StatsRecorder
from opencensus.stats.base_exporter import StatsExporter
from prometheus_client.core import Metric as PrometheusMetric
from opencensus.stats.aggregation_data import CountAggregationData, DistributionAggregationData, LastValueAggregationData
from opencensus.stats.view import View
from opencensus.tags import tag_key as tag_key_module
from opencensus.tags import tag_map as tag_map_module
from opencensus.tags import tag_value as tag_value_module
import ray
from ray._raylet import GcsClient
from ray.core.generated.metrics_pb2 import Metric
logger = logging.getLogger(__name__)
RAY_WORKER_TIMEOUT_S = 'RAY_WORKER_TIMEOUT_S'
GLOBAL_COMPONENT_KEY = 'CORE'
RE_NON_ALPHANUMS = re.compile('[^a-zA-Z0-9]')

class Gauge(View):
    """Gauge representation of opencensus view.

    This class is used to collect process metrics from the reporter agent.
    Cpp metrics should be collected in a different way.
    """

    def __init__(self, name, description, unit, tags: List[str]):
        if False:
            print('Hello World!')
        self._measure = measure_module.MeasureInt(name, description, unit)
        tags = [tag_key_module.TagKey(tag) for tag in tags]
        self._view = View(name, description, tags, self.measure, aggregation.LastValueAggregation())

    @property
    def measure(self):
        if False:
            for i in range(10):
                print('nop')
        return self._measure

    @property
    def view(self):
        if False:
            while True:
                i = 10
        return self._view

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self.measure.name
Record = namedtuple('Record', ['gauge', 'value', 'tags'])

def fix_grpc_metric(metric: Metric):
    if False:
        while True:
            i = 10
    "\n    Fix the inbound `opencensus.proto.metrics.v1.Metric` protos to make it acceptable\n    by opencensus.stats.DistributionAggregationData.\n\n    - metric name: gRPC OpenCensus metrics have names with slashes and dots, e.g.\n    `grpc.io/client/server_latency`[1]. However Prometheus metric names only take\n    alphanums,underscores and colons[2]. We santinize the name by replacing non-alphanum\n    chars to underscore, like the official opencensus prometheus exporter[3].\n    - distribution bucket bounds: The Metric proto asks distribution bucket bounds to\n    be > 0 [4]. However, gRPC OpenCensus metrics have their first bucket bound == 0 [1].\n    This makes the `DistributionAggregationData` constructor to raise Exceptions. This\n    applies to all bytes and milliseconds (latencies). The fix: we update the initial 0\n    bounds to be 0.000_000_1. This will not affect the precision of the metrics, since\n    we don't expect any less-than-1 bytes, or less-than-1-nanosecond times.\n\n    [1] https://github.com/census-instrumentation/opencensus-specs/blob/master/stats/gRPC.md#units  # noqa: E501\n    [2] https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels\n    [3] https://github.com/census-instrumentation/opencensus-cpp/blob/50eb5de762e5f87e206c011a4f930adb1a1775b1/opencensus/exporters/stats/prometheus/internal/prometheus_utils.cc#L39 # noqa: E501\n    [4] https://github.com/census-instrumentation/opencensus-proto/blob/master/src/opencensus/proto/metrics/v1/metrics.proto#L218 # noqa: E501\n    "
    if not metric.metric_descriptor.name.startswith('grpc.io/'):
        return
    metric.metric_descriptor.name = RE_NON_ALPHANUMS.sub('_', metric.metric_descriptor.name)
    for series in metric.timeseries:
        for point in series.points:
            if point.HasField('distribution_value'):
                dist_value = point.distribution_value
                bucket_bounds = dist_value.bucket_options.explicit.bounds
                if len(bucket_bounds) > 0 and bucket_bounds[0] == 0:
                    bucket_bounds[0] = 1e-07

class OpencensusProxyMetric:

    def __init__(self, name: str, desc: str, unit: str, label_keys: List[str]):
        if False:
            while True:
                i = 10
        'Represents the OpenCensus metrics that will be proxy exported.'
        self._name = name
        self._desc = desc
        self._unit = unit
        self._label_keys = label_keys
        self._data = {}

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self._name

    @property
    def desc(self):
        if False:
            for i in range(10):
                print('nop')
        return self._desc

    @property
    def unit(self):
        if False:
            return 10
        return self._unit

    @property
    def label_keys(self):
        if False:
            return 10
        return self._label_keys

    @property
    def data(self):
        if False:
            return 10
        return self._data

    def record(self, metric: Metric):
        if False:
            for i in range(10):
                print('nop')
        'Parse the Opencensus Protobuf and store the data.\n\n        The data can be accessed via `data` API once recorded.\n        '
        timeseries = metric.timeseries
        if len(timeseries) == 0:
            return
        for series in timeseries:
            labels = tuple((val.value for val in series.label_values))
            for point in series.points:
                if point.HasField('int64_value'):
                    data = CountAggregationData(point.int64_value)
                elif point.HasField('double_value'):
                    data = LastValueAggregationData(ValueDouble, point.double_value)
                elif point.HasField('distribution_value'):
                    dist_value = point.distribution_value
                    counts_per_bucket = [bucket.count for bucket in dist_value.buckets]
                    bucket_bounds = dist_value.bucket_options.explicit.bounds
                    data = DistributionAggregationData(dist_value.sum / dist_value.count, dist_value.count, dist_value.sum_of_squared_deviation, counts_per_bucket, bucket_bounds)
                else:
                    raise ValueError('Summary is not supported')
                self._data[labels] = data

class Component:

    def __init__(self, id: str):
        if False:
            print('Hello World!')
        'Represent a component that requests to proxy export metrics\n\n        Args:\n            id: Id of this component.\n        '
        self.id = id
        self._last_reported_time = time.monotonic()
        self._metrics = {}

    @property
    def metrics(self) -> Dict[str, OpencensusProxyMetric]:
        if False:
            while True:
                i = 10
        'Return the metrics requested to proxy export from this component.'
        return self._metrics

    @property
    def last_reported_time(self):
        if False:
            print('Hello World!')
        return self._last_reported_time

    def record(self, metrics: List[Metric]):
        if False:
            return 10
        'Parse the Opencensus protobuf and store metrics.\n\n        Metrics can be accessed via `metrics` API for proxy export.\n\n        Args:\n            metrics: A list of Opencensus protobuf for proxy export.\n        '
        self._last_reported_time = time.monotonic()
        for metric in metrics:
            fix_grpc_metric(metric)
            descriptor = metric.metric_descriptor
            name = descriptor.name
            label_keys = [label_key.key for label_key in descriptor.label_keys]
            if name not in self._metrics:
                self._metrics[name] = OpencensusProxyMetric(name, descriptor.description, descriptor.unit, label_keys)
            self._metrics[name].record(metric)

class OpenCensusProxyCollector:

    def __init__(self, namespace: str, component_timeout_s: int=60):
        if False:
            print('Hello World!')
        'Prometheus collector implementation for opencensus proxy export.\n\n        Prometheus collector requires to implement `collect` which is\n        invoked whenever Prometheus queries the endpoint.\n\n        The class is thread-safe.\n\n        Args:\n            namespace: Prometheus namespace.\n        '
        self._components_lock = threading.Lock()
        self._component_timeout_s = component_timeout_s
        self._namespace = namespace
        self._components = {}

    def record(self, metrics: List[Metric], worker_id_hex: str=None):
        if False:
            for i in range(10):
                print('nop')
        'Record the metrics reported from the component that reports it.\n\n        Args:\n            metrics: A list of opencensus protobuf to proxy export metrics.\n            worker_id_hex: A worker id that reports these metrics.\n                If None, it means they are reported from Raylet or GCS.\n        '
        key = GLOBAL_COMPONENT_KEY if not worker_id_hex else worker_id_hex
        with self._components_lock:
            if key not in self._components:
                self._components[key] = Component(key)
            self._components[key].record(metrics)

    def clean_stale_components(self):
        if False:
            return 10
        "Clean up stale components.\n\n        Stale means the component is dead or unresponsive.\n\n        Stale components won't be reported to Prometheus anymore.\n        "
        with self._components_lock:
            stale_components = []
            stale_component_ids = []
            for (id, component) in self._components.items():
                elapsed = time.monotonic() - component.last_reported_time
                if elapsed > self._component_timeout_s:
                    stale_component_ids.append(id)
                    logger.info('Metrics from a worker ({}) is cleaned up due to timeout. Time since last report {}s'.format(id, elapsed))
            for id in stale_component_ids:
                stale_components.append(self._components.pop(id))
            return stale_components

    def to_metric(self, metric_name: str, metric_description: str, label_keys: List[str], metric_units: str, label_values: Tuple[tag_value_module.TagValue], agg_data: Any, metrics_map: Dict[str, PrometheusMetric]) -> PrometheusMetric:
        if False:
            while True:
                i = 10
        'to_metric translate the data that OpenCensus create\n        to Prometheus format, using Prometheus Metric object.\n\n        This method is from Opencensus Prometheus Exporter.\n\n        Args:\n            metric_name: Name of the metric.\n            metric_description: Description of the metric.\n            label_keys: The fixed label keys of the metric.\n            metric_units: Units of the metric.\n            label_values: The values of `label_keys`.\n            agg_data: `opencensus.stats.aggregation_data.AggregationData` object.\n                Aggregated data that needs to be converted as Prometheus samples\n\n        Returns:\n            A Prometheus metric object\n        '
        assert self._components_lock.locked()
        metric_name = f'{self._namespace}_{metric_name}'
        assert len(label_values) == len(label_keys), (label_values, label_keys)
        label_values = [tv if tv else '' for tv in label_values]
        if isinstance(agg_data, CountAggregationData):
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = CounterMetricFamily(name=metric_name, documentation=metric_description, unit=metric_units, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=label_values, value=agg_data.count_data)
            return metric
        elif isinstance(agg_data, DistributionAggregationData):
            assert agg_data.bounds == sorted(agg_data.bounds)
            buckets = []
            cum_count = 0
            for (ii, bound) in enumerate(agg_data.bounds):
                cum_count += agg_data.counts_per_bucket[ii]
                bucket = [str(bound), cum_count]
                buckets.append(bucket)
            buckets.append(['+Inf', agg_data.count_data])
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = HistogramMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=label_values, buckets=buckets, sum_value=agg_data.sum)
            return metric
        elif isinstance(agg_data, LastValueAggregationData):
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = GaugeMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=label_values, value=agg_data.value)
            return metric
        else:
            raise ValueError(f'unsupported aggregation type {type(agg_data)}')

    def collect(self):
        if False:
            for i in range(10):
                print('nop')
        'Collect fetches the statistics from OpenCensus\n        and delivers them as Prometheus Metrics.\n        Collect is invoked every time a prometheus.Gatherer is run\n        for example when the HTTP endpoint is invoked by Prometheus.\n\n        This method is required as a Prometheus Collector.\n        '
        with self._components_lock:
            metrics_map = {}
            for component in self._components.values():
                for metric in component.metrics.values():
                    for (label_values, data) in metric.data.items():
                        self.to_metric(metric.name, metric.desc, metric.label_keys, metric.unit, label_values, data, metrics_map)
        for metric in metrics_map.values():
            yield metric

class MetricsAgent:

    def __init__(self, view_manager: ViewManager, stats_recorder: StatsRecorder, stats_exporter: StatsExporter=None):
        if False:
            for i in range(10):
                print('nop')
        'A class to record and export metrics.\n\n        The class exports metrics in 2 different ways.\n        - Directly record and export metrics using OpenCensus.\n        - Proxy metrics from other core components\n            (e.g., raylet, GCS, core workers).\n\n        This class is thread-safe.\n        '
        self._lock = threading.Lock()
        self.view_manager = view_manager
        self.stats_recorder = stats_recorder
        self.stats_exporter = stats_exporter
        self.proxy_exporter_collector = None
        if self.stats_exporter is None:
            self.view_manager = None
        else:
            self.view_manager.register_exporter(stats_exporter)
            self.proxy_exporter_collector = OpenCensusProxyCollector(self.stats_exporter.options.namespace, component_timeout_s=int(os.getenv(RAY_WORKER_TIMEOUT_S, 120)))

    def record_and_export(self, records: List[Record], global_tags=None):
        if False:
            return 10
        'Directly record and export stats from the same process.'
        global_tags = global_tags or {}
        with self._lock:
            if not self.view_manager:
                return
            for record in records:
                gauge = record.gauge
                value = record.value
                tags = record.tags
                self._record_gauge(gauge, value, {**tags, **global_tags})

    def _record_gauge(self, gauge: Gauge, value: float, tags: dict):
        if False:
            print('Hello World!')
        view_data = self.view_manager.get_view(gauge.name)
        if not view_data:
            self.view_manager.register_view(gauge.view)
        view = self.view_manager.get_view(gauge.name).view
        measurement_map = self.stats_recorder.new_measurement_map()
        tag_map = tag_map_module.TagMap()
        for (key, tag_val) in tags.items():
            tag_key = tag_key_module.TagKey(key)
            tag_value = tag_value_module.TagValue(tag_val)
            tag_map.insert(tag_key, tag_value)
        measurement_map.measure_float_put(view.measure, value)
        measurement_map.record(tag_map)

    def proxy_export_metrics(self, metrics: List[Metric], worker_id_hex: str=None):
        if False:
            while True:
                i = 10
        'Proxy export metrics specified by a Opencensus Protobuf.\n\n        This API is used to export metrics emitted from\n        core components.\n\n        Args:\n            metrics: A list of protobuf Metric defined from OpenCensus.\n            worker_id_hex: The worker ID it proxies metrics export. None\n                if the metric is not from a worker (i.e., raylet, GCS).\n        '
        with self._lock:
            if not self.view_manager:
                return
        self._proxy_export_metrics(metrics, worker_id_hex)

    def _proxy_export_metrics(self, metrics: List[Metric], worker_id_hex: str=None):
        if False:
            for i in range(10):
                print('nop')
        self.proxy_exporter_collector.record(metrics, worker_id_hex)

    def clean_all_dead_worker_metrics(self):
        if False:
            print('Hello World!')
        "Clean dead worker's metrics.\n\n        Worker metrics are cleaned up and won't be exported once\n        it is considered as dead.\n\n        This method has to be periodically called by a caller.\n        "
        with self._lock:
            if not self.view_manager:
                return
        self.proxy_exporter_collector.clean_stale_components()

class PrometheusServiceDiscoveryWriter(threading.Thread):
    """A class to support Prometheus service discovery.

    It supports file-based service discovery. Checkout
    https://prometheus.io/docs/guides/file-sd/ for more details.

    Args:
        gcs_address: Gcs address for this cluster.
        temp_dir: Temporary directory used by
            Ray to store logs and metadata.
    """

    def __init__(self, gcs_address, temp_dir):
        if False:
            i = 10
            return i + 15
        gcs_client_options = ray._raylet.GcsClientOptions.from_gcs_address(gcs_address)
        self.gcs_address = gcs_address
        ray._private.state.state._initialize_global_state(gcs_client_options)
        self.temp_dir = temp_dir
        self.default_service_discovery_flush_period = 5
        super().__init__()

    def get_file_discovery_content(self):
        if False:
            print('Hello World!')
        'Return the content for Prometheus service discovery.'
        nodes = ray.nodes()
        metrics_export_addresses = ['{}:{}'.format(node['NodeManagerAddress'], node['MetricsExportPort']) for node in nodes if node['alive'] is True]
        gcs_client = GcsClient(address=self.gcs_address)
        autoscaler_addr = gcs_client.internal_kv_get(b'AutoscalerMetricsAddress', None)
        if autoscaler_addr:
            metrics_export_addresses.append(autoscaler_addr.decode('utf-8'))
        dashboard_addr = gcs_client.internal_kv_get(b'DashboardMetricsAddress', None)
        if dashboard_addr:
            metrics_export_addresses.append(dashboard_addr.decode('utf-8'))
        return json.dumps([{'labels': {'job': 'ray'}, 'targets': metrics_export_addresses}])

    def write(self):
        if False:
            i = 10
            return i + 15
        temp_file_name = self.get_temp_file_name()
        with open(temp_file_name, 'w') as json_file:
            json_file.write(self.get_file_discovery_content())
        os.replace(temp_file_name, self.get_target_file_name())

    def get_target_file_name(self):
        if False:
            return 10
        return os.path.join(self.temp_dir, ray._private.ray_constants.PROMETHEUS_SERVICE_DISCOVERY_FILE)

    def get_temp_file_name(self):
        if False:
            print('Hello World!')
        return os.path.join(self.temp_dir, '{}_{}'.format('tmp', ray._private.ray_constants.PROMETHEUS_SERVICE_DISCOVERY_FILE))

    def run(self):
        if False:
            i = 10
            return i + 15
        while True:
            try:
                self.write()
            except Exception as e:
                logger.warning('Writing a service discovery file, {},failed.'.format(self.get_target_file_name()))
                logger.warning(traceback.format_exc())
                logger.warning(f'Error message: {e}')
            time.sleep(self.default_service_discovery_flush_period)