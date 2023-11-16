import re
from prometheus_client import start_http_server
from prometheus_client.core import REGISTRY, CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily, UnknownMetricFamily
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
logger = logging.getLogger(__name__)

class Options(object):
    """Options contains options for configuring the exporter.
    The address can be empty as the prometheus client will
    assume it's localhost
    :type namespace: str
    :param namespace: The prometheus namespace to be used. Defaults to ''.
    :type port: int
    :param port: The Prometheus port to be used. Defaults to 8000.
    :type address: str
    :param address: The Prometheus address to be used. Defaults to ''.
    :type registry: registry
    :param registry: The Prometheus address to be used. Defaults to ''.
    :type registry: :class:`~prometheus_client.core.CollectorRegistry`
    :param registry: A Prometheus collector registry instance.
    """

    def __init__(self, namespace='', port=8000, address='', registry=REGISTRY):
        if False:
            return 10
        self._namespace = namespace
        self._registry = registry
        self._port = int(port)
        self._address = address

    @property
    def registry(self):
        if False:
            return 10
        'Prometheus Collector Registry instance'
        return self._registry

    @property
    def namespace(self):
        if False:
            while True:
                i = 10
        'Prefix to be used with view name'
        return self._namespace

    @property
    def port(self):
        if False:
            print('Hello World!')
        'Port number to listen'
        return self._port

    @property
    def address(self):
        if False:
            print('Hello World!')
        'Endpoint address (default is localhost)'
        return self._address

class Collector(object):
    """Collector represents the Prometheus Collector object"""

    def __init__(self, options=Options(), view_name_to_data_map=None):
        if False:
            print('Hello World!')
        if view_name_to_data_map is None:
            view_name_to_data_map = {}
        self._options = options
        self._registry = options.registry
        self._view_name_to_data_map = view_name_to_data_map
        self._registered_views = {}

    @property
    def options(self):
        if False:
            while True:
                i = 10
        'Options to be used to configure the exporter'
        return self._options

    @property
    def registry(self):
        if False:
            print('Hello World!')
        'Prometheus Collector Registry instance'
        return self._registry

    @property
    def view_name_to_data_map(self):
        if False:
            return 10
        'Map with all view data objects\n        that will be sent to Prometheus\n        '
        return self._view_name_to_data_map

    @property
    def registered_views(self):
        if False:
            i = 10
            return i + 15
        'Map with all registered views'
        return self._registered_views

    def register_view(self, view):
        if False:
            return 10
        'register_view will create the needed structure\n        in order to be able to sent all data to Prometheus\n        '
        v_name = get_view_name(self.options.namespace, view)
        if v_name not in self.registered_views:
            desc = {'name': v_name, 'documentation': view.description, 'labels': list(map(sanitize, view.columns)), 'units': view.measure.unit}
            self.registered_views[v_name] = desc

    def add_view_data(self, view_data):
        if False:
            while True:
                i = 10
        'Add view data object to be sent to server'
        self.register_view(view_data.view)
        v_name = get_view_name(self.options.namespace, view_data.view)
        self.view_name_to_data_map[v_name] = view_data

    def to_metric(self, desc, tag_values, agg_data, metrics_map):
        if False:
            while True:
                i = 10
        'to_metric translate the data that OpenCensus create\n        to Prometheus format, using Prometheus Metric object\n        :type desc: dict\n        :param desc: The map that describes view definition\n        :type tag_values: tuple of :class:\n            `~opencensus.tags.tag_value.TagValue`\n        :param object of opencensus.tags.tag_value.TagValue:\n            TagValue object used as label values\n        :type agg_data: object of :class:\n            `~opencensus.stats.aggregation_data.AggregationData`\n        :param object of opencensus.stats.aggregation_data.AggregationData:\n            Aggregated data that needs to be converted as Prometheus samples\n        :rtype: :class:`~prometheus_client.core.CounterMetricFamily` or\n                :class:`~prometheus_client.core.HistogramMetricFamily` or\n                :class:`~prometheus_client.core.UnknownMetricFamily` or\n                :class:`~prometheus_client.core.GaugeMetricFamily`\n        :returns: A Prometheus metric object\n        '
        metric_name = desc['name']
        metric_description = desc['documentation']
        label_keys = desc['labels']
        metric_units = desc['units']
        assert len(tag_values) == len(label_keys), (tag_values, label_keys)
        tag_values = [tv if tv else '' for tv in tag_values]
        if isinstance(agg_data, aggregation_data_module.CountAggregationData):
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = CounterMetricFamily(name=metric_name, documentation=metric_description, unit=metric_units, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=tag_values, value=agg_data.count_data)
            return metric
        elif isinstance(agg_data, aggregation_data_module.DistributionAggregationData):
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
            metric.add_metric(labels=tag_values, buckets=buckets, sum_value=agg_data.sum)
            return metric
        elif isinstance(agg_data, aggregation_data_module.SumAggregationData):
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = UnknownMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=tag_values, value=agg_data.sum_data)
            return metric
        elif isinstance(agg_data, aggregation_data_module.LastValueAggregationData):
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = GaugeMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=tag_values, value=agg_data.value)
            return metric
        else:
            raise ValueError(f'unsupported aggregation type {type(agg_data)}')

    def collect(self):
        if False:
            while True:
                i = 10
        'Collect fetches the statistics from OpenCensus\n        and delivers them as Prometheus Metrics.\n        Collect is invoked every time a prometheus.Gatherer is run\n        for example when the HTTP endpoint is invoked by Prometheus.\n        '
        metrics_map = {}
        for (v_name, view_data) in self._view_name_to_data_map.copy().items():
            if v_name not in self.registered_views:
                continue
            desc = self.registered_views[v_name]
            for tag_values in view_data.tag_value_aggregation_data_map:
                agg_data = view_data.tag_value_aggregation_data_map[tag_values]
                metric = self.to_metric(desc, tag_values, agg_data, metrics_map)
        for metric in metrics_map.values():
            yield metric

class PrometheusStatsExporter(base_exporter.StatsExporter):
    """Exporter exports stats to Prometheus, users need
        to register the exporter as an HTTP Handler to be
        able to export.
    :type options:
        :class:`~opencensus.ext.prometheus.stats_exporter.Options`
    :param options: An options object with the parameters to instantiate the
                         prometheus exporter.
    :type gatherer: :class:`~prometheus_client.core.CollectorRegistry`
    :param gatherer: A Prometheus collector registry instance.
    :type transport:
        :class:`opencensus.common.transports.sync.SyncTransport` or
        :class:`opencensus.common.transports.async_.AsyncTransport`
    :param transport: An instance of a Transpor to send data with.
    :type collector:
        :class:`~opencensus.ext.prometheus.stats_exporter.Collector`
    :param collector: An instance of the Prometheus Collector object.
    """

    def __init__(self, options, gatherer, transport=sync.SyncTransport, collector=Collector()):
        if False:
            i = 10
            return i + 15
        self._options = options
        self._gatherer = gatherer
        self._collector = collector
        self._transport = transport(self)
        self.serve_http()
        REGISTRY.register(self._collector)

    @property
    def transport(self):
        if False:
            while True:
                i = 10
        'The transport way to be sent data to server\n        (default is sync).\n        '
        return self._transport

    @property
    def collector(self):
        if False:
            i = 10
            return i + 15
        'Collector class instance to be used\n        to communicate with Prometheus\n        '
        return self._collector

    @property
    def gatherer(self):
        if False:
            while True:
                i = 10
        'Prometheus Collector Registry instance'
        return self._gatherer

    @property
    def options(self):
        if False:
            for i in range(10):
                print('nop')
        'Options to be used to configure the exporter'
        return self._options

    def export(self, view_data):
        if False:
            for i in range(10):
                print('nop')
        'export send the data to the transport class\n        in order to be sent to Prometheus in a sync or async way.\n        '
        if view_data is not None:
            self.transport.export(view_data)

    def on_register_view(self, view):
        if False:
            print('Hello World!')
        return NotImplementedError('Not supported by Prometheus')

    def emit(self, view_data):
        if False:
            return 10
        'Emit exports to the Prometheus if view data has one or more rows.\n        Each OpenCensus AggregationData will be converted to\n        corresponding Prometheus Metric: SumData will be converted\n        to Untyped Metric, CountData will be a Counter Metric\n        DistributionData will be a Histogram Metric.\n        '
        for v_data in view_data:
            if v_data.tag_value_aggregation_data_map is None:
                v_data.tag_value_aggregation_data_map = {}
            self.collector.add_view_data(v_data)

    def serve_http(self):
        if False:
            i = 10
            return i + 15
        'serve_http serves the Prometheus endpoint.'
        address = str(self.options.address)
        kwargs = {'addr': address} if address else {}
        start_http_server(port=self.options.port, **kwargs)

def new_stats_exporter(option):
    if False:
        i = 10
        return i + 15
    'new_stats_exporter returns an exporter\n    that exports stats to Prometheus.\n    '
    if option.namespace == '':
        raise ValueError('Namespace can not be empty string.')
    collector = new_collector(option)
    exporter = PrometheusStatsExporter(options=option, gatherer=option.registry, collector=collector)
    return exporter

def new_collector(options):
    if False:
        i = 10
        return i + 15
    'new_collector should be used\n    to create instance of Collector class in order to\n    prevent the usage of constructor directly\n    '
    return Collector(options=options)

def get_view_name(namespace, view):
    if False:
        i = 10
        return i + 15
    'create the name for the view'
    name = ''
    if namespace != '':
        name = namespace + '_'
    return sanitize(name + view.name)
_NON_LETTERS_NOR_DIGITS_RE = re.compile('[^\\w]', re.UNICODE | re.IGNORECASE)

def sanitize(key):
    if False:
        return 10
    "sanitize the given metric name or label according to Prometheus rule.\n    Replace all characters other than [A-Za-z0-9_] with '_'.\n    "
    return _NON_LETTERS_NOR_DIGITS_RE.sub('_', key)