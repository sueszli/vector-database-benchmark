from typing import Optional
from ray.dashboard.consts import COMPONENT_METRICS_TAG_KEYS

class NullMetric:
    """Mock metric class to be used in case of prometheus_client import error."""

    def set(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def observe(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def inc(self, *args, **kwargs):
        if False:
            return 10
        pass
try:
    from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge

    class DashboardPrometheusMetrics:

        def __init__(self, registry: Optional[CollectorRegistry]=None):
            if False:
                while True:
                    i = 10
            self.registry: CollectorRegistry = registry or CollectorRegistry(auto_describe=True)
            histogram_buckets_s = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20, 40, 60]
            self.metrics_request_duration = Histogram('dashboard_api_requests_duration_seconds', 'Total duration in seconds per endpoint', ('endpoint', 'http_status', 'SessionName', 'Component'), unit='seconds', namespace='ray', registry=self.registry, buckets=histogram_buckets_s)
            self.metrics_request_count = Counter('dashboard_api_requests_count', 'Total requests count per endpoint', ('method', 'endpoint', 'http_status', 'SessionName', 'Component'), unit='requests', namespace='ray', registry=self.registry)
            self.metrics_dashboard_cpu = Gauge('component_cpu', 'Dashboard CPU percentage usage.', tuple(COMPONENT_METRICS_TAG_KEYS), unit='percentage', namespace='ray', registry=self.registry)
            self.metrics_dashboard_mem = Gauge('component_uss', 'USS usage of all components on the node.', tuple(COMPONENT_METRICS_TAG_KEYS), unit='mb', namespace='ray', registry=self.registry)
except ImportError:

    class DashboardPrometheusMetrics(object):

        def __getattr__(self, attr):
            if False:
                i = 10
                return i + 15
            return NullMetric()