import logging
from apache_beam.metrics import monitoring_infos
from apache_beam.metrics.execution import MetricKey
from apache_beam.metrics.metric import MetricName
_LOGGER = logging.getLogger(__name__)

def from_monitoring_infos(monitoring_info_list, user_metrics_only=False):
    if False:
        print('Hello World!')
    'Groups MonitoringInfo objects into counters, distributions and gauges.\n\n  Args:\n    monitoring_info_list: An iterable of MonitoringInfo objects.\n    user_metrics_only: If true, includes user metrics only.\n  Returns:\n    A tuple containing three dictionaries: counters, distributions and gauges,\n    respectively. Each dictionary contains (MetricKey, metric result) pairs.\n  '
    counters = {}
    distributions = {}
    gauges = {}
    for mi in monitoring_info_list:
        if user_metrics_only and (not monitoring_infos.is_user_monitoring_info(mi)):
            continue
        try:
            key = _create_metric_key(mi)
        except ValueError as e:
            _LOGGER.debug(str(e))
            continue
        metric_result = monitoring_infos.extract_metric_result_map_value(mi)
        if monitoring_infos.is_counter(mi):
            counters[key] = metric_result
        elif monitoring_infos.is_distribution(mi):
            distributions[key] = metric_result
        elif monitoring_infos.is_gauge(mi):
            gauges[key] = metric_result
    return (counters, distributions, gauges)

def _create_metric_key(monitoring_info):
    if False:
        for i in range(10):
            print('nop')
    step_name = monitoring_infos.get_step_name(monitoring_info)
    (namespace, name) = monitoring_infos.parse_namespace_and_name(monitoring_info)
    return MetricKey(step_name, MetricName(namespace, name))