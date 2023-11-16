import datetime
import sacred.optional as opt
from queue import Queue, Empty

class MetricsLogger:
    """MetricsLogger collects metrics measured during experiments.

    MetricsLogger is the (only) part of the Metrics API.
    An instance of the class should be created for the Run class, such that the
    log_scalar_metric method is accessible from running experiments using
    _run.metrics.log_scalar_metric.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._logged_metrics = Queue()
        self._metric_step_counter = {}
        'Remembers the last number of each metric.'

    def log_scalar_metric(self, metric_name, value, step=None):
        if False:
            return 10
        '\n        Add a new measurement.\n\n        The measurement will be processed by the MongoDB observer\n        during a heartbeat event.\n        Other observers are not yet supported.\n\n        :param metric_name: The name of the metric, e.g. training.loss.\n        :param value: The measured value.\n        :param step: The step number (integer), e.g. the iteration number\n                    If not specified, an internal counter for each metric\n                    is used, incremented by one.\n        '
        if opt.has_numpy:
            np = opt.np
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(step, np.generic):
                step = step.item()
        if step is None:
            step = self._metric_step_counter.get(metric_name, -1) + 1
        self._logged_metrics.put(ScalarMetricLogEntry(metric_name, step, datetime.datetime.utcnow(), value))
        self._metric_step_counter[metric_name] = step

    def get_last_metrics(self):
        if False:
            for i in range(10):
                print('nop')
        'Read all measurement events since last call of the method.\n\n        :return List[ScalarMetricLogEntry]\n        '
        read_up_to = self._logged_metrics.qsize()
        messages = []
        for i in range(read_up_to):
            try:
                messages.append(self._logged_metrics.get_nowait())
            except Empty:
                pass
        return messages

class ScalarMetricLogEntry:
    """Container for measurements of scalar metrics.

    There is exactly one ScalarMetricLogEntry per logged scalar metric value.
    """

    def __init__(self, name, step, timestamp, value):
        if False:
            return 10
        self.name = name
        self.step = step
        self.timestamp = timestamp
        self.value = value

def linearize_metrics(logged_metrics):
    if False:
        for i in range(10):
            print('nop')
    '\n    Group metrics by name.\n\n    Takes a list of individual measurements, possibly belonging\n    to different metrics and groups them by name.\n\n    :param logged_metrics: A list of ScalarMetricLogEntries\n    :return: Measured values grouped by the metric name:\n    {"metric_name1": {"steps": [0,1,2], "values": [4, 5, 6],\n    "timestamps": [datetime, datetime, datetime]},\n    "metric_name2": {...}}\n    '
    metrics_by_name = {}
    for metric_entry in logged_metrics:
        if metric_entry.name not in metrics_by_name:
            metrics_by_name[metric_entry.name] = {'steps': [], 'values': [], 'timestamps': [], 'name': metric_entry.name}
        metrics_by_name[metric_entry.name]['steps'].append(metric_entry.step)
        metrics_by_name[metric_entry.name]['values'].append(metric_entry.value)
        metrics_by_name[metric_entry.name]['timestamps'].append(metric_entry.timestamp)
    return metrics_by_name