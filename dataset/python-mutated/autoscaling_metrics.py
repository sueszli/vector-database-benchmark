import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional
from ray.serve._private.constants import SERVE_LOGGER_NAME
logger = logging.getLogger(SERVE_LOGGER_NAME)

@dataclass(order=True)
class TimeStampedValue:
    timestamp: float
    value: float = field(compare=False)

class InMemoryMetricsStore:
    """A very simple, in memory time series database"""

    def __init__(self):
        if False:
            return 10
        self.data: DefaultDict[str, List[TimeStampedValue]] = defaultdict(list)

    def add_metrics_point(self, data_points: Dict[str, float], timestamp: float):
        if False:
            while True:
                i = 10
        'Push new data points to the store.\n\n        Args:\n            data_points: dictionary containing the metrics values. The\n              key should be a string that uniquely identifies this time series\n              and to be used to perform aggregation.\n            timestamp: the unix epoch timestamp the metrics are\n              collected at.\n        '
        for (name, value) in data_points.items():
            bisect.insort(a=self.data[name], x=TimeStampedValue(timestamp, value))

    def _get_datapoints(self, key: str, window_start_timestamp_s: float) -> List[float]:
        if False:
            i = 10
            return i + 15
        'Get all data points given key after window_start_timestamp_s'
        datapoints = self.data[key]
        idx = bisect.bisect(a=datapoints, x=TimeStampedValue(timestamp=window_start_timestamp_s, value=0))
        return datapoints[idx:]

    def window_average(self, key: str, window_start_timestamp_s: float, do_compact: bool=True) -> Optional[float]:
        if False:
            return 10
        "Perform a window average operation for metric `key`\n\n        Args:\n            key: the metric name.\n            window_start_timestamp_s: the unix epoch timestamp for the\n              start of the window. The computed average will use all datapoints\n              from this timestamp until now.\n            do_compact: whether or not to delete the datapoints that's\n              before `window_start_timestamp_s` to save memory. Default is\n              true.\n        Returns:\n            The average of all the datapoints for the key on and after time\n            window_start_timestamp_s, or None if there are no such points.\n        "
        points_after_idx = self._get_datapoints(key, window_start_timestamp_s)
        if do_compact:
            self.data[key] = points_after_idx
        if len(points_after_idx) == 0:
            return
        return sum((point.value for point in points_after_idx)) / len(points_after_idx)

    def max(self, key: str, window_start_timestamp_s: float, do_compact: bool=True):
        if False:
            return 10
        "Perform a max operation for metric `key`.\n\n        Args:\n            key: the metric name.\n            window_start_timestamp_s: the unix epoch timestamp for the\n              start of the window. The computed average will use all datapoints\n              from this timestamp until now.\n            do_compact: whether or not to delete the datapoints that's\n              before `window_start_timestamp_s` to save memory. Default is\n              true.\n        Returns:\n            Max value of the data points for the key on and after time\n            window_start_timestamp_s, or None if there are no such points.\n        "
        points_after_idx = self._get_datapoints(key, window_start_timestamp_s)
        if do_compact:
            self.data[key] = points_after_idx
        return max((point.value for point in points_after_idx), default=None)