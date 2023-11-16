import logging
import threading
import traceback
from typing import Dict, Mapping, Set, Tuple
from prometheus_client.core import Counter, Histogram
from synapse.logging.context import current_context
from synapse.metrics import LaterGauge
logger = logging.getLogger(__name__)
response_count = Counter('synapse_http_server_response_count', '', ['method', 'servlet', 'tag'])
requests_counter = Counter('synapse_http_server_requests_received', '', ['method', 'servlet'])
outgoing_responses_counter = Counter('synapse_http_server_responses', '', ['method', 'code'])
response_timer = Histogram('synapse_http_server_response_time_seconds', 'sec', ['method', 'servlet', 'tag', 'code'])
response_ru_utime = Counter('synapse_http_server_response_ru_utime_seconds', 'sec', ['method', 'servlet', 'tag'])
response_ru_stime = Counter('synapse_http_server_response_ru_stime_seconds', 'sec', ['method', 'servlet', 'tag'])
response_db_txn_count = Counter('synapse_http_server_response_db_txn_count', '', ['method', 'servlet', 'tag'])
response_db_txn_duration = Counter('synapse_http_server_response_db_txn_duration_seconds', '', ['method', 'servlet', 'tag'])
response_db_sched_duration = Counter('synapse_http_server_response_db_sched_duration_seconds', '', ['method', 'servlet', 'tag'])
response_size = Counter('synapse_http_server_response_size', '', ['method', 'servlet', 'tag'])
in_flight_requests_ru_utime = Counter('synapse_http_server_in_flight_requests_ru_utime_seconds', '', ['method', 'servlet'])
in_flight_requests_ru_stime = Counter('synapse_http_server_in_flight_requests_ru_stime_seconds', '', ['method', 'servlet'])
in_flight_requests_db_txn_count = Counter('synapse_http_server_in_flight_requests_db_txn_count', '', ['method', 'servlet'])
in_flight_requests_db_txn_duration = Counter('synapse_http_server_in_flight_requests_db_txn_duration_seconds', '', ['method', 'servlet'])
in_flight_requests_db_sched_duration = Counter('synapse_http_server_in_flight_requests_db_sched_duration_seconds', '', ['method', 'servlet'])
_in_flight_requests: Set['RequestMetrics'] = set()
_in_flight_requests_lock = threading.Lock()

def _get_in_flight_counts() -> Mapping[Tuple[str, ...], int]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a count of all in flight requests by (method, server_name)'
    with _in_flight_requests_lock:
        reqs = list(_in_flight_requests)
    for rm in reqs:
        rm.update_metrics()
    counts: Dict[Tuple[str, ...], int] = {}
    for rm in reqs:
        key = (rm.method, rm.name)
        counts[key] = counts.get(key, 0) + 1
    return counts
LaterGauge('synapse_http_server_in_flight_requests_count', '', ['method', 'servlet'], _get_in_flight_counts)

class RequestMetrics:

    def start(self, time_sec: float, name: str, method: str) -> None:
        if False:
            print('Hello World!')
        self.start_ts = time_sec
        self.start_context = current_context()
        self.name = name
        self.method = method
        if self.start_context:
            self._request_stats = self.start_context.get_resource_usage()
        else:
            logger.error('Tried to start a RequestMetric from the sentinel context.\n%s', ''.join(traceback.format_stack()))
        with _in_flight_requests_lock:
            _in_flight_requests.add(self)

    def stop(self, time_sec: float, response_code: int, sent_bytes: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        with _in_flight_requests_lock:
            _in_flight_requests.discard(self)
        context = current_context()
        tag = ''
        if context:
            tag = context.tag
            if context != self.start_context:
                logger.error('Context have unexpectedly changed %r, %r', context, self.start_context)
                return
        else:
            logger.error('Trying to stop RequestMetrics in the sentinel context.\n%s', ''.join(traceback.format_stack()))
            return
        response_code_str = str(response_code)
        outgoing_responses_counter.labels(self.method, response_code_str).inc()
        response_count.labels(self.method, self.name, tag).inc()
        response_timer.labels(self.method, self.name, tag, response_code_str).observe(time_sec - self.start_ts)
        resource_usage = context.get_resource_usage()
        response_ru_utime.labels(self.method, self.name, tag).inc(resource_usage.ru_utime)
        response_ru_stime.labels(self.method, self.name, tag).inc(resource_usage.ru_stime)
        response_db_txn_count.labels(self.method, self.name, tag).inc(resource_usage.db_txn_count)
        response_db_txn_duration.labels(self.method, self.name, tag).inc(resource_usage.db_txn_duration_sec)
        response_db_sched_duration.labels(self.method, self.name, tag).inc(resource_usage.db_sched_duration_sec)
        response_size.labels(self.method, self.name, tag).inc(sent_bytes)
        self.update_metrics()

    def update_metrics(self) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the in flight metrics with values from this request.'
        if not self.start_context:
            logger.error('Tried to update a RequestMetric from the sentinel context.\n%s', ''.join(traceback.format_stack()))
            return
        new_stats = self.start_context.get_resource_usage()
        diff = new_stats - self._request_stats
        self._request_stats = new_stats
        in_flight_requests_ru_utime.labels(self.method, self.name).inc(max(diff.ru_utime, 0))
        in_flight_requests_ru_stime.labels(self.method, self.name).inc(max(diff.ru_stime, 0))
        in_flight_requests_db_txn_count.labels(self.method, self.name).inc(diff.db_txn_count)
        in_flight_requests_db_txn_duration.labels(self.method, self.name).inc(diff.db_txn_duration_sec)
        in_flight_requests_db_sched_duration.labels(self.method, self.name).inc(diff.db_sched_duration_sec)