import queue
import threading
from typing import Tuple, Union, List, Callable
from sslyze import ServerScanRequest, ServerTlsProbingResult
from sslyze.errors import ConnectionToServerFailed
from sslyze.server_connectivity import check_connectivity_to_server
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias
_ServerConnectivityTestingResult = Union[ServerTlsProbingResult, ConnectionToServerFailed]
ServerConnectivityTestCompletedCallback = Callable[[ServerScanRequest, ServerTlsProbingResult], None]
ServerConnectivityTestErrorCallback = Callable[[ServerScanRequest, ConnectionToServerFailed], None]

class _NoMoreWorkSentinel:
    pass
_ScanRequestsQueueType: TypeAlias = 'queue.Queue[Union[_NoMoreWorkSentinel, ServerScanRequest]]'
_ResultsQueueType: TypeAlias = 'queue.Queue[Union[_NoMoreWorkSentinel, Tuple[ServerScanRequest, _ServerConnectivityTestingResult]]]'

class MassConnectivityTester:
    """Wrapper around check_connectivity_to_server() for concurrent/mass testing."""

    def __init__(self, concurrent_server_scans_count: int) -> None:
        if False:
            print('Hello World!')
        if concurrent_server_scans_count < 1:
            raise ValueError()
        self._scan_requests_queue: _ScanRequestsQueueType = queue.Queue()
        self._results_queue: _ResultsQueueType = queue.Queue()
        self._all_worker_threads = [_ConnectivityTesterThread(scan_requests_queue_in=self._scan_requests_queue, results_queue_out=self._results_queue) for _ in range(concurrent_server_scans_count)]
        self.has_started_work = False

    def start_work(self, server_scan_requests: List[ServerScanRequest]) -> None:
        if False:
            print('Hello World!')
        assert not self.has_started_work
        self.has_started_work = True
        for worker_thread in self._all_worker_threads:
            worker_thread.start()
        for request in server_scan_requests:
            self._scan_requests_queue.put(request)
        for _ in self._all_worker_threads:
            self._scan_requests_queue.put(_NoMoreWorkSentinel())

    def wait_until_all_work_was_processed(self, server_connectivity_test_completed_callback: ServerConnectivityTestCompletedCallback, server_connectivity_test_error_callback: ServerConnectivityTestErrorCallback) -> None:
        if False:
            for i in range(10):
                print('nop')
        shutdown_workers_count = 0
        while shutdown_workers_count < len(self._all_worker_threads):
            result = self._results_queue.get(block=True)
            if isinstance(result, _NoMoreWorkSentinel):
                shutdown_workers_count += 1
            else:
                (scan_request, connectivity_result) = result
                if isinstance(connectivity_result, ConnectionToServerFailed):
                    server_connectivity_test_error_callback(scan_request, connectivity_result)
                elif isinstance(connectivity_result, ServerTlsProbingResult):
                    server_connectivity_test_completed_callback(scan_request, connectivity_result)
                else:
                    raise TypeError()
            self._results_queue.task_done()
        self._scan_requests_queue.join()
        self._results_queue.join()
        for worker_thread in self._all_worker_threads:
            worker_thread.join()

class _ConnectivityTesterThread(threading.Thread):

    def __init__(self, scan_requests_queue_in: _ScanRequestsQueueType, results_queue_out: _ResultsQueueType):
        if False:
            while True:
                i = 10
        super().__init__()
        self._scan_requests_queue_in = scan_requests_queue_in
        self._results_queue_out = results_queue_out
        self.daemon = True

    def run(self) -> None:
        if False:
            print('Hello World!')
        while True:
            scan_request = self._scan_requests_queue_in.get(block=True)
            if isinstance(scan_request, _NoMoreWorkSentinel):
                self._results_queue_out.put(_NoMoreWorkSentinel())
                self._scan_requests_queue_in.task_done()
                return
            try:
                tls_probing_result = check_connectivity_to_server(server_location=scan_request.server_location, network_configuration=scan_request.network_configuration)
                self._results_queue_out.put((scan_request, tls_probing_result))
            except ConnectionToServerFailed as e:
                self._results_queue_out.put((scan_request, e))
            self._scan_requests_queue_in.task_done()