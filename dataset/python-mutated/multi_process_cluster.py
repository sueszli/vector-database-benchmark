"""tf.data service test-cluster with local and remote workers."""
import tempfile
from tensorflow.core.protobuf import data_service_pb2
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
_WORKER_SHUTDOWN_QUIET_PERIOD_MS = 100

class _RemoteWorkerProcess(multi_process_lib.Process):
    """Runs a worker server in a new process to simulate a remote worker."""

    def __init__(self, dispatcher_address, port, worker_tags, pipe_writer):
        if False:
            return 10
        super(_RemoteWorkerProcess, self).__init__()
        self._dispatcher_address = dispatcher_address
        self._port = port
        self._worker_tags = worker_tags
        self._pipe_writer = pipe_writer

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.start_worker()

    def start_worker(self):
        if False:
            print('Hello World!')
        self._worker = data_service_test_base.TestWorker(self._dispatcher_address, _WORKER_SHUTDOWN_QUIET_PERIOD_MS, port=self._port, worker_tags=self._worker_tags)
        self._worker.start()
        self._pipe_writer.send(self._worker.worker_address())
        self._worker.join()

class MultiProcessCluster:
    """tf.data service cluster with local and remote workers.

  Represents a cluster with a dispatcher, `num_local_workers` local workers, and
  `num_remote_workers` remote workers. Remote workers run in separate processes.
  This is useful to test reading from local in-process workers. For example:

  ```
  cluster = multi_process_cluster.MultiProcessCluster(
      num_local_workers=1, num_remote_workers=3)
  num_elements = 10
  dataset = self.make_distributed_range_dataset(
      num_elements, cluster, target_workers="LOCAL")
  self.assertDatasetProduces(dataset, list(range(num_elements)))
  ```
  """

    def __init__(self, num_local_workers, num_remote_workers, worker_tags=None, worker_addresses=None, deployment_mode=data_service_pb2.DEPLOYMENT_MODE_COLOCATED):
        if False:
            i = 10
            return i + 15
        self._work_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
        self._deployment_mode = deployment_mode
        self._start_dispatcher(worker_addresses)
        self._start_local_workers(num_local_workers, worker_tags)
        self._start_remote_workers(num_remote_workers, worker_tags)

    def _start_dispatcher(self, worker_addresses, port=0):
        if False:
            for i in range(10):
                print('nop')
        if port == 0:
            port = test_util.pick_unused_port()
        self._dispatcher = server_lib.DispatchServer(service_config_pb2.DispatcherConfig(port=port, protocol='grpc', work_dir=self._work_dir, fault_tolerant_mode=True, worker_addresses=worker_addresses, deployment_mode=self._deployment_mode), start=True)

    def _start_local_workers(self, num_workers, worker_tags=None):
        if False:
            for i in range(10):
                print('nop')
        self._local_workers = []
        for _ in range(num_workers):
            self.start_local_worker(worker_tags)

    def _start_remote_workers(self, num_workers, worker_tags=None):
        if False:
            while True:
                i = 10
        self._remote_workers = []
        for _ in range(num_workers):
            self.start_remote_worker(worker_tags)

    def start_local_worker(self, worker_tags=None):
        if False:
            print('Hello World!')
        worker = data_service_test_base.TestWorker(self.dispatcher_address(), _WORKER_SHUTDOWN_QUIET_PERIOD_MS, port=test_util.pick_unused_port(), worker_tags=worker_tags)
        worker.start()
        self._local_workers.append(worker)

    def start_remote_worker(self, worker_tags=None):
        if False:
            for i in range(10):
                print('nop')
        'Runs a tf.data service worker in a remote process.'
        (pipe_reader, pipe_writer) = multi_process_lib.multiprocessing.Pipe(duplex=False)
        worker_process = _RemoteWorkerProcess(self.dispatcher_address(), port=test_util.pick_unused_port(), worker_tags=worker_tags, pipe_writer=pipe_writer)
        worker_process.start()
        worker_address = pipe_reader.recv()
        self._remote_workers.append((worker_address, worker_process))

    def restart_dispatcher(self):
        if False:
            while True:
                i = 10
        port = int(self.dispatcher_address().split(':')[1])
        self._dispatcher._stop()
        self._start_dispatcher(worker_addresses=self.local_worker_addresses() + self.remote_worker_addresses(), port=port)

    def restart_local_workers(self):
        if False:
            print('Hello World!')
        for worker in self._local_workers:
            worker.restart()

    def dispatcher_address(self):
        if False:
            print('Hello World!')
        return self._dispatcher._address

    def local_worker_addresses(self):
        if False:
            while True:
                i = 10
        return [worker.worker_address() for worker in self._local_workers]

    def remote_worker_addresses(self):
        if False:
            print('Hello World!')
        return [worker_address for (worker_address, _) in self._remote_workers]

    def _stop(self):
        if False:
            i = 10
            return i + 15
        for worker in self._local_workers:
            worker.stop()
        for (_, worker_process) in self._remote_workers:
            worker_process.kill()
        self._dispatcher._stop()

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self._stop()

def test_main():
    if False:
        for i in range(10):
            print('nop')
    'Main function to be called within `__main__` of a test file.'
    multi_process_lib.test_main()