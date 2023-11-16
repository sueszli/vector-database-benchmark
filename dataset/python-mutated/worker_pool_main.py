"""
Worker pool entry point.

The worker pool exposes an RPC service that is used with EXTERNAL
environment to start and stop the SDK workers.

The worker pool uses child processes for parallelism; threads are
subject to the GIL and not sufficient.

This entry point is used by the Python SDK container in worker pool mode.
"""
import argparse
import atexit
import logging
import subprocess
import sys
import threading
import time
import traceback
from typing import Dict
from typing import Optional
from typing import Tuple
import grpc
from apache_beam.portability.api import beam_fn_api_pb2
from apache_beam.portability.api import beam_fn_api_pb2_grpc
from apache_beam.runners.worker import sdk_worker
from apache_beam.utils import thread_pool_executor
_LOGGER = logging.getLogger(__name__)

def kill_process_gracefully(proc, timeout=10):
    if False:
        return 10
    '\n  Kill a worker process gracefully by sending a SIGTERM and waiting for\n  it to finish. A SIGKILL will be sent if the process has not finished\n  after ``timeout`` seconds.\n  '

    def _kill():
        if False:
            i = 10
            return i + 15
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _LOGGER.warning('Worker process did not respond, killing it.')
            proc.kill()
            proc.wait()
    kill_thread = threading.Thread(target=_kill)
    kill_thread.start()
    kill_thread.join()

class BeamFnExternalWorkerPoolServicer(beam_fn_api_pb2_grpc.BeamFnExternalWorkerPoolServicer):

    def __init__(self, use_process=False, container_executable=None, state_cache_size=0, data_buffer_time_limit_ms=0):
        if False:
            for i in range(10):
                print('nop')
        self._use_process = use_process
        self._container_executable = container_executable
        self._state_cache_size = state_cache_size
        self._data_buffer_time_limit_ms = data_buffer_time_limit_ms
        self._worker_processes = {}

    @classmethod
    def start(cls, use_process=False, port=0, state_cache_size=0, data_buffer_time_limit_ms=-1, container_executable=None):
        if False:
            print('Hello World!')
        options = [('grpc.http2.max_pings_without_data', 0), ('grpc.http2.max_ping_strikes', 0)]
        worker_server = grpc.server(thread_pool_executor.shared_unbounded_instance(), options=options)
        worker_address = 'localhost:%s' % worker_server.add_insecure_port('[::]:%s' % port)
        worker_pool = cls(use_process=use_process, container_executable=container_executable, state_cache_size=state_cache_size, data_buffer_time_limit_ms=data_buffer_time_limit_ms)
        beam_fn_api_pb2_grpc.add_BeamFnExternalWorkerPoolServicer_to_server(worker_pool, worker_server)
        worker_server.start()
        _LOGGER.info('Listening for workers at %s', worker_address)

        def kill_worker_processes():
            if False:
                while True:
                    i = 10
            for worker_process in worker_pool._worker_processes.values():
                kill_process_gracefully(worker_process)
        atexit.register(kill_worker_processes)
        return (worker_address, worker_server)

    def StartWorker(self, start_worker_request, unused_context):
        if False:
            print('Hello World!')
        try:
            if self._use_process:
                command = ['python', '-c', 'from apache_beam.runners.worker.sdk_worker import SdkHarness; SdkHarness("%s",worker_id="%s",state_cache_size=%d,data_buffer_time_limit_ms=%d).run()' % (start_worker_request.control_endpoint.url, start_worker_request.worker_id, self._state_cache_size, self._data_buffer_time_limit_ms)]
                if self._container_executable:
                    command = [self._container_executable, '--id=%s' % start_worker_request.worker_id, '--logging_endpoint=%s' % start_worker_request.logging_endpoint.url, '--artifact_endpoint=%s' % start_worker_request.artifact_endpoint.url, '--provision_endpoint=%s' % start_worker_request.provision_endpoint.url, '--control_endpoint=%s' % start_worker_request.control_endpoint.url]
                _LOGGER.warning('Starting worker with command %s' % command)
                worker_process = subprocess.Popen(command, stdout=subprocess.PIPE, close_fds=True)
                self._worker_processes[start_worker_request.worker_id] = worker_process
            else:
                worker = sdk_worker.SdkHarness(start_worker_request.control_endpoint.url, worker_id=start_worker_request.worker_id, state_cache_size=self._state_cache_size, data_buffer_time_limit_ms=self._data_buffer_time_limit_ms)
                worker_thread = threading.Thread(name='run_worker_%s' % start_worker_request.worker_id, target=worker.run)
                worker_thread.daemon = True
                worker_thread.start()
            return beam_fn_api_pb2.StartWorkerResponse()
        except Exception:
            return beam_fn_api_pb2.StartWorkerResponse(error=traceback.format_exc())

    def StopWorker(self, stop_worker_request, unused_context):
        if False:
            i = 10
            return i + 15
        worker_process = self._worker_processes.pop(stop_worker_request.worker_id, None)
        if worker_process:
            _LOGGER.info('Stopping worker %s' % stop_worker_request.worker_id)
            kill_process_gracefully(worker_process)
        return beam_fn_api_pb2.StopWorkerResponse()

def main(argv=None):
    if False:
        print('Hello World!')
    'Entry point for worker pool service for external environments.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--container_executable', type=str, default=None, help='Executable that implements the Beam SDK container contract.')
    parser.add_argument('--service_port', type=int, required=True, dest='port', help='Bind port for the worker pool service.')
    (args, _) = parser.parse_known_args(argv)
    (address, server) = BeamFnExternalWorkerPoolServicer.start(use_process=True, **vars(args))
    logging.getLogger().setLevel(logging.INFO)
    _LOGGER.info('Started worker pool servicer at port: %s with executable: %s', address, args.container_executable)
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)
if __name__ == '__main__':
    main(sys.argv)