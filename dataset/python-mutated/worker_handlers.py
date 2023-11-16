"""Code for communicating with the Workers."""
import collections
import contextlib
import copy
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import BinaryIO
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import cast
from typing import overload
import grpc
from apache_beam.io import filesystems
from apache_beam.io.filesystems import CompressionTypes
from apache_beam.portability import common_urns
from apache_beam.portability import python_urns
from apache_beam.portability.api import beam_artifact_api_pb2_grpc
from apache_beam.portability.api import beam_fn_api_pb2
from apache_beam.portability.api import beam_fn_api_pb2_grpc
from apache_beam.portability.api import beam_provision_api_pb2
from apache_beam.portability.api import beam_provision_api_pb2_grpc
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.portability.api import endpoints_pb2
from apache_beam.runners.portability import artifact_service
from apache_beam.runners.portability.fn_api_runner.execution import Buffer
from apache_beam.runners.worker import data_plane
from apache_beam.runners.worker import sdk_worker
from apache_beam.runners.worker.channel_factory import GRPCChannelFactory
from apache_beam.runners.worker.log_handler import LOGENTRY_TO_LOG_LEVEL_MAP
from apache_beam.runners.worker.sdk_worker import _Future
from apache_beam.runners.worker.statecache import StateCache
from apache_beam.utils import proto_utils
from apache_beam.utils import thread_pool_executor
from apache_beam.utils.interactive_utils import is_in_notebook
from apache_beam.utils.sentinel import Sentinel
if TYPE_CHECKING:
    from grpc import ServicerContext
    from google.protobuf import message
    from apache_beam.runners.portability.fn_api_runner.fn_runner import ExtendedProvisionInfo
STATE_CACHE_SIZE_MB = 100
MB_TO_BYTES = 1 << 20
DATA_BUFFER_TIME_LIMIT_MS = 1000
_LOGGER = logging.getLogger(__name__)
T = TypeVar('T')
ConstructorFn = Callable[[Union['message.Message', bytes], 'sdk_worker.StateHandler', 'ExtendedProvisionInfo', 'GrpcServer'], 'WorkerHandler']

class ControlConnection(object):
    _uid_counter = 0
    _lock = threading.Lock()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._push_queue = queue.Queue()
        self._input = None
        self._futures_by_id = {}
        self._read_thread = threading.Thread(name='beam_control_read', target=self._read)
        self._state = BeamFnControlServicer.UNSTARTED_STATE

    def _read(self):
        if False:
            print('Hello World!')
        assert self._input is not None
        for data in self._input:
            self._futures_by_id.pop(data.instruction_id).set(data)

    @overload
    def push(self, req):
        if False:
            return 10
        pass

    @overload
    def push(self, req):
        if False:
            while True:
                i = 10
        pass

    def push(self, req):
        if False:
            return 10
        if req is BeamFnControlServicer._DONE_MARKER:
            self._push_queue.put(req)
            return None
        if not req.instruction_id:
            with ControlConnection._lock:
                ControlConnection._uid_counter += 1
                req.instruction_id = 'control_%s' % ControlConnection._uid_counter
        future = ControlFuture(req.instruction_id)
        self._futures_by_id[req.instruction_id] = future
        self._push_queue.put(req)
        return future

    def get_req(self):
        if False:
            return 10
        return self._push_queue.get()

    def set_input(self, input):
        if False:
            while True:
                i = 10
        with ControlConnection._lock:
            if self._input:
                raise RuntimeError('input is already set.')
            self._input = input
            self._read_thread.start()
            self._state = BeamFnControlServicer.STARTED_STATE

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        with ControlConnection._lock:
            if self._state == BeamFnControlServicer.STARTED_STATE:
                self.push(BeamFnControlServicer._DONE_MARKER)
                self._read_thread.join()
            self._state = BeamFnControlServicer.DONE_STATE

    def abort(self, exn):
        if False:
            return 10
        for future in self._futures_by_id.values():
            future.abort(exn)

class BeamFnControlServicer(beam_fn_api_pb2_grpc.BeamFnControlServicer):
    """Implementation of BeamFnControlServicer for clients."""
    UNSTARTED_STATE = 'unstarted'
    STARTED_STATE = 'started'
    DONE_STATE = 'done'
    _DONE_MARKER = Sentinel.sentinel

    def __init__(self, worker_manager):
        if False:
            i = 10
            return i + 15
        self._worker_manager = worker_manager
        self._lock = threading.Lock()
        self._uid_counter = 0
        self._state = self.UNSTARTED_STATE
        self._req_sent = collections.defaultdict(int)
        self._log_req = logging.getLogger().getEffectiveLevel() <= logging.DEBUG
        self._connections_by_worker_id = collections.defaultdict(ControlConnection)

    def get_conn_by_worker_id(self, worker_id):
        if False:
            i = 10
            return i + 15
        with self._lock:
            return self._connections_by_worker_id[worker_id]

    def Control(self, iterator, context):
        if False:
            return 10
        with self._lock:
            if self._state == self.DONE_STATE:
                return
            else:
                self._state = self.STARTED_STATE
        worker_id = dict(context.invocation_metadata()).get('worker_id')
        if not worker_id:
            raise RuntimeError('All workers communicate through gRPC should have worker_id. Received None.')
        control_conn = self.get_conn_by_worker_id(worker_id)
        control_conn.set_input(iterator)
        while True:
            to_push = control_conn.get_req()
            if to_push is self._DONE_MARKER:
                return
            yield to_push
            if self._log_req:
                self._req_sent[to_push.instruction_id] += 1

    def done(self):
        if False:
            i = 10
            return i + 15
        self._state = self.DONE_STATE
        _LOGGER.debug('Runner: Requests sent by runner: %s', [(str(req), cnt) for (req, cnt) in self._req_sent.items()])

    def GetProcessBundleDescriptor(self, id, context=None):
        if False:
            i = 10
            return i + 15
        return self._worker_manager.get_process_bundle_descriptor(id)

class WorkerHandler(object):
    """worker_handler for a worker.

  It provides utilities to start / stop the worker, provision any resources for
  it, as well as provide descriptors for the data, state and logging APIs for
  it.
  """
    _registered_environments = {}
    _worker_id_counter = -1
    _lock = threading.Lock()
    control_conn = None
    data_conn = None

    def __init__(self, control_handler, data_plane_handler, state, provision_info):
        if False:
            i = 10
            return i + 15
        'Initialize a WorkerHandler.\n\n    Args:\n      control_handler:\n      data_plane_handler (data_plane.DataChannel):\n      state:\n      provision_info:\n    '
        self.control_handler = control_handler
        self.data_plane_handler = data_plane_handler
        self.state = state
        self.provision_info = provision_info
        with WorkerHandler._lock:
            WorkerHandler._worker_id_counter += 1
            self.worker_id = 'worker_%s' % WorkerHandler._worker_id_counter

    def close(self):
        if False:
            while True:
                i = 10
        self.stop_worker()

    def start_worker(self):
        if False:
            return 10
        raise NotImplementedError

    def stop_worker(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def control_api_service_descriptor(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def artifact_api_service_descriptor(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def data_api_service_descriptor(self):
        if False:
            return 10
        raise NotImplementedError

    def state_api_service_descriptor(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def logging_api_service_descriptor(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @classmethod
    def register_environment(cls, urn, payload_type):
        if False:
            while True:
                i = 10

        def wrapper(constructor):
            if False:
                while True:
                    i = 10
            cls._registered_environments[urn] = (constructor, payload_type)
            return constructor
        return wrapper

    @classmethod
    def create(cls, environment, state, provision_info, grpc_server):
        if False:
            while True:
                i = 10
        (constructor, payload_type) = cls._registered_environments[environment.urn]
        return constructor(proto_utils.parse_Bytes(environment.payload, payload_type), state, provision_info, grpc_server)

@WorkerHandler.register_environment(python_urns.EMBEDDED_PYTHON, None)
class EmbeddedWorkerHandler(WorkerHandler):
    """An in-memory worker_handler for fn API control, state and data planes."""

    def __init__(self, unused_payload, state, provision_info, worker_manager):
        if False:
            print('Hello World!')
        super().__init__(self, data_plane.InMemoryDataChannel(), state, provision_info)
        self.control_conn = self
        self.data_conn = self.data_plane_handler
        state_cache = StateCache(STATE_CACHE_SIZE_MB * MB_TO_BYTES)
        self.bundle_processor_cache = sdk_worker.BundleProcessorCache(SingletonStateHandlerFactory(sdk_worker.GlobalCachingStateHandler(state_cache, state)), data_plane.InMemoryDataChannelFactory(self.data_plane_handler.inverse()), worker_manager._process_bundle_descriptors)
        self.worker = sdk_worker.SdkWorker(self.bundle_processor_cache)
        self._uid_counter = 0

    def push(self, request):
        if False:
            for i in range(10):
                print('nop')
        if not request.instruction_id:
            self._uid_counter += 1
            request.instruction_id = 'control_%s' % self._uid_counter
        response = self.worker.do_instruction(request)
        return ControlFuture(request.instruction_id, response)

    def start_worker(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def stop_worker(self):
        if False:
            while True:
                i = 10
        self.bundle_processor_cache.shutdown()

    def done(self):
        if False:
            return 10
        pass

    def data_api_service_descriptor(self):
        if False:
            while True:
                i = 10
        return endpoints_pb2.ApiServiceDescriptor(url='fake')

    def state_api_service_descriptor(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    def logging_api_service_descriptor(self):
        if False:
            i = 10
            return i + 15
        return None

class BasicLoggingService(beam_fn_api_pb2_grpc.BeamFnLoggingServicer):

    def Logging(self, log_messages, context=None):
        if False:
            for i in range(10):
                print('nop')
        yield beam_fn_api_pb2.LogControl()
        for log_message in log_messages:
            for log in log_message.log_entries:
                logging.log(LOGENTRY_TO_LOG_LEVEL_MAP[log.severity], str(log))

class BasicProvisionService(beam_provision_api_pb2_grpc.ProvisionServiceServicer):

    def __init__(self, base_info, worker_manager):
        if False:
            return 10
        self._base_info = base_info
        self._worker_manager = worker_manager

    def GetProvisionInfo(self, request, context=None):
        if False:
            print('Hello World!')
        if context:
            worker_id = dict(context.invocation_metadata())['worker_id']
            worker = self._worker_manager.get_worker(worker_id)
            info = copy.copy(worker.provision_info.provision_info)
            info.logging_endpoint.CopyFrom(worker.logging_api_service_descriptor())
            info.artifact_endpoint.CopyFrom(worker.artifact_api_service_descriptor())
            info.control_endpoint.CopyFrom(worker.control_api_service_descriptor())
        else:
            info = self._base_info
        return beam_provision_api_pb2.GetProvisionInfoResponse(info=info)

class GrpcServer(object):
    _DEFAULT_SHUTDOWN_TIMEOUT_SECS = 5

    def __init__(self, state, provision_info, worker_manager):
        if False:
            print('Hello World!')
        options = [('grpc.max_receive_message_length', -1), ('grpc.max_send_message_length', -1), ('grpc.http2.max_pings_without_data', 0), ('grpc.http2.max_ping_strikes', 0)]
        self.state = state
        self.provision_info = provision_info
        self.control_server = grpc.server(thread_pool_executor.shared_unbounded_instance(), options=options)
        self.control_port = self.control_server.add_insecure_port('[::]:0')
        self.control_address = 'localhost:%s' % self.control_port
        self.data_server = grpc.server(thread_pool_executor.shared_unbounded_instance(), options=options)
        self.data_port = self.data_server.add_insecure_port('[::]:0')
        self.state_server = grpc.server(thread_pool_executor.shared_unbounded_instance(), options=options)
        self.state_port = self.state_server.add_insecure_port('[::]:0')
        self.control_handler = BeamFnControlServicer(worker_manager)
        beam_fn_api_pb2_grpc.add_BeamFnControlServicer_to_server(self.control_handler, self.control_server)
        if self.provision_info:
            if self.provision_info.provision_info:
                beam_provision_api_pb2_grpc.add_ProvisionServiceServicer_to_server(BasicProvisionService(self.provision_info.provision_info, worker_manager), self.control_server)

            def open_uncompressed(f):
                if False:
                    return 10
                return filesystems.FileSystems.open(f, compression_type=CompressionTypes.UNCOMPRESSED)
            beam_artifact_api_pb2_grpc.add_ArtifactRetrievalServiceServicer_to_server(artifact_service.ArtifactRetrievalService(file_reader=open_uncompressed), self.control_server)
        self.data_plane_handler = data_plane.BeamFnDataServicer(DATA_BUFFER_TIME_LIMIT_MS)
        beam_fn_api_pb2_grpc.add_BeamFnDataServicer_to_server(self.data_plane_handler, self.data_server)
        beam_fn_api_pb2_grpc.add_BeamFnStateServicer_to_server(GrpcStateServicer(state), self.state_server)
        self.logging_server = grpc.server(thread_pool_executor.shared_unbounded_instance(), options=options)
        self.logging_port = self.logging_server.add_insecure_port('[::]:0')
        beam_fn_api_pb2_grpc.add_BeamFnLoggingServicer_to_server(BasicLoggingService(), self.logging_server)
        _LOGGER.info('starting control server on port %s', self.control_port)
        _LOGGER.info('starting data server on port %s', self.data_port)
        _LOGGER.info('starting state server on port %s', self.state_port)
        _LOGGER.info('starting logging server on port %s', self.logging_port)
        self.logging_server.start()
        self.state_server.start()
        self.data_server.start()
        self.control_server.start()

    def close(self):
        if False:
            i = 10
            return i + 15
        self.control_handler.done()
        to_wait = [self.control_server.stop(self._DEFAULT_SHUTDOWN_TIMEOUT_SECS), self.data_server.stop(self._DEFAULT_SHUTDOWN_TIMEOUT_SECS), self.state_server.stop(self._DEFAULT_SHUTDOWN_TIMEOUT_SECS), self.logging_server.stop(self._DEFAULT_SHUTDOWN_TIMEOUT_SECS)]
        for w in to_wait:
            w.wait()

class GrpcWorkerHandler(WorkerHandler):
    """An grpc based worker_handler for fn API control, state and data planes."""

    def __init__(self, state, provision_info, grpc_server):
        if False:
            return 10
        self._grpc_server = grpc_server
        super().__init__(self._grpc_server.control_handler, self._grpc_server.data_plane_handler, state, provision_info)
        self.state = state
        self.control_address = self.port_from_worker(self._grpc_server.control_port)
        self.control_conn = self._grpc_server.control_handler.get_conn_by_worker_id(self.worker_id)
        self.data_conn = self._grpc_server.data_plane_handler.get_conn_by_worker_id(self.worker_id)

    def control_api_service_descriptor(self):
        if False:
            print('Hello World!')
        return endpoints_pb2.ApiServiceDescriptor(url=self.port_from_worker(self._grpc_server.control_port))

    def artifact_api_service_descriptor(self):
        if False:
            print('Hello World!')
        return endpoints_pb2.ApiServiceDescriptor(url=self.port_from_worker(self._grpc_server.control_port))

    def data_api_service_descriptor(self):
        if False:
            while True:
                i = 10
        return endpoints_pb2.ApiServiceDescriptor(url=self.port_from_worker(self._grpc_server.data_port))

    def state_api_service_descriptor(self):
        if False:
            i = 10
            return i + 15
        return endpoints_pb2.ApiServiceDescriptor(url=self.port_from_worker(self._grpc_server.state_port))

    def logging_api_service_descriptor(self):
        if False:
            while True:
                i = 10
        return endpoints_pb2.ApiServiceDescriptor(url=self.port_from_worker(self._grpc_server.logging_port))

    def close(self):
        if False:
            while True:
                i = 10
        self.control_conn.close()
        self.data_conn.close()
        super().close()

    def port_from_worker(self, port):
        if False:
            i = 10
            return i + 15
        return '%s:%s' % (self.host_from_worker(), port)

    def host_from_worker(self):
        if False:
            return 10
        return 'localhost'

@WorkerHandler.register_environment(common_urns.environments.EXTERNAL.urn, beam_runner_api_pb2.ExternalPayload)
class ExternalWorkerHandler(GrpcWorkerHandler):

    def __init__(self, external_payload, state, provision_info, grpc_server):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(state, provision_info, grpc_server)
        self._external_payload = external_payload

    def start_worker(self):
        if False:
            for i in range(10):
                print('nop')
        _LOGGER.info('Requesting worker at %s', self._external_payload.endpoint.url)
        stub = beam_fn_api_pb2_grpc.BeamFnExternalWorkerPoolStub(GRPCChannelFactory.insecure_channel(self._external_payload.endpoint.url))
        control_descriptor = endpoints_pb2.ApiServiceDescriptor(url=self.control_address)
        response = stub.StartWorker(beam_fn_api_pb2.StartWorkerRequest(worker_id=self.worker_id, control_endpoint=control_descriptor, artifact_endpoint=control_descriptor, provision_endpoint=control_descriptor, logging_endpoint=self.logging_api_service_descriptor(), params=self._external_payload.params))
        if response.error:
            raise RuntimeError('Error starting worker: %s' % response.error)

    def stop_worker(self):
        if False:
            return 10
        pass

    def host_from_worker(self):
        if False:
            while True:
                i = 10
        if sys.platform in ['win32', 'darwin']:
            return 'localhost'
        import socket
        return socket.getfqdn()

@WorkerHandler.register_environment(python_urns.EMBEDDED_PYTHON_GRPC, bytes)
class EmbeddedGrpcWorkerHandler(GrpcWorkerHandler):

    def __init__(self, payload, state, provision_info, grpc_server):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(state, provision_info, grpc_server)
        from apache_beam.transforms.environments import EmbeddedPythonGrpcEnvironment
        config = EmbeddedPythonGrpcEnvironment.parse_config(payload.decode('utf-8'))
        self._state_cache_size = (config.get('state_cache_size') or STATE_CACHE_SIZE_MB) << 20
        self._data_buffer_time_limit_ms = config.get('data_buffer_time_limit_ms') or DATA_BUFFER_TIME_LIMIT_MS

    def start_worker(self):
        if False:
            i = 10
            return i + 15
        self.worker = sdk_worker.SdkHarness(self.control_address, state_cache_size=self._state_cache_size, data_buffer_time_limit_ms=self._data_buffer_time_limit_ms, worker_id=self.worker_id)
        self.worker_thread = threading.Thread(name='run_worker', target=self.worker.run)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop_worker(self):
        if False:
            while True:
                i = 10
        self.worker_thread.join()
SUBPROCESS_LOCK = threading.Lock()

@WorkerHandler.register_environment(python_urns.SUBPROCESS_SDK, bytes)
class SubprocessSdkWorkerHandler(GrpcWorkerHandler):

    def __init__(self, worker_command_line, state, provision_info, grpc_server):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(state, provision_info, grpc_server)
        self._worker_command_line = worker_command_line

    def start_worker(self):
        if False:
            while True:
                i = 10
        from apache_beam.runners.portability import local_job_service
        self.worker = local_job_service.SubprocessSdkWorker(self._worker_command_line, self.control_address, self.provision_info, self.worker_id)
        self.worker_thread = threading.Thread(name='run_worker', target=self.worker.run)
        self.worker_thread.start()

    def stop_worker(self):
        if False:
            i = 10
            return i + 15
        self.worker_thread.join()

@WorkerHandler.register_environment(common_urns.environments.DOCKER.urn, beam_runner_api_pb2.DockerPayload)
class DockerSdkWorkerHandler(GrpcWorkerHandler):

    def __init__(self, payload, state, provision_info, grpc_server):
        if False:
            return 10
        super().__init__(state, provision_info, grpc_server)
        self._container_image = payload.container_image
        self._container_id = None

    def host_from_worker(self):
        if False:
            for i in range(10):
                print('nop')
        if sys.platform == 'darwin':
            return 'host.docker.internal'
        if sys.platform == 'linux' and is_in_notebook():
            import socket
            return socket.gethostbyname(socket.getfqdn())
        return super().host_from_worker()

    def start_worker(self):
        if False:
            for i in range(10):
                print('nop')
        credential_options = []
        try:
            import google.auth
        except ImportError:
            pass
        else:
            from google.auth import environment_vars
            from google.auth import _cloud_sdk
            gcloud_cred_file = os.environ.get(environment_vars.CREDENTIALS, _cloud_sdk.get_application_default_credentials_path())
            if os.path.exists(gcloud_cred_file):
                docker_cred_file = '/docker_cred_file.json'
                credential_options.extend(['--mount', f'type=bind,source={gcloud_cred_file},target={docker_cred_file}', '--env', f'{environment_vars.CREDENTIALS}={docker_cred_file}'])
        with SUBPROCESS_LOCK:
            try:
                _LOGGER.info('Attempting to pull image %s', self._container_image)
                subprocess.check_call(['docker', 'pull', self._container_image])
            except Exception:
                _LOGGER.info('Unable to pull image %s, defaulting to local image if it exists' % self._container_image)
            self._container_id = subprocess.check_output(['docker', 'run', '-d', '--network=host'] + credential_options + [self._container_image, '--id=%s' % self.worker_id, '--logging_endpoint=%s' % self.logging_api_service_descriptor().url, '--control_endpoint=%s' % self.control_address, '--artifact_endpoint=%s' % self.control_address, '--provision_endpoint=%s' % self.control_address]).strip()
            assert self._container_id is not None
            while True:
                status = subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Status}}', self._container_id]).strip()
                _LOGGER.info('Waiting for docker to start up. Current status is %s' % status.decode('utf-8'))
                if status == b'running':
                    _LOGGER.info('Docker container is running. container_id = %s, worker_id = %s', self._container_id, self.worker_id)
                    break
                elif status in (b'dead', b'exited'):
                    subprocess.call(['docker', 'container', 'logs', self._container_id])
                    raise RuntimeError('SDK failed to start. Final status is %s' % status.decode('utf-8'))
            time.sleep(1)
        self._done = False
        t = threading.Thread(target=self.watch_container)
        t.daemon = True
        t.start()

    def watch_container(self):
        if False:
            while True:
                i = 10
        while not self._done:
            assert self._container_id is not None
            status = subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Status}}', self._container_id]).strip()
            if status != b'running':
                if not self._done:
                    logs = subprocess.check_output(['docker', 'container', 'logs', '--tail', '10', self._container_id], stderr=subprocess.STDOUT)
                    _LOGGER.info(logs)
                    self.control_conn.abort(RuntimeError('SDK exited unexpectedly. Final status is %s. Final log line is %s' % (status.decode('utf-8'), logs.decode('utf-8').strip().rsplit('\n', maxsplit=1)[-1])))
            time.sleep(5)

    def stop_worker(self):
        if False:
            i = 10
            return i + 15
        self._done = True
        if self._container_id:
            with SUBPROCESS_LOCK:
                subprocess.call(['docker', 'kill', self._container_id])

class WorkerHandlerManager(object):
    """
  Manages creation of ``WorkerHandler``s.

  Caches ``WorkerHandler``s based on environment id.
  """

    def __init__(self, environments, job_provision_info):
        if False:
            print('Hello World!')
        self._environments = environments
        self._job_provision_info = job_provision_info
        self._cached_handlers = collections.defaultdict(list)
        self._workers_by_id = {}
        self.state_servicer = StateServicer()
        self._grpc_server = None
        self._process_bundle_descriptors = {}

    def register_process_bundle_descriptor(self, process_bundle_descriptor):
        if False:
            while True:
                i = 10
        self._process_bundle_descriptors[process_bundle_descriptor.id] = process_bundle_descriptor

    def get_process_bundle_descriptor(self, request):
        if False:
            for i in range(10):
                print('nop')
        return self._process_bundle_descriptors[request.process_bundle_descriptor_id]

    def get_worker_handlers(self, environment_id, num_workers):
        if False:
            while True:
                i = 10
        if environment_id is None:
            environment_id = next(iter(self._environments.keys()))
        environment = self._environments[environment_id]
        if environment.urn == python_urns.EMBEDDED_PYTHON:
            grpc_server = cast(GrpcServer, self)
        elif self._grpc_server is None:
            self._grpc_server = GrpcServer(self.state_servicer, self._job_provision_info, self)
            grpc_server = self._grpc_server
        else:
            grpc_server = self._grpc_server
        worker_handler_list = self._cached_handlers[environment_id]
        if len(worker_handler_list) < num_workers:
            for _ in range(len(worker_handler_list), num_workers):
                worker_handler = WorkerHandler.create(environment, self.state_servicer, self._job_provision_info.for_environment(environment), grpc_server)
                _LOGGER.info('Created Worker handler %s for environment %s (%s, %r)', worker_handler, environment_id, environment.urn, environment.payload)
                self._cached_handlers[environment_id].append(worker_handler)
                self._workers_by_id[worker_handler.worker_id] = worker_handler
                worker_handler.start_worker()
        return self._cached_handlers[environment_id][:num_workers]

    def close_all(self):
        if False:
            for i in range(10):
                print('nop')
        for worker_handler_list in self._cached_handlers.values():
            for worker_handler in set(worker_handler_list):
                try:
                    worker_handler.close()
                except Exception:
                    _LOGGER.error('Error closing worker_handler %s' % worker_handler, exc_info=True)
        self._cached_handlers = {}
        self._workers_by_id = {}
        if self._grpc_server is not None:
            self._grpc_server.close()
            self._grpc_server = None

    def get_worker(self, worker_id):
        if False:
            for i in range(10):
                print('nop')
        return self._workers_by_id[worker_id]

class StateServicer(beam_fn_api_pb2_grpc.BeamFnStateServicer, sdk_worker.StateHandler):

    class CopyOnWriteState(object):

        def __init__(self, underlying):
            if False:
                print('Hello World!')
            self._underlying = underlying
            self._overlay = {}

        def __getitem__(self, key):
            if False:
                for i in range(10):
                    print('nop')
            if key in self._overlay:
                return self._overlay[key]
            else:
                return StateServicer.CopyOnWriteList(self._underlying, self._overlay, key)

        def __delitem__(self, key):
            if False:
                for i in range(10):
                    print('nop')
            self._overlay[key] = []

        def commit(self):
            if False:
                i = 10
                return i + 15
            self._underlying.update(self._overlay)
            return self._underlying

    class CopyOnWriteList(object):

        def __init__(self, underlying, overlay, key):
            if False:
                return 10
            self._underlying = underlying
            self._overlay = overlay
            self._key = key

        def __iter__(self):
            if False:
                print('Hello World!')
            if self._key in self._overlay:
                return iter(self._overlay[self._key])
            else:
                return iter(self._underlying[self._key])

        def append(self, item):
            if False:
                print('Hello World!')
            if self._key not in self._overlay:
                self._overlay[self._key] = list(self._underlying[self._key])
            self._overlay[self._key].append(item)

        def extend(self, other: Buffer) -> None:
            if False:
                print('Hello World!')
            raise NotImplementedError()
    StateType = Union[CopyOnWriteState, DefaultDict[bytes, Buffer]]

    def __init__(self):
        if False:
            return 10
        self._lock = threading.Lock()
        self._state = collections.defaultdict(list)
        self._checkpoint = None
        self._use_continuation_tokens = False
        self._continuations = {}

    def checkpoint(self):
        if False:
            print('Hello World!')
        assert self._checkpoint is None and (not isinstance(self._state, StateServicer.CopyOnWriteState))
        self._checkpoint = self._state
        self._state = StateServicer.CopyOnWriteState(self._state)

    def commit(self):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(self._state, StateServicer.CopyOnWriteState) and isinstance(self._checkpoint, StateServicer.CopyOnWriteState)
        self._state.commit()
        self._state = self._checkpoint.commit()
        self._checkpoint = None

    def restore(self):
        if False:
            while True:
                i = 10
        assert self._checkpoint is not None
        self._state = self._checkpoint
        self._checkpoint = None

    @contextlib.contextmanager
    def process_instruction_id(self, unused_instruction_id):
        if False:
            while True:
                i = 10
        yield

    def get_raw(self, state_key, continuation_token=None):
        if False:
            while True:
                i = 10
        with self._lock:
            full_state = self._state[self._to_key(state_key)]
            if self._use_continuation_tokens:
                if not continuation_token:
                    token_base = b'token_%x' % len(self._continuations)
                    self._continuations[token_base] = tuple(full_state)
                    return (b'', b'%s:0' % token_base)
                else:
                    (token_base, index) = continuation_token.split(b':')
                    ix = int(index)
                    full_state_cont = self._continuations[token_base]
                    if ix == len(full_state_cont):
                        return (b'', None)
                    else:
                        return (full_state_cont[ix], b'%s:%d' % (token_base, ix + 1))
            else:
                assert not continuation_token
                return (b''.join(full_state), None)

    def append_raw(self, state_key, data):
        if False:
            i = 10
            return i + 15
        with self._lock:
            self._state[self._to_key(state_key)].append(data)
        return _Future.done()

    def clear(self, state_key):
        if False:
            return 10
        with self._lock:
            try:
                del self._state[self._to_key(state_key)]
            except KeyError:
                pass
        return _Future.done()

    def done(self):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def _to_key(state_key):
        if False:
            i = 10
            return i + 15
        return state_key.SerializeToString()

class GrpcStateServicer(beam_fn_api_pb2_grpc.BeamFnStateServicer):

    def __init__(self, state):
        if False:
            i = 10
            return i + 15
        self._state = state

    def State(self, request_stream, context=None):
        if False:
            while True:
                i = 10
        for request in request_stream:
            request_type = request.WhichOneof('request')
            if request_type == 'get':
                (data, continuation_token) = self._state.get_raw(request.state_key, request.get.continuation_token)
                yield beam_fn_api_pb2.StateResponse(id=request.id, get=beam_fn_api_pb2.StateGetResponse(data=data, continuation_token=continuation_token))
            elif request_type == 'append':
                self._state.append_raw(request.state_key, request.append.data)
                yield beam_fn_api_pb2.StateResponse(id=request.id, append=beam_fn_api_pb2.StateAppendResponse())
            elif request_type == 'clear':
                self._state.clear(request.state_key)
                yield beam_fn_api_pb2.StateResponse(id=request.id, clear=beam_fn_api_pb2.StateClearResponse())
            else:
                raise NotImplementedError('Unknown state request: %s' % request_type)

class SingletonStateHandlerFactory(sdk_worker.StateHandlerFactory):
    """A singleton cache for a StateServicer."""

    def __init__(self, state_handler):
        if False:
            print('Hello World!')
        self._state_handler = state_handler

    def create_state_handler(self, api_service_descriptor):
        if False:
            for i in range(10):
                print('nop')
        'Returns the singleton state handler.'
        return self._state_handler

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        'Does nothing.'
        pass

class ControlFuture(object):

    def __init__(self, instruction_id, response=None):
        if False:
            for i in range(10):
                print('nop')
        self.instruction_id = instruction_id
        self._response = response
        if response is None:
            self._condition = threading.Condition()
        self._exception = None

    def is_done(self):
        if False:
            for i in range(10):
                print('nop')
        return self._response is not None

    def set(self, response):
        if False:
            for i in range(10):
                print('nop')
        with self._condition:
            self._response = response
            self._condition.notify_all()

    def get(self, timeout=None):
        if False:
            while True:
                i = 10
        if not self._response and (not self._exception):
            with self._condition:
                if not self._response and (not self._exception):
                    self._condition.wait(timeout)
        if self._exception:
            raise self._exception
        else:
            assert self._response is not None
            return self._response

    def abort(self, exception):
        if False:
            i = 10
            return i + 15
        with self._condition:
            self._exception = exception
            self._condition.notify_all()