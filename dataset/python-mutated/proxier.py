import atexit
import json
import logging
import socket
import sys
import time
import traceback
from concurrent import futures
from dataclasses import dataclass
from itertools import chain
import urllib
from threading import Event, Lock, RLock, Thread
from typing import Callable, Dict, List, Optional, Tuple
import grpc
import psutil
import ray
import ray.core.generated.agent_manager_pb2 as agent_manager_pb2
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
import ray.core.generated.runtime_env_agent_pb2 as runtime_env_agent_pb2
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.parameter import RayParams
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.services import ProcessInfo, start_ray_client_server
from ray._private.tls_utils import add_port_to_grpc_server
from ray._private.utils import detect_fate_sharing_support
from ray.cloudpickle.compat import pickle
from ray.job_config import JobConfig
from ray.util.client.common import CLIENT_SERVER_MAX_THREADS, GRPC_OPTIONS, ClientServerHandle, _get_client_id_from_context, _propagate_error_in_context
from ray.util.client.server.dataservicer import _get_reconnecting_from_context
logger = logging.getLogger(__name__)
CHECK_PROCESS_INTERVAL_S = 30
MIN_SPECIFIC_SERVER_PORT = 23000
MAX_SPECIFIC_SERVER_PORT = 24000
CHECK_CHANNEL_TIMEOUT_S = 30
LOGSTREAM_RETRIES = 5
LOGSTREAM_RETRY_INTERVAL_SEC = 2

@dataclass
class SpecificServer:
    port: int
    process_handle_future: futures.Future
    channel: 'grpc._channel.Channel'

    def is_ready(self) -> bool:
        if False:
            i = 10
            return i + 15
        "Check if the server is ready or not (doesn't block)."
        return self.process_handle_future.done()

    def wait_ready(self, timeout: Optional[float]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Wait for the server to actually start up.\n        '
        res = self.process_handle_future.result(timeout=timeout)
        if res is None:
            raise RuntimeError('Server startup failed.')

    def poll(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        'Check if the process has exited.'
        try:
            proc = self.process_handle_future.result(timeout=0.1)
            if proc is not None:
                return proc.process.poll()
        except futures.TimeoutError:
            return

    def kill(self) -> None:
        if False:
            while True:
                i = 10
        'Try to send a KILL signal to the process.'
        try:
            proc = self.process_handle_future.result(timeout=0.1)
            if proc is not None:
                proc.process.kill()
        except futures.TimeoutError:
            pass

    def set_result(self, proc: Optional[ProcessInfo]) -> None:
        if False:
            while True:
                i = 10
        'Set the result of the internal future if it is currently unset.'
        if not self.is_ready():
            self.process_handle_future.set_result(proc)

def _match_running_client_server(command: List[str]) -> bool:
    if False:
        return 10
    '\n    Detects if the main process in the given command is the RayClient Server.\n    This works by ensuring that the the first three arguments are similar to:\n        <python> -m ray.util.client.server\n    '
    flattened = ' '.join(command)
    rejoined = flattened.split()
    if len(rejoined) < 3:
        return False
    return rejoined[1:3] == ['-m', 'ray.util.client.server']

class ProxyManager:

    def __init__(self, address: Optional[str], runtime_env_agent_address: str, *, session_dir: Optional[str]=None, redis_password: Optional[str]=None, runtime_env_agent_port: int=0):
        if False:
            for i in range(10):
                print('nop')
        self.servers: Dict[str, SpecificServer] = dict()
        self.server_lock = RLock()
        self._address = address
        self._redis_password = redis_password
        self._free_ports: List[int] = list(range(MIN_SPECIFIC_SERVER_PORT, MAX_SPECIFIC_SERVER_PORT))
        self._runtime_env_agent_address = runtime_env_agent_address
        self._check_thread = Thread(target=self._check_processes, daemon=True)
        self._check_thread.start()
        self.fate_share = bool(detect_fate_sharing_support())
        self._node: Optional[ray._private.node.Node] = None
        atexit.register(self._cleanup)

    def _get_unused_port(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Search for a port in _free_ports that is unused.\n        '
        with self.server_lock:
            num_ports = len(self._free_ports)
            for _ in range(num_ports):
                port = self._free_ports.pop(0)
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.bind(('', port))
                except OSError:
                    self._free_ports.append(port)
                    continue
                finally:
                    s.close()
                return port
        raise RuntimeError('Unable to succeed in selecting a random port.')

    @property
    def address(self) -> str:
        if False:
            return 10
        '\n        Returns the provided Ray bootstrap address, or creates a new cluster.\n        '
        if self._address:
            return self._address
        connection_tuple = ray.init()
        self._address = connection_tuple['address']
        self._session_dir = connection_tuple['session_dir']
        return self._address

    @property
    def node(self) -> ray._private.node.Node:
        if False:
            while True:
                i = 10
        "Gets a 'ray.Node' object for this node (the head node).\n        If it does not already exist, one is created using the bootstrap\n        address.\n        "
        if self._node:
            return self._node
        ray_params = RayParams(gcs_address=self.address)
        self._node = ray._private.node.Node(ray_params, head=False, shutdown_at_exit=False, spawn_reaper=False, connect_only=True)
        return self._node

    def create_specific_server(self, client_id: str) -> SpecificServer:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create, but not start a SpecificServer for a given client. This\n        method must be called once per client.\n        '
        with self.server_lock:
            assert self.servers.get(client_id) is None, f'Server already created for Client: {client_id}'
            port = self._get_unused_port()
            server = SpecificServer(port=port, process_handle_future=futures.Future(), channel=ray._private.utils.init_grpc_channel(f'127.0.0.1:{port}', options=GRPC_OPTIONS))
            self.servers[client_id] = server
            return server

    def _create_runtime_env(self, serialized_runtime_env: str, runtime_env_config: str, specific_server: SpecificServer):
        if False:
            return 10
        "Increase the runtime_env reference by sending an RPC to the agent.\n\n        Includes retry logic to handle the case when the agent is\n        temporarily unreachable (e.g., hasn't been started up yet).\n        "
        logger.info(f'Increasing runtime env reference for ray_client_server_{specific_server.port}.Serialized runtime env is {serialized_runtime_env}.')
        assert len(self._runtime_env_agent_address) > 0, 'runtime_env_agent_address not set'
        create_env_request = runtime_env_agent_pb2.GetOrCreateRuntimeEnvRequest(serialized_runtime_env=serialized_runtime_env, runtime_env_config=runtime_env_config, job_id=f'ray_client_server_{specific_server.port}'.encode('utf-8'), source_process='client_server')
        retries = 0
        max_retries = 5
        wait_time_s = 0.5
        last_exception = None
        while retries <= max_retries:
            try:
                url = urllib.parse.urljoin(self._runtime_env_agent_address, '/get_or_create_runtime_env')
                data = create_env_request.SerializeToString()
                req = urllib.request.Request(url, data=data, method='POST')
                req.add_header('Content-Type', 'application/octet-stream')
                response = urllib.request.urlopen(req, timeout=None)
                response_data = response.read()
                r = runtime_env_agent_pb2.GetOrCreateRuntimeEnvReply()
                r.ParseFromString(response_data)
                if r.status == agent_manager_pb2.AgentRpcStatus.AGENT_RPC_STATUS_OK:
                    return r.serialized_runtime_env_context
                elif r.status == agent_manager_pb2.AgentRpcStatus.AGENT_RPC_STATUS_FAILED:
                    raise RuntimeError(f'Failed to create runtime_env for Ray client server, it is caused by:\n{r.error_message}')
                else:
                    assert False, f'Unknown status: {r.status}.'
            except urllib.error.URLError as e:
                last_exception = e
                logger.warning(f'GetOrCreateRuntimeEnv request failed: {e}. Retrying after {wait_time_s}s. {max_retries - retries} retries remaining.')
            time.sleep(wait_time_s)
            retries += 1
            wait_time_s *= 2
        raise TimeoutError(f'GetOrCreateRuntimeEnv request failed after {max_retries} attempts. Last exception: {last_exception}')

    def start_specific_server(self, client_id: str, job_config: JobConfig) -> bool:
        if False:
            return 10
        '\n        Start up a RayClient Server for an incoming client to\n        communicate with. Returns whether creation was successful.\n        '
        specific_server = self._get_server_for_client(client_id)
        assert specific_server, f'Server has not been created for: {client_id}'
        (output, error) = self.node.get_log_file_handles(f'ray_client_server_{specific_server.port}', unique=True)
        serialized_runtime_env = job_config._get_serialized_runtime_env()
        runtime_env_config = job_config._get_proto_runtime_env_config()
        if not serialized_runtime_env or serialized_runtime_env == '{}':
            serialized_runtime_env_context = RuntimeEnvContext().serialize()
        else:
            serialized_runtime_env_context = self._create_runtime_env(serialized_runtime_env=serialized_runtime_env, runtime_env_config=runtime_env_config, specific_server=specific_server)
        proc = start_ray_client_server(self.address, self.node.node_ip_address, specific_server.port, stdout_file=output, stderr_file=error, fate_share=self.fate_share, server_type='specific-server', serialized_runtime_env_context=serialized_runtime_env_context, redis_password=self._redis_password)
        pid = proc.process.pid
        if sys.platform != 'win32':
            psutil_proc = psutil.Process(pid)
        else:
            psutil_proc = None
        while psutil_proc is not None:
            if proc.process.poll() is not None:
                logger.error(f'SpecificServer startup failed for client: {client_id}')
                break
            cmd = psutil_proc.cmdline()
            if _match_running_client_server(cmd):
                break
            logger.debug('Waiting for Process to reach the actual client server.')
            time.sleep(0.5)
        specific_server.set_result(proc)
        logger.info(f'SpecificServer started on port: {specific_server.port} with PID: {pid} for client: {client_id}')
        return proc.process.poll() is None

    def _get_server_for_client(self, client_id: str) -> Optional[SpecificServer]:
        if False:
            print('Hello World!')
        with self.server_lock:
            client = self.servers.get(client_id)
            if client is None:
                logger.error(f'Unable to find channel for client: {client_id}')
            return client

    def has_channel(self, client_id: str) -> bool:
        if False:
            i = 10
            return i + 15
        server = self._get_server_for_client(client_id)
        if server is None:
            return False
        return server.is_ready()

    def get_channel(self, client_id: str) -> Optional['grpc._channel.Channel']:
        if False:
            i = 10
            return i + 15
        '\n        Find the gRPC Channel for the given client_id. This will block until\n        the server process has started.\n        '
        server = self._get_server_for_client(client_id)
        if server is None:
            return None
        server.wait_ready()
        try:
            grpc.channel_ready_future(server.channel).result(timeout=CHECK_CHANNEL_TIMEOUT_S)
            return server.channel
        except grpc.FutureTimeoutError:
            logger.exception(f'Timeout waiting for channel for {client_id}')
            return None

    def _check_processes(self):
        if False:
            i = 10
            return i + 15
        '\n        Keeps the internal servers dictionary up-to-date with running servers.\n        '
        while True:
            with self.server_lock:
                for (client_id, specific_server) in list(self.servers.items()):
                    if specific_server.poll() is not None:
                        logger.info(f'Specific server {client_id} is no longer running, freeing its port {specific_server.port}')
                        del self.servers[client_id]
                        self._free_ports.append(specific_server.port)
            time.sleep(CHECK_PROCESS_INTERVAL_S)

    def _cleanup(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Forcibly kill all spawned RayClient Servers. This ensures cleanup\n        for platforms where fate sharing is not supported.\n        '
        for server in self.servers.values():
            server.kill()

class RayletServicerProxy(ray_client_pb2_grpc.RayletDriverServicer):

    def __init__(self, ray_connect_handler: Callable, proxy_manager: ProxyManager):
        if False:
            print('Hello World!')
        self.proxy_manager = proxy_manager
        self.ray_connect_handler = ray_connect_handler

    def _call_inner_function(self, request, context, method: str) -> Optional[ray_client_pb2_grpc.RayletDriverStub]:
        if False:
            for i in range(10):
                print('nop')
        client_id = _get_client_id_from_context(context)
        chan = self.proxy_manager.get_channel(client_id)
        if not chan:
            logger.error(f'Channel for Client: {client_id} not found!')
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return None
        stub = ray_client_pb2_grpc.RayletDriverStub(chan)
        try:
            metadata = [('client_id', client_id)]
            if context:
                metadata = context.invocation_metadata()
            return getattr(stub, method)(request, metadata=metadata)
        except Exception as e:
            logger.exception(f'Proxying call to {method} failed!')
            _propagate_error_in_context(e, context)

    def _has_channel_for_request(self, context):
        if False:
            i = 10
            return i + 15
        client_id = _get_client_id_from_context(context)
        return self.proxy_manager.has_channel(client_id)

    def Init(self, request, context=None) -> ray_client_pb2.InitResponse:
        if False:
            return 10
        return self._call_inner_function(request, context, 'Init')

    def KVPut(self, request, context=None) -> ray_client_pb2.KVPutResponse:
        if False:
            i = 10
            return i + 15
        "Proxies internal_kv.put.\n\n        This is used by the working_dir code to upload to the GCS before\n        ray.init is called. In that case (if we don't have a server yet)\n        we directly make the internal KV call from the proxier.\n\n        Otherwise, we proxy the call to the downstream server as usual.\n        "
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'KVPut')
        with disable_client_hook():
            already_exists = ray.experimental.internal_kv._internal_kv_put(request.key, request.value, overwrite=request.overwrite)
        return ray_client_pb2.KVPutResponse(already_exists=already_exists)

    def KVGet(self, request, context=None) -> ray_client_pb2.KVGetResponse:
        if False:
            print('Hello World!')
        "Proxies internal_kv.get.\n\n        This is used by the working_dir code to upload to the GCS before\n        ray.init is called. In that case (if we don't have a server yet)\n        we directly make the internal KV call from the proxier.\n\n        Otherwise, we proxy the call to the downstream server as usual.\n        "
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'KVGet')
        with disable_client_hook():
            value = ray.experimental.internal_kv._internal_kv_get(request.key)
        return ray_client_pb2.KVGetResponse(value=value)

    def KVDel(self, request, context=None) -> ray_client_pb2.KVDelResponse:
        if False:
            for i in range(10):
                print('nop')
        "Proxies internal_kv.delete.\n\n        This is used by the working_dir code to upload to the GCS before\n        ray.init is called. In that case (if we don't have a server yet)\n        we directly make the internal KV call from the proxier.\n\n        Otherwise, we proxy the call to the downstream server as usual.\n        "
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'KVDel')
        with disable_client_hook():
            ray.experimental.internal_kv._internal_kv_del(request.key)
        return ray_client_pb2.KVDelResponse()

    def KVList(self, request, context=None) -> ray_client_pb2.KVListResponse:
        if False:
            for i in range(10):
                print('nop')
        "Proxies internal_kv.list.\n\n        This is used by the working_dir code to upload to the GCS before\n        ray.init is called. In that case (if we don't have a server yet)\n        we directly make the internal KV call from the proxier.\n\n        Otherwise, we proxy the call to the downstream server as usual.\n        "
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'KVList')
        with disable_client_hook():
            keys = ray.experimental.internal_kv._internal_kv_list(request.prefix)
        return ray_client_pb2.KVListResponse(keys=keys)

    def KVExists(self, request, context=None) -> ray_client_pb2.KVExistsResponse:
        if False:
            for i in range(10):
                print('nop')
        "Proxies internal_kv.exists.\n\n        This is used by the working_dir code to upload to the GCS before\n        ray.init is called. In that case (if we don't have a server yet)\n        we directly make the internal KV call from the proxier.\n\n        Otherwise, we proxy the call to the downstream server as usual.\n        "
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'KVExists')
        with disable_client_hook():
            exists = ray.experimental.internal_kv._internal_kv_exists(request.key)
        return ray_client_pb2.KVExistsResponse(exists=exists)

    def PinRuntimeEnvURI(self, request, context=None) -> ray_client_pb2.ClientPinRuntimeEnvURIResponse:
        if False:
            for i in range(10):
                print('nop')
        "Proxies internal_kv.pin_runtime_env_uri.\n\n        This is used by the working_dir code to upload to the GCS before\n        ray.init is called. In that case (if we don't have a server yet)\n        we directly make the internal KV call from the proxier.\n\n        Otherwise, we proxy the call to the downstream server as usual.\n        "
        if self._has_channel_for_request(context):
            return self._call_inner_function(request, context, 'PinRuntimeEnvURI')
        with disable_client_hook():
            ray.experimental.internal_kv._pin_runtime_env_uri(request.uri, expiration_s=request.expiration_s)
        return ray_client_pb2.ClientPinRuntimeEnvURIResponse()

    def ListNamedActors(self, request, context=None) -> ray_client_pb2.ClientListNamedActorsResponse:
        if False:
            i = 10
            return i + 15
        return self._call_inner_function(request, context, 'ListNamedActors')

    def ClusterInfo(self, request, context=None) -> ray_client_pb2.ClusterInfoResponse:
        if False:
            i = 10
            return i + 15
        if request.type == ray_client_pb2.ClusterInfoType.PING:
            resp = ray_client_pb2.ClusterInfoResponse(json=json.dumps({}))
            return resp
        return self._call_inner_function(request, context, 'ClusterInfo')

    def Terminate(self, req, context=None):
        if False:
            return 10
        return self._call_inner_function(req, context, 'Terminate')

    def GetObject(self, request, context=None):
        if False:
            return 10
        try:
            yield from self._call_inner_function(request, context, 'GetObject')
        except Exception as e:
            logger.exception('Proxying call to GetObject failed!')
            _propagate_error_in_context(e, context)

    def PutObject(self, request: ray_client_pb2.PutRequest, context=None) -> ray_client_pb2.PutResponse:
        if False:
            for i in range(10):
                print('nop')
        return self._call_inner_function(request, context, 'PutObject')

    def WaitObject(self, request, context=None) -> ray_client_pb2.WaitResponse:
        if False:
            print('Hello World!')
        return self._call_inner_function(request, context, 'WaitObject')

    def Schedule(self, task, context=None) -> ray_client_pb2.ClientTaskTicket:
        if False:
            for i in range(10):
                print('nop')
        return self._call_inner_function(task, context, 'Schedule')

def ray_client_server_env_prep(job_config: JobConfig) -> JobConfig:
    if False:
        while True:
            i = 10
    return job_config

def prepare_runtime_init_req(init_request: ray_client_pb2.DataRequest) -> Tuple[ray_client_pb2.DataRequest, JobConfig]:
    if False:
        return 10
    '\n    Extract JobConfig and possibly mutate InitRequest before it is passed to\n    the specific RayClient Server.\n    '
    init_type = init_request.WhichOneof('type')
    assert init_type == 'init', f"Received initial message of type {init_type}, not 'init'."
    req = init_request.init
    job_config = JobConfig()
    if req.job_config:
        job_config = pickle.loads(req.job_config)
    new_job_config = ray_client_server_env_prep(job_config)
    modified_init_req = ray_client_pb2.InitRequest(job_config=pickle.dumps(new_job_config), ray_init_kwargs=init_request.init.ray_init_kwargs, reconnect_grace_period=init_request.init.reconnect_grace_period)
    init_request.init.CopyFrom(modified_init_req)
    return (init_request, new_job_config)

class RequestIteratorProxy:

    def __init__(self, request_iterator):
        if False:
            i = 10
            return i + 15
        self.request_iterator = request_iterator

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            return 10
        try:
            return next(self.request_iterator)
        except grpc.RpcError as e:
            if type(e) != grpc.RpcError:
                raise e
            logger.exception('Stop iterating cancelled request stream with the following exception:')
            raise StopIteration

class DataServicerProxy(ray_client_pb2_grpc.RayletDataStreamerServicer):

    def __init__(self, proxy_manager: ProxyManager):
        if False:
            print('Hello World!')
        self.num_clients = 0
        self.clients_last_seen: Dict[str, float] = {}
        self.reconnect_grace_periods: Dict[str, float] = {}
        self.clients_lock = Lock()
        self.proxy_manager = proxy_manager
        self.stopped = Event()

    def modify_connection_info_resp(self, init_resp: ray_client_pb2.DataResponse) -> ray_client_pb2.DataResponse:
        if False:
            print('Hello World!')
        '\n        Modify the `num_clients` returned the ConnectionInfoResponse because\n        individual SpecificServers only have **one** client.\n        '
        init_type = init_resp.WhichOneof('type')
        if init_type != 'connection_info':
            return init_resp
        modified_resp = ray_client_pb2.DataResponse()
        modified_resp.CopyFrom(init_resp)
        with self.clients_lock:
            modified_resp.connection_info.num_clients = self.num_clients
        return modified_resp

    def Datapath(self, request_iterator, context):
        if False:
            while True:
                i = 10
        request_iterator = RequestIteratorProxy(request_iterator)
        cleanup_requested = False
        start_time = time.time()
        client_id = _get_client_id_from_context(context)
        if client_id == '':
            return
        reconnecting = _get_reconnecting_from_context(context)
        if reconnecting:
            with self.clients_lock:
                if client_id not in self.clients_last_seen:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details('Attempted to reconnect a session that has already been cleaned up')
                    return
                self.clients_last_seen[client_id] = start_time
            server = self.proxy_manager._get_server_for_client(client_id)
            channel = self.proxy_manager.get_channel(client_id)
            new_iter = request_iterator
        else:
            server = self.proxy_manager.create_specific_server(client_id)
            with self.clients_lock:
                self.clients_last_seen[client_id] = start_time
                self.num_clients += 1
        try:
            if not reconnecting:
                logger.info(f'New data connection from client {client_id}: ')
                init_req = next(request_iterator)
                with self.clients_lock:
                    self.reconnect_grace_periods[client_id] = init_req.init.reconnect_grace_period
                try:
                    (modified_init_req, job_config) = prepare_runtime_init_req(init_req)
                    if not self.proxy_manager.start_specific_server(client_id, job_config):
                        logger.error(f'Server startup failed for client: {client_id}, using JobConfig: {job_config}!')
                        raise RuntimeError(f'Starting Ray client server failed. See ray_client_server_{server.port}.err for detailed logs.')
                    channel = self.proxy_manager.get_channel(client_id)
                    if channel is None:
                        logger.error(f'Channel not found for {client_id}')
                        raise RuntimeError(f'Proxy failed to Connect to backend! Check `ray_client_server.err` and `ray_client_server_{server.port}.err` on the head node of the cluster for the relevant logs. By default these are located at /tmp/ray/session_latest/logs.')
                except Exception:
                    init_resp = ray_client_pb2.DataResponse(init=ray_client_pb2.InitResponse(ok=False, msg=traceback.format_exc()))
                    init_resp.req_id = init_req.req_id
                    yield init_resp
                    return None
                new_iter = chain([modified_init_req], request_iterator)
            stub = ray_client_pb2_grpc.RayletDataStreamerStub(channel)
            metadata = [('client_id', client_id), ('reconnecting', str(reconnecting))]
            resp_stream = stub.Datapath(new_iter, metadata=metadata)
            for resp in resp_stream:
                resp_type = resp.WhichOneof('type')
                if resp_type == 'connection_cleanup':
                    cleanup_requested = True
                yield self.modify_connection_info_resp(resp)
        except Exception as e:
            logger.exception('Proxying Datapath failed!')
            recoverable = _propagate_error_in_context(e, context)
            if not recoverable:
                cleanup_requested = True
        finally:
            cleanup_delay = self.reconnect_grace_periods.get(client_id)
            if not cleanup_requested and cleanup_delay is not None:
                self.stopped.wait(timeout=cleanup_delay)
            with self.clients_lock:
                if client_id not in self.clients_last_seen:
                    logger.info(f'{client_id} not found. Skipping clean up.')
                    return
                last_seen = self.clients_last_seen[client_id]
                logger.info(f'{client_id} last started stream at {last_seen}. Current stream started at {start_time}.')
                if last_seen > start_time:
                    logger.info('Client reconnected. Skipping cleanup.')
                    return
                logger.debug(f'Client detached: {client_id}')
                self.num_clients -= 1
                del self.clients_last_seen[client_id]
                if client_id in self.reconnect_grace_periods:
                    del self.reconnect_grace_periods[client_id]
                server.set_result(None)

class LogstreamServicerProxy(ray_client_pb2_grpc.RayletLogStreamerServicer):

    def __init__(self, proxy_manager: ProxyManager):
        if False:
            return 10
        super().__init__()
        self.proxy_manager = proxy_manager

    def Logstream(self, request_iterator, context):
        if False:
            for i in range(10):
                print('nop')
        request_iterator = RequestIteratorProxy(request_iterator)
        client_id = _get_client_id_from_context(context)
        if client_id == '':
            return
        logger.debug(f'New logstream connection from client {client_id}: ')
        channel = None
        for i in range(LOGSTREAM_RETRIES):
            channel = self.proxy_manager.get_channel(client_id)
            if channel is not None:
                break
            logger.warning(f'Retrying Logstream connection. {i + 1} attempts failed.')
            time.sleep(LOGSTREAM_RETRY_INTERVAL_SEC)
        if channel is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f'Logstream proxy failed to connect. Channel for client {client_id} not found.')
            return None
        stub = ray_client_pb2_grpc.RayletLogStreamerStub(channel)
        resp_stream = stub.Logstream(request_iterator, metadata=[('client_id', client_id)])
        try:
            for resp in resp_stream:
                yield resp
        except Exception:
            logger.exception('Proxying Logstream failed!')

def serve_proxier(connection_str: str, address: Optional[str], *, redis_password: Optional[str]=None, session_dir: Optional[str]=None, runtime_env_agent_address: Optional[str]=None):
    if False:
        print('Hello World!')
    if address is not None:
        gcs_cli = GcsClient(address=address)
        ray.experimental.internal_kv._initialize_internal_kv(gcs_cli)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=CLIENT_SERVER_MAX_THREADS), options=GRPC_OPTIONS)
    proxy_manager = ProxyManager(address, session_dir=session_dir, redis_password=redis_password, runtime_env_agent_address=runtime_env_agent_address)
    task_servicer = RayletServicerProxy(None, proxy_manager)
    data_servicer = DataServicerProxy(proxy_manager)
    logs_servicer = LogstreamServicerProxy(proxy_manager)
    ray_client_pb2_grpc.add_RayletDriverServicer_to_server(task_servicer, server)
    ray_client_pb2_grpc.add_RayletDataStreamerServicer_to_server(data_servicer, server)
    ray_client_pb2_grpc.add_RayletLogStreamerServicer_to_server(logs_servicer, server)
    add_port_to_grpc_server(server, connection_str)
    server.start()
    return ClientServerHandle(task_servicer=task_servicer, data_servicer=data_servicer, logs_servicer=logs_servicer, grpc_server=server)