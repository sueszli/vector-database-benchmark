"""Launches PODs."""
from __future__ import annotations
import enum
import itertools
import json
import math
import time
import warnings
from collections.abc import Iterable
from contextlib import closing, suppress
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, Generator, Literal, Protocol, cast
import pendulum
import tenacity
from kubernetes import client, watch
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream as kubernetes_stream
from pendulum import DateTime
from pendulum.parsing.exceptions import ParserError
from urllib3.exceptions import HTTPError as BaseHTTPError
from airflow.exceptions import AirflowException, AirflowProviderDeprecationWarning
from airflow.providers.cncf.kubernetes.pod_generator import PodDefaults
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.timezone import utcnow
if TYPE_CHECKING:
    from kubernetes.client.models.core_v1_event_list import CoreV1EventList
    from kubernetes.client.models.v1_container_status import V1ContainerStatus
    from kubernetes.client.models.v1_pod import V1Pod
    from urllib3.response import HTTPResponse
EMPTY_XCOM_RESULT = '__airflow_xcom_result_empty__'
'\nSentinel for no xcom result.\n\n:meta private:\n'

class PodLaunchFailedException(AirflowException):
    """When pod launching fails in KubernetesPodOperator."""

def should_retry_start_pod(exception: BaseException) -> bool:
    if False:
        print('Hello World!')
    'Check if an Exception indicates a transient error and warrants retrying.'
    if isinstance(exception, ApiException):
        return exception.status == 409
    return False

class PodPhase:
    """
    Possible pod phases.

    See https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase.
    """
    PENDING = 'Pending'
    RUNNING = 'Running'
    FAILED = 'Failed'
    SUCCEEDED = 'Succeeded'
    terminal_states = {FAILED, SUCCEEDED}

class PodOperatorHookProtocol(Protocol):
    """
    Protocol to define methods relied upon by KubernetesPodOperator.

    Subclasses of KubernetesPodOperator, such as GKEStartPodOperator, may use
    hooks that don't extend KubernetesHook.  We use this protocol to document the
    methods used by KPO and ensure that these methods exist on such other hooks.
    """

    @property
    def core_v1_client(self) -> client.CoreV1Api:
        if False:
            i = 10
            return i + 15
        'Get authenticated client object.'

    @property
    def is_in_cluster(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Expose whether the hook is configured with ``load_incluster_config`` or not.'

    def get_pod(self, name: str, namespace: str) -> V1Pod:
        if False:
            while True:
                i = 10
        'Read pod object from kubernetes API.'

    def get_namespace(self) -> str | None:
        if False:
            print('Hello World!')
        'Return the namespace that defined in the connection.'

    def get_xcom_sidecar_container_image(self) -> str | None:
        if False:
            i = 10
            return i + 15
        'Return the xcom sidecar image that defined in the connection.'

    def get_xcom_sidecar_container_resources(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        'Return the xcom sidecar resources that defined in the connection.'

def get_container_status(pod: V1Pod, container_name: str) -> V1ContainerStatus | None:
    if False:
        i = 10
        return i + 15
    'Retrieve container status.'
    container_statuses = pod.status.container_statuses if pod and pod.status else None
    if container_statuses:
        return next((x for x in container_statuses if x.name == container_name), None)
    return None

def container_is_running(pod: V1Pod, container_name: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Examine V1Pod ``pod`` to determine whether ``container_name`` is running.\n\n    If that container is present and running, returns True.  Returns False otherwise.\n    '
    container_status = get_container_status(pod, container_name)
    if not container_status:
        return False
    return container_status.state.running is not None

def container_is_completed(pod: V1Pod, container_name: str) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Examine V1Pod ``pod`` to determine whether ``container_name`` is completed.\n\n    If that container is present and completed, returns True.  Returns False otherwise.\n    '
    container_status = get_container_status(pod, container_name)
    if not container_status:
        return False
    return container_status.state.terminated is not None

def container_is_succeeded(pod: V1Pod, container_name: str) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Examine V1Pod ``pod`` to determine whether ``container_name`` is completed and succeeded.\n\n    If that container is present and completed and succeeded, returns True.  Returns False otherwise.\n    '
    if not container_is_completed(pod, container_name):
        return False
    container_status = get_container_status(pod, container_name)
    if not container_status:
        return False
    return container_status.state.terminated.exit_code == 0

def container_is_terminated(pod: V1Pod, container_name: str) -> bool:
    if False:
        while True:
            i = 10
    '\n    Examine V1Pod ``pod`` to determine whether ``container_name`` is terminated.\n\n    If that container is present and terminated, returns True.  Returns False otherwise.\n    '
    container_statuses = pod.status.container_statuses if pod and pod.status else None
    if not container_statuses:
        return False
    container_status = next((x for x in container_statuses if x.name == container_name), None)
    if not container_status:
        return False
    return container_status.state.terminated is not None

def get_container_termination_message(pod: V1Pod, container_name: str):
    if False:
        for i in range(10):
            print('nop')
    with suppress(AttributeError, TypeError):
        container_statuses = pod.status.container_statuses
        container_status = next((x for x in container_statuses if x.name == container_name), None)
        return container_status.state.terminated.message if container_status else None

class PodLogsConsumer:
    """
    Responsible for pulling pod logs from a stream with checking a container status before reading data.

    This class is a workaround for the issue https://github.com/apache/airflow/issues/23497.

    :param response: HTTP response with logs
    :param pod: Pod instance from Kubernetes client
    :param pod_manager: Pod manager instance
    :param container_name: Name of the container that we're reading logs from
    :param post_termination_timeout: (Optional) The period of time in seconds representing for how long time
        logs are available after the container termination.
    :param read_pod_cache_timeout: (Optional) The container's status cache lifetime.
        The container status is cached to reduce API calls.

    :meta private:
    """

    def __init__(self, response: HTTPResponse, pod: V1Pod, pod_manager: PodManager, container_name: str, post_termination_timeout: int=120, read_pod_cache_timeout: int=120):
        if False:
            print('Hello World!')
        self.response = response
        self.pod = pod
        self.pod_manager = pod_manager
        self.container_name = container_name
        self.post_termination_timeout = post_termination_timeout
        self.last_read_pod_at = None
        self.read_pod_cache = None
        self.read_pod_cache_timeout = read_pod_cache_timeout

    def __iter__(self) -> Generator[bytes, None, None]:
        if False:
            print('Hello World!')
        "Yield log items divided by the '\\n' symbol."
        incomplete_log_item: list[bytes] = []
        if self.logs_available():
            for data_chunk in self.response.stream(amt=None, decode_content=True):
                if b'\n' in data_chunk:
                    log_items = data_chunk.split(b'\n')
                    yield from self._extract_log_items(incomplete_log_item, log_items)
                    incomplete_log_item = self._save_incomplete_log_item(log_items[-1])
                else:
                    incomplete_log_item.append(data_chunk)
                if not self.logs_available():
                    break
        if incomplete_log_item:
            yield b''.join(incomplete_log_item)

    @staticmethod
    def _extract_log_items(incomplete_log_item: list[bytes], log_items: list[bytes]):
        if False:
            for i in range(10):
                print('nop')
        yield (b''.join(incomplete_log_item) + log_items[0] + b'\n')
        for x in log_items[1:-1]:
            yield (x + b'\n')

    @staticmethod
    def _save_incomplete_log_item(sub_chunk: bytes):
        if False:
            i = 10
            return i + 15
        return [sub_chunk] if [sub_chunk] else []

    def logs_available(self):
        if False:
            return 10
        remote_pod = self.read_pod()
        if container_is_running(pod=remote_pod, container_name=self.container_name):
            return True
        container_status = get_container_status(pod=remote_pod, container_name=self.container_name)
        state = container_status.state if container_status else None
        terminated = state.terminated if state else None
        if terminated:
            termination_time = terminated.finished_at
            if termination_time:
                return termination_time + timedelta(seconds=self.post_termination_timeout) > utcnow()
        return False

    def read_pod(self):
        if False:
            for i in range(10):
                print('nop')
        _now = utcnow()
        if self.read_pod_cache is None or self.last_read_pod_at + timedelta(seconds=self.read_pod_cache_timeout) < _now:
            self.read_pod_cache = self.pod_manager.read_pod(self.pod)
            self.last_read_pod_at = _now
        return self.read_pod_cache

class PodManager(LoggingMixin):
    """Create, monitor, and otherwise interact with Kubernetes pods for use with the KubernetesPodOperator."""

    def __init__(self, kube_client: client.CoreV1Api, progress_callback: Callable[[str], None] | None=None):
        if False:
            while True:
                i = 10
        '\n        Create the launcher.\n\n        :param kube_client: kubernetes client\n        :param progress_callback: Callback function invoked when fetching container log.\n        '
        super().__init__()
        self._client = kube_client
        self._progress_callback = progress_callback
        self._watch = watch.Watch()

    def run_pod_async(self, pod: V1Pod, **kwargs) -> V1Pod:
        if False:
            while True:
                i = 10
        'Run POD asynchronously.'
        sanitized_pod = self._client.api_client.sanitize_for_serialization(pod)
        json_pod = json.dumps(sanitized_pod, indent=2)
        self.log.debug('Pod Creation Request: \n%s', json_pod)
        try:
            resp = self._client.create_namespaced_pod(body=sanitized_pod, namespace=pod.metadata.namespace, **kwargs)
            self.log.debug('Pod Creation Response: %s', resp)
        except Exception as e:
            self.log.exception('Exception when attempting to create Namespaced Pod: %s', str(json_pod).replace('\n', ' '))
            raise e
        return resp

    def delete_pod(self, pod: V1Pod) -> None:
        if False:
            while True:
                i = 10
        'Delete POD.'
        try:
            self._client.delete_namespaced_pod(pod.metadata.name, pod.metadata.namespace, body=client.V1DeleteOptions())
        except ApiException as e:
            if e.status != 404:
                raise

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_random_exponential(), reraise=True, retry=tenacity.retry_if_exception(should_retry_start_pod))
    def create_pod(self, pod: V1Pod) -> V1Pod:
        if False:
            i = 10
            return i + 15
        'Launch the pod asynchronously.'
        return self.run_pod_async(pod)

    def await_pod_start(self, pod: V1Pod, startup_timeout: int=120, startup_check_interval: int=1) -> None:
        if False:
            return 10
        '\n        Wait for the pod to reach phase other than ``Pending``.\n\n        :param pod:\n        :param startup_timeout: Timeout (in seconds) for startup of the pod\n            (if pod is pending for too long, fails task)\n        :param startup_check_interval: Interval (in seconds) between checks\n        :return:\n        '
        curr_time = time.time()
        while True:
            remote_pod = self.read_pod(pod)
            if remote_pod.status.phase != PodPhase.PENDING:
                break
            self.log.warning('Pod not yet started: %s', pod.metadata.name)
            if time.time() - curr_time >= startup_timeout:
                msg = f'Pod took longer than {startup_timeout} seconds to start. Check the pod events in kubernetes to determine why.'
                raise PodLaunchFailedException(msg)
            time.sleep(startup_check_interval)

    def follow_container_logs(self, pod: V1Pod, container_name: str) -> None:
        if False:
            while True:
                i = 10
        warnings.warn('Method `follow_container_logs` is deprecated.  Use `fetch_container_logs` insteadwith option `follow=True`.', category=AirflowProviderDeprecationWarning, stacklevel=2)
        return self.fetch_container_logs(pod=pod, container_name=container_name, follow=True)

    def fetch_container_logs(self, pod: V1Pod, container_name: str, *, follow=False, since_time: DateTime | None=None, post_termination_timeout: int=120) -> None:
        if False:
            print('Hello World!')
        '\n        Follow the logs of container and stream to airflow logging.\n\n        Returns when container exits.\n\n        Between when the pod starts and logs being available, there might be a delay due to CSR not approved\n        and signed yet. In such situation, ApiException is thrown. This is why we are retrying on this\n        specific exception.\n\n        :meta private:\n        '

        def consume_logs(*, since_time: DateTime | None=None) -> DateTime | None:
            if False:
                while True:
                    i = 10
            '\n            Try to follow container logs until container completes.\n\n            For a long-running container, sometimes the log read may be interrupted\n            Such errors of this kind are suppressed.\n\n            Returns the last timestamp observed in logs.\n            '
            last_captured_timestamp = None
            try:
                logs = self.read_pod_logs(pod=pod, container_name=container_name, timestamps=True, since_seconds=math.ceil((pendulum.now() - since_time).total_seconds()) if since_time else None, follow=follow, post_termination_timeout=post_termination_timeout)
                message_to_log = None
                message_timestamp = None
                progress_callback_lines = []
                try:
                    for raw_line in logs:
                        line = raw_line.decode('utf-8', errors='backslashreplace')
                        (line_timestamp, message) = self.parse_log_line(line)
                        if line_timestamp:
                            if message_to_log is None:
                                message_to_log = message
                                message_timestamp = line_timestamp
                                progress_callback_lines.append(line)
                            else:
                                if self._progress_callback:
                                    for line in progress_callback_lines:
                                        self._progress_callback(line)
                                self.log.info('[%s] %s', container_name, message_to_log)
                                last_captured_timestamp = message_timestamp
                                message_to_log = message
                                message_timestamp = line_timestamp
                                progress_callback_lines = [line]
                        else:
                            message_to_log = f'{message_to_log}\n{message}'
                            progress_callback_lines.append(line)
                finally:
                    if self._progress_callback:
                        for line in progress_callback_lines:
                            self._progress_callback(line)
                    self.log.info('[%s] %s', container_name, message_to_log)
                    last_captured_timestamp = message_timestamp
            except BaseHTTPError:
                self.log.exception('Reading of logs interrupted for container %r; will retry.', container_name)
            return last_captured_timestamp or since_time
        last_log_time = since_time
        while True:
            last_log_time = consume_logs(since_time=last_log_time)
            if not follow:
                return
            if self.container_is_running(pod, container_name=container_name):
                self.log.warning('Follow requested but pod log read interrupted and container %s still running', container_name)
                time.sleep(1)
            else:
                break

    def _reconcile_requested_log_containers(self, requested: Iterable[str] | str | bool, actual: list[str], pod_name) -> list[str]:
        if False:
            i = 10
            return i + 15
        'Return actual containers based on requested.'
        containers_to_log = []
        if actual:
            if isinstance(requested, str):
                if requested in actual:
                    containers_to_log.append(requested)
                else:
                    self.log.error('container %s whose logs were requested not found in the pod %s', requested, pod_name)
            elif isinstance(requested, bool):
                if requested is True:
                    containers_to_log.extend(actual)
                else:
                    self.log.error('False is not a valid value for container_logs')
            elif isinstance(requested, Iterable):
                for container in requested:
                    if container in actual:
                        containers_to_log.append(container)
                    else:
                        self.log.error('Container %s whose logs were requests not found in the pod %s', container, pod_name)
            else:
                self.log.error('Invalid type %s specified for container names input parameter', type(requested))
        else:
            self.log.error('Could not retrieve containers for the pod: %s', pod_name)
        return containers_to_log

    def fetch_requested_container_logs(self, pod: V1Pod, containers: Iterable[str] | str | Literal[True], follow_logs=False) -> None:
        if False:
            print('Hello World!')
        '\n        Follow the logs of containers in the specified pod and publish it to airflow logging.\n\n        Returns when all the containers exit.\n\n        :meta private:\n        '
        all_containers = self.get_container_names(pod)
        containers_to_log = self._reconcile_requested_log_containers(requested=containers, actual=all_containers, pod_name=pod.metadata.name)
        for c in containers_to_log:
            self.fetch_container_logs(pod=pod, container_name=c, follow=follow_logs)

    def await_container_completion(self, pod: V1Pod, container_name: str) -> None:
        if False:
            print('Hello World!')
        '\n        Wait for the given container in the given pod to be completed.\n\n        :param pod: pod spec that will be monitored\n        :param container_name: name of the container within the pod to monitor\n        '
        while True:
            remote_pod = self.read_pod(pod)
            terminated = container_is_completed(remote_pod, container_name)
            if terminated:
                break
            self.log.info("Waiting for container '%s' state to be completed", container_name)
            time.sleep(1)

    def await_pod_completion(self, pod: V1Pod, istio_enabled: bool=False, container_name: str='base') -> V1Pod:
        if False:
            return 10
        '\n        Monitor a pod and return the final state.\n\n        :param istio_enabled: whether istio is enabled in the namespace\n        :param pod: pod spec that will be monitored\n        :param container_name: name of the container within the pod\n        :return: tuple[State, str | None]\n        '
        while True:
            remote_pod = self.read_pod(pod)
            if remote_pod.status.phase in PodPhase.terminal_states:
                break
            if istio_enabled and container_is_completed(remote_pod, container_name):
                break
            self.log.info('Pod %s has phase %s', pod.metadata.name, remote_pod.status.phase)
            time.sleep(2)
        return remote_pod

    def parse_log_line(self, line: str) -> tuple[DateTime | None, str]:
        if False:
            i = 10
            return i + 15
        '\n        Parse K8s log line and returns the final state.\n\n        :param line: k8s log line\n        :return: timestamp and log message\n        '
        (timestamp, sep, message) = line.strip().partition(' ')
        if not sep:
            return (None, line)
        try:
            last_log_time = cast(DateTime, pendulum.parse(timestamp))
        except ParserError:
            return (None, line)
        return (last_log_time, message)

    def container_is_running(self, pod: V1Pod, container_name: str) -> bool:
        if False:
            print('Hello World!')
        'Read pod and checks if container is running.'
        remote_pod = self.read_pod(pod)
        return container_is_running(pod=remote_pod, container_name=container_name)

    def container_is_terminated(self, pod: V1Pod, container_name: str) -> bool:
        if False:
            return 10
        'Read pod and checks if container is terminated.'
        remote_pod = self.read_pod(pod)
        return container_is_terminated(pod=remote_pod, container_name=container_name)

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(), reraise=True)
    def read_pod_logs(self, pod: V1Pod, container_name: str, tail_lines: int | None=None, timestamps: bool=False, since_seconds: int | None=None, follow=True, post_termination_timeout: int=120) -> PodLogsConsumer:
        if False:
            i = 10
            return i + 15
        'Read log from the POD.'
        additional_kwargs = {}
        if since_seconds:
            additional_kwargs['since_seconds'] = since_seconds
        if tail_lines:
            additional_kwargs['tail_lines'] = tail_lines
        try:
            logs = self._client.read_namespaced_pod_log(name=pod.metadata.name, namespace=pod.metadata.namespace, container=container_name, follow=follow, timestamps=timestamps, _preload_content=False, **additional_kwargs)
        except BaseHTTPError:
            self.log.exception('There was an error reading the kubernetes API.')
            raise
        return PodLogsConsumer(response=logs, pod=pod, pod_manager=self, container_name=container_name, post_termination_timeout=post_termination_timeout)

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(), reraise=True)
    def get_container_names(self, pod: V1Pod) -> list[str]:
        if False:
            i = 10
            return i + 15
        'Return container names from the POD except for the airflow-xcom-sidecar container.'
        pod_info = self.read_pod(pod)
        return [container_spec.name for container_spec in pod_info.spec.containers if container_spec.name != PodDefaults.SIDECAR_CONTAINER_NAME]

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(), reraise=True)
    def read_pod_events(self, pod: V1Pod) -> CoreV1EventList:
        if False:
            for i in range(10):
                print('nop')
        'Read events from the POD.'
        try:
            return self._client.list_namespaced_event(namespace=pod.metadata.namespace, field_selector=f'involvedObject.name={pod.metadata.name}')
        except BaseHTTPError as e:
            raise AirflowException(f'There was an error reading the kubernetes API: {e}')

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(), reraise=True)
    def read_pod(self, pod: V1Pod) -> V1Pod:
        if False:
            for i in range(10):
                print('nop')
        'Read POD information.'
        try:
            return self._client.read_namespaced_pod(pod.metadata.name, pod.metadata.namespace)
        except BaseHTTPError as e:
            raise AirflowException(f'There was an error reading the kubernetes API: {e}')

    def await_xcom_sidecar_container_start(self, pod: V1Pod) -> None:
        if False:
            return 10
        self.log.info('Checking if xcom sidecar container is started.')
        for attempt in itertools.count():
            if self.container_is_running(pod, PodDefaults.SIDECAR_CONTAINER_NAME):
                self.log.info('The xcom sidecar container is started.')
                break
            if not attempt:
                self.log.warning('The xcom sidecar container is not yet started.')
            time.sleep(1)

    def extract_xcom(self, pod: V1Pod) -> str:
        if False:
            return 10
        'Retrieve XCom value and kill xcom sidecar container.'
        try:
            result = self.extract_xcom_json(pod)
            return result
        finally:
            self.extract_xcom_kill(pod)

    @tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    def extract_xcom_json(self, pod: V1Pod) -> str:
        if False:
            return 10
        'Retrieve XCom value and also check if xcom json is valid.'
        with closing(kubernetes_stream(self._client.connect_get_namespaced_pod_exec, pod.metadata.name, pod.metadata.namespace, container=PodDefaults.SIDECAR_CONTAINER_NAME, command=['/bin/sh'], stdin=True, stdout=True, stderr=True, tty=False, _preload_content=False)) as resp:
            result = self._exec_pod_command(resp, f'if [ -s {PodDefaults.XCOM_MOUNT_PATH}/return.json ]; then cat {PodDefaults.XCOM_MOUNT_PATH}/return.json; else echo {EMPTY_XCOM_RESULT}; fi')
            if result and result.rstrip() != EMPTY_XCOM_RESULT:
                json.loads(result)
        if result is None:
            raise AirflowException(f'Failed to extract xcom from pod: {pod.metadata.name}')
        return result

    @tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    def extract_xcom_kill(self, pod: V1Pod):
        if False:
            while True:
                i = 10
        'Kill xcom sidecar container.'
        with closing(kubernetes_stream(self._client.connect_get_namespaced_pod_exec, pod.metadata.name, pod.metadata.namespace, container=PodDefaults.SIDECAR_CONTAINER_NAME, command=['/bin/sh'], stdin=True, stdout=True, stderr=True, tty=False, _preload_content=False)) as resp:
            self._exec_pod_command(resp, 'kill -s SIGINT 1')

    def _exec_pod_command(self, resp, command: str) -> str | None:
        if False:
            while True:
                i = 10
        res = ''
        if not resp.is_open():
            return None
        self.log.info('Running command... %s', command)
        resp.write_stdin(f'{command}\n')
        while resp.is_open():
            resp.update(timeout=1)
            while resp.peek_stdout():
                res += resp.read_stdout()
            error_res = ''
            while resp.peek_stderr():
                error_res += resp.read_stderr()
            if error_res:
                self.log.info('stderr from command: %s', error_res)
                break
            if res:
                return res
        return None

class OnFinishAction(str, enum.Enum):
    """Action to take when the pod finishes."""
    KEEP_POD = 'keep_pod'
    DELETE_POD = 'delete_pod'
    DELETE_SUCCEEDED_POD = 'delete_succeeded_pod'