import binascii
import logging
import os
import random
import string
import time
from datetime import date, datetime
from enum import Enum, auto
from threading import RLock, Timer
from typing import Callable, Dict, Optional
from localstack import config
from localstack.aws.api.lambda_ import TracingMode
from localstack.aws.connect import connect_to
from localstack.services.lambda_.invocation.lambda_models import Credentials, FunctionVersion, InitializationType, Invocation, InvocationResult
from localstack.services.lambda_.invocation.runtime_executor import RuntimeExecutor, get_runtime_executor
from localstack.utils.strings import to_str
STARTUP_TIMEOUT_SEC = config.LAMBDA_RUNTIME_ENVIRONMENT_TIMEOUT
HEX_CHARS = [str(num) for num in range(10)] + ['a', 'b', 'c', 'd', 'e', 'f']
LOG = logging.getLogger(__name__)

class RuntimeStatus(Enum):
    INACTIVE = auto()
    STARTING = auto()
    READY = auto()
    RUNNING = auto()
    STARTUP_FAILED = auto()
    STARTUP_TIMED_OUT = auto()
    STOPPED = auto()

class InvalidStatusException(Exception):

    def __init__(self, message: str):
        if False:
            while True:
                i = 10
        super().__init__(message)

class EnvironmentStartupTimeoutException(Exception):

    def __init__(self, message: str):
        if False:
            print('Hello World!')
        super().__init__(message)

def generate_runtime_id() -> str:
    if False:
        i = 10
        return i + 15
    return ''.join(random.choices(string.hexdigits[:16], k=32)).lower()

class ExecutionEnvironment:
    runtime_executor: RuntimeExecutor
    status_lock: RLock
    status: RuntimeStatus
    initialization_type: InitializationType
    last_returned: datetime
    startup_timer: Optional[Timer]
    keepalive_timer: Optional[Timer]

    def __init__(self, function_version: FunctionVersion, initialization_type: InitializationType, on_timeout: Callable[[str, str], None]):
        if False:
            i = 10
            return i + 15
        self.id = generate_runtime_id()
        self.status = RuntimeStatus.INACTIVE
        self.status_lock = RLock()
        self.function_version = function_version
        self.initialization_type = initialization_type
        self.runtime_executor = get_runtime_executor()(self.id, function_version)
        self.last_returned = datetime.min
        self.startup_timer = None
        self.keepalive_timer = Timer(0, lambda *args, **kwargs: None)
        self.on_timeout = on_timeout

    def get_log_group_name(self) -> str:
        if False:
            print('Hello World!')
        return f'/aws/lambda/{self.function_version.id.function_name}'

    def get_log_stream_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{date.today():%Y/%m/%d}/[{self.function_version.id.qualifier}]{self.id}'

    def get_environment_variables(self) -> Dict[str, str]:
        if False:
            while True:
                i = 10
        '\n        Returns the environment variable set for the runtime container\n        :return: Dict of environment variables\n        '
        credentials = self.get_credentials()
        env_vars = {'AWS_DEFAULT_REGION': self.function_version.id.region, 'AWS_REGION': self.function_version.id.region, 'AWS_LAMBDA_FUNCTION_NAME': self.function_version.id.function_name, 'AWS_LAMBDA_FUNCTION_MEMORY_SIZE': self.function_version.config.memory_size, 'AWS_LAMBDA_FUNCTION_VERSION': self.function_version.id.qualifier, 'AWS_LAMBDA_INITIALIZATION_TYPE': self.initialization_type, 'AWS_LAMBDA_LOG_GROUP_NAME': self.get_log_group_name(), 'AWS_LAMBDA_LOG_STREAM_NAME': self.get_log_stream_name(), 'AWS_ACCESS_KEY_ID': credentials['AccessKeyId'], 'AWS_SECRET_ACCESS_KEY': credentials['SecretAccessKey'], 'AWS_SESSION_TOKEN': credentials['SessionToken'], 'LAMBDA_TASK_ROOT': '/var/task', 'LAMBDA_RUNTIME_DIR': '/var/runtime', 'AWS_XRAY_CONTEXT_MISSING': 'LOG_ERROR', 'AWS_XRAY_DAEMON_ADDRESS': '127.0.0.1:2000', '_AWS_XRAY_DAEMON_PORT': '2000', '_AWS_XRAY_DAEMON_ADDRESS': '127.0.0.1', 'TZ': ':UTC', 'AWS_LAMBDA_FUNCTION_TIMEOUT': self.function_version.config.timeout, 'LOCALSTACK_HOSTNAME': self.runtime_executor.get_endpoint_from_executor(), 'EDGE_PORT': str(config.GATEWAY_LISTEN[0].port), 'AWS_ENDPOINT_URL': f'http://{self.runtime_executor.get_endpoint_from_executor()}:{config.GATEWAY_LISTEN[0].port}', 'LOCALSTACK_RUNTIME_ID': self.id, 'LOCALSTACK_RUNTIME_ENDPOINT': self.runtime_executor.get_runtime_endpoint()}
        if self.function_version.config.handler:
            env_vars['_HANDLER'] = self.function_version.config.handler
        if self.function_version.config.runtime:
            env_vars['AWS_EXECUTION_ENV'] = f'Aws_Lambda_{self.function_version.config.runtime}'
        if self.function_version.config.environment:
            env_vars.update(self.function_version.config.environment)
        if config.LAMBDA_INIT_DEBUG:
            env_vars['LOCALSTACK_USER'] = 'root'
        if config.LAMBDA_INIT_USER is not None:
            env_vars['LOCALSTACK_USER'] = config.LAMBDA_INIT_USER
        if config.DEBUG:
            env_vars['LOCALSTACK_INIT_LOG_LEVEL'] = 'debug'
        if config.LAMBDA_INIT_POST_INVOKE_WAIT_MS:
            env_vars['LOCALSTACK_POST_INVOKE_WAIT_MS'] = int(config.LAMBDA_INIT_POST_INVOKE_WAIT_MS)
        return env_vars

    def start(self) -> None:
        if False:
            print('Hello World!')
        '\n        Starting the runtime environment\n        '
        with self.status_lock:
            if self.status != RuntimeStatus.INACTIVE:
                raise InvalidStatusException(f'Execution environment {self.id} can only be started when inactive. Current status: {self.status}')
            self.status = RuntimeStatus.STARTING
        self.startup_timer = Timer(STARTUP_TIMEOUT_SEC, self.timed_out)
        self.startup_timer.start()
        try:
            time_before = time.perf_counter()
            self.runtime_executor.start(self.get_environment_variables())
            LOG.debug('Start of execution environment %s for function %s took %0.2fms', self.id, self.function_version.qualified_arn, (time.perf_counter() - time_before) * 1000)
            with self.status_lock:
                self.status = RuntimeStatus.READY
        except Exception as e:
            if self.status == RuntimeStatus.STARTUP_TIMED_OUT:
                raise EnvironmentStartupTimeoutException('Execution environment timed out during startup.') from e
            else:
                LOG.warning('Failed to start execution environment %s: %s', self.id, e)
                self.errored()
            raise
        finally:
            if self.startup_timer:
                self.startup_timer.cancel()
                self.startup_timer = None

    def stop(self) -> None:
        if False:
            return 10
        '\n        Stopping the runtime environment\n        '
        with self.status_lock:
            if self.status in [RuntimeStatus.INACTIVE, RuntimeStatus.STOPPED]:
                raise InvalidStatusException(f'Execution environment {self.id} cannot be stopped when inactive or already stopped. Current status: {self.status}')
            self.status = RuntimeStatus.STOPPED
        self.runtime_executor.stop()
        self.keepalive_timer.cancel()

    def release(self) -> None:
        if False:
            print('Hello World!')
        self.last_returned = datetime.now()
        with self.status_lock:
            if self.status != RuntimeStatus.RUNNING:
                raise InvalidStatusException(f'Execution environment {self.id} can only be set to status ready while running. Current status: {self.status}')
            self.status = RuntimeStatus.READY
        if self.initialization_type == 'on-demand':
            self.keepalive_timer = Timer(config.LAMBDA_KEEPALIVE_MS / 1000, self.keepalive_passed)
            self.keepalive_timer.start()

    def reserve(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.status_lock:
            if self.status != RuntimeStatus.READY:
                raise InvalidStatusException(f'Execution environment {self.id} can only be reserved if ready.  Current status: {self.status}')
            self.status = RuntimeStatus.RUNNING
        self.keepalive_timer.cancel()

    def keepalive_passed(self) -> None:
        if False:
            return 10
        LOG.debug('Execution environment %s for function %s has not received any invocations in a while. Stopping.', self.id, self.function_version.qualified_arn)
        self.stop()
        self.on_timeout(self.function_version.qualified_arn, self.id)

    def timed_out(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handle status updates if the startup of an execution environment times out.\n        Invoked asynchronously by the startup timer in a separate thread.'
        LOG.warning('Execution environment %s for function %s timed out during startup. Check for errors during the startup of your Lambda function and consider increasing the startup timeout via LAMBDA_RUNTIME_ENVIRONMENT_TIMEOUT.', self.id, self.function_version.qualified_arn)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f'Logs from the execution environment {self.id} after startup timeout:\n{self.get_prefixed_logs()}')
        with self.status_lock:
            if self.status != RuntimeStatus.STARTING:
                raise InvalidStatusException(f'Execution environment {self.id} can only time out while starting. Current status: {self.status}')
            self.status = RuntimeStatus.STARTUP_TIMED_OUT
        try:
            self.runtime_executor.stop()
        except Exception as e:
            LOG.debug('Unable to shutdown execution environment %s after timeout: %s', self.id, e)

    def errored(self) -> None:
        if False:
            return 10
        'Handle status updates if the startup of an execution environment fails.\n        Invoked synchronously when an unexpected error occurs during startup.'
        LOG.warning('Execution environment %s for function %s failed during startup. Check for errors during the startup of your Lambda function.', self.id, self.function_version.qualified_arn)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f'Logs from the execution environment {self.id} after startup error:\n{self.get_prefixed_logs()}')
        with self.status_lock:
            if self.status != RuntimeStatus.STARTING:
                raise InvalidStatusException(f'Execution environment {self.id} can only error while starting. Current status: {self.status}')
            self.status = RuntimeStatus.STARTUP_FAILED
        try:
            self.runtime_executor.stop()
        except Exception as e:
            LOG.debug('Unable to shutdown execution environment %s after error: %s', self.id, e)

    def get_prefixed_logs(self) -> str:
        if False:
            print('Hello World!')
        'Returns prefixed lambda containers logs'
        logs = self.runtime_executor.get_logs()
        prefix = f'[lambda {self.id}] '
        prefixed_logs = logs.replace('\n', f'\n{prefix}')
        return f'{prefix}{prefixed_logs}'

    def invoke(self, invocation: Invocation) -> InvocationResult:
        if False:
            i = 10
            return i + 15
        assert self.status == RuntimeStatus.RUNNING
        invoke_payload = {'invoke-id': invocation.request_id, 'invoked-function-arn': invocation.invoked_arn, 'payload': to_str(invocation.payload), 'trace-id': self._generate_trace_header()}
        return self.runtime_executor.invoke(payload=invoke_payload)

    def get_credentials(self) -> Credentials:
        if False:
            print('Hello World!')
        sts_client = connect_to().sts.request_metadata(service_principal='lambda')
        role_session_name = self.function_version.id.function_name
        if len(role_session_name) == 1:
            role_session_name += '@lambda_function'
        return sts_client.assume_role(RoleArn=self.function_version.config.role, RoleSessionName=role_session_name, DurationSeconds=43200)['Credentials']

    def _generate_trace_id(self):
        if False:
            while True:
                i = 10
        'https://docs.aws.amazon.com/xray/latest/devguide/xray-api-sendingdata.html#xray-api-traceids'
        original_request_epoch = int(time.time())
        timestamp_hex = hex(original_request_epoch)[2:]
        version_number = '1'
        unique_id = binascii.hexlify(os.urandom(12)).decode('utf-8')
        return f'{version_number}-{timestamp_hex}-{unique_id}'

    def _generate_trace_header(self):
        if False:
            return 10
        '\n        https://docs.aws.amazon.com/lambda/latest/dg/services-xray.html\n\n        "The sampling rate is 1 request per second and 5 percent of additional requests."\n\n        Currently we implement a simpler, more predictable strategy.\n        If TracingMode is "Active", we always sample the request. (Sampled=1)\n\n        TODO: implement passive tracing\n        TODO: use xray sdk here\n        '
        if self.function_version.config.tracing_config_mode == TracingMode.Active:
            sampled = '1'
        else:
            sampled = '0'
        root_trace_id = self._generate_trace_id()
        parent = binascii.b2a_hex(os.urandom(8)).decode('utf-8')
        return f'Root={root_trace_id};Parent={parent};Sampled={sampled}'