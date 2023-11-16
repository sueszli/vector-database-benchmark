import logging
import os
import signal
import sys
import threading
import traceback
import boto3
from moto.core import BaseModel
from moto.core.base_backend import InstanceTrackerMeta
from localstack import config, constants
from localstack.aws.accounts import get_aws_account_id
from localstack.constants import AWS_REGION_US_EAST_1, ENV_DEV, LOCALSTACK_INFRA_PROCESS
from localstack.http.duplex_socket import enable_duplex_socket
from localstack.runtime import events, hooks
from localstack.runtime.exceptions import LocalstackExit
from localstack.services.plugins import SERVICE_PLUGINS, ServiceDisabled, wait_for_infra_shutdown
from localstack.utils import files, objects
from localstack.utils.analytics import usage
from localstack.utils.aws.request_context import patch_moto_request_handling
from localstack.utils.bootstrap import get_enabled_apis, log_duration, setup_logging, should_eager_load_api
from localstack.utils.container_networking import get_main_container_id
from localstack.utils.files import cleanup_tmp_files
from localstack.utils.net import is_port_open
from localstack.utils.patch import patch
from localstack.utils.platform import in_docker
from localstack.utils.sync import poll_condition
from localstack.utils.threads import cleanup_threads_and_processes, start_thread
READY_MARKER_OUTPUT = constants.READY_MARKER_OUTPUT
DEFAULT_BACKEND_HOST = '127.0.0.1'
LOG = logging.getLogger(__name__)
INFRA_READY = events.infra_ready
SHUTDOWN_INFRA = threading.Event()
EXIT_CODE: objects.Value[int] = objects.Value(0)

def patch_urllib3_connection_pool(**constructor_kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Override the default parameters of HTTPConnectionPool, e.g., set the pool size via maxsize=16\n    '
    try:
        from urllib3 import connectionpool, poolmanager

        class MyHTTPSConnectionPool(connectionpool.HTTPSConnectionPool):

            def __init__(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                kwargs.update(constructor_kwargs)
                super(MyHTTPSConnectionPool, self).__init__(*args, **kwargs)
        poolmanager.pool_classes_by_scheme['https'] = MyHTTPSConnectionPool

        class MyHTTPConnectionPool(connectionpool.HTTPConnectionPool):

            def __init__(self, *args, **kwargs):
                if False:
                    return 10
                kwargs.update(constructor_kwargs)
                super(MyHTTPConnectionPool, self).__init__(*args, **kwargs)
        poolmanager.pool_classes_by_scheme['http'] = MyHTTPConnectionPool
    except Exception:
        pass

def patch_instance_tracker_meta():
    if False:
        for i in range(10):
            print('nop')
    'Avoid instance collection for moto dashboard'
    if hasattr(InstanceTrackerMeta, '_ls_patch_applied'):
        return

    @patch(InstanceTrackerMeta.__new__, pass_target=False)
    def new_instance(meta, name, bases, dct):
        if False:
            i = 10
            return i + 15
        cls = super(InstanceTrackerMeta, meta).__new__(meta, name, bases, dct)
        if name == 'BaseModel':
            return cls
        cls.instances = []
        return cls

    @patch(BaseModel.__new__, pass_target=False)
    def new_basemodel(cls, *args, **kwargs):
        if False:
            return 10
        instance = super(BaseModel, cls).__new__(cls)
        return instance
    InstanceTrackerMeta._ls_patch_applied = True

def exit_infra(code: int):
    if False:
        print('Hello World!')
    '\n    Triggers an orderly shutdown of the localstack infrastructure and sets the code the main process should\n    exit with to a specific value.\n\n    :param code: the exit code the main process should return with\n    '
    EXIT_CODE.set(code)
    SHUTDOWN_INFRA.set()

def stop_infra():
    if False:
        for i in range(10):
            print('nop')
    if events.infra_stopping.is_set():
        return
    usage.aggregate_and_send()
    events.infra_stopping.set()
    try:
        LOG.debug('[shutdown] Running shutdown hooks ...')
        hooks.on_infra_shutdown.run()
        LOG.debug('[shutdown] Cleaning up resources ...')
        cleanup_resources()
        if config.FORCE_SHUTDOWN:
            LOG.debug('[shutdown] Force shutdown, not waiting for infrastructure to shut down')
            return
        LOG.debug('[shutdown] Waiting for infrastructure to shut down ...')
        wait_for_infra_shutdown()
        LOG.debug('[shutdown] Infrastructure is shut down')
    finally:
        events.infra_stopped.set()

def cleanup_resources():
    if False:
        return 10
    cleanup_tmp_files()
    cleanup_threads_and_processes()
    if config.CLEAR_TMP_FOLDER:
        try:
            files.rm_rf(config.dirs.tmp)
        except PermissionError as e:
            LOG.error('unable to delete temp folder %s: %s, please delete manually or you will keep seeing these errors', config.dirs.tmp, e)
        try:
            files.rm_rf(config.dirs.mounted_tmp)
        except PermissionError as e:
            LOG.error('unable to delete mounted temp folder %s: %s, please delete manually or you will keep seeing these errors', config.dirs.mounted_tmp, e)

def gateway_listen_ports_info() -> str:
    if False:
        print('Hello World!')
    'Example: http port [4566,443]'
    gateway_listen_ports = [gw_listen.port for gw_listen in config.GATEWAY_LISTEN]
    return f'{config.get_protocol()} port {gateway_listen_ports}'

def log_startup_message(service):
    if False:
        for i in range(10):
            print('nop')
    LOG.info('Starting mock %s service on %s ...', service, gateway_listen_ports_info())

def check_aws_credentials():
    if False:
        return 10
    os.environ['AWS_ACCESS_KEY_ID'] = get_aws_account_id()
    os.environ['AWS_SECRET_ACCESS_KEY'] = constants.INTERNAL_AWS_SECRET_ACCESS_KEY
    session = boto3.Session()
    credentials = session.get_credentials()
    assert credentials

def signal_supervisor_restart():
    if False:
        for i in range(10):
            print('nop')
    if (pid := os.environ.get('SUPERVISOR_PID')):
        os.kill(int(pid), signal.SIGUSR1)
    else:
        LOG.warning('could not signal supervisor to restart localstack')

def print_runtime_information(in_docker=False):
    if False:
        i = 10
        return i + 15
    print()
    print('LocalStack version: %s' % constants.VERSION)
    if in_docker:
        id = get_main_container_id()
        if id:
            print('LocalStack Docker container id: %s' % id[:12])
    if config.LOCALSTACK_BUILD_DATE:
        print('LocalStack build date: %s' % config.LOCALSTACK_BUILD_DATE)
    if config.LOCALSTACK_BUILD_GIT_HASH:
        print('LocalStack build git hash: %s' % config.LOCALSTACK_BUILD_GIT_HASH)
    print()

def start_infra(asynchronous=False, apis=None):
    if False:
        for i in range(10):
            print('nop')
    if config.CLEAR_TMP_FOLDER:
        try:
            files.rm_rf(config.dirs.tmp)
        except PermissionError as e:
            LOG.error('unable to delete temp folder %s: %s, please delete manually or you will keep seeing these errors', config.dirs.tmp, e)
    config.dirs.mkdirs()
    events.infra_starting.set()
    try:
        os.environ[LOCALSTACK_INFRA_PROCESS] = '1'
        is_in_docker = in_docker()
        print_runtime_information(is_in_docker)
        patch_urllib3_connection_pool(maxsize=128)
        patch_instance_tracker_meta()
        setup_logging()
        hooks.on_infra_start.run()
        thread = do_start_infra(asynchronous, apis, is_in_docker)
        if not asynchronous and thread:
            SHUTDOWN_INFRA.wait()
        return thread
    except KeyboardInterrupt:
        print('Shutdown')
    except LocalstackExit as e:
        print(f'Localstack returning with exit code {e.code}. Reason: {e}')
        raise
    except Exception as e:
        print('Unexpected exception while starting infrastructure: %s %s' % (e, traceback.format_exc()))
        raise e
    finally:
        sys.stdout.flush()
        if not asynchronous:
            stop_infra()

def do_start_infra(asynchronous, apis, is_in_docker):
    if False:
        while True:
            i = 10
    if config.DEVELOP:
        from localstack.packages.debugpy import debugpy_package
        debugpy_package.install()
        import debugpy
        LOG.info('Starting debug server at: %s:%s', constants.BIND_HOST, config.DEVELOP_PORT)
        debugpy.listen((constants.BIND_HOST, config.DEVELOP_PORT))
        if config.WAIT_FOR_DEBUGGER:
            debugpy.wait_for_client()

    @log_duration()
    def prepare_environment():
        if False:
            for i in range(10):
                print('nop')
        enable_duplex_socket()
        os.environ['AWS_REGION'] = AWS_REGION_US_EAST_1
        os.environ['ENV'] = ENV_DEV
        check_aws_credentials()
        patch_moto_request_handling()

    @log_duration()
    def preload_services():
        if False:
            print('Hello World!')
        '\n        Preload services - restore persistence, and initialize services if EAGER_SERVICE_LOADING=1.\n        '
        available_services = get_enabled_apis()
        if not config.EAGER_SERVICE_LOADING:
            return
        for api in available_services:
            if should_eager_load_api(api):
                try:
                    SERVICE_PLUGINS.require(api)
                except ServiceDisabled as e:
                    LOG.debug('%s', e)
                except Exception:
                    LOG.exception('could not load service plugin %s', api)

    @log_duration()
    def start_runtime_components():
        if False:
            i = 10
            return i + 15
        from localstack.services.edge import start_edge
        t = start_thread(start_edge, quiet=False)
        if not poll_condition(lambda : is_port_open(config.GATEWAY_LISTEN[0].port), timeout=15, interval=0.3):
            if LOG.isEnabledFor(logging.DEBUG):
                is_port_open(config.GATEWAY_LISTEN[0].port, quiet=False)
            raise TimeoutError(f'gave up waiting for edge server on {config.GATEWAY_LISTEN[0].host_and_port()}')
        return t
    prepare_environment()
    thread = start_runtime_components()
    preload_services()
    print(READY_MARKER_OUTPUT)
    sys.stdout.flush()
    events.infra_ready.set()
    hooks.on_infra_ready.run()
    return thread