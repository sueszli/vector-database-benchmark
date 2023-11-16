import os
import sys
import types
from metaflow.exception import MetaflowException
from metaflow.metaflow_config_funcs import from_conf, get_validate_choice_fn
if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
DEFAULT_DATASTORE = from_conf('DEFAULT_DATASTORE', 'local')
DEFAULT_ENVIRONMENT = from_conf('DEFAULT_ENVIRONMENT', 'local')
DEFAULT_EVENT_LOGGER = from_conf('DEFAULT_EVENT_LOGGER', 'nullSidecarLogger')
DEFAULT_METADATA = from_conf('DEFAULT_METADATA', 'local')
DEFAULT_MONITOR = from_conf('DEFAULT_MONITOR', 'nullSidecarMonitor')
DEFAULT_PACKAGE_SUFFIXES = from_conf('DEFAULT_PACKAGE_SUFFIXES', '.py,.R,.RDS')
DEFAULT_AWS_CLIENT_PROVIDER = from_conf('DEFAULT_AWS_CLIENT_PROVIDER', 'boto3')
DEFAULT_SECRETS_BACKEND_TYPE = from_conf('DEFAULT_SECRETS_BACKEND_TYPE')
DEFAULT_SECRETS_ROLE = from_conf('DEFAULT_SECRETS_ROLE')
USER = from_conf('USER')
DATASTORE_LOCAL_DIR = '.metaflow'
DATASTORE_SYSROOT_LOCAL = from_conf('DATASTORE_SYSROOT_LOCAL')
DATASTORE_SYSROOT_S3 = from_conf('DATASTORE_SYSROOT_S3')
DATASTORE_SYSROOT_AZURE = from_conf('DATASTORE_SYSROOT_AZURE')
DATASTORE_SYSROOT_GS = from_conf('DATASTORE_SYSROOT_GS')
CLIENT_CACHE_PATH = from_conf('CLIENT_CACHE_PATH', '/tmp/metaflow_client')
CLIENT_CACHE_MAX_SIZE = from_conf('CLIENT_CACHE_MAX_SIZE', 10000)
CLIENT_CACHE_MAX_FLOWDATASTORE_COUNT = from_conf('CLIENT_CACHE_MAX_FLOWDATASTORE_COUNT', 50)
CLIENT_CACHE_MAX_TASKDATASTORE_COUNT = from_conf('CLIENT_CACHE_MAX_TASKDATASTORE_COUNT', CLIENT_CACHE_MAX_FLOWDATASTORE_COUNT * 100)
S3_ENDPOINT_URL = from_conf('S3_ENDPOINT_URL')
S3_VERIFY_CERTIFICATE = from_conf('S3_VERIFY_CERTIFICATE')
S3_SERVER_SIDE_ENCRYPTION = from_conf('S3_SERVER_SIDE_ENCRYPTION')
S3_RETRY_COUNT = from_conf('S3_RETRY_COUNT', 7)
S3_TRANSIENT_RETRY_COUNT = from_conf('S3_TRANSIENT_RETRY_COUNT', 20)
RETRY_WARNING_THRESHOLD = 3
DATATOOLS_SUFFIX = from_conf('DATATOOLS_SUFFIX', 'data')
DATATOOLS_S3ROOT = from_conf('DATATOOLS_S3ROOT', os.path.join(DATASTORE_SYSROOT_S3, DATATOOLS_SUFFIX) if DATASTORE_SYSROOT_S3 else None)
TEMPDIR = from_conf('TEMPDIR', '.')
DATATOOLS_CLIENT_PARAMS = from_conf('DATATOOLS_CLIENT_PARAMS', {})
if S3_ENDPOINT_URL:
    DATATOOLS_CLIENT_PARAMS['endpoint_url'] = S3_ENDPOINT_URL
if S3_VERIFY_CERTIFICATE:
    DATATOOLS_CLIENT_PARAMS['verify'] = S3_VERIFY_CERTIFICATE
DATATOOLS_SESSION_VARS = from_conf('DATATOOLS_SESSION_VARS', {})
DATATOOLS_AZUREROOT = from_conf('DATATOOLS_AZUREROOT', os.path.join(DATASTORE_SYSROOT_AZURE, DATATOOLS_SUFFIX) if DATASTORE_SYSROOT_AZURE else None)
DATATOOLS_GSROOT = from_conf('DATATOOLS_GSROOT', os.path.join(DATASTORE_SYSROOT_GS, DATATOOLS_SUFFIX) if DATASTORE_SYSROOT_GS else None)
DATATOOLS_LOCALROOT = from_conf('DATATOOLS_LOCALROOT', os.path.join(DATASTORE_SYSROOT_LOCAL, DATATOOLS_SUFFIX) if DATASTORE_SYSROOT_LOCAL else None)
AWS_SECRETS_MANAGER_DEFAULT_REGION = from_conf('AWS_SECRETS_MANAGER_DEFAULT_REGION')
ARTIFACT_LOCALROOT = from_conf('ARTIFACT_LOCALROOT', os.getcwd())
CARD_SUFFIX = 'mf.cards'
CARD_LOCALROOT = from_conf('CARD_LOCALROOT')
CARD_S3ROOT = from_conf('CARD_S3ROOT', os.path.join(DATASTORE_SYSROOT_S3, CARD_SUFFIX) if DATASTORE_SYSROOT_S3 else None)
CARD_AZUREROOT = from_conf('CARD_AZUREROOT', os.path.join(DATASTORE_SYSROOT_AZURE, CARD_SUFFIX) if DATASTORE_SYSROOT_AZURE else None)
CARD_GSROOT = from_conf('CARD_GSROOT', os.path.join(DATASTORE_SYSROOT_GS, CARD_SUFFIX) if DATASTORE_SYSROOT_GS else None)
CARD_NO_WARNING = from_conf('CARD_NO_WARNING', False)
SKIP_CARD_DUALWRITE = from_conf('SKIP_CARD_DUALWRITE', False)
AZURE_STORAGE_BLOB_SERVICE_ENDPOINT = from_conf('AZURE_STORAGE_BLOB_SERVICE_ENDPOINT')
AZURE_STORAGE_WORKLOAD_TYPE = from_conf('AZURE_STORAGE_WORKLOAD_TYPE', default='general', validate_fn=get_validate_choice_fn(['general', 'high_throughput']))
GS_STORAGE_WORKLOAD_TYPE = from_conf('GS_STORAGE_WORKLOAD_TYPE', 'general', validate_fn=get_validate_choice_fn(['general', 'high_throughput']))
SERVICE_URL = from_conf('SERVICE_URL')
SERVICE_RETRY_COUNT = from_conf('SERVICE_RETRY_COUNT', 5)
SERVICE_AUTH_KEY = from_conf('SERVICE_AUTH_KEY')
SERVICE_HEADERS = from_conf('SERVICE_HEADERS', {})
if SERVICE_AUTH_KEY is not None:
    SERVICE_HEADERS['x-api-key'] = SERVICE_AUTH_KEY
SERVICE_VERSION_CHECK = from_conf('SERVICE_VERSION_CHECK', True)
DEFAULT_CONTAINER_IMAGE = from_conf('DEFAULT_CONTAINER_IMAGE')
DEFAULT_CONTAINER_REGISTRY = from_conf('DEFAULT_CONTAINER_REGISTRY')
UI_URL = from_conf('UI_URL')
CONTACT_INFO = from_conf('CONTACT_INFO', {'Read the documentation': 'http://docs.metaflow.org', 'Chat with us': 'http://chat.metaflow.org', 'Get help by email': 'help@metaflow.org'})
ECS_S3_ACCESS_IAM_ROLE = from_conf('ECS_S3_ACCESS_IAM_ROLE')
ECS_FARGATE_EXECUTION_ROLE = from_conf('ECS_FARGATE_EXECUTION_ROLE')
BATCH_JOB_QUEUE = from_conf('BATCH_JOB_QUEUE')
BATCH_CONTAINER_IMAGE = from_conf('BATCH_CONTAINER_IMAGE', DEFAULT_CONTAINER_IMAGE)
BATCH_CONTAINER_REGISTRY = from_conf('BATCH_CONTAINER_REGISTRY', DEFAULT_CONTAINER_REGISTRY)
SERVICE_INTERNAL_URL = from_conf('SERVICE_INTERNAL_URL', SERVICE_URL)
BATCH_EMIT_TAGS = from_conf('BATCH_EMIT_TAGS', False)
SFN_IAM_ROLE = from_conf('SFN_IAM_ROLE')
SFN_DYNAMO_DB_TABLE = from_conf('SFN_DYNAMO_DB_TABLE')
EVENTS_SFN_ACCESS_IAM_ROLE = from_conf('EVENTS_SFN_ACCESS_IAM_ROLE')
SFN_STATE_MACHINE_PREFIX = from_conf('SFN_STATE_MACHINE_PREFIX')
SFN_EXECUTION_LOG_GROUP_ARN = from_conf('SFN_EXECUTION_LOG_GROUP_ARN')
KUBERNETES_NAMESPACE = from_conf('KUBERNETES_NAMESPACE', 'default')
KUBERNETES_SERVICE_ACCOUNT = from_conf('KUBERNETES_SERVICE_ACCOUNT')
KUBERNETES_NODE_SELECTOR = from_conf('KUBERNETES_NODE_SELECTOR', '')
KUBERNETES_TOLERATIONS = from_conf('KUBERNETES_TOLERATIONS', '')
KUBERNETES_PERSISTENT_VOLUME_CLAIMS = from_conf('KUBERNETES_PERSISTENT_VOLUME_CLAIMS', '')
KUBERNETES_SECRETS = from_conf('KUBERNETES_SECRETS', '')
KUBERNETES_LABELS = from_conf('KUBERNETES_LABELS', '')
KUBERNETES_GPU_VENDOR = from_conf('KUBERNETES_GPU_VENDOR', 'nvidia')
KUBERNETES_CONTAINER_IMAGE = from_conf('KUBERNETES_CONTAINER_IMAGE', DEFAULT_CONTAINER_IMAGE)
KUBERNETES_IMAGE_PULL_POLICY = from_conf('KUBERNETES_IMAGE_PULL_POLICY', None)
KUBERNETES_CONTAINER_REGISTRY = from_conf('KUBERNETES_CONTAINER_REGISTRY', DEFAULT_CONTAINER_REGISTRY)
KUBERNETES_FETCH_EC2_METADATA = from_conf('KUBERNETES_FETCH_EC2_METADATA', False)
ARGO_WORKFLOWS_KUBERNETES_SECRETS = from_conf('ARGO_WORKFLOWS_KUBERNETES_SECRETS', '')
ARGO_WORKFLOWS_ENV_VARS_TO_SKIP = from_conf('ARGO_WORKFLOWS_ENV_VARS_TO_SKIP', '')
ARGO_EVENTS_SERVICE_ACCOUNT = from_conf('ARGO_EVENTS_SERVICE_ACCOUNT')
ARGO_EVENTS_EVENT_BUS = from_conf('ARGO_EVENTS_EVENT_BUS', 'default')
ARGO_EVENTS_EVENT_SOURCE = from_conf('ARGO_EVENTS_EVENT_SOURCE')
ARGO_EVENTS_EVENT = from_conf('ARGO_EVENTS_EVENT')
ARGO_EVENTS_WEBHOOK_URL = from_conf('ARGO_EVENTS_WEBHOOK_URL')
ARGO_EVENTS_INTERNAL_WEBHOOK_URL = from_conf('ARGO_EVENTS_INTERNAL_WEBHOOK_URL', ARGO_EVENTS_WEBHOOK_URL)
ARGO_EVENTS_WEBHOOK_AUTH = from_conf('ARGO_EVENTS_WEBHOOK_AUTH', 'none')
ARGO_WORKFLOWS_UI_URL = from_conf('ARGO_WORKFLOWS_UI_URL')
AIRFLOW_KUBERNETES_STARTUP_TIMEOUT_SECONDS = from_conf('AIRFLOW_KUBERNETES_STARTUP_TIMEOUT_SECONDS', 60 * 60)
AIRFLOW_KUBERNETES_CONN_ID = from_conf('AIRFLOW_KUBERNETES_CONN_ID')
AIRFLOW_KUBERNETES_KUBECONFIG_FILE = from_conf('AIRFLOW_KUBERNETES_KUBECONFIG_FILE')
AIRFLOW_KUBERNETES_KUBECONFIG_CONTEXT = from_conf('AIRFLOW_KUBERNETES_KUBECONFIG_CONTEXT')
CONDA_PACKAGE_S3ROOT = from_conf('CONDA_PACKAGE_S3ROOT')
CONDA_PACKAGE_AZUREROOT = from_conf('CONDA_PACKAGE_AZUREROOT')
CONDA_PACKAGE_GSROOT = from_conf('CONDA_PACKAGE_GSROOT')
CONDA_DEPENDENCY_RESOLVER = from_conf('CONDA_DEPENDENCY_RESOLVER', 'conda')
DEBUG_OPTIONS = ['subcommand', 'sidecar', 's3client', 'tracing']
for typ in DEBUG_OPTIONS:
    vars()['DEBUG_%s' % typ.upper()] = from_conf('DEBUG_%s' % typ.upper(), False)
AWS_SANDBOX_ENABLED = from_conf('AWS_SANDBOX_ENABLED', False)
AWS_SANDBOX_STS_ENDPOINT_URL = SERVICE_URL
AWS_SANDBOX_API_KEY = from_conf('AWS_SANDBOX_API_KEY')
AWS_SANDBOX_INTERNAL_SERVICE_URL = from_conf('AWS_SANDBOX_INTERNAL_SERVICE_URL')
AWS_SANDBOX_REGION = from_conf('AWS_SANDBOX_REGION')
if AWS_SANDBOX_ENABLED:
    os.environ['AWS_DEFAULT_REGION'] = AWS_SANDBOX_REGION
    SERVICE_INTERNAL_URL = AWS_SANDBOX_INTERNAL_SERVICE_URL
    SERVICE_HEADERS['x-api-key'] = AWS_SANDBOX_API_KEY
    SFN_STATE_MACHINE_PREFIX = from_conf('AWS_SANDBOX_STACK_NAME')
KUBERNETES_SANDBOX_INIT_SCRIPT = from_conf('KUBERNETES_SANDBOX_INIT_SCRIPT')
OTEL_ENDPOINT = from_conf('OTEL_ENDPOINT')
ZIPKIN_ENDPOINT = from_conf('ZIPKIN_ENDPOINT')
CONSOLE_TRACE_ENABLED = from_conf('CONSOLE_TRACE_ENABLED', False)
DISABLE_TRACING = bool(os.environ.get('DISABLE_TRACING', False))
MAX_ATTEMPTS = 6

def get_pinned_conda_libs(python_version, datastore_type):
    if False:
        return 10
    pins = {'requests': '>=2.21.0'}
    if datastore_type == 's3':
        pins['boto3'] = '>=1.14.0'
    elif datastore_type == 'azure':
        pins['azure-identity'] = '>=1.10.0'
        pins['azure-storage-blob'] = '>=12.12.0'
    elif datastore_type == 'gs':
        pins['google-cloud-storage'] = '>=2.5.0'
        pins['google-auth'] = '>=2.11.0'
    elif datastore_type == 'local':
        pass
    else:
        raise MetaflowException(msg='conda lib pins for datastore %s are undefined' % (datastore_type,))
    return pins
try:
    from metaflow.extension_support import get_modules
    ext_modules = get_modules('config')
    for m in ext_modules:
        for (n, o) in m.module.__dict__.items():
            if n == 'DEBUG_OPTIONS':
                DEBUG_OPTIONS.extend(o)
                for typ in o:
                    vars()['DEBUG_%s' % typ.upper()] = from_conf('DEBUG_%s' % typ.upper(), False)
            elif n == 'get_pinned_conda_libs':

                def _new_get_pinned_conda_libs(python_version, datastore_type, f1=globals()[n], f2=o):
                    if False:
                        return 10
                    d1 = f1(python_version, datastore_type)
                    d2 = f2(python_version, datastore_type)
                    for (k, v) in d2.items():
                        d1[k] = v if k not in d1 else ','.join([d1[k], v])
                    return d1
                globals()[n] = _new_get_pinned_conda_libs
            elif not n.startswith('__') and (not isinstance(o, types.ModuleType)):
                globals()[n] = o
finally:
    for _n in ['m', 'n', 'o', 'typ', 'ext_modules', 'get_modules', '_new_get_pinned_conda_libs', 'd1', 'd2', 'k', 'v', 'f1', 'f2']:
        try:
            del globals()[_n]
        except KeyError:
            pass
    del globals()['_n']