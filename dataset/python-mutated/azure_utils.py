import sys
import time
from metaflow.plugins.azure.azure_exceptions import MetaflowAzureAuthenticationError, MetaflowAzureResourceError, MetaflowAzurePackageError
from metaflow.exception import MetaflowInternalError, MetaflowException

def _check_and_init_azure_deps():
    if False:
        while True:
            i = 10
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import azure.storage.blob
        import azure.identity
        import logging
        logging.getLogger('azure.identity').setLevel(logging.ERROR)
        logging.getLogger('msrest.serialization').setLevel(logging.ERROR)
    except ImportError:
        raise MetaflowAzurePackageError()
    if sys.version_info[:2] < (3, 6):
        raise MetaflowException(msg='Metaflow may only use Azure Blob Storage with Python 3.6 or newer')

def check_azure_deps(func):
    if False:
        print('Hello World!')
    'The decorated function checks Azure dependencies (as needed for Azure storage backend). This includes\n    various Azure SDK packages, as well as a Python version of >3.6\n\n    We also tune some warning and logging configurations to reduce excessive log lines from Azure SDK.\n    '

    def _inner_func(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        _check_and_init_azure_deps()
        return func(*args, **kwargs)
    return _inner_func

def parse_azure_full_path(blob_full_uri):
    if False:
        while True:
            i = 10
    "\n    Parse an Azure Blob Storage path str into a tuple (container_name, blob).\n\n    Expected format is: <container_name>/<blob>\n\n    This is sometimes used to parse an Azure sys root, in which case:\n\n    - <container_name> is the Azure Blob Storage container name\n    - <blob> is effectively a blob_prefix, a subpath within the container in which blobs will live\n\n    Blob may be None, if input looks like <container_name>. I.e. no slashes present.\n\n    We take a strict validation approach, doing no implicit string manipulations on\n    the user's behalf.  Path manipulations by themselves are complicated enough without\n    adding magic.\n\n    We provide clear error messages so the user knows exactly how to fix any validation error.\n    "
    if blob_full_uri.endswith('/'):
        raise ValueError('sysroot may not end with slash (got %s)' % blob_full_uri)
    if blob_full_uri.startswith('/'):
        raise ValueError('sysroot may not start with slash (got %s)' % blob_full_uri)
    if '//' in blob_full_uri:
        raise ValueError('sysroot may not contain any consecutive slashes (got %s)' % blob_full_uri)
    parts = blob_full_uri.split('/', 1)
    container_name = parts[0]
    if container_name == '':
        raise ValueError('Container name part of sysroot may not be empty (tried to parse %s)' % (blob_full_uri,))
    if len(parts) == 1:
        blob_name = None
    else:
        blob_name = parts[1]
    return (container_name, blob_name)

@check_azure_deps
def process_exception(e):
    if False:
        return 10
    '\n    Translate errors to Metaflow errors for standardized messaging. The intent is that all\n    Azure Blob Storage integration logic should send errors to this function for\n    translation.\n\n    We explicitly EXCLUDE executor related errors here.  See handle_executor_exceptions\n    '
    if isinstance(e, MetaflowException):
        raise
    if isinstance(e, ImportError):
        raise
    from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError, ResourceExistsError, AzureError
    if isinstance(e, ClientAuthenticationError):
        raise MetaflowAzureAuthenticationError(msg=str(e).splitlines()[-1])
    elif isinstance(e, (ResourceNotFoundError, ResourceExistsError)):
        raise MetaflowAzureResourceError(msg=str(e))
    elif isinstance(e, AzureError):
        raise MetaflowInternalError(msg='Azure error: %s' % str(e))
    else:
        raise MetaflowInternalError(msg=str(e))

def handle_exceptions(func):
    if False:
        while True:
            i = 10
    'This is a decorator leveraging the logic from process_exception()'

    def _inner_func(*args, **kwargs):
        if False:
            print('Hello World!')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            process_exception(e)
    return _inner_func

@check_azure_deps
def create_cacheable_default_azure_credentials(*args, **kwargs):
    if False:
        while True:
            i = 10
    'azure.identity.DefaultAzureCredential is not readily cacheable in a dictionary\n    because it does not have a content based hash and equality implementations.\n\n    We implement a subclass CacheableDefaultAzureCredential to add them.\n\n    We need this because credentials will be part of the cache key in _ClientCache.\n    '
    from azure.identity import DefaultAzureCredential

    class CacheableDefaultAzureCredential(DefaultAzureCredential):

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            super(CacheableDefaultAzureCredential, self).__init__(*args, **kwargs)
            self._hash_code = hash((args, tuple(sorted(kwargs.items()))))

        def __hash__(self):
            if False:
                return 10
            return self._hash_code

        def __eq__(self, other):
            if False:
                i = 10
                return i + 15
            return hash(self) == hash(other)
    return CacheableDefaultAzureCredential(*args, **kwargs)

@check_azure_deps
def create_static_token_credential(token_):
    if False:
        for i in range(10):
            print('nop')
    from azure.core.credentials import TokenCredential

    class StaticTokenCredential(TokenCredential):

        def __init__(self, token):
            if False:
                print('Hello World!')
            self._cached_token = token
            self._credential = None

        def get_token(self, *_scopes, **_kwargs):
            if False:
                while True:
                    i = 10
            if self._cached_token.expires_on - time.time() < 300:
                from azure.identity import DefaultAzureCredential
                self._credential = DefaultAzureCredential()
            if self._credential:
                return self._credential.get_token(*_scopes, **_kwargs)
            return self._cached_token

        def __hash__(self):
            if False:
                print('Hello World!')
            return hash(self._cached_token)

        def __eq__(self, other):
            if False:
                while True:
                    i = 10
            return self._cached_token == other._cached_token
    return StaticTokenCredential(token_)