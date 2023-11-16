import sys
from metaflow.exception import MetaflowException, MetaflowInternalError
from metaflow.plugins.gcp.gs_exceptions import MetaflowGSPackageError

def parse_gs_full_path(gs_uri):
    if False:
        print('Hello World!')
    from urllib.parse import urlparse
    (scheme, netloc, path, _, _, _) = urlparse(gs_uri)
    assert scheme == 'gs'
    assert netloc is not None
    bucket = netloc
    path = path.lstrip('/').rstrip('/')
    if path == '':
        path = None
    return (bucket, path)

def _check_and_init_gs_deps():
    if False:
        for i in range(10):
            print('nop')
    try:
        from google.cloud import storage
        import google.auth
    except ImportError:
        raise MetaflowGSPackageError()
    if sys.version_info[:2] < (3, 7):
        raise MetaflowException(msg='Metaflow may only use Google Cloud Storage with Python 3.7 or newer')

def check_gs_deps(func):
    if False:
        print('Hello World!')
    'The decorated function checks GS dependencies (as needed for Azure storage backend). This includes\n    various GCP SDK packages, as well as a Python version of >=3.7\n    '

    def _inner_func(*args, **kwargs):
        if False:
            return 10
        _check_and_init_gs_deps()
        return func(*args, **kwargs)
    return _inner_func

@check_gs_deps
def process_gs_exception(e):
    if False:
        return 10
    '\n    Translate errors to Metaflow errors for standardized messaging. The intent is that all\n    Google Cloud Storage integration logic should send errors to this function for\n    translation.\n\n    We explicitly EXCLUDE executor related errors here.  See handle_executor_exceptions\n    '
    if isinstance(e, MetaflowException):
        raise
    if isinstance(e, ImportError):
        raise
    raise MetaflowInternalError(msg=str(e))