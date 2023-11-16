import contextlib
import threading
import warnings
_thread_local = threading.local()

class DeviceSynchronized(RuntimeError):
    """Raised when device synchronization is detected while disallowed.

    .. warning::

       This API has been deprecated in CuPy v10 and will be removed in future
       releases.

    .. seealso:: :func:`cupyx.allow_synchronize`

    """

    def __init__(self, message=None):
        if False:
            print('Hello World!')
        if message is None:
            message = 'Device synchronization was detected while disallowed.'
        super().__init__(message)

def _is_allowed():
    if False:
        return 10
    try:
        return _thread_local.allowed
    except AttributeError:
        _thread_local.allowed = True
        return True

def _declare_synchronize():
    if False:
        while True:
            i = 10
    if not _is_allowed():
        raise DeviceSynchronized()

@contextlib.contextmanager
def allow_synchronize(allow):
    if False:
        print('Hello World!')
    'Allows or disallows device synchronization temporarily in the current thread.\n\n    .. warning::\n\n       This API has been deprecated in CuPy v10 and will be removed in future\n       releases.\n\n    If device synchronization is detected, :class:`cupyx.DeviceSynchronized`\n    will be raised.\n\n    Note that there can be false negatives and positives.\n    Device synchronization outside CuPy will not be detected.\n    '
    warnings.warn('cupyx.allow_synchronize will be removed in future releases as it is not possible to reliably detect synchronizations.')
    old = _is_allowed()
    _thread_local.allowed = allow
    try:
        yield
    finally:
        _thread_local.allowed = old