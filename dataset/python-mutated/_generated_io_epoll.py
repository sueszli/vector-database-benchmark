from __future__ import annotations
from typing import TYPE_CHECKING
from ._ki import LOCALS_KEY_KI_PROTECTION_ENABLED
from ._run import GLOBAL_RUN_CONTEXT
if TYPE_CHECKING:
    from .._file_io import _HasFileNo
import sys
assert not TYPE_CHECKING or sys.platform == 'linux'

async def wait_readable(fd: int | _HasFileNo) -> None:
    """Block until the kernel reports that the given object is readable.

    On Unix systems, ``fd`` must either be an integer file descriptor,
    or else an object with a ``.fileno()`` method which returns an
    integer file descriptor. Any kind of file descriptor can be passed,
    though the exact semantics will depend on your kernel. For example,
    this probably won't do anything useful for on-disk files.

    On Windows systems, ``fd`` must either be an integer ``SOCKET``
    handle, or else an object with a ``.fileno()`` method which returns
    an integer ``SOCKET`` handle. File descriptors aren't supported,
    and neither are handles that refer to anything besides a
    ``SOCKET``.

    :raises trio.BusyResourceError:
        if another task is already waiting for the given socket to
        become readable.
    :raises trio.ClosedResourceError:
        if another task calls :func:`notify_closing` while this
        function is still working.
    """
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.wait_readable(fd)
    except AttributeError:
        raise RuntimeError('must be called from async context') from None

async def wait_writable(fd: int | _HasFileNo) -> None:
    """Block until the kernel reports that the given object is writable.

    See `wait_readable` for the definition of ``fd``.

    :raises trio.BusyResourceError:
        if another task is already waiting for the given socket to
        become writable.
    :raises trio.ClosedResourceError:
        if another task calls :func:`notify_closing` while this
        function is still working.
    """
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.wait_writable(fd)
    except AttributeError:
        raise RuntimeError('must be called from async context') from None

def notify_closing(fd: int | _HasFileNo) -> None:
    if False:
        return 10
    "Notify waiters of the given object that it will be closed.\n\n    Call this before closing a file descriptor (on Unix) or socket (on\n    Windows). This will cause any `wait_readable` or `wait_writable`\n    calls on the given object to immediately wake up and raise\n    `~trio.ClosedResourceError`.\n\n    This doesn't actually close the object – you still have to do that\n    yourself afterwards. Also, you want to be careful to make sure no\n    new tasks start waiting on the object in between when you call this\n    and when it's actually closed. So to close something properly, you\n    usually want to do these steps in order:\n\n    1. Explicitly mark the object as closed, so that any new attempts\n       to use it will abort before they start.\n    2. Call `notify_closing` to wake up any already-existing users.\n    3. Actually close the object.\n\n    It's also possible to do them in a different order if that's more\n    convenient, *but only if* you make sure not to have any checkpoints in\n    between the steps. This way they all happen in a single atomic\n    step, so other tasks won't be able to tell what order they happened\n    in anyway.\n    "
    locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = True
    try:
        return GLOBAL_RUN_CONTEXT.runner.io_manager.notify_closing(fd)
    except AttributeError:
        raise RuntimeError('must be called from async context') from None