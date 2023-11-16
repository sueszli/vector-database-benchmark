from __future__ import annotations
import math
import trio
from ._core._windows_cffi import CData, ErrorCodes, _handle, ffi, handle_array, kernel32, raise_winerror

async def WaitForSingleObject(obj: int | CData) -> None:
    """Async and cancellable variant of WaitForSingleObject. Windows only.

    Args:
      handle: A Win32 handle, as a Python integer.

    Raises:
      OSError: If the handle is invalid, e.g. when it is already closed.

    """
    handle = _handle(obj)
    retcode = kernel32.WaitForSingleObject(handle, 0)
    if retcode == ErrorCodes.WAIT_FAILED:
        raise_winerror()
    elif retcode != ErrorCodes.WAIT_TIMEOUT:
        return
    cancel_handle = kernel32.CreateEventA(ffi.NULL, True, False, ffi.NULL)
    try:
        await trio.to_thread.run_sync(WaitForMultipleObjects_sync, handle, cancel_handle, abandon_on_cancel=True, limiter=trio.CapacityLimiter(math.inf))
    finally:
        kernel32.SetEvent(cancel_handle)
        kernel32.CloseHandle(cancel_handle)

def WaitForMultipleObjects_sync(*handles: int | CData) -> None:
    if False:
        i = 10
        return i + 15
    'Wait for any of the given Windows handles to be signaled.'
    n = len(handles)
    handle_arr = handle_array(n)
    for i in range(n):
        handle_arr[i] = handles[i]
    timeout = 4294967295
    retcode = kernel32.WaitForMultipleObjects(n, handle_arr, False, timeout)
    if retcode == ErrorCodes.WAIT_FAILED:
        raise_winerror()