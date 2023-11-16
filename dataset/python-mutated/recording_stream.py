from __future__ import annotations
from rerun import bindings

class RecordingStream:
    """
    A RecordingStream is used to send data to Rerun.

    You can instantiate a RecordingStream by calling either [`rerun.init`][] (to create a global
    recording) or [`rerun.new_recording`][] (for more advanced use cases).

    Multithreading
    --------------

    A RecordingStream can safely be copied and sent to other threads.
    You can also set a recording as the global active one for all threads ([`rerun.set_global_data_recording`][])
    or just for the current thread ([`rerun.set_thread_local_data_recording`][]).

    Similarly, the `with` keyword can be used to temporarily set the active recording for the
    current thread, e.g.:
    ```
    with rec:
        rr.log(...)
    ```

    See also: [`rerun.get_data_recording`][], [`rerun.get_global_data_recording`][],
    [`rerun.get_thread_local_data_recording`][].

    Available methods
    -----------------

    Every function in the Rerun SDK that takes an optional RecordingStream as a parameter can also
    be called as a method on RecordingStream itself.

    This includes, but isn't limited to:

    - Metadata-related functions:
        [`rerun.is_enabled`][], [`rerun.get_recording_id`][], …
    - Sink-related functions:
        [`rerun.connect`][], [`rerun.spawn`][], …
    - Time-related functions:
        [`rerun.set_time_seconds`][], [`rerun.set_time_sequence`][], …
    - Log-related functions:
        [`rerun.log`][], [`rerun.log_components`][], …

    For an exhaustive list, see `help(rerun.RecordingStream)`.

    Micro-batching
    --------------

    Micro-batching using both space and time triggers (whichever comes first) is done automatically
    in a dedicated background thread.

    You can configure the frequency of the batches using the following environment variables:

    - `RERUN_FLUSH_TICK_SECS`:
        Flush frequency in seconds (default: `0.05` (50ms)).
    - `RERUN_FLUSH_NUM_BYTES`:
        Flush threshold in bytes (default: `1048576` (1MiB)).
    - `RERUN_FLUSH_NUM_ROWS`:
        Flush threshold in number of rows (default: `18446744073709551615` (u64::MAX)).

    """

    def __init__(self, inner: bindings.PyRecordingStream) -> None:
        if False:
            print('Hello World!')
        self.inner = inner
        self._prev: RecordingStream | None = None

    def __enter__(self):
        if False:
            while True:
                i = 10
        self._prev = set_thread_local_data_recording(self)
        return self

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        self._prev = set_thread_local_data_recording(self._prev)

    def to_native(self: RecordingStream | None) -> bindings.PyRecordingStream | None:
        if False:
            print('Hello World!')
        return self.inner if self is not None else None

    def __del__(self):
        if False:
            while True:
                i = 10
        recording = RecordingStream.to_native(self)
        bindings.flush(blocking=False, recording=recording)

def _patch(funcs):
    if False:
        while True:
            i = 10
    'Adds the given functions as methods to the `RecordingStream` class; injects `recording=self` in passing.'
    import functools
    import os
    from typing import Any
    if os.environ.get('RERUN_APP_ONLY'):
        return

    def eager_wrap(fn):
        if False:
            while True:
                i = 10

        @functools.wraps(fn)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            if False:
                print('Hello World!')
            kwargs['recording'] = self
            return fn(*args, **kwargs)
        return wrapper
    for fn in funcs:
        wrapper = eager_wrap(fn)
        setattr(RecordingStream, fn.__name__, wrapper)

def is_enabled(recording: RecordingStream | None=None) -> bool:
    if False:
        return 10
    '\n    Is this Rerun recording enabled.\n\n    If false, all calls to the recording are ignored.\n\n    The default can be set in [`rerun.init`][], but is otherwise `True`.\n\n    This can be controlled with the environment variable `RERUN` (e.g. `RERUN=on` or `RERUN=off`).\n\n    '
    return bindings.is_enabled(recording=RecordingStream.to_native(recording))

def get_application_id(recording: RecordingStream | None=None) -> str | None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the application ID that this recording is associated with, if any.\n\n    Parameters\n    ----------\n    recording:\n        Specifies the [`rerun.RecordingStream`][] to use.\n        If left unspecified, defaults to the current active data recording, if there is one.\n        See also: [`rerun.init`][], [`rerun.set_global_data_recording`][].\n\n    Returns\n    -------\n    str\n        The application ID that this recording is associated with.\n\n    '
    app_id = bindings.get_application_id(recording=RecordingStream.to_native(recording))
    return str(app_id) if app_id is not None else None

def get_recording_id(recording: RecordingStream | None=None) -> str | None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the recording ID that this recording is logging to, as a UUIDv4, if any.\n\n    The default recording_id is based on `multiprocessing.current_process().authkey`\n    which means that all processes spawned with `multiprocessing`\n    will have the same default recording_id.\n\n    If you are not using `multiprocessing` and still want several different Python\n    processes to log to the same Rerun instance (and be part of the same recording),\n    you will need to manually assign them all the same recording_id.\n    Any random UUIDv4 will work, or copy the recording id for the parent process.\n\n    Parameters\n    ----------\n    recording:\n        Specifies the [`rerun.RecordingStream`][] to use.\n        If left unspecified, defaults to the current active data recording, if there is one.\n        See also: [`rerun.init`][], [`rerun.set_global_data_recording`][].\n\n    Returns\n    -------\n    str\n        The recording ID that this recording is logging to.\n\n    '
    rec_id = bindings.get_recording_id(recording=RecordingStream.to_native(recording))
    return str(rec_id) if rec_id is not None else None
_patch([is_enabled, get_application_id, get_recording_id])

def get_data_recording(recording: RecordingStream | None=None) -> RecordingStream | None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the most appropriate recording to log data to, in the current context, if any.\n\n    * If `recording` is specified, returns that one;\n    * Otherwise, falls back to the currently active thread-local recording, if there is one;\n    * Otherwise, falls back to the currently active global recording, if there is one;\n    * Otherwise, returns None.\n\n    Parameters\n    ----------\n    recording:\n        Specifies the [`rerun.RecordingStream`][] to use.\n        If left unspecified, defaults to the current active data recording, if there is one.\n        See also: [`rerun.init`][], [`rerun.set_global_data_recording`][].\n\n    Returns\n    -------\n    Optional[RecordingStream]\n        The most appropriate recording to log data to, in the current context, if any.\n    '
    result = bindings.get_data_recording(recording=recording)
    return RecordingStream(result) if result is not None else None

def get_global_data_recording() -> RecordingStream | None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the currently active global recording, if any.\n\n    Returns\n    -------\n    Optional[RecordingStream]\n        The currently active global recording, if any.\n    '
    result = bindings.get_global_data_recording()
    return RecordingStream(result) if result is not None else None

def set_global_data_recording(recording: RecordingStream) -> RecordingStream | None:
    if False:
        return 10
    '\n    Replaces the currently active global recording with the specified one.\n\n    Parameters\n    ----------\n    recording:\n        The newly active global recording.\n    '
    result = bindings.set_global_data_recording(RecordingStream.to_native(recording))
    return RecordingStream(result) if result is not None else None

def get_thread_local_data_recording() -> RecordingStream | None:
    if False:
        while True:
            i = 10
    '\n    Returns the currently active thread-local recording, if any.\n\n    Returns\n    -------\n    Optional[RecordingStream]\n        The currently active thread-local recording, if any.\n    '
    result = bindings.get_thread_local_data_recording()
    return RecordingStream(result) if result is not None else None

def set_thread_local_data_recording(recording: RecordingStream) -> RecordingStream | None:
    if False:
        print('Hello World!')
    '\n    Replaces the currently active thread-local recording with the specified one.\n\n    Parameters\n    ----------\n    recording:\n        The newly active thread-local recording.\n    '
    result = bindings.set_thread_local_data_recording(recording=RecordingStream.to_native(recording))
    return RecordingStream(result) if result is not None else None