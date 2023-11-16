from __future__ import annotations
import gc
import io
import textwrap
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple
import rmm.mr
from cudf.core.buffer.spillable_buffer import SpillableBuffer
from cudf.options import get_option
from cudf.utils.nvtx_annotation import _cudf_nvtx_annotate
from cudf.utils.string import format_bytes
_spill_cudf_nvtx_annotate = partial(_cudf_nvtx_annotate, domain='cudf_python-spill')

def get_traceback() -> str:
    if False:
        while True:
            i = 10
    'Pretty print current traceback to a string'
    with io.StringIO() as f:
        traceback.print_stack(file=f)
        f.seek(0)
        return f.read()

def get_rmm_memory_resource_stack(mr: rmm.mr.DeviceMemoryResource) -> List[rmm.mr.DeviceMemoryResource]:
    if False:
        return 10
    'Get the RMM resource stack\n\n    Parameters\n    ----------\n    mr : rmm.mr.DeviceMemoryResource\n        Top of the resource stack\n\n    Return\n    ------\n    list\n        List of RMM resources\n    '
    if hasattr(mr, 'upstream_mr'):
        return [mr] + get_rmm_memory_resource_stack(mr.upstream_mr)
    return [mr]

class SpillStatistics:
    """Gather spill statistics

    Levels of information gathered:
      0  - disabled (no overhead).
      1+ - duration and number of bytes spilled (very low overhead).
      2+ - a traceback for each time a spillable buffer is exposed
           permanently (potential high overhead).

    The statistics are printed when spilling-on-demand fails to find
    any buffer to spill. It is possible to retrieve the statistics
    manually through the spill manager, see example below.

    Parameters
    ----------
    level : int
        If not 0, enables statistics at the specified level.

    Examples
    --------
    >>> import cudf
    >>> from cudf.core.buffer.spill_manager import get_global_manager
    >>> manager = get_global_manager()
    >>> manager.statistics
    <SpillStatistics level=1>
    >>> df = cudf.DataFrame({"a": [1,2,3]})
    >>> manager.spill_to_device_limit(1)  # Spill df
    24
    >>> print(get_global_manager().statistics)
    Spill Statistics (level=1):
     Spilling (level >= 1):
      gpu => cpu: 24B in 0.0033579860000827466s
    """

    @dataclass
    class Expose:
        traceback: str
        count: int = 1
        total_nbytes: int = 0
        spilled_nbytes: int = 0
    spill_totals: Dict[Tuple[str, str], Tuple[int, float]]

    def __init__(self, level) -> None:
        if False:
            print('Hello World!')
        self.lock = threading.Lock()
        self.level = level
        self.spill_totals = defaultdict(lambda : (0, 0))
        self.exposes: Dict[str, SpillStatistics.Expose] = {}

    def log_spill(self, src: str, dst: str, nbytes: int, time: float) -> None:
        if False:
            while True:
                i = 10
        'Log a (un-)spilling event\n\n        Parameters\n        ----------\n        src : str\n            The memory location before spilling.\n        dst : str\n            The memory location after spilling.\n        nbytes : int\n            Number of bytes (un-)spilled.\n        nbytes : float\n            Elapsed time the event took in seconds.\n        '
        if self.level < 1:
            return
        with self.lock:
            (total_nbytes, total_time) = self.spill_totals[src, dst]
            self.spill_totals[src, dst] = (total_nbytes + nbytes, total_time + time)

    def log_expose(self, buf: SpillableBuffer) -> None:
        if False:
            return 10
        'Log an expose event\n\n        We track logged exposes by grouping them by their traceback such\n        that `self.exposes` maps tracebacks (as strings) to their logged\n        data (as `Expose`).\n\n        Parameters\n        ----------\n        buf : spillabe-buffer\n            The buffer being exposed.\n        '
        if self.level < 2:
            return
        with self.lock:
            tb = get_traceback()
            stat = self.exposes.get(tb, None)
            spilled_nbytes = buf.nbytes if buf.is_spilled else 0
            if stat is None:
                self.exposes[tb] = self.Expose(traceback=tb, total_nbytes=buf.nbytes, spilled_nbytes=spilled_nbytes)
            else:
                stat.count += 1
                stat.total_nbytes += buf.nbytes
                stat.spilled_nbytes += spilled_nbytes

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<SpillStatistics level={self.level}>'

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        with self.lock:
            ret = f'Spill Statistics (level={self.level}):\n'
            if self.level == 0:
                return ret[:-1] + ' N/A'
            ret += '  Spilling (level >= 1):'
            if len(self.spill_totals) == 0:
                ret += ' None'
            ret += '\n'
            for ((src, dst), (nbytes, time)) in self.spill_totals.items():
                ret += f'    {src} => {dst}: '
                ret += f'{format_bytes(nbytes)} in {time:.3f}s\n'
            ret += '  Exposed buffers (level >= 2): '
            if self.level < 2:
                return ret + 'disabled'
            if len(self.exposes) == 0:
                ret += 'None'
            ret += '\n'
            for s in sorted(self.exposes.values(), key=lambda x: -x.count):
                ret += textwrap.indent(f'exposed {s.count} times, total: {format_bytes(s.total_nbytes)}, spilled: {format_bytes(s.spilled_nbytes)}, traceback:\n{s.traceback}', prefix=' ' * 4)
            return ret[:-1]

class SpillManager:
    """Manager of spillable buffers.

    This class implements tracking of all known spillable buffers, on-demand
    spilling of said buffers, and (optionally) maintains a memory usage limit.

    When `spill_on_demand=True`, the manager registers an RMM out-of-memory
    error handler, which will spill spillable buffers in order to free up
    memory.

    When `device_memory_limit=<limit-in-bytes>`, the manager will try keep
    the device memory usage below the specified limit by spilling of spillable
    buffers continuously, which will introduce a modest overhead.
    Notice, this is a soft limit. The memory usage might exceed the limit if
    too many buffers are unspillable.

    Parameters
    ----------
    spill_on_demand : bool
        Enable spill on demand.
    device_memory_limit: int, optional
        If not None, this is the device memory limit in bytes that triggers
        device to host spilling. The global manager sets this to the value
        of `CUDF_SPILL_DEVICE_LIMIT` or None.
    statistic_level: int, optional
        If not 0, enables statistics at the specified level. See
        SpillStatistics for the different levels.
    """
    _buffers: weakref.WeakValueDictionary[int, SpillableBuffer]
    statistics: SpillStatistics

    def __init__(self, *, spill_on_demand: bool=False, device_memory_limit: Optional[int]=None, statistic_level: int=0) -> None:
        if False:
            i = 10
            return i + 15
        self._lock = threading.Lock()
        self._buffers = weakref.WeakValueDictionary()
        self._id_counter = 0
        self._spill_on_demand = spill_on_demand
        self._device_memory_limit = device_memory_limit
        self.statistics = SpillStatistics(statistic_level)
        if self._spill_on_demand:
            mr = rmm.mr.get_current_device_resource()
            if all((not isinstance(m, rmm.mr.FailureCallbackResourceAdaptor) for m in get_rmm_memory_resource_stack(mr))):
                rmm.mr.set_current_device_resource(rmm.mr.FailureCallbackResourceAdaptor(mr, self._out_of_memory_handle))

    def _out_of_memory_handle(self, nbytes: int, *, retry_once=True) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "Try to handle an out-of-memory error by spilling\n\n        This can by used as the callback function to RMM's\n        `FailureCallbackResourceAdaptor`\n\n        Parameters\n        ----------\n        nbytes : int\n            Number of bytes to try to spill.\n        retry_once : bool, optional\n            If True, call `gc.collect()` and retry once.\n\n        Return\n        ------\n        bool\n            True if any buffers were freed otherwise False.\n\n        Warning\n        -------\n        In order to avoid deadlock, this function should not lock\n        already locked buffers.\n        "
        spilled = self.spill_device_memory(nbytes=nbytes)
        if spilled > 0:
            return True
        if retry_once:
            gc.collect()
            return self._out_of_memory_handle(nbytes, retry_once=False)
        print(f"[WARNING] RMM allocation of {format_bytes(nbytes)} bytes failed, spill-on-demand couldn't find any device memory to spill:\n{repr(self)}\ntraceback:\n{get_traceback()}\n{self.statistics}")
        return False

    def add(self, buffer: SpillableBuffer) -> None:
        if False:
            return 10
        'Add buffer to the set of managed buffers\n\n        The manager keeps a weak reference to the buffer\n\n        Parameters\n        ----------\n        buffer : SpillableBuffer\n            The buffer to manage\n        '
        if buffer.size > 0 and (not buffer.exposed):
            with self._lock:
                self._buffers[self._id_counter] = buffer
                self._id_counter += 1
        self.spill_to_device_limit()

    def buffers(self, order_by_access_time: bool=False) -> Tuple[SpillableBuffer, ...]:
        if False:
            return 10
        'Get all managed buffers\n\n        Parameters\n        ----------\n        order_by_access_time : bool, optional\n            Order the buffer by access time (ascending order)\n\n        Return\n        ------\n        tuple\n            Tuple of buffers\n        '
        with self._lock:
            ret = tuple(self._buffers.values())
        if order_by_access_time:
            ret = tuple(sorted(ret, key=lambda b: b.last_accessed))
        return ret

    @_spill_cudf_nvtx_annotate
    def spill_device_memory(self, nbytes: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Try to spill device memory\n\n        This function is safe to call doing spill-on-demand\n        since it does not lock buffers already locked.\n\n        Parameters\n        ----------\n        nbytes : int\n            Number of bytes to try to spill\n\n        Return\n        ------\n        int\n            Number of actually bytes spilled.\n        '
        spilled = 0
        for buf in self.buffers(order_by_access_time=True):
            if buf.lock.acquire(blocking=False):
                try:
                    if not buf.is_spilled and buf.spillable:
                        buf.spill(target='cpu')
                        spilled += buf.size
                        if spilled >= nbytes:
                            break
                finally:
                    buf.lock.release()
        return spilled

    def spill_to_device_limit(self, device_limit: Optional[int]=None) -> int:
        if False:
            while True:
                i = 10
        'Try to spill device memory until device limit\n\n        Notice, by default this is a no-op.\n\n        Parameters\n        ----------\n        device_limit : int, optional\n            Limit in bytes. If None, the value of the environment variable\n            `CUDF_SPILL_DEVICE_LIMIT` is used. If this is not set, the method\n            does nothing and returns 0.\n\n        Return\n        ------\n        int\n            The number of bytes spilled.\n        '
        limit = self._device_memory_limit if device_limit is None else device_limit
        if limit is None:
            return 0
        unspilled = sum((buf.size for buf in self.buffers() if not buf.is_spilled))
        return self.spill_device_memory(nbytes=unspilled - limit)

    def __repr__(self) -> str:
        if False:
            return 10
        spilled = sum((buf.size for buf in self.buffers() if buf.is_spilled))
        unspilled = sum((buf.size for buf in self.buffers() if not buf.is_spilled))
        unspillable = 0
        for buf in self.buffers():
            if not (buf.is_spilled or buf.spillable):
                unspillable += buf.size
        unspillable_ratio = unspillable / unspilled if unspilled else 0
        dev_limit = 'N/A'
        if self._device_memory_limit is not None:
            dev_limit = format_bytes(self._device_memory_limit)
        return f'<SpillManager spill_on_demand={self._spill_on_demand} device_memory_limit={dev_limit} | {format_bytes(spilled)} spilled | {format_bytes(unspilled)} ({unspillable_ratio:.0%}) unspilled (unspillable)>'
_global_manager_uninitialized: bool = True
_global_manager: Optional[SpillManager] = None

def set_global_manager(manager: Optional[SpillManager]) -> None:
    if False:
        i = 10
        return i + 15
    'Set the global manager, which if None disables spilling'
    global _global_manager, _global_manager_uninitialized
    if _global_manager is not None:
        gc.collect()
        buffers = _global_manager.buffers()
        if len(buffers) > 0:
            warnings.warn(f'overwriting non-empty manager: {buffers}')
    _global_manager = manager
    _global_manager_uninitialized = False

def get_global_manager() -> Optional[SpillManager]:
    if False:
        while True:
            i = 10
    'Get the global manager or None if spilling is disabled'
    global _global_manager_uninitialized
    if _global_manager_uninitialized:
        manager = None
        if get_option('spill'):
            manager = SpillManager(spill_on_demand=get_option('spill_on_demand'), device_memory_limit=get_option('spill_device_limit'), statistic_level=get_option('spill_stats'))
        set_global_manager(manager)
    return _global_manager