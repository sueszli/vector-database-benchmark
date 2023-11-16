from __future__ import annotations
import collections.abc
import pickle
import time
import weakref
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple
import numpy
from typing_extensions import Self
import rmm
from cudf.core.buffer.buffer import Buffer, cuda_array_interface_wrapper, get_ptr_and_size, host_memory_allocation
from cudf.utils.nvtx_annotation import _get_color_for_nvtx, annotate
from cudf.utils.string import format_bytes
if TYPE_CHECKING:
    from cudf.core.buffer.spill_manager import SpillManager

def get_spillable_owner(data) -> Optional[SpillableBuffer]:
    if False:
        i = 10
        return i + 15
    'Get the spillable owner of `data`, if any exist\n\n    Search through the stack of data owners in order to find an\n    owner of type `SpillableBuffer` (not subclasses).\n\n    Parameters\n    ----------\n    data : buffer-like or array-like\n        A buffer-like or array-like object that represent C-contiguous memory.\n\n    Return\n    ------\n    SpillableBuffer or None\n        The owner of `data` if spillable or None.\n    '
    if type(data) is SpillableBuffer:
        return data
    if hasattr(data, 'owner'):
        return get_spillable_owner(data.owner)
    return None

def as_spillable_buffer(data, exposed: bool) -> SpillableBuffer:
    if False:
        return 10
    'Factory function to wrap `data` in a SpillableBuffer object.\n\n    If `data` isn\'t a buffer already, a new buffer that points to the memory of\n    `data` is created. If `data` represents host memory, it is copied to a new\n    `rmm.DeviceBuffer` device allocation. Otherwise, the memory of `data` is\n    **not** copied, instead the new buffer keeps a reference to `data` in order\n    to retain its lifetime.\n\n    If `data` is owned by a spillable buffer, a "slice" of the buffer is\n    returned. In this case, the spillable buffer must either be "exposed" or\n    spilled locked (called within an acquire_spill_lock context). This is to\n    guarantee that the memory of `data` isn\'t spilled before this function gets\n    to calculate the offset of the new slice.\n\n    It is illegal for a spillable buffer to own another spillable buffer.\n\n    Parameters\n    ----------\n    data : buffer-like or array-like\n        A buffer-like or array-like object that represent C-contiguous memory.\n    exposed : bool, optional\n        Mark the buffer as permanently exposed (unspillable).\n\n    Return\n    ------\n    SpillableBuffer\n        A spillabe buffer instance that represents the device memory of `data`.\n    '
    from cudf.core.buffer.utils import get_spill_lock
    if not hasattr(data, '__cuda_array_interface__'):
        if exposed:
            raise ValueError('cannot created exposed host memory')
        return SpillableBuffer._from_host_memory(data)
    spillable_owner = get_spillable_owner(data)
    if spillable_owner is None:
        return SpillableBuffer._from_device_memory(data, exposed=exposed)
    if not spillable_owner.exposed and get_spill_lock() is None:
        raise ValueError('A owning spillable buffer must either be exposed or spilled locked.')
    (ptr, size) = get_ptr_and_size(data.__cuda_array_interface__)
    base_ptr = spillable_owner.memory_info()[0]
    return SpillableBufferSlice(spillable_owner, offset=ptr - base_ptr, size=size)

class SpillLock:
    pass

class DelayedPointerTuple(collections.abc.Sequence):
    """
    A delayed version of the "data" field in __cuda_array_interface__.

    The idea is to delay the access to `Buffer.ptr` until the user
    actually accesses the data pointer.

    For instance, in many cases __cuda_array_interface__ is accessed
    only to determine whether an object is a CUDA object or not.

    TODO: this doesn't support libraries such as PyTorch that declare
    the tuple of __cuda_array_interface__["data"] in Cython. In such
    cases, Cython will raise an error because DelayedPointerTuple
    isn't a "real" tuple.
    """

    def __init__(self, buffer) -> None:
        if False:
            print('Hello World!')
        self._buf = buffer

    def __len__(self):
        if False:
            while True:
                i = 10
        return 2

    def __getitem__(self, i):
        if False:
            print('Hello World!')
        if i == 0:
            return self._buf.get_ptr(mode='write')
        elif i == 1:
            return False
        raise IndexError('tuple index out of range')

class SpillableBuffer(Buffer):
    """A Buffer that supports spilling memory off the GPU to avoid OOMs.

    This buffer supports spilling the represented data to host memory.
    Spilling can be done manually by calling `.spill(target="cpu")` but
    usually the associated spilling manager triggers spilling based on current
    device memory usage see `cudf.core.buffer.spill_manager.SpillManager`.
    Unspill is triggered automatically when accessing the data of the buffer.

    The buffer might not be spillable, which is based on the "expose" status
    of the buffer. We say that the buffer has been exposed if the device
    pointer (integer or void*) has been accessed outside of SpillableBuffer.
    In this case, we cannot invalidate the device pointer by moving the data
    to host.

    A buffer can be exposed permanently at creation or by accessing the `.ptr`
    property. To avoid this, one can use `.get_ptr()` instead, which support
    exposing the buffer temporarily.

    Use the factory function `as_buffer` to create a SpillableBuffer instance.
    """
    lock: RLock
    _spill_locks: weakref.WeakSet
    _last_accessed: float
    _ptr_desc: Dict[str, Any]
    _exposed: bool
    _manager: SpillManager

    def _finalize_init(self, ptr_desc: Dict[str, Any], exposed: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Finish initialization of the spillable buffer\n\n        This implements the common initialization that `_from_device_memory`\n        and `_from_host_memory` are missing.\n\n        Parameters\n        ----------\n        ptr_desc : dict\n            Description of the memory.\n        exposed : bool, optional\n            Mark the buffer as permanently exposed (unspillable).\n        '
        from cudf.core.buffer.spill_manager import get_global_manager
        self.lock = RLock()
        self._spill_locks = weakref.WeakSet()
        self._last_accessed = time.monotonic()
        self._ptr_desc = ptr_desc
        self._exposed = exposed
        manager = get_global_manager()
        if manager is None:
            raise ValueError(f'cannot create {self.__class__} without a global spill manager')
        self._manager = manager
        self._manager.add(self)

    @classmethod
    def _from_device_memory(cls, data: Any, *, exposed: bool=False) -> Self:
        if False:
            print('Hello World!')
        'Create a spillabe buffer from device memory.\n\n        No data is being copied.\n\n        Parameters\n        ----------\n        data : device-buffer-like\n            An object implementing the CUDA Array Interface.\n        exposed : bool, optional\n            Mark the buffer as permanently exposed (unspillable).\n\n        Returns\n        -------\n        SpillableBuffer\n            Buffer representing the same device memory as `data`\n        '
        ret = super()._from_device_memory(data)
        ret._finalize_init(ptr_desc={'type': 'gpu'}, exposed=exposed)
        return ret

    @classmethod
    def _from_host_memory(cls, data: Any) -> Self:
        if False:
            return 10
        "Create a spillabe buffer from host memory.\n\n        Data must implement `__array_interface__`, the buffer protocol, and/or\n        be convertible to a buffer object using `numpy.array()`\n\n        The new buffer is marked as spilled to host memory already.\n\n        Raises ValueError if array isn't C-contiguous.\n\n        Parameters\n        ----------\n        data : Any\n            An object that represens host memory.\n\n        Returns\n        -------\n        SpillableBuffer\n            Buffer representing a copy of `data`.\n        "
        data = memoryview(numpy.array(data, copy=False, subok=True))
        if not data.c_contiguous:
            raise ValueError('Buffer data must be C-contiguous')
        data = data.cast('B')
        ret = cls.__new__(cls)
        ret._owner = None
        ret._ptr = 0
        ret._size = data.nbytes
        ret._finalize_init(ptr_desc={'type': 'cpu', 'memoryview': data}, exposed=False)
        return ret

    @property
    def is_spilled(self) -> bool:
        if False:
            print('Hello World!')
        return self._ptr_desc['type'] != 'gpu'

    def copy(self, deep: bool=True) -> Self:
        if False:
            return 10
        spill_lock = SpillLock()
        self.spill_lock(spill_lock=spill_lock)
        return super().copy(deep=deep)

    def spill(self, target: str='cpu') -> None:
        if False:
            print('Hello World!')
        'Spill or un-spill this buffer in-place\n\n        Parameters\n        ----------\n        target : str\n            The target of the spilling.\n        '
        time_start = time.perf_counter()
        with self.lock:
            ptr_type = self._ptr_desc['type']
            if ptr_type == target:
                return
            if not self.spillable:
                raise ValueError(f'Cannot in-place move an unspillable buffer: {self}')
            if (ptr_type, target) == ('gpu', 'cpu'):
                with annotate(message='SpillDtoH', color=_get_color_for_nvtx('SpillDtoH'), domain='cudf_python-spill'):
                    host_mem = host_memory_allocation(self.size)
                    rmm._lib.device_buffer.copy_ptr_to_host(self._ptr, host_mem)
                self._ptr_desc['memoryview'] = host_mem
                self._ptr = 0
                self._owner = None
            elif (ptr_type, target) == ('cpu', 'gpu'):
                with annotate(message='SpillHtoD', color=_get_color_for_nvtx('SpillHtoD'), domain='cudf_python-spill'):
                    dev_mem = rmm.DeviceBuffer.to_device(self._ptr_desc.pop('memoryview'))
                self._ptr = dev_mem.ptr
                self._owner = dev_mem
                assert self._size == dev_mem.size
            else:
                raise ValueError(f'Unknown target: {target}')
            self._ptr_desc['type'] = target
        time_end = time.perf_counter()
        self._manager.statistics.log_spill(src=ptr_type, dst=target, nbytes=self.size, time=time_end - time_start)

    def mark_exposed(self) -> None:
        if False:
            while True:
                i = 10
        'Mark the buffer as "exposed" and make it unspillable permanently.\n\n        This also unspills the buffer (unspillable buffers cannot be spilled!).\n        '
        self._manager.spill_to_device_limit()
        with self.lock:
            if not self._exposed:
                self._manager.statistics.log_expose(self)
            self.spill(target='gpu')
            self._exposed = True
            self._last_accessed = time.monotonic()

    def spill_lock(self, spill_lock: SpillLock) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Spill lock the buffer\n\n        Mark the buffer as unspillable while `spill_lock` is alive,\n        which is tracked by monitoring a weakref to `spill_lock`.\n\n        Parameters\n        ----------\n        spill_lock : SpillLock\n            The object that defines the scope of the lock.\n        '
        with self.lock:
            self.spill(target='gpu')
            self._spill_locks.add(spill_lock)

    def get_ptr(self, *, mode: Literal['read', 'write']) -> int:
        if False:
            print('Hello World!')
        'Get a device pointer to the memory of the buffer.\n\n        If this is called within an `acquire_spill_lock` context,\n        a reference to this buffer is added to spill_lock, which\n        disable spilling of this buffer while in the context.\n\n        If this is *not* called within a `acquire_spill_lock` context,\n        this buffer is marked as unspillable permanently.\n\n        Returns\n        -------\n        int\n            The device pointer as an integer\n        '
        from cudf.core.buffer.utils import get_spill_lock
        spill_lock = get_spill_lock()
        if spill_lock is None:
            self.mark_exposed()
        else:
            self.spill_lock(spill_lock)
            self._last_accessed = time.monotonic()
        return self._ptr

    def memory_info(self) -> Tuple[int, int, str]:
        if False:
            for i in range(10):
                print('nop')
        'Get pointer, size, and device type of this buffer.\n\n        Warning, it is not safe to access the pointer value without\n        spill lock the buffer manually. This method neither exposes\n        nor spill locks the buffer.\n\n        Return\n        ------\n        int\n            The memory pointer as an integer (device or host memory)\n        int\n            The size of the memory in bytes\n        str\n            The device type as a string ("cpu" or "gpu")\n        '
        if self._ptr_desc['type'] == 'gpu':
            ptr = self._ptr
        elif self._ptr_desc['type'] == 'cpu':
            ptr = numpy.array(self._ptr_desc['memoryview'], copy=False).__array_interface__['data'][0]
        return (ptr, self.nbytes, self._ptr_desc['type'])

    @property
    def owner(self) -> Any:
        if False:
            print('Hello World!')
        return self._owner

    @property
    def exposed(self) -> bool:
        if False:
            while True:
                i = 10
        return self._exposed

    @property
    def spillable(self) -> bool:
        if False:
            print('Hello World!')
        return not self._exposed and len(self._spill_locks) == 0

    @property
    def size(self) -> int:
        if False:
            print('Hello World!')
        return self._size

    @property
    def nbytes(self) -> int:
        if False:
            while True:
                i = 10
        return self._size

    @property
    def last_accessed(self) -> float:
        if False:
            return 10
        return self._last_accessed

    @property
    def __cuda_array_interface__(self) -> dict:
        if False:
            i = 10
            return i + 15
        return {'data': DelayedPointerTuple(self), 'shape': (self.size,), 'strides': None, 'typestr': '|u1', 'version': 0}

    def memoryview(self, *, offset: int=0, size: Optional[int]=None) -> memoryview:
        if False:
            return 10
        size = self._size if size is None else size
        with self.lock:
            if self.spillable:
                self.spill(target='cpu')
                return self._ptr_desc['memoryview'][offset:offset + size]
            else:
                assert self._ptr_desc['type'] == 'gpu'
                ret = host_memory_allocation(size)
                rmm._lib.device_buffer.copy_ptr_to_host(self._ptr + offset, ret)
                return ret

    def _getitem(self, offset: int, size: int) -> SpillableBufferSlice:
        if False:
            while True:
                i = 10
        return SpillableBufferSlice(base=self, offset=offset, size=size)

    def serialize(self) -> Tuple[dict, list]:
        if False:
            return 10
        "Serialize the Buffer\n\n        Normally, we would use `[self]` as the frames. This would work but\n        also mean that `self` becomes exposed permanently if the frames are\n        later accessed through `__cuda_array_interface__`, which is exactly\n        what libraries like Dask+UCX would do when communicating!\n\n        The sound solution is to modify Dask et al. so that they access the\n        frames through `.get_ptr()` and holds on to the `spill_lock` until\n        the frame has been transferred. However, until this adaptation we\n        use a hack where the frame is a `Buffer` with a `spill_lock` as the\n        owner, which makes `self` unspillable while the frame is alive but\n        doesn't expose `self` when `__cuda_array_interface__` is accessed.\n\n        Warning, this hack means that the returned frame must be copied before\n        given to `.deserialize()`, otherwise we would have a `Buffer` pointing\n        to memory already owned by an existing `SpillableBuffer`.\n        "
        header: Dict[Any, Any]
        frames: List[Buffer | memoryview]
        with self.lock:
            header = {}
            header['type-serialized'] = pickle.dumps(self.__class__)
            header['frame_count'] = 1
            if self.is_spilled:
                frames = [self.memoryview()]
            else:
                spill_lock = SpillLock()
                self.spill_lock(spill_lock)
                (ptr, size, _) = self.memory_info()
                frames = [Buffer._from_device_memory(cuda_array_interface_wrapper(ptr=ptr, size=size, owner=(self._owner, spill_lock)))]
            return (header, frames)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self._ptr_desc['type'] != 'gpu':
            ptr_info = str(self._ptr_desc)
        else:
            ptr_info = str(hex(self._ptr))
        return f'<SpillableBuffer size={format_bytes(self._size)} spillable={self.spillable} exposed={self.exposed} num-spill-locks={len(self._spill_locks)} ptr={ptr_info} owner={repr(self._owner)}>'

class SpillableBufferSlice(SpillableBuffer):
    """A slice of a spillable buffer

    This buffer applies the slicing and then delegates all
    operations to its base buffer.

    Parameters
    ----------
    base : SpillableBuffer
        The base of the view
    offset : int
        Memory offset into the base buffer
    size : int
        Size of the view (in bytes)
    """

    def __init__(self, base: SpillableBuffer, offset: int, size: int) -> None:
        if False:
            i = 10
            return i + 15
        if size < 0:
            raise ValueError('size cannot be negative')
        if offset < 0:
            raise ValueError('offset cannot be negative')
        if offset + size > base.size:
            raise ValueError('offset+size cannot be greater than the size of base')
        self._base = base
        self._offset = offset
        self._size = size
        self._owner = base
        self.lock = base.lock

    def get_ptr(self, *, mode: Literal['read', 'write']) -> int:
        if False:
            while True:
                i = 10
        '\n        A passthrough method to `SpillableBuffer.get_ptr`\n        with factoring in the `offset`.\n        '
        return self._base.get_ptr(mode=mode) + self._offset

    def _getitem(self, offset: int, size: int) -> SpillableBufferSlice:
        if False:
            print('Hello World!')
        return SpillableBufferSlice(base=self._base, offset=offset + self._offset, size=size)

    @classmethod
    def deserialize(cls, header: dict, frames: list):
        if False:
            print('Hello World!')
        return SpillableBuffer.deserialize(header, frames)

    def memoryview(self, *, offset: int=0, size: Optional[int]=None) -> memoryview:
        if False:
            print('Hello World!')
        size = self._size if size is None else size
        return self._base.memoryview(offset=self._offset + offset, size=size)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<SpillableBufferSlice size={format_bytes(self._size)} offset={format_bytes(self._offset)} of {self._base} '

    def spill(self, target: str='cpu') -> None:
        if False:
            i = 10
            return i + 15
        return self._base.spill(target=target)

    @property
    def is_spilled(self) -> bool:
        if False:
            while True:
                i = 10
        return self._base.is_spilled

    @property
    def exposed(self) -> bool:
        if False:
            print('Hello World!')
        return self._base.exposed

    @property
    def spillable(self) -> bool:
        if False:
            print('Hello World!')
        return self._base.spillable

    def spill_lock(self, spill_lock: SpillLock) -> None:
        if False:
            print('Hello World!')
        self._base.spill_lock(spill_lock=spill_lock)

    def memory_info(self) -> Tuple[int, int, str]:
        if False:
            return 10
        (ptr, _, device_type) = self._base.memory_info()
        return (ptr + self._offset, self.nbytes, device_type)