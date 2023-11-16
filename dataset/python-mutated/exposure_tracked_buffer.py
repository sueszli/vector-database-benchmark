from __future__ import annotations
import weakref
from typing import Any, Container, Literal, Mapping, Optional, Type, TypeVar, cast
from typing_extensions import Self
import cudf
from cudf.core.buffer.buffer import Buffer, get_ptr_and_size
from cudf.utils.string import format_bytes
T = TypeVar('T', bound='ExposureTrackedBuffer')

def get_owner(data, klass: Type[T]) -> Optional[T]:
    if False:
        for i in range(10):
            print('nop')
    'Get the owner of `data`, if any exist\n\n    Search through the stack of data owners in order to find an\n    owner of type `klass` (not subclasses).\n\n    Parameters\n    ----------\n    data\n        The data object\n\n    Return\n    ------\n    klass or None\n        The owner of `data` if `klass` or None.\n    '
    if type(data) is klass:
        return data
    if hasattr(data, 'owner'):
        return get_owner(data.owner, klass)
    return None

def as_exposure_tracked_buffer(data, exposed: bool, subclass: Optional[Type[T]]=None) -> BufferSlice:
    if False:
        print('Hello World!')
    'Factory function to wrap `data` in a slice of an exposure tracked buffer\n\n    If `subclass` is None, a new ExposureTrackedBuffer that points to the\n    memory of `data` is created and a BufferSlice that points to all of the\n    new ExposureTrackedBuffer is returned.\n\n    If `subclass` is not None, a new `subclass` is created instead. Still,\n    a BufferSlice that points to all of the new `subclass` is returned\n\n    It is illegal for an exposure tracked buffer to own another exposure\n    tracked buffer. When representing the same memory, we should have a single\n    exposure tracked buffer and multiple buffer slices.\n\n    Developer Notes\n    ---------------\n    This function always returns slices thus all buffers in cudf will use\n    `BufferSlice` when copy-on-write is enabled. The slices implement\n    copy-on-write by trigging deep copies when write access is detected\n    and multiple slices points to the same exposure tracked buffer.\n\n    Parameters\n    ----------\n    data : buffer-like or array-like\n        A buffer-like or array-like object that represents C-contiguous memory.\n    exposed\n        Mark the buffer as permanently exposed.\n    subclass\n        If not None, a subclass of ExposureTrackedBuffer to wrap `data`.\n\n    Return\n    ------\n    BufferSlice\n        A buffer slice that points to a ExposureTrackedBuffer (or `subclass`),\n        which in turn wraps `data`.\n    '
    if not hasattr(data, '__cuda_array_interface__'):
        if exposed:
            raise ValueError('cannot created exposed host memory')
        return cast(BufferSlice, ExposureTrackedBuffer._from_host_memory(data)[:])
    owner = get_owner(data, subclass or ExposureTrackedBuffer)
    if owner is None:
        return cast(BufferSlice, ExposureTrackedBuffer._from_device_memory(data, exposed=exposed)[:])
    (ptr, size) = get_ptr_and_size(data.__cuda_array_interface__)
    if size > 0 and owner._ptr == 0:
        raise ValueError('Cannot create a non-empty slice of a null buffer')
    return BufferSlice(base=owner, offset=ptr - owner._ptr, size=size)

class ExposureTrackedBuffer(Buffer):
    """A Buffer that tracks its "expose" status.

    In order to implement copy-on-write and spillable buffers, we need the
    ability to detect external access to the underlying memory. We say that
    the buffer has been exposed if the device pointer (integer or void*) has
    been accessed outside of ExposureTrackedBuffer. In this case, we have no
    control over knowing if the data is being modified by a third-party.

    Attributes
    ----------
    _exposed
        The current exposure status of the buffer. Notice, once the exposure
        status becomes True, it should never change back.
    _slices
        The set of BufferSlice instances that point to this buffer.
    """
    _exposed: bool
    _slices: weakref.WeakSet[BufferSlice]

    @property
    def exposed(self) -> bool:
        if False:
            print('Hello World!')
        return self._exposed

    def mark_exposed(self) -> None:
        if False:
            print('Hello World!')
        'Mark the buffer as "exposed" permanently'
        self._exposed = True

    @classmethod
    def _from_device_memory(cls, data: Any, *, exposed: bool=False) -> Self:
        if False:
            i = 10
            return i + 15
        'Create an exposure tracked buffer from device memory.\n\n        No data is being copied.\n\n        Parameters\n        ----------\n        data : device-buffer-like\n            An object implementing the CUDA Array Interface.\n        exposed : bool, optional\n            Mark the buffer as permanently exposed.\n\n        Returns\n        -------\n        ExposureTrackedBuffer\n            Buffer representing the same device memory as `data`\n        '
        ret = super()._from_device_memory(data)
        ret._exposed = exposed
        ret._slices = weakref.WeakSet()
        return ret

    def _getitem(self, offset: int, size: int) -> BufferSlice:
        if False:
            for i in range(10):
                print('nop')
        return BufferSlice(base=self, offset=offset, size=size)

    @property
    def __cuda_array_interface__(self) -> Mapping:
        if False:
            for i in range(10):
                print('nop')
        self.mark_exposed()
        return super().__cuda_array_interface__

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<ExposureTrackedBuffer exposed={self.exposed} size={format_bytes(self._size)} ptr={hex(self._ptr)} owner={repr(self._owner)}>'

class BufferSlice(ExposureTrackedBuffer):
    """A slice (aka. a view) of a exposure tracked buffer.

    Parameters
    ----------
    base
        The exposure tracked buffer this slice refers to.
    offset
        The offset relative to the start memory of base (in bytes).
    size
        The size of the slice (in bytes)
    passthrough_attributes
        Name of attributes that are passed through to the base as-is.
    """

    def __init__(self, base: ExposureTrackedBuffer, offset: int, size: int, *, passthrough_attributes: Container[str]=('exposed',)) -> None:
        if False:
            return 10
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
        self._passthrough_attributes = passthrough_attributes
        base._slices.add(self)

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        if name in self._passthrough_attributes:
            return getattr(self._base, name)
        raise AttributeError(f'{self.__class__.__name__} object has no attribute {name}')

    def _getitem(self, offset: int, size: int) -> BufferSlice:
        if False:
            for i in range(10):
                print('nop')
        return BufferSlice(base=self._base, offset=offset + self._offset, size=size)

    def get_ptr(self, *, mode: Literal['read', 'write']) -> int:
        if False:
            i = 10
            return i + 15
        if mode == 'write' and cudf.get_option('copy_on_write'):
            self.make_single_owner_inplace()
        return self._base.get_ptr(mode=mode) + self._offset

    def memoryview(self, *, offset: int=0, size: Optional[int]=None) -> memoryview:
        if False:
            while True:
                i = 10
        return self._base.memoryview(offset=self._offset + offset, size=size)

    def copy(self, deep: bool=True) -> Self:
        if False:
            while True:
                i = 10
        'Return a copy of Buffer.\n\n        What actually happens when `deep == False` depends on the\n        "copy_on_write" option. When copy-on-write is enabled, a shallow copy\n        becomes a deep copy if the buffer has been exposed. This is because we\n        have no control over knowing if the data is being modified when the\n        buffer has been exposed to third-party.\n\n        Parameters\n        ----------\n        deep : bool, default True\n            The semantics when copy-on-write is disabled:\n                - If deep=True, returns a deep copy of the underlying data.\n                - If deep=False, returns a shallow copy of the Buffer pointing\n                  to the same underlying data.\n            The semantics when copy-on-write is enabled:\n                - From the users perspective, always a deep copy of the\n                  underlying data. However, the data isn\'t actually copied\n                  until someone writers to the returned buffer.\n\n        Returns\n        -------\n        BufferSlice\n            A slice pointing to either a new or the existing base buffer\n            depending on the expose status of the base buffer and the\n            copy-on-write option (see above).\n        '
        if cudf.get_option('copy_on_write'):
            base_copy = self._base.copy(deep=deep or self.exposed)
        else:
            base_copy = self._base.copy(deep=deep)
        return cast(Self, base_copy[self._offset:self._offset + self._size])

    @property
    def __cuda_array_interface__(self) -> Mapping:
        if False:
            for i in range(10):
                print('nop')
        if cudf.get_option('copy_on_write'):
            self.make_single_owner_inplace()
        return super().__cuda_array_interface__

    def make_single_owner_inplace(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Make sure this slice is the only one pointing to the base.\n\n        This is used by copy-on-write to trigger a deep copy when write\n        access is detected.\n\n        Parameters\n        ----------\n        data : device-buffer-like\n            An object implementing the CUDA Array Interface.\n\n        Returns\n        -------\n        Buffer\n            Buffer representing the same device memory as `data`\n        '
        if len(self._base._slices) > 1:
            t = self.copy(deep=True)
            self._base = t._base
            self._offset = t._offset
            self._size = t._size
            self._owner = t._base
            self._base._slices.add(self)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'<BufferSlice size={format_bytes(self._size)} offset={format_bytes(self._offset)} of {self._base}>'