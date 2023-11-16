import os
from nvidia.dali import backend as _b

class SharedMem:
    """SharedMem allows you to allocate and access shared memory.
    Provides memory view of the allocated memory via buf property.
    You can transfer access to the same shared memory chunk by sending related shared memory
    handle (file descriptor on Unix) available as handle property. Use SharedMem.allocate
    to allocate new chunk of shared memory and SharedMem.open if you received handle to already
    existing memory chunk.

    There is out of the box support for shared memory starting from Python3.8, though
    the only way there to transfer the memory to other processes is via filename,
    which might 'leak' if process was closed abruptly.

    Parameters
    ----------
    `handle` : int
        Handle identifying related shared memory object. Pass None to allocate new memory chunk.
    `size` : int
        When handle=None it is the size of shared memory to allocate in bytes, otherwise it must be
        the size of shared memory objects that provided handle represents.
    """

    def __init__(self, handle, size):
        if False:
            return 10
        if handle is None:
            handle = -1
        self.shm = _b.SharedMem(handle, size)
        self.capacity = size

    def __getattr__(self, key):
        if False:
            return 10
        if key == 'buf':
            buf = self.shm.buf()
            self.__dict__['buf'] = buf
            return buf
        raise AttributeError

    @classmethod
    def allocate(cls, size):
        if False:
            while True:
                i = 10
        'Creates new SharedMem instance representing freshly allocated\n        shared memory of ``size`` bytes.\n\n        Parameters\n        ----------\n        `size` : int\n            Number of bytes to allocate.\n        '
        return cls(None, size)

    @classmethod
    def open(cls, handle, size):
        if False:
            return 10
        'Creates new SharedMem instance that points to already allocated shared\n        memory chunk accessible via provided shared memory ``handle``.\n\n        Parameters\n        ----------\n        `handle`: int\n            Handle pointing to already existing shared memory chunk.\n        `size` : int\n            Size of the existing shared memory chunk.\n        '
        instance = cls(handle, size)
        assert os.fstat(handle).st_size >= size
        return instance

    @property
    def handle(self):
        if False:
            for i in range(10):
                print('nop')
        'Shared memory handle (file descriptor on Unix), use it to transfer access\n        to the shared memory object to another process.\n        You can transfer it between processes via socket using multiprocessing.reduction.send_handle\n        '
        return self.shm.handle

    def resize(self, size, trunc=False):
        if False:
            while True:
                i = 10
        'Resize already allocated shared memory chunk. If you want to resize the underlying\n        shared memory chunk pass trunc=True, if the memory chunk has already been resized\n        via another SharedMem instance (possibly in another process), pass new size and\n        trunc=False to simply adjust mmaping of the memory into the current process address space.\n        '
        if 'buf' in self.__dict__:
            del self.__dict__['buf']
        self.shm.resize(size, trunc)
        self.capacity = size

    def close(self):
        if False:
            return 10
        "Removes maping of the memory into process address space and closes related handle.\n        If all processes sharing given chunk close it, it will be automatically released by the OS.\n        You don't have to call this method, as corresponding clean up is performed when instance\n        gets garbage collected but you can call it as soon as you no longer need it for more\n        effective resources handling.\n        "
        self.buf = None
        self.shm.close()

    def close_handle(self):
        if False:
            print('Hello World!')
        'Closes OS handle for underlying shared memory. From now on, the process cannot resize the\n           underlying memory with this handle but still can adjust the mapping if the underlying\n           shared memory is resized, for instance, by another process.\n           This means that call to resize with ``trunc``= True will be illegal.\n        '
        self.shm.close_handle()