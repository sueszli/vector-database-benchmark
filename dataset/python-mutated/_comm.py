import abc
from cupyx.distributed import _store

class _Backend(abc.ABC):

    def __init__(self, n_devices, rank, host=_store._DEFAULT_HOST, port=_store._DEFAULT_PORT):
        if False:
            for i in range(10):
                print('nop')
        self._n_devices = n_devices
        self.rank = rank
        self._store_proxy = _store.TCPStoreProxy(host, port)
        if rank == 0:
            self._store = _store.TCPStore(n_devices)

    @abc.abstractmethod
    def all_reduce(self, in_array, out_array, op='sum', stream=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def reduce(self, in_array, out_array, root=0, op='sum', stream=None):
        if False:
            print('Hello World!')
        pass

    @abc.abstractmethod
    def broadcast(self, in_out_array, root=0, stream=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def reduce_scatter(self, in_array, out_array, count, op='sum', stream=None):
        if False:
            return 10
        pass

    @abc.abstractmethod
    def all_gather(self, in_array, out_array, count, stream=None):
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def send(self, array, peer, stream=None):
        if False:
            i = 10
            return i + 15
        pass

    @abc.abstractmethod
    def recv(self, out_array, peer, stream=None):
        if False:
            i = 10
            return i + 15
        pass

    @abc.abstractmethod
    def send_recv(self, in_array, out_array, peer, stream=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def scatter(self, in_array, out_array, root=0, stream=None):
        if False:
            print('Hello World!')
        pass

    @abc.abstractmethod
    def gather(self, in_array, out_array, root=0, stream=None):
        if False:
            print('Hello World!')
        pass

    @abc.abstractmethod
    def all_to_all(self, in_array, out_array, stream=None):
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def barrier(self):
        if False:
            print('Hello World!')
        pass

    def stop(self):
        if False:
            print('Hello World!')
        if self.rank == 0:
            self._store.stop()