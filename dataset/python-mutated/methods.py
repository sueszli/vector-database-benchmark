import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back

class broadcast:

    @to_ivy_arrays_and_back
    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        data = ivy.broadcast_arrays(*map(ivy.array, args))
        self._shape = data[0].shape
        self._ndim = data[0].ndim
        self._index = 0
        self._numiter = len(data)
        self._size = data[0].size
        self._data = tuple((*zip(*(ivy.flatten(i) for i in data)),))
        self._iters = tuple((iter(ivy.flatten(i)) for i in data))

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        return self._shape

    @property
    def ndim(self):
        if False:
            i = 10
            return i + 15
        return self._ndim

    @property
    def nd(self):
        if False:
            while True:
                i = 10
        return self._ndim

    @property
    def numiter(self):
        if False:
            print('Hello World!')
        return self._numiter

    @property
    def size(self):
        if False:
            print('Hello World!')
        return self._size

    @property
    def iters(self):
        if False:
            print('Hello World!')
        return self._iters

    @property
    def index(self):
        if False:
            i = 10
            return i + 15
        return self._index

    def __next__(self):
        if False:
            i = 10
            return i + 15
        if self.index < self.size:
            self._index += 1
            return self._data[self.index - 1]
        raise StopIteration

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def reset(self):
        if False:
            print('Hello World!')
        self._index = 0