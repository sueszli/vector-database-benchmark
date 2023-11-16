"""Parameter Vector Class to simplify management of parameter lists."""
from uuid import uuid4, UUID
from .parameter import Parameter

class ParameterVectorElement(Parameter):
    """An element of a ParameterVector."""
    ___slots__ = ('_vector', '_index')

    def __init__(self, vector, index, uuid=None):
        if False:
            print('Hello World!')
        super().__init__(f'{vector.name}[{index}]', uuid=uuid)
        self._vector = vector
        self._index = index

    @property
    def index(self):
        if False:
            print('Hello World!')
        'Get the index of this element in the parent vector.'
        return self._index

    @property
    def vector(self):
        if False:
            print('Hello World!')
        'Get the parent vector instance.'
        return self._vector

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return super().__getstate__() + (self._vector, self._index)

    def __setstate__(self, state):
        if False:
            return 10
        (*super_state, vector, index) = state
        super().__setstate__(super_state)
        self._vector = vector
        self._index = index

class ParameterVector:
    """ParameterVector class to quickly generate lists of parameters."""
    __slots__ = ('_name', '_params', '_size', '_root_uuid')

    def __init__(self, name, length=0):
        if False:
            return 10
        self._name = name
        self._size = length
        self._root_uuid = uuid4()
        root_uuid_int = self._root_uuid.int
        self._params = [ParameterVectorElement(self, i, UUID(int=root_uuid_int + i)) for i in range(length)]

    @property
    def name(self):
        if False:
            return 10
        'Returns the name of the ParameterVector.'
        return self._name

    @property
    def params(self):
        if False:
            i = 10
            return i + 15
        'Returns the list of parameters in the ParameterVector.'
        return self._params

    def index(self, value):
        if False:
            return 10
        'Returns first index of value.'
        return self._params.index(value)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        if isinstance(key, slice):
            (start, stop, step) = key.indices(self._size)
            return self.params[start:stop:step]
        if key > self._size:
            raise IndexError(f'Index out of range: {key} > {self._size}')
        return self.params[key]

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.params[:self._size])

    def __len__(self):
        if False:
            return 10
        return self._size

    def __str__(self):
        if False:
            print('Hello World!')
        return f'{self.name}, {[str(item) for item in self.params[:self._size]]}'

    def __repr__(self):
        if False:
            return 10
        return f'{self.__class__.__name__}(name={self.name}, length={len(self)})'

    def resize(self, length):
        if False:
            while True:
                i = 10
        'Resize the parameter vector.\n\n        If necessary, new elements are generated. If length is smaller than before, the\n        previous elements are cached and not re-generated if the vector is enlarged again.\n        This is to ensure that the parameter instances do not change.\n        '
        if length > len(self._params):
            root_uuid_int = self._root_uuid.int
            self._params.extend([ParameterVectorElement(self, i, UUID(int=root_uuid_int + i)) for i in range(len(self._params), length)])
        self._size = length