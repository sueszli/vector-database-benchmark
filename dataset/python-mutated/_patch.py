def _operation(name, location, **kwargs):
    if False:
        i = 10
        return i + 15
    return {'operation': name, 'location': location, 'params': dict(**kwargs)}
_noop = object()

def validate_slice(obj):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(obj, slice):
        raise TypeError('a slice is not a valid index for patch')

class Patch:
    """
    Patch a callback output value

    Act like a proxy of the output prop value on the frontend.

    Supported prop types: Dictionaries and lists.
    """

    def __init__(self, location=None, parent=None):
        if False:
            return 10
        if location is not None:
            self._location = location
        else:
            self._location = parent and parent._location or []
        if parent is not None:
            self._operations = parent._operations
        else:
            self._operations = []

    def __getstate__(self):
        if False:
            print('Hello World!')
        return vars(self)

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        vars(self).update(state)

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        validate_slice(item)
        return Patch(location=self._location + [item], parent=self)

    def __getattr__(self, item):
        if False:
            print('Hello World!')
        if item == 'tolist':
            raise AttributeError
        if item == '_location':
            return self._location
        if item == '_operations':
            return self._operations
        return self.__getitem__(item)

    def __setattr__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if key in ('_location', '_operations'):
            self.__dict__[key] = value
        else:
            self.__setitem__(key, value)

    def __delattr__(self, item):
        if False:
            for i in range(10):
                print('nop')
        self.__delitem__(item)

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        validate_slice(key)
        if value is _noop:
            return
        self._operations.append(_operation('Assign', self._location + [key], value=value))

    def __delitem__(self, key):
        if False:
            while True:
                i = 10
        validate_slice(key)
        self._operations.append(_operation('Delete', self._location + [key]))

    def __iadd__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, (list, tuple)):
            self.extend(other)
        else:
            self._operations.append(_operation('Add', self._location, value=other))
        return _noop

    def __isub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        self._operations.append(_operation('Sub', self._location, value=other))
        return _noop

    def __imul__(self, other):
        if False:
            return 10
        self._operations.append(_operation('Mul', self._location, value=other))
        return _noop

    def __itruediv__(self, other):
        if False:
            while True:
                i = 10
        self._operations.append(_operation('Div', self._location, value=other))
        return _noop

    def __ior__(self, other):
        if False:
            for i in range(10):
                print('nop')
        self.update(E=other)
        return _noop

    def __iter__(self):
        if False:
            print('Hello World!')
        raise TypeError('Patch objects are write-only, you cannot iterate them.')

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'<write-only dash.Patch object at {self._location}>'

    def append(self, item):
        if False:
            i = 10
            return i + 15
        'Add the item to the end of a list'
        self._operations.append(_operation('Append', self._location, value=item))

    def prepend(self, item):
        if False:
            print('Hello World!')
        'Add the item to the start of a list'
        self._operations.append(_operation('Prepend', self._location, value=item))

    def insert(self, index, item):
        if False:
            while True:
                i = 10
        'Add the item at the index of a list'
        self._operations.append(_operation('Insert', self._location, value=item, index=index))

    def clear(self):
        if False:
            return 10
        'Remove all items in a list'
        self._operations.append(_operation('Clear', self._location))

    def reverse(self):
        if False:
            for i in range(10):
                print('nop')
        'Reversal of the order of items in a list'
        self._operations.append(_operation('Reverse', self._location))

    def extend(self, item):
        if False:
            return 10
        'Add all the items to the end of a list'
        if not isinstance(item, (list, tuple)):
            raise TypeError(f'{item} should be a list or tuple')
        self._operations.append(_operation('Extend', self._location, value=item))

    def remove(self, item):
        if False:
            i = 10
            return i + 15
        'filter the item out of a list on the frontend'
        self._operations.append(_operation('Remove', self._location, value=item))

    def update(self, E=None, **F):
        if False:
            return 10
        'Merge a dict or keyword arguments with another dictionary'
        value = E or {}
        value.update(F)
        self._operations.append(_operation('Merge', self._location, value=value))

    def sort(self):
        if False:
            while True:
                i = 10
        raise KeyError('sort is reserved for future use, use brackets to access this key on your object')

    def to_plotly_json(self):
        if False:
            return 10
        return {'__dash_patch_update': '__dash_patch_update', 'operations': self._operations}