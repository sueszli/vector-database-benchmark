import pickle
from collections import namedtuple

class RecordLevel:
    __slots__ = ('name', 'no', 'icon')

    def __init__(self, name, no, icon):
        if False:
            print('Hello World!')
        self.name = name
        self.no = no
        self.icon = icon

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '(name=%r, no=%r, icon=%r)' % (self.name, self.no, self.icon)

    def __format__(self, spec):
        if False:
            return 10
        return self.name.__format__(spec)

class RecordFile:
    __slots__ = ('name', 'path')

    def __init__(self, name, path):
        if False:
            return 10
        self.name = name
        self.path = path

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '(name=%r, path=%r)' % (self.name, self.path)

    def __format__(self, spec):
        if False:
            for i in range(10):
                print('nop')
        return self.name.__format__(spec)

class RecordThread:
    __slots__ = ('id', 'name')

    def __init__(self, id_, name):
        if False:
            print('Hello World!')
        self.id = id_
        self.name = name

    def __repr__(self):
        if False:
            print('Hello World!')
        return '(id=%r, name=%r)' % (self.id, self.name)

    def __format__(self, spec):
        if False:
            i = 10
            return i + 15
        return self.id.__format__(spec)

class RecordProcess:
    __slots__ = ('id', 'name')

    def __init__(self, id_, name):
        if False:
            print('Hello World!')
        self.id = id_
        self.name = name

    def __repr__(self):
        if False:
            return 10
        return '(id=%r, name=%r)' % (self.id, self.name)

    def __format__(self, spec):
        if False:
            i = 10
            return i + 15
        return self.id.__format__(spec)

class RecordException(namedtuple('RecordException', ('type', 'value', 'traceback'))):

    def __repr__(self):
        if False:
            print('Hello World!')
        return '(type=%r, value=%r, traceback=%r)' % (self.type, self.value, self.traceback)

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        try:
            pickled_value = pickle.dumps(self.value)
        except Exception:
            return (RecordException, (self.type, None, None))
        else:
            return (RecordException._from_pickled_value, (self.type, pickled_value, None))

    @classmethod
    def _from_pickled_value(cls, type_, pickled_value, traceback_):
        if False:
            print('Hello World!')
        try:
            value = pickle.loads(pickled_value)
        except Exception:
            return cls(type_, None, traceback_)
        else:
            return cls(type_, value, traceback_)