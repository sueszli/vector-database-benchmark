import inspect
import pickle
import io
from nvidia.dali import reducers

class _DaliPickle:

    @staticmethod
    def dumps(obj, protocol=None, **kwargs):
        if False:
            print('Hello World!')
        f = io.BytesIO()
        reducers.DaliCallbackPickler(f, protocol, **kwargs).dump(obj)
        return f.getvalue()

    @staticmethod
    def loads(s, **kwargs):
        if False:
            print('Hello World!')
        return pickle.loads(s, **kwargs)

class _CustomPickler:

    @classmethod
    def create(cls, py_callback_pickler):
        if False:
            for i in range(10):
                print('nop')
        if py_callback_pickler is None or isinstance(py_callback_pickler, cls):
            return py_callback_pickler
        if hasattr(py_callback_pickler, 'dumps') and hasattr(py_callback_pickler, 'loads'):
            return cls.create_from_reducer(py_callback_pickler)
        if isinstance(py_callback_pickler, (tuple, list)):
            params = [None] * 3
            for (i, item) in enumerate(py_callback_pickler):
                params[i] = item
            (reducer, kwargs_dumps, kwargs_loads) = params
            return cls.create_from_reducer(reducer, kwargs_dumps, kwargs_loads)
        raise ValueError('Unsupported py_callback_pickler value provided.')

    @classmethod
    def create_from_reducer(cls, reducer, dumps_kwargs=None, loads_kwargs=None):
        if False:
            while True:
                i = 10
        return cls(reducer.dumps, reducer.loads, dumps_kwargs, loads_kwargs)

    def __init__(self, dumps, loads, dumps_kwargs, loads_kwargs):
        if False:
            return 10
        self._dumps = dumps
        self._loads = loads
        self.dumps_kwargs = dumps_kwargs or {}
        self.loads_kwargs = loads_kwargs or {}

    def dumps(self, obj):
        if False:
            return 10
        return self._dumps(obj, **self.dumps_kwargs)

    def loads(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return self._loads(obj, **self.loads_kwargs)

def pickle_by_value(fun):
    if False:
        while True:
            i = 10
    "\n    Hints parallel external source to serialize a decorated global function by value\n    rather than by reference, which would be a default behavior of Python's pickler.\n    "
    if inspect.isfunction(fun):
        setattr(fun, '_dali_pickle_by_value', True)
        return fun
    else:
        raise TypeError('Only functions can be explicitely set to be pickled by value')