import inspect
from functools import partial
from joblib.externals.cloudpickle import dumps, loads
WRAP_CACHE = {}

class CloudpickledObjectWrapper:

    def __init__(self, obj, keep_wrapper=False):
        if False:
            return 10
        self._obj = obj
        self._keep_wrapper = keep_wrapper

    def __reduce__(self):
        if False:
            return 10
        _pickled_object = dumps(self._obj)
        if not self._keep_wrapper:
            return (loads, (_pickled_object,))
        return (_reconstruct_wrapper, (_pickled_object, self._keep_wrapper))

    def __getattr__(self, attr):
        if False:
            return 10
        if attr not in ['_obj', '_keep_wrapper']:
            return getattr(self._obj, attr)
        return getattr(self, attr)

class CallableObjectWrapper(CloudpickledObjectWrapper):

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self._obj(*args, **kwargs)

def _wrap_non_picklable_objects(obj, keep_wrapper):
    if False:
        i = 10
        return i + 15
    if callable(obj):
        return CallableObjectWrapper(obj, keep_wrapper=keep_wrapper)
    return CloudpickledObjectWrapper(obj, keep_wrapper=keep_wrapper)

def _reconstruct_wrapper(_pickled_object, keep_wrapper):
    if False:
        return 10
    obj = loads(_pickled_object)
    return _wrap_non_picklable_objects(obj, keep_wrapper)

def _wrap_objects_when_needed(obj):
    if False:
        i = 10
        return i + 15
    need_wrap = '__main__' in getattr(obj, '__module__', '')
    if isinstance(obj, partial):
        return partial(_wrap_objects_when_needed(obj.func), *[_wrap_objects_when_needed(a) for a in obj.args], **{k: _wrap_objects_when_needed(v) for (k, v) in obj.keywords.items()})
    if callable(obj):
        func_code = getattr(obj, '__code__', '')
        need_wrap |= getattr(func_code, 'co_flags', 0) & inspect.CO_NESTED
        func_name = getattr(obj, '__name__', '')
        need_wrap |= '<lambda>' in func_name
    if not need_wrap:
        return obj
    wrapped_obj = WRAP_CACHE.get(obj)
    if wrapped_obj is None:
        wrapped_obj = _wrap_non_picklable_objects(obj, keep_wrapper=False)
        WRAP_CACHE[obj] = wrapped_obj
    return wrapped_obj

def wrap_non_picklable_objects(obj, keep_wrapper=True):
    if False:
        for i in range(10):
            print('nop')
    'Wrapper for non-picklable object to use cloudpickle to serialize them.\n\n    Note that this wrapper tends to slow down the serialization process as it\n    is done with cloudpickle which is typically slower compared to pickle. The\n    proper way to solve serialization issues is to avoid defining functions and\n    objects in the main scripts and to implement __reduce__ functions for\n    complex classes.\n    '
    if inspect.isclass(obj):

        class CloudpickledClassWrapper(CloudpickledObjectWrapper):

            def __init__(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                self._obj = obj(*args, **kwargs)
                self._keep_wrapper = keep_wrapper
        CloudpickledClassWrapper.__name__ = obj.__name__
        return CloudpickledClassWrapper
    return _wrap_non_picklable_objects(obj, keep_wrapper=keep_wrapper)