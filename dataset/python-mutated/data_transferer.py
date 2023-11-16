import base64
import functools
import pickle
import sys
from collections import OrderedDict, defaultdict, namedtuple
from copy import copy
from datetime import datetime, timedelta
ObjReference = namedtuple('ObjReference', 'value_type class_name identifier')
if sys.version_info[0] >= 3:

    class InvalidLong:
        pass

    class InvalidUnicode:
        pass
_types = [type(None), bool, int, float, complex, str, list, tuple, bytearray, bytes, set, frozenset, dict, defaultdict, OrderedDict, datetime, timedelta]
_container_types = (list, tuple, set, frozenset, dict, defaultdict, OrderedDict)
if sys.version_info[0] >= 3:
    _types.extend([InvalidLong, InvalidUnicode])
    _simple_types = (bool, int, float, complex, bytearray, bytes, datetime, timedelta)
else:
    _types.extend([long, unicode])
    _simple_types = (bool, int, float, complex, bytearray, bytes, unicode, long, datetime, timedelta)
_types_to_encoding = {x: idx for (idx, x) in enumerate(_types)}
_dumpers = {}
_loaders = {}
FIELD_TYPE = 't'
FIELD_ANNOTATION = 'a'
FIELD_INLINE_VALUE = 'v'
FIELD_INLINE_KEY = 'k'
defaultProtocol = pickle.HIGHEST_PROTOCOL

def _register_dumper(what):
    if False:
        i = 10
        return i + 15

    def wrapper(func):
        if False:
            print('Hello World!')
        for w in what:
            _dumpers[w] = functools.partial(func, w)
        return func
    return wrapper

def _register_loader(what):
    if False:
        while True:
            i = 10

    def wrapper(func):
        if False:
            i = 10
            return i + 15
        for w in what:
            _loaders[_types_to_encoding[w]] = functools.partial(func, w)
        return func
    return wrapper

@_register_dumper((type(None),))
def _dump_none(obj_type, transferer, obj):
    if False:
        print('Hello World!')
    return (None, False)

@_register_loader((type(None),))
def _load_none(obj_type, transferer, json_annotation, json_obj):
    if False:
        return 10
    return None

@_register_dumper(_simple_types)
def _dump_simple(obj_type, transferer, obj):
    if False:
        i = 10
        return i + 15
    return (None, base64.b64encode(pickle.dumps(obj, protocol=defaultProtocol)).decode('utf-8'))

@_register_loader(_simple_types)
def _load_simple(obj_type, transferer, json_annotation, json_obj):
    if False:
        i = 10
        return i + 15
    new_obj = pickle.loads(base64.b64decode(json_obj), encoding='utf-8')
    if not isinstance(new_obj, obj_type):
        raise RuntimeError("Pickle didn't create an object of the proper type")
    return new_obj

@_register_dumper(_container_types)
def _dump_container(obj_type, transferer, obj):
    if False:
        while True:
            i = 10
    try:
        new_obj = transferer.pickle_container(obj)
    except RuntimeError as e:
        raise RuntimeError('Cannot dump container %s: %s' % (str(obj), e))
    if new_obj is None:
        return _dump_simple(obj_type, transferer, obj)
    else:
        (_, dump) = _dump_simple(obj_type, transferer, new_obj)
        return (True, dump)

@_register_loader(_container_types)
def _load_container(obj_type, transferer, json_annotation, json_obj):
    if False:
        while True:
            i = 10
    obj = _load_simple(obj_type, transferer, json_annotation, json_obj)
    if json_annotation:
        obj = transferer.unpickle_container(obj)
    return obj
if sys.version_info[0] >= 3:

    @_register_dumper((str,))
    def _dump_str(obj_type, transferer, obj):
        if False:
            print('Hello World!')
        return _dump_simple(obj_type, transferer, obj)

    @_register_loader((str,))
    def _load_str(obj_type, transferer, json_annotation, json_obj):
        if False:
            for i in range(10):
                print('nop')
        return _load_simple(obj_type, transferer, json_annotation, json_obj)

    @_register_dumper((InvalidLong,))
    def _dump_invalidlong(obj_type, transferer, obj):
        if False:
            return 10
        return _dump_simple(int, transferer, obj)

    @_register_loader((InvalidLong,))
    def _load_invalidlong(obj_type, transferer, json_annotation, json_obj):
        if False:
            for i in range(10):
                print('nop')
        return _load_simple(int, transferer, json_annotation, json_obj)

    @_register_dumper((InvalidUnicode,))
    def _dump_invalidunicode(obj_type, transferer, obj):
        if False:
            for i in range(10):
                print('nop')
        return _dump_simple(str, transferer, obj)

    @_register_loader((InvalidUnicode,))
    def _load_invalidunicode(obj_type, transferer, json_annotation, json_obj):
        if False:
            print('Hello World!')
        return _load_simple(str, transferer, json_annotation, json_obj)
else:

    @_register_dumper((str,))
    def _dump_str(obj_type, transferer, obj):
        if False:
            print('Hello World!')
        return _dump_simple(obj_type, transferer, obj.encode('utf-8'))

    @_register_loader((str,))
    def _load_str(obj_type, transferer, json_annotation, json_obj):
        if False:
            return 10
        return _load_simple(bytes, json_annotation, json_obj).decode('utf-8')

    @_register_dumper((unicode, long))
    def _dump_py2_simple(obj_type, transferer, obj):
        if False:
            return 10
        return _dump_simple(obj_type, transferer, obj)

    @_register_loader((unicode, long))
    def _load_py2_simple(obj_type, transferer, json_annotation, json_obj):
        if False:
            while True:
                i = 10
        return _load_simple(obj_type, transferer, json_annotation, json_obj)

class DataTransferer(object):

    def __init__(self, connection):
        if False:
            i = 10
            return i + 15
        self._dumpers = _dumpers.copy()
        self._loaders = _loaders.copy()
        self._types_to_encoding = _types_to_encoding.copy()
        self._connection = connection

    @staticmethod
    def can_simple_dump(obj):
        if False:
            return 10
        return DataTransferer._can_dump(DataTransferer.can_simple_dump, obj)

    def can_dump(self, obj):
        if False:
            return 10
        r = DataTransferer._can_dump(self.can_dump, obj)
        if not r:
            return self._connection.can_encode(obj)
        return False

    def dump(self, obj):
        if False:
            return 10
        obj_type = type(obj)
        handler = self._dumpers.get(type(obj))
        if handler:
            (attr, v) = handler(self, obj)
            return {FIELD_TYPE: self._types_to_encoding[obj_type], FIELD_ANNOTATION: attr, FIELD_INLINE_VALUE: v}
        else:
            try:
                json_obj = base64.b64encode(pickle.dumps(self._connection.pickle_object(obj), protocol=defaultProtocol)).decode('utf-8')
            except ValueError as e:
                raise RuntimeError('Unable to dump non base type: %s' % e)
            return {FIELD_TYPE: -1, FIELD_INLINE_VALUE: json_obj}

    def load(self, json_obj):
        if False:
            for i in range(10):
                print('nop')
        obj_type = json_obj.get(FIELD_TYPE)
        if obj_type is None:
            raise RuntimeError('Malformed message -- missing %s: %s' % (FIELD_TYPE, str(json_obj)))
        if obj_type == -1:
            try:
                return self._connection.unpickle_object(pickle.loads(base64.b64decode(json_obj[FIELD_INLINE_VALUE]), encoding='utf-8'))
            except ValueError as e:
                raise RuntimeError('Unable to load non base type: %s' % e)
        handler = self._loaders.get(obj_type)
        if handler:
            json_subobj = json_obj.get(FIELD_INLINE_VALUE)
            if json_subobj is not None:
                return handler(self, json_obj.get(FIELD_ANNOTATION), json_obj[FIELD_INLINE_VALUE])
            raise RuntimeError('Non inline value not supported')
        raise RuntimeError('Unable to find handler for type %s' % obj_type)

    def _transform_container(self, checker, processor, recursor, obj, in_place=True):
        if False:
            return 10

        def _sub_process(obj):
            if False:
                while True:
                    i = 10
            obj_type = type(obj)
            if obj is None or obj_type in _simple_types or obj_type == str:
                return None
            elif obj_type in _container_types:
                return recursor(obj)
            elif checker(obj):
                return processor(obj)
            else:
                raise RuntimeError('Cannot pickle object of type %s: %s' % (obj_type, str(obj)))
        cast_to = None
        key_change_allowed = True
        update_default_factory = False
        has_changes = False
        if isinstance(obj, (tuple, set, frozenset)):
            cast_to = type(obj)
            obj = list(obj)
            in_place = True
        if isinstance(obj, OrderedDict):
            key_change_allowed = False
        if isinstance(obj, defaultdict):
            if callable(obj.default_factory):
                if not in_place:
                    obj = copy(obj)
                    in_place = True
                obj['__default_factory'] = obj.default_factory
                obj.default_factory = None
            elif obj.get('__default_factory') is not None:
                update_default_factory = True
            has_changes = True
        if isinstance(obj, list):
            for idx in range(len(obj)):
                sub_obj = _sub_process(obj[idx])
                if sub_obj is not None:
                    has_changes = True
                    if not in_place:
                        obj = list(obj)
                        in_place = True
                    obj[idx] = sub_obj
        elif isinstance(obj, dict):
            new_items = {}
            del_keys = []
            for (k, v) in obj.items():
                sub_key = _sub_process(k)
                if sub_key is not None:
                    if not key_change_allowed:
                        raise RuntimeError('OrderedDict key cannot contain references -- this would change the order')
                    has_changes = True
                sub_val = _sub_process(v)
                if sub_val is not None:
                    has_changes = True
                if has_changes and (not in_place):
                    obj = copy(obj)
                    in_place = True
                if sub_key:
                    if sub_val:
                        new_items[sub_key] = sub_val
                    else:
                        new_items[sub_key] = v
                    del_keys.append(k)
                elif sub_val:
                    obj[k] = sub_val
            for k in del_keys:
                del obj[k]
            obj.update(new_items)
        else:
            raise RuntimeError('Unknown container type: %s' % type(obj))
        if update_default_factory:
            obj.default_factory = obj['__default_factory']
            del obj['__default_factory']
        if has_changes:
            if cast_to:
                return cast_to(obj)
            return obj
        return None

    def pickle_container(self, obj):
        if False:
            return 10
        return self._transform_container(self._connection.can_pickle, self._connection.pickle_object, self.pickle_container, obj, in_place=False)

    def unpickle_container(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return self._transform_container(lambda x: isinstance(x, ObjReference), self._connection.unpickle_object, self.unpickle_container, obj)

    @staticmethod
    def _can_dump(recursive_func, obj):
        if False:
            i = 10
            return i + 15
        obj_type = type(obj)
        if obj is None:
            return True
        if obj_type in _simple_types:
            return True
        if obj_type == str:
            return True
        if obj_type == dict or obj_type == OrderedDict:
            return all((recursive_func(k) and recursive_func(v) for (k, v) in obj.items()))
        if obj_type in _container_types:
            return all((recursive_func(x) for x in obj))
        return False