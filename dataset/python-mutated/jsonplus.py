"""Custom datatypes (like datetime) serialization to/from JSON."""
import numpy as np
import simplejson as json
import threading
import uuid
from collections import namedtuple
from datetime import date, datetime, time, timedelta
from dateutil.parser import parse as parse_datetime
from decimal import Decimal
from fractions import Fraction
from functools import partial, wraps
from operator import attrgetter, methodcaller
from sortedcontainers import SortedList
try:
    from moneyed import Currency, Money
except ImportError:
    pass
__all__ = ['loads', 'dumps', 'pretty', 'json_loads', 'json_dumps', 'json_prettydump', 'encoder', 'decoder']
EXACT = 1
COMPAT = 2
CODING_DEFAULT = EXACT
_local = threading.local()

def prefer(coding):
    if False:
        while True:
            i = 10
    _local.coding = coding

def prefer_exact():
    if False:
        for i in range(10):
            print('nop')
    prefer(EXACT)

def prefer_compat():
    if False:
        while True:
            i = 10
    prefer(COMPAT)

def getattrs(value, attrs):
    if False:
        print('Hello World!')
    "Helper function that extracts a list of attributes from\n    `value` object in a `dict`/mapping of (attr, value[attr]).\n\n    Args:\n        value (object):\n            Any Python object upon which `getattr` can act.\n\n        attrs (iterable):\n            Any iterable containing attribute names for extract.\n\n    Returns:\n        `dict` of attr -> val mappings.\n\n    Example:\n        >>> getattrs(complex(2,3), ['imag', 'real'])\n        {'imag': 3.0, 'real': 2.0}\n    "
    return dict([(attr, getattr(value, attr)) for attr in attrs])

def kwargified(constructor):
    if False:
        return 10
    "Function decorator that wraps a function receiving\n    keyword arguments into a function receiving a dictionary\n    of arguments.\n\n    Example:\n        @kwargified\n        def test(a=1, b=2):\n            return a + b\n\n        >>> test({'b': 3})\n        4\n    "

    @wraps(constructor)
    def kwargs_constructor(kwargs):
        if False:
            i = 10
            return i + 15
        return constructor(**kwargs)
    return kwargs_constructor
_PredicatedEncoder = namedtuple('_PredicatedEncoder', 'priority predicate encoder typename')

def encoder(classname, predicate=None, priority=None, exact=True):
    if False:
        while True:
            i = 10
    "A decorator for registering a new encoder for object type\n    defined either by a `classname`, or detected via `predicate`.\n\n    Predicates are tested according to priority (low to high),\n    but always before classname.\n\n    Args:\n        classname (str):\n            Classname of the object serialized, equal to\n            ``type(obj).__name__``.\n\n        predicate (callable, default=None):\n            A predicate for testing if object is of certain type.\n            The predicate shall receive a single argument, the object\n            being encoded, and it has to return a boolean `True/False`.\n            See examples below.\n\n        priority (int, default=None):\n            Predicate priority. If undefined, encoder is added at\n            the end, with lowest priority.\n\n        exact (bool, default=True):\n            Determines the kind of encoder registered, an exact\n            (default), or a compact representation encoder.\n\n    Examples:\n        @encoder('mytype')\n        def mytype_exact_encoder(myobj):\n            return myobj.to_json()\n\n        Functional discriminator usage is appropriate for serialization\n        of objects with a different classname, but which can be encoded\n        with the same encoder:\n\n        @encoder('BaseClass', lambda obj: isinstance(obj, BaseClass))\n        def all_derived_classes_encoder(derived):\n            return derived.base_encoder()\n    "
    if exact:
        subregistry = _encode_handlers['exact']
    else:
        subregistry = _encode_handlers['compat']
    if priority is None:
        if len(subregistry['predicate']) > 0:
            priority = subregistry['predicate'][-1].priority + 100
        else:
            priority = 1000

    def _decorator(f):
        if False:
            i = 10
            return i + 15
        if predicate:
            subregistry['predicate'].add(_PredicatedEncoder(priority, predicate, f, classname))
        else:
            subregistry['classname'].setdefault(classname, f)
        return f
    return _decorator

def _json_default_exact(obj):
    if False:
        while True:
            i = 10
    'Serialization handlers for types unsupported by `simplejson` \n    that try to preserve the exact data types.\n    '
    for handler in _encode_handlers['exact']['predicate']:
        if handler.predicate(obj):
            return {'__class__': handler.typename, '__value__': handler.encoder(obj)}
    classname = type(obj).__name__
    if classname in _encode_handlers['exact']['classname']:
        return {'__class__': classname, '__value__': _encode_handlers['exact']['classname'][classname](obj)}
    raise TypeError(repr(obj) + ' is not JSON serializable')

def _json_default_compat(obj):
    if False:
        print('Hello World!')
    'Serialization handlers that try to dump objects in\n    compatibility mode. Similar to above.\n    '
    for handler in _encode_handlers['compat']['predicate']:
        if handler.predicate(obj):
            return handler.encoder(obj)
    classname = type(obj).__name__
    if classname in _encode_handlers['compat']['classname']:
        return _encode_handlers['compat']['classname'][classname](obj)
    raise TypeError(repr(obj) + ' is not JSON serializable')

def decoder(classname):
    if False:
        for i in range(10):
            print('nop')
    "A decorator for registering a new decoder for `classname`.\n    Only ``exact`` decoders can be registered, since it is an assumption\n    the ``compat`` mode serializes to standard JSON.\n\n    Example:\n        @decoder('mytype')\n        def mytype_decoder(value):\n            return mytype(value, reconstruct=True)\n    "

    def _decorator(f):
        if False:
            print('Hello World!')
        _decode_handlers.setdefault(classname, f)
    return _decorator

def _json_object_hook(dict):
    if False:
        return 10
    'Deserialization handlers for types unsupported by `simplejson`.\n    '
    classname = dict.get('__class__')
    if classname:
        constructor = _decode_handlers.get(classname)
        value = dict.get('__value__')
        if constructor:
            return constructor(value)
        raise TypeError("Unknown class: '%s'" % classname)
    return dict

def _encoder_default_args(kw):
    if False:
        while True:
            i = 10
    'Shape default arguments for encoding functions.'
    if kw.pop('exact', getattr(_local, 'coding', CODING_DEFAULT) == EXACT):
        kw.update({'default': _json_default_exact, 'use_decimal': False, 'tuple_as_array': False, 'namedtuple_as_object': False})
    else:
        kw.update({'default': _json_default_compat, 'ignore_nan': True})
    kw.setdefault('separators', (',', ':'))
    kw.setdefault('for_json', True)

def _decoder_default_args(kw):
    if False:
        return 10
    'Shape default arguments for decoding functions.'
    kw.update({'object_hook': _json_object_hook})

class JSONEncoder(json.JSONEncoder):

    def __init__(self, **kw):
        if False:
            while True:
                i = 10
        'Constructor for simplejson.JSONEncoder, with defaults overriden\n        for jsonplus.\n        '
        _encoder_default_args(kw)
        super(JSONEncoder, self).__init__(**kw)

class JSONDecoder(json.JSONDecoder):

    def __init__(self, **kw):
        if False:
            print('Hello World!')
        'Constructor for simplejson.JSONDecoder, with defaults overriden\n        for jsonplus.\n        '
        _decoder_default_args(kw)
        super(JSONDecoder, self).__init__(**kw)

def dumps(*pa, **kw):
    if False:
        return 10
    _encoder_default_args(kw)
    return json.dumps(*pa, **kw)

def loads(*pa, **kw):
    if False:
        print('Hello World!')
    _decoder_default_args(kw)
    return json.loads(*pa, **kw)

def pretty(x, sort_keys=True, indent=4 * ' ', separators=(',', ': '), **kw):
    if False:
        while True:
            i = 10
    kw.setdefault('sort_keys', sort_keys)
    kw.setdefault('indent', indent)
    kw.setdefault('separators', separators)
    return dumps(x, **kw)
json_dumps = dumps
json_loads = loads
json_prettydump = pretty

def np_to_list(value):
    if False:
        i = 10
        return i + 15
    return value.tolist()

def generic_to_item(value):
    if False:
        for i in range(10):
            print('nop')
    return value.item()
_encode_handlers = {'exact': {'classname': {'datetime': methodcaller('isoformat'), 'date': methodcaller('isoformat'), 'time': methodcaller('isoformat'), 'timedelta': partial(getattrs, attrs=['days', 'seconds', 'microseconds']), 'tuple': list, 'set': list, 'ndarray': np_to_list, 'float16': generic_to_item, 'float32': generic_to_item, 'frozenset': list, 'complex': partial(getattrs, attrs=['real', 'imag']), 'Decimal': str, 'Fraction': partial(getattrs, attrs=['numerator', 'denominator']), 'UUID': partial(getattrs, attrs=['hex']), 'Money': partial(getattrs, attrs=['amount', 'currency'])}, 'predicate': SortedList(key=attrgetter('priority'))}, 'compat': {'classname': {'datetime': methodcaller('isoformat'), 'date': methodcaller('isoformat'), 'time': methodcaller('isoformat'), 'set': list, 'ndarray': np_to_list, 'float16': generic_to_item, 'float32': generic_to_item, 'frozenset': list, 'complex': partial(getattrs, attrs=['real', 'imag']), 'Fraction': partial(getattrs, attrs=['numerator', 'denominator']), 'UUID': str, 'Currency': str, 'Money': str}, 'predicate': SortedList(key=attrgetter('priority'))}}
_decode_handlers = {'datetime': parse_datetime, 'date': lambda v: parse_datetime(v).date(), 'time': lambda v: parse_datetime(v).timetz(), 'timedelta': kwargified(timedelta), 'tuple': tuple, 'set': set, 'ndarray': np.asarray, 'float16': np.float16, 'float32': np.float32, 'frozenset': frozenset, 'complex': kwargified(complex), 'Decimal': Decimal, 'Fraction': kwargified(Fraction), 'UUID': kwargified(uuid.UUID)}

@encoder('namedtuple', lambda obj: isinstance(obj, tuple) and hasattr(obj, '_fields'))
def _dump_namedtuple(obj):
    if False:
        i = 10
        return i + 15
    return {'name': type(obj).__name__, 'fields': list(obj._fields), 'values': list(obj)}

@decoder('namedtuple')
def _load_namedtuple(val):
    if False:
        for i in range(10):
            print('nop')
    cls = namedtuple(val['name'], val['fields'])
    return cls(*val['values'])

@encoder('timedelta', exact=False)
def _timedelta_total_seconds(td):
    if False:
        while True:
            i = 10
    return (td.microseconds + (td.seconds + td.days * 24 * 3600.0) * 10 ** 6) / 10 ** 6

@encoder('Currency')
def _dump_currency(obj):
    if False:
        i = 10
        return i + 15
    'Serialize standard (ISO-defined) currencies to currency code only,\n    and non-standard (user-added) currencies in full.\n    '
    from moneyed import CurrencyDoesNotExist, get_currency
    try:
        get_currency(obj.code)
        return obj.code
    except CurrencyDoesNotExist:
        return getattrs(obj, ['code', 'numeric', 'name', 'countries'])

@decoder('Currency')
def _load_currency(val):
    if False:
        return 10
    'Deserialize string values as standard currencies, but\n    manually define fully-defined currencies (with code/name/numeric/countries).\n    '
    from moneyed import get_currency
    try:
        return get_currency(code=val)
    except:
        return Currency(**val)

@decoder('Money')
def _load_money(val):
    if False:
        while True:
            i = 10
    return Money(**val)