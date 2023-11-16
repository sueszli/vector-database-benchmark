"""
This module centralizes all functionality related to json encoding and decoding in Connexion.
"""
import datetime
import functools
import json
import typing as t
import uuid
from decimal import Decimal

def wrap_default(default_fn: t.Callable) -> t.Callable:
    if False:
        for i in range(10):
            print('nop')
    'The Connexion defaults for JSON encoding. Handles extra types compared to the\n    built-in :class:`json.JSONEncoder`.\n\n    -   :class:`datetime.datetime` and :class:`datetime.date` are\n        serialized to :rfc:`822` strings. This is the same as the HTTP\n        date format.\n    -   :class:`decimal.Decimal` is serialized to a float.\n    -   :class:`uuid.UUID` is serialized to a string.\n    '

    @functools.wraps(default_fn)
    def wrapped_default(self, o):
        if False:
            i = 10
            return i + 15
        if isinstance(o, datetime.datetime):
            if o.tzinfo:
                return o.isoformat('T')
            else:
                return o.isoformat('T') + 'Z'
        if isinstance(o, datetime.date):
            return o.isoformat()
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, uuid.UUID):
            return str(o)
        return default_fn(self, o)
    return wrapped_default

class JSONEncoder(json.JSONEncoder):
    """The default Connexion JSON encoder. Handles extra types compared to the
    built-in :class:`json.JSONEncoder`.

    -   :class:`datetime.datetime` and :class:`datetime.date` are
        serialized to :rfc:`822` strings. This is the same as the HTTP
        date format.
    -   :class:`uuid.UUID` is serialized to a string.
    """

    @wrap_default
    def default(self, o):
        if False:
            for i in range(10):
                print('nop')
        return super().default(o)

class Jsonifier:
    """
    Central point to serialize and deserialize to/from JSon in Connexion.
    """

    def __init__(self, json_=json, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        :param json_: json library to use. Must have loads() and dumps() method  # NOQA\n        :param kwargs: default arguments to pass to json.dumps()\n        '
        self.json = json_
        self.dumps_args = kwargs
        self.dumps_args.setdefault('cls', JSONEncoder)

    def dumps(self, data, **kwargs):
        if False:
            return 10
        'Central point where JSON serialization happens inside\n        Connexion.\n        '
        for (k, v) in self.dumps_args.items():
            kwargs.setdefault(k, v)
        return self.json.dumps(data, **kwargs) + '\n'

    def loads(self, data):
        if False:
            i = 10
            return i + 15
        'Central point where JSON deserialization happens inside\n        Connexion.\n        '
        if isinstance(data, bytes):
            data = data.decode()
        try:
            return self.json.loads(data)
        except Exception:
            if isinstance(data, str):
                return data