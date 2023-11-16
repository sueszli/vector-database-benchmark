"""
Utilities to resolve a string to Mongo host, or a Arctic library.
"""
import logging
import re
from weakref import WeakValueDictionary
__all__ = ['get_arctic_lib']
logger = logging.getLogger(__name__)
arctic_cache = WeakValueDictionary()
CONNECTION_STR = re.compile('(^\\w+\\.?\\w+)@([^\\s:]+:?\\w+)$')

def get_arctic_lib(connection_string, **kwargs):
    if False:
        print('Hello World!')
    '\n    Returns a mongo library for the given connection string\n\n    Parameters\n    ---------\n    connection_string: `str`\n        Format must be one of the following:\n            library@trading for known mongo servers\n            library@hostname:port\n\n    Returns:\n    --------\n    Arctic library\n    '
    m = CONNECTION_STR.match(connection_string)
    if not m:
        raise ValueError('connection string incorrectly formed: %s' % connection_string)
    (library, host) = (m.group(1), m.group(2))
    return _get_arctic(host, **kwargs)[library]

def _get_arctic(instance, **kwargs):
    if False:
        i = 10
        return i + 15
    key = (instance, frozenset(kwargs.items()))
    arctic = arctic_cache.get(key, None)
    if not arctic:
        from .arctic import Arctic
        arctic = Arctic(instance, **kwargs)
        arctic_cache[key] = arctic
    return arctic