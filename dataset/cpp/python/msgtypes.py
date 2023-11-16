# -*- coding: utf-8 -*-

__all__ = (
    'MSG_TYPES_PACK', 'MSG_TYPES_UNPACK',
    'msgpack_exthook'
)

from collections import namedtuple

from network.conf import transports

from network.lib import Proxy
from network.lib.proxies import ProxyInfo
from network.lib.utils import HostInfo, TransportInfo

from umsgpack import Ext, packb, unpackb


MSG_TYPES_PACK = {}
MSG_TYPES_UNPACK = {}

KNOWN_NAMED_TUPLES = (
    Proxy, ProxyInfo, HostInfo, TransportInfo
)


def register_named_tuple(code, type):
    MSG_TYPES_PACK[type] = lambda obj: Ext(
        code, packb(tuple(x for x in obj)))
    MSG_TYPES_UNPACK[code] = lambda obj: type(*unpackb(obj.data))


def register_string(type, code, name):
    MSG_TYPES_PACK[type] = lambda obj: Ext(code, '')
    MSG_TYPES_UNPACK[code] = lambda obj: name


for idx, type in enumerate(KNOWN_NAMED_TUPLES):
    register_named_tuple(idx, type)

SPECIAL_TYPES_OFFT = len(KNOWN_NAMED_TUPLES)

for idx, (name, type) in enumerate(transports.iteritems()):
    register_string(idx, type, name)


wrapext = namedtuple('Ext', ('code', 'data'))

def msgpack_exthook(code, data):
    if code in MSG_TYPES_UNPACK:
        obj = wrapext(code, data)
        return MSG_TYPES_UNPACK[code](obj)
