"""Tools for working with MongoDB ObjectIds."""
from __future__ import annotations
import binascii
import calendar
import datetime
import os
import struct
import threading
import time
from random import SystemRandom
from typing import Any, NoReturn, Optional, Type, Union
from bson.errors import InvalidId
from bson.tz_util import utc
_MAX_COUNTER_VALUE = 16777215

def _raise_invalid_id(oid: str) -> NoReturn:
    if False:
        while True:
            i = 10
    raise InvalidId('%r is not a valid ObjectId, it must be a 12-byte input or a 24-character hex string' % oid)

def _random_bytes() -> bytes:
    if False:
        return 10
    'Get the 5-byte random field of an ObjectId.'
    return os.urandom(5)

class ObjectId:
    """A MongoDB ObjectId."""
    _pid = os.getpid()
    _inc = SystemRandom().randint(0, _MAX_COUNTER_VALUE)
    _inc_lock = threading.Lock()
    __random = _random_bytes()
    __slots__ = ('__id',)
    _type_marker = 7

    def __init__(self, oid: Optional[Union[str, ObjectId, bytes]]=None) -> None:
        if False:
            return 10
        "Initialize a new ObjectId.\n\n        An ObjectId is a 12-byte unique identifier consisting of:\n\n          - a 4-byte value representing the seconds since the Unix epoch,\n          - a 5-byte random value,\n          - a 3-byte counter, starting with a random value.\n\n        By default, ``ObjectId()`` creates a new unique identifier. The\n        optional parameter `oid` can be an :class:`ObjectId`, or any 12\n        :class:`bytes`.\n\n        For example, the 12 bytes b'foo-bar-quux' do not follow the ObjectId\n        specification but they are acceptable input::\n\n          >>> ObjectId(b'foo-bar-quux')\n          ObjectId('666f6f2d6261722d71757578')\n\n        `oid` can also be a :class:`str` of 24 hex digits::\n\n          >>> ObjectId('0123456789ab0123456789ab')\n          ObjectId('0123456789ab0123456789ab')\n\n        Raises :class:`~bson.errors.InvalidId` if `oid` is not 12 bytes nor\n        24 hex digits, or :class:`TypeError` if `oid` is not an accepted type.\n\n        :Parameters:\n          - `oid` (optional): a valid ObjectId.\n\n        .. seealso:: The MongoDB documentation on  `ObjectIds <http://dochub.mongodb.org/core/objectids>`_.\n\n        .. versionchanged:: 3.8\n           :class:`~bson.objectid.ObjectId` now implements the `ObjectID\n           specification version 0.2\n           <https://github.com/mongodb/specifications/blob/master/source/\n           objectid.rst>`_.\n        "
        if oid is None:
            self.__generate()
        elif isinstance(oid, bytes) and len(oid) == 12:
            self.__id = oid
        else:
            self.__validate(oid)

    @classmethod
    def from_datetime(cls: Type[ObjectId], generation_time: datetime.datetime) -> ObjectId:
        if False:
            i = 10
            return i + 15
        'Create a dummy ObjectId instance with a specific generation time.\n\n        This method is useful for doing range queries on a field\n        containing :class:`ObjectId` instances.\n\n        .. warning::\n           It is not safe to insert a document containing an ObjectId\n           generated using this method. This method deliberately\n           eliminates the uniqueness guarantee that ObjectIds\n           generally provide. ObjectIds generated with this method\n           should be used exclusively in queries.\n\n        `generation_time` will be converted to UTC. Naive datetime\n        instances will be treated as though they already contain UTC.\n\n        An example using this helper to get documents where ``"_id"``\n        was generated before January 1, 2010 would be:\n\n        >>> gen_time = datetime.datetime(2010, 1, 1)\n        >>> dummy_id = ObjectId.from_datetime(gen_time)\n        >>> result = collection.find({"_id": {"$lt": dummy_id}})\n\n        :Parameters:\n          - `generation_time`: :class:`~datetime.datetime` to be used\n            as the generation time for the resulting ObjectId.\n        '
        offset = generation_time.utcoffset()
        if offset is not None:
            generation_time = generation_time - offset
        timestamp = calendar.timegm(generation_time.timetuple())
        oid = struct.pack('>I', int(timestamp)) + b'\x00\x00\x00\x00\x00\x00\x00\x00'
        return cls(oid)

    @classmethod
    def is_valid(cls: Type[ObjectId], oid: Any) -> bool:
        if False:
            while True:
                i = 10
        'Checks if a `oid` string is valid or not.\n\n        :Parameters:\n          - `oid`: the object id to validate\n\n        .. versionadded:: 2.3\n        '
        if not oid:
            return False
        try:
            ObjectId(oid)
            return True
        except (InvalidId, TypeError):
            return False

    @classmethod
    def _random(cls) -> bytes:
        if False:
            i = 10
            return i + 15
        'Generate a 5-byte random number once per process.'
        pid = os.getpid()
        if pid != cls._pid:
            cls._pid = pid
            cls.__random = _random_bytes()
        return cls.__random

    def __generate(self) -> None:
        if False:
            while True:
                i = 10
        'Generate a new value for this ObjectId.'
        oid = struct.pack('>I', int(time.time()))
        oid += ObjectId._random()
        with ObjectId._inc_lock:
            oid += struct.pack('>I', ObjectId._inc)[1:4]
            ObjectId._inc = (ObjectId._inc + 1) % (_MAX_COUNTER_VALUE + 1)
        self.__id = oid

    def __validate(self, oid: Any) -> None:
        if False:
            return 10
        'Validate and use the given id for this ObjectId.\n\n        Raises TypeError if id is not an instance of :class:`str`,\n        :class:`bytes`, or ObjectId. Raises InvalidId if it is not a\n        valid ObjectId.\n\n        :Parameters:\n          - `oid`: a valid ObjectId\n        '
        if isinstance(oid, ObjectId):
            self.__id = oid.binary
        elif isinstance(oid, str):
            if len(oid) == 24:
                try:
                    self.__id = bytes.fromhex(oid)
                except (TypeError, ValueError):
                    _raise_invalid_id(oid)
            else:
                _raise_invalid_id(oid)
        else:
            raise TypeError(f'id must be an instance of (bytes, str, ObjectId), not {type(oid)}')

    @property
    def binary(self) -> bytes:
        if False:
            i = 10
            return i + 15
        '12-byte binary representation of this ObjectId.'
        return self.__id

    @property
    def generation_time(self) -> datetime.datetime:
        if False:
            for i in range(10):
                print('nop')
        'A :class:`datetime.datetime` instance representing the time of\n        generation for this :class:`ObjectId`.\n\n        The :class:`datetime.datetime` is timezone aware, and\n        represents the generation time in UTC. It is precise to the\n        second.\n        '
        timestamp = struct.unpack('>I', self.__id[0:4])[0]
        return datetime.datetime.fromtimestamp(timestamp, utc)

    def __getstate__(self) -> bytes:
        if False:
            i = 10
            return i + 15
        'Return value of object for pickling.\n        needed explicitly because __slots__() defined.\n        '
        return self.__id

    def __setstate__(self, value: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Explicit state set from pickling'
        if isinstance(value, dict):
            oid = value['_ObjectId__id']
        else:
            oid = value
        if isinstance(oid, str):
            self.__id = oid.encode('latin-1')
        else:
            self.__id = oid

    def __str__(self) -> str:
        if False:
            return 10
        return binascii.hexlify(self.__id).decode()

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f"ObjectId('{self!s}')"

    def __eq__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(other, ObjectId):
            return self.__id == other.binary
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        if isinstance(other, ObjectId):
            return self.__id != other.binary
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(other, ObjectId):
            return self.__id < other.binary
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, ObjectId):
            return self.__id <= other.binary
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(other, ObjectId):
            return self.__id > other.binary
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        if isinstance(other, ObjectId):
            return self.__id >= other.binary
        return NotImplemented

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Get a hash value for this :class:`ObjectId`.'
        return hash(self.__id)