from __future__ import annotations
from collections import deque
from datetime import date, datetime, time
from decimal import Decimal
from functools import partial
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path, PurePath
from re import Pattern
from typing import TYPE_CHECKING, Any, Callable, Mapping, TypeVar, overload
from uuid import UUID
import msgspec
from litestar.exceptions import SerializationException
from litestar.types import Empty, EmptyType, Serializer, TypeDecodersSequence
if TYPE_CHECKING:
    from litestar.types import TypeEncodersMap
__all__ = ('decode_json', 'decode_msgpack', 'default_deserializer', 'default_serializer', 'encode_json', 'encode_msgpack', 'get_serializer')
T = TypeVar('T')
DEFAULT_TYPE_ENCODERS: TypeEncodersMap = {Path: str, PurePath: str, IPv4Address: str, IPv4Interface: str, IPv4Network: str, IPv6Address: str, IPv6Interface: str, IPv6Network: str, datetime: lambda val: val.isoformat(), date: lambda val: val.isoformat(), time: lambda val: val.isoformat(), deque: list, Decimal: lambda val: int(val) if val.as_tuple().exponent >= 0 else float(val), Pattern: lambda val: val.pattern, str: str, int: int, float: float, set: set, frozenset: frozenset, bytes: bytes}

def default_serializer(value: Any, type_encoders: Mapping[Any, Callable[[Any], Any]] | None=None) -> Any:
    if False:
        while True:
            i = 10
    'Transform values non-natively supported by ``msgspec``\n\n    Args:\n        value: A value to serialized\n        type_encoders: Mapping of types to callables to transforming types\n    Returns:\n        A serialized value\n    Raises:\n        TypeError: if value is not supported\n    '
    type_encoders = {**DEFAULT_TYPE_ENCODERS, **(type_encoders or {})}
    for base in value.__class__.__mro__[:-1]:
        try:
            encoder = type_encoders[base]
            return encoder(value)
        except KeyError:
            continue
    raise TypeError(f'Unsupported type: {type(value)!r}')

def default_deserializer(target_type: Any, value: Any, type_decoders: TypeDecodersSequence | None=None) -> Any:
    if False:
        i = 10
        return i + 15
    'Transform values non-natively supported by ``msgspec``\n\n    Args:\n        target_type: Encountered type\n        value: Value to coerce\n        type_decoders: Optional sequence of type decoders\n\n    Returns:\n        A ``msgspec``-supported type\n    '
    from litestar.datastructures.state import ImmutableState
    if isinstance(value, target_type):
        return value
    if type_decoders:
        for (predicate, decoder) in type_decoders:
            if predicate(target_type):
                return decoder(target_type, value)
    if issubclass(target_type, (Path, PurePath, ImmutableState, UUID)):
        return target_type(value)
    raise TypeError(f'Unsupported type: {type(value)!r}')
_msgspec_json_encoder = msgspec.json.Encoder(enc_hook=default_serializer)
_msgspec_json_decoder = msgspec.json.Decoder(dec_hook=default_deserializer)
_msgspec_msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=default_serializer)
_msgspec_msgpack_decoder = msgspec.msgpack.Decoder(dec_hook=default_deserializer)

def encode_json(value: Any, serializer: Callable[[Any], Any] | None=None) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    'Encode a value into JSON.\n\n    Args:\n        value: Value to encode\n        serializer: Optional callable to support non-natively supported types.\n\n    Returns:\n        JSON as bytes\n\n    Raises:\n        SerializationException: If error encoding ``obj``.\n    '
    try:
        return msgspec.json.encode(value, enc_hook=serializer) if serializer else _msgspec_json_encoder.encode(value)
    except (TypeError, msgspec.EncodeError) as msgspec_error:
        raise SerializationException(str(msgspec_error)) from msgspec_error

@overload
def decode_json(value: str | bytes) -> Any:
    if False:
        print('Hello World!')
    ...

@overload
def decode_json(value: str | bytes, type_decoders: TypeDecodersSequence | None) -> Any:
    if False:
        i = 10
        return i + 15
    ...

@overload
def decode_json(value: str | bytes, target_type: type[T]) -> T:
    if False:
        return 10
    ...

@overload
def decode_json(value: str | bytes, target_type: type[T], type_decoders: TypeDecodersSequence | None) -> T:
    if False:
        for i in range(10):
            print('nop')
    ...

def decode_json(value: str | bytes, target_type: type[T] | EmptyType=Empty, type_decoders: TypeDecodersSequence | None=None) -> Any:
    if False:
        while True:
            i = 10
    'Decode a JSON string/bytes into an object.\n\n    Args:\n        value: Value to decode\n        target_type: An optional type to decode the data into\n        type_decoders: Optional sequence of type decoders\n\n    Returns:\n        An object\n\n    Raises:\n        SerializationException: If error decoding ``value``.\n    '
    try:
        if target_type is Empty:
            return _msgspec_json_decoder.decode(value)
        return msgspec.json.decode(value, dec_hook=partial(default_deserializer, type_decoders=type_decoders), type=target_type)
    except msgspec.DecodeError as msgspec_error:
        raise SerializationException(str(msgspec_error)) from msgspec_error

def encode_msgpack(value: Any, serializer: Callable[[Any], Any] | None=default_serializer) -> bytes:
    if False:
        while True:
            i = 10
    'Encode a value into MessagePack.\n\n    Args:\n        value: Value to encode\n        serializer: Optional callable to support non-natively supported types\n\n    Returns:\n        MessagePack as bytes\n\n    Raises:\n        SerializationException: If error encoding ``obj``.\n    '
    try:
        if serializer is None or serializer is default_serializer:
            return _msgspec_msgpack_encoder.encode(value)
        return msgspec.msgpack.encode(value, enc_hook=serializer)
    except (TypeError, msgspec.EncodeError) as msgspec_error:
        raise SerializationException(str(msgspec_error)) from msgspec_error

@overload
def decode_msgpack(value: bytes) -> Any:
    if False:
        print('Hello World!')
    ...

@overload
def decode_msgpack(value: bytes, type_decoders: TypeDecodersSequence | None) -> Any:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def decode_msgpack(value: bytes, target_type: type[T]) -> T:
    if False:
        return 10
    ...

@overload
def decode_msgpack(value: bytes, target_type: type[T], type_decoders: TypeDecodersSequence | None) -> T:
    if False:
        print('Hello World!')
    ...

def decode_msgpack(value: bytes, target_type: type[T] | EmptyType=Empty, type_decoders: TypeDecodersSequence | None=None) -> Any:
    if False:
        for i in range(10):
            print('nop')
    'Decode a MessagePack string/bytes into an object.\n\n    Args:\n        value: Value to decode\n        target_type: An optional type to decode the data into\n        type_decoders: Optional sequence of type decoders\n\n    Returns:\n        An object\n\n    Raises:\n        SerializationException: If error decoding ``value``.\n    '
    try:
        if target_type is Empty:
            return _msgspec_msgpack_decoder.decode(value)
        return msgspec.msgpack.decode(value, dec_hook=partial(default_deserializer, type_decoders=type_decoders), type=target_type)
    except msgspec.DecodeError as msgspec_error:
        raise SerializationException(str(msgspec_error)) from msgspec_error

def get_serializer(type_encoders: TypeEncodersMap | None=None) -> Serializer:
    if False:
        for i in range(10):
            print('nop')
    'Get the serializer for the given type encoders.'
    if type_encoders:
        return partial(default_serializer, type_encoders=type_encoders)
    return default_serializer