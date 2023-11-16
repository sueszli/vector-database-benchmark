import importlib
import struct
import json
from typing import Iterator, Tuple, Union

import aim.ext.transport.proto.remote_tracking_pb2 as rpc_messages
from aim.storage.object import CustomObject
from aim.storage.types import BLOB

Message = Union[rpc_messages.ResourceRequest, rpc_messages.ResourceResponse]


def pack_args(args: Iterator[Tuple[bytes, bytes]]) -> bytes:
    result = [struct.pack('I', len(key)) + key + struct.pack('I', len(val)) + val for key, val in args]
    return b''.join(result)


def unpack_args(args: bytes) -> Tuple[bytes, bytes]:
    while args:
        (key_size,), args_tail = struct.unpack('I', args[:4]), args[4:]
        key, args_tail = args_tail[:key_size], args_tail[key_size:]

        (value_size,), args_tail = struct.unpack('I', args_tail[:4]), args_tail[4:]
        value, args_tail = args_tail[:value_size], args_tail[value_size:]
        args = args_tail
        yield key, value


def pack_stream(tree: Iterator[Tuple[bytes, bytes]]) -> bytes:
    # TODO: [MV] check the performance diff of current version vs collecting the whole tree as a chunk
    for key, val in tree:
        if not isinstance(val, BLOB):
            yield struct.pack('I', len(key)) + key + struct.pack('?', False) + struct.pack('I', len(val)) + val
        else:
            val = val.load()
            yield struct.pack('I', len(key)) + key + struct.pack('?', True) + struct.pack('I', len(val)) + val


def unpack_helper(msg: bytes) -> Tuple[bytes, bytes]:
    (key_size,), tail = struct.unpack('I', msg[:4]), msg[4:]
    key, tail = tail[:key_size], tail[key_size:]
    (is_blob,), tail = struct.unpack('?', tail[:1]), tail[1:]
    (value_size,), tail = struct.unpack('I', tail[:4]), tail[4:]
    value, tail = tail[:value_size], tail[value_size:]
    assert len(tail) == 0
    if is_blob:
        yield key, BLOB(data=value)
    else:
        yield key, value


def unpack_bytes(stream: Iterator[bytes]) -> Tuple[bytes, bytes]:
    for msg in stream:
        yield from unpack_helper(msg)


def unpack_stream(stream: Iterator[Message]) -> Tuple[bytes, bytes]:
    for msg in stream:
        if msg.WhichOneof('instruction') == 'header':
            # can be header in case of exceptions on server side
            assert msg.header.status == rpc_messages.ResponseHeader.Status.ERROR
            raise_exception(msg.header.exception)
            return

        assert msg.WhichOneof('instruction') == 'message'
        msg = msg.message

        yield from unpack_helper(msg)


def raise_exception(grpc_exception):
    assert grpc_exception is not None
    module = importlib.import_module(grpc_exception.module_name)
    exception = getattr(module, grpc_exception.class_name)
    args = json.loads(grpc_exception.args or [])
    raise exception(*args) if args else exception()


def build_exception(exception: Exception):
    return rpc_messages.ExceptionResponse(
        module_name=exception.__class__.__module__,
        class_name=exception.__class__.__name__,
        args=json.dumps(exception.args),
    )


class UnauthorizedRequestError(RuntimeError):
    def __init__(self, handler, *args, **kwargs):
        self.handler = handler


@CustomObject.alias('aim.resource')
class ResourceObject(CustomObject):
    AIM_NAME = 'aim.resource'

    def __init__(self, handler):
        self.storage['handler'] = handler
