from io import BytesIO
from typing import Any
from ..tl_object import TLObject

class Int(bytes, TLObject):
    SIZE = 4

    @classmethod
    def read(cls, data: BytesIO, signed: bool=True, *args: Any) -> int:
        if False:
            while True:
                i = 10
        return int.from_bytes(data.read(cls.SIZE), 'little', signed=signed)

    def __new__(cls, value: int, signed: bool=True) -> bytes:
        if False:
            i = 10
            return i + 15
        return value.to_bytes(cls.SIZE, 'little', signed=signed)

class Long(Int):
    SIZE = 8

class Int128(Int):
    SIZE = 16

class Int256(Int):
    SIZE = 32