from io import BytesIO
from typing import Any
from .primitives.int import Int, Long
from .tl_object import TLObject

class FutureSalt(TLObject):
    ID = 155834844
    __slots__ = ['valid_since', 'valid_until', 'salt']
    QUALNAME = 'FutureSalt'

    def __init__(self, valid_since: int, valid_until: int, salt: int):
        if False:
            for i in range(10):
                print('nop')
        self.valid_since = valid_since
        self.valid_until = valid_until
        self.salt = salt

    @staticmethod
    def read(data: BytesIO, *args: Any) -> 'FutureSalt':
        if False:
            while True:
                i = 10
        valid_since = Int.read(data)
        valid_until = Int.read(data)
        salt = Long.read(data)
        return FutureSalt(valid_since, valid_until, salt)

    def write(self, *args: Any) -> bytes:
        if False:
            print('Hello World!')
        b = BytesIO()
        b.write(Int(self.valid_since))
        b.write(Int(self.valid_until))
        b.write(Long(self.salt))
        return b.getvalue()