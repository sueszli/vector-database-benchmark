from io import BytesIO
from typing import List, Any
from .message import Message
from .primitives.int import Int
from .tl_object import TLObject

class MsgContainer(TLObject):
    ID = 1945237724
    __slots__ = ['messages']
    QUALNAME = 'MsgContainer'

    def __init__(self, messages: List[Message]):
        if False:
            print('Hello World!')
        self.messages = messages

    @staticmethod
    def read(data: BytesIO, *args: Any) -> 'MsgContainer':
        if False:
            while True:
                i = 10
        count = Int.read(data)
        return MsgContainer([Message.read(data) for _ in range(count)])

    def write(self, *args: Any) -> bytes:
        if False:
            print('Hello World!')
        b = BytesIO()
        b.write(Int(self.ID, False))
        count = len(self.messages)
        b.write(Int(count))
        for message in self.messages:
            b.write(message.write())
        return b.getvalue()