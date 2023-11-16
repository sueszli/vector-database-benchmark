from pyrogram.raw.core import Message, MsgContainer, TLObject
from pyrogram.raw.functions import Ping
from pyrogram.raw.types import MsgsAck, HttpWait
from .msg_id import MsgId
from .seq_no import SeqNo
not_content_related = (Ping, HttpWait, MsgsAck, MsgContainer)

class MsgFactory:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.seq_no = SeqNo()

    def __call__(self, body: TLObject) -> Message:
        if False:
            while True:
                i = 10
        return Message(body, MsgId(), self.seq_no(not isinstance(body, not_content_related)), len(body))