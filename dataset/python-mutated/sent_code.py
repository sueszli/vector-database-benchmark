from pyrogram import raw, enums
from ..object import Object

class SentCode(Object):
    """Contains info on a sent confirmation code.

    Parameters:
        type (:obj:`~pyrogram.enums.SentCodeType`):
            Type of the current sent code.

        phone_code_hash (``str``):
            Confirmation code identifier useful for the next authorization steps (either
            :meth:`~pyrogram.Client.sign_in` or :meth:`~pyrogram.Client.sign_up`).

        next_type (:obj:`~pyrogram.enums.NextCodeType`, *optional*):
            Type of the next code to be sent with :meth:`~pyrogram.Client.resend_code`.

        timeout (``int``, *optional*):
            Delay in seconds before calling :meth:`~pyrogram.Client.resend_code`.
    """

    def __init__(self, *, type: 'enums.SentCodeType', phone_code_hash: str, next_type: 'enums.NextCodeType'=None, timeout: int=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.type = type
        self.phone_code_hash = phone_code_hash
        self.next_type = next_type
        self.timeout = timeout

    @staticmethod
    def _parse(sent_code: raw.types.auth.SentCode) -> 'SentCode':
        if False:
            while True:
                i = 10
        return SentCode(type=enums.SentCodeType(type(sent_code.type)), phone_code_hash=sent_code.phone_code_hash, next_type=enums.NextCodeType(type(sent_code.next_type)) if sent_code.next_type else None, timeout=sent_code.timeout)