from pyrogram import raw
from pyrogram.session.internals import MsgId
from ..object import Object

class InputPhoneContact(Object):
    """A Phone Contact to be added in your Telegram address book.
    It is intended to be used with :meth:`~pyrogram.Client.add_contacts()`

    Parameters:
        phone (``str``):
            Contact's phone number

        first_name (``str``):
            Contact's first name

        last_name (``str``, *optional*):
            Contact's last name
    """

    def __init__(self, phone: str, first_name: str, last_name: str=''):
        if False:
            while True:
                i = 10
        super().__init__(None)

    def __new__(cls, phone: str, first_name: str, last_name: str=''):
        if False:
            while True:
                i = 10
        return raw.types.InputPhoneContact(client_id=MsgId(), phone='+' + phone.strip('+'), first_name=first_name, last_name=last_name)