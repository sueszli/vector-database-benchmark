"""This module contains an object that represents a Telegram Contact."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class Contact(TelegramObject):
    """This object represents a phone contact.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`phone_number` is equal.

    Args:
        phone_number (:obj:`str`): Contact's phone number.
        first_name (:obj:`str`): Contact's first name.
        last_name (:obj:`str`, optional): Contact's last name.
        user_id (:obj:`int`, optional): Contact's user identifier in Telegram.
        vcard (:obj:`str`, optional): Additional data about the contact in the form of a vCard.

    Attributes:
        phone_number (:obj:`str`): Contact's phone number.
        first_name (:obj:`str`): Contact's first name.
        last_name (:obj:`str`): Optional. Contact's last name.
        user_id (:obj:`int`): Optional. Contact's user identifier in Telegram.
        vcard (:obj:`str`): Optional. Additional data about the contact in the form of a vCard.

    """
    __slots__ = ('vcard', 'user_id', 'first_name', 'last_name', 'phone_number')

    def __init__(self, phone_number: str, first_name: str, last_name: Optional[str]=None, user_id: Optional[int]=None, vcard: Optional[str]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.phone_number: str = str(phone_number)
        self.first_name: str = first_name
        self.last_name: Optional[str] = last_name
        self.user_id: Optional[int] = user_id
        self.vcard: Optional[str] = vcard
        self._id_attrs = (self.phone_number,)
        self._freeze()