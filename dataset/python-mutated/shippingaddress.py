"""This module contains an object that represents a Telegram ShippingAddress."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class ShippingAddress(TelegramObject):
    """This object represents a Telegram ShippingAddress.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their  :attr:`country_code`, :attr:`state`, :attr:`city`,
    :attr:`street_line1`, :attr:`street_line2` and :attr:`post_code` are equal.

    Args:
        country_code (:obj:`str`): ISO 3166-1 alpha-2 country code.
        state (:obj:`str`): State, if applicable.
        city (:obj:`str`): City.
        street_line1 (:obj:`str`): First line for the address.
        street_line2 (:obj:`str`): Second line for the address.
        post_code (:obj:`str`): Address post code.

    Attributes:
        country_code (:obj:`str`): ISO 3166-1 alpha-2 country code.
        state (:obj:`str`): State, if applicable.
        city (:obj:`str`): City.
        street_line1 (:obj:`str`): First line for the address.
        street_line2 (:obj:`str`): Second line for the address.
        post_code (:obj:`str`): Address post code.

    """
    __slots__ = ('post_code', 'city', 'country_code', 'street_line2', 'street_line1', 'state')

    def __init__(self, country_code: str, state: str, city: str, street_line1: str, street_line2: str, post_code: str, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self.country_code: str = country_code
        self.state: str = state
        self.city: str = city
        self.street_line1: str = street_line1
        self.street_line2: str = street_line2
        self.post_code: str = post_code
        self._id_attrs = (self.country_code, self.state, self.city, self.street_line1, self.street_line2, self.post_code)
        self._freeze()