from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .base import TelegramObject

class ShippingAddress(TelegramObject):
    """
    This object represents a shipping address.

    Source: https://core.telegram.org/bots/api#shippingaddress
    """
    country_code: str
    'Two-letter ISO 3166-1 alpha-2 country code'
    state: str
    'State, if applicable'
    city: str
    'City'
    street_line1: str
    'First line for the address'
    street_line2: str
    'Second line for the address'
    post_code: str
    'Address post code'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, country_code: str, state: str, city: str, street_line1: str, street_line2: str, post_code: str, **__pydantic_kwargs: Any) -> None:
            if False:
                while True:
                    i = 10
            super().__init__(country_code=country_code, state=state, city=city, street_line1=street_line1, street_line2=street_line2, post_code=post_code, **__pydantic_kwargs)