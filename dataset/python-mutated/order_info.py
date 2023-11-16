from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from .base import TelegramObject
if TYPE_CHECKING:
    from .shipping_address import ShippingAddress

class OrderInfo(TelegramObject):
    """
    This object represents information about an order.

    Source: https://core.telegram.org/bots/api#orderinfo
    """
    name: Optional[str] = None
    '*Optional*. User name'
    phone_number: Optional[str] = None
    "*Optional*. User's phone number"
    email: Optional[str] = None
    '*Optional*. User email'
    shipping_address: Optional[ShippingAddress] = None
    '*Optional*. User shipping address'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, name: Optional[str]=None, phone_number: Optional[str]=None, email: Optional[str]=None, shipping_address: Optional[ShippingAddress]=None, **__pydantic_kwargs: Any) -> None:
            if False:
                while True:
                    i = 10
            super().__init__(name=name, phone_number=phone_number, email=email, shipping_address=shipping_address, **__pydantic_kwargs)