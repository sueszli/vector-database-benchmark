from __future__ import annotations
from typing import TYPE_CHECKING, Any, List
from .base import TelegramObject
if TYPE_CHECKING:
    from .labeled_price import LabeledPrice

class ShippingOption(TelegramObject):
    """
    This object represents one shipping option.

    Source: https://core.telegram.org/bots/api#shippingoption
    """
    id: str
    'Shipping option identifier'
    title: str
    'Option title'
    prices: List[LabeledPrice]
    'List of price portions'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, id: str, title: str, prices: List[LabeledPrice], **__pydantic_kwargs: Any) -> None:
            if False:
                i = 10
                return i + 15
            super().__init__(id=id, title=title, prices=prices, **__pydantic_kwargs)