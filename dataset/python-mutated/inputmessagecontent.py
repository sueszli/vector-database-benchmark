"""This module contains the classes that represent Telegram InputMessageContent."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class InputMessageContent(TelegramObject):
    """Base class for Telegram InputMessageContent Objects.

    See: :class:`telegram.InputContactMessageContent`,
    :class:`telegram.InputInvoiceMessageContent`,
    :class:`telegram.InputLocationMessageContent`, :class:`telegram.InputTextMessageContent` and
    :class:`telegram.InputVenueMessageContent` for more details.

    """
    __slots__ = ()

    def __init__(self, *, api_kwargs: Optional[JSONDict]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self._freeze()