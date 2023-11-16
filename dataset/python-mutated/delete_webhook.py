from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from .base import TelegramMethod

class DeleteWebhook(TelegramMethod[bool]):
    """
    Use this method to remove webhook integration if you decide to switch back to :class:`aiogram.methods.get_updates.GetUpdates`. Returns :code:`True` on success.

    Source: https://core.telegram.org/bots/api#deletewebhook
    """
    __returning__ = bool
    __api_method__ = 'deleteWebhook'
    drop_pending_updates: Optional[bool] = None
    'Pass :code:`True` to drop all pending updates'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, drop_pending_updates: Optional[bool]=None, **__pydantic_kwargs: Any) -> None:
            if False:
                while True:
                    i = 10
            super().__init__(drop_pending_updates=drop_pending_updates, **__pydantic_kwargs)