from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union
from .base import TelegramMethod

class DeleteChatStickerSet(TelegramMethod[bool]):
    """
    Use this method to delete a group sticker set from a supergroup. The bot must be an administrator in the chat for this to work and must have the appropriate administrator rights. Use the field *can_set_sticker_set* optionally returned in :class:`aiogram.methods.get_chat.GetChat` requests to check if the bot can use this method. Returns :code:`True` on success.

    Source: https://core.telegram.org/bots/api#deletechatstickerset
    """
    __returning__ = bool
    __api_method__ = 'deleteChatStickerSet'
    chat_id: Union[int, str]
    'Unique identifier for the target chat or username of the target supergroup (in the format :code:`@supergroupusername`)'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, chat_id: Union[int, str], **__pydantic_kwargs: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(chat_id=chat_id, **__pydantic_kwargs)