from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union
from .base import TelegramMethod

class SetChatTitle(TelegramMethod[bool]):
    """
    Use this method to change the title of a chat. Titles can't be changed for private chats. The bot must be an administrator in the chat for this to work and must have the appropriate administrator rights. Returns :code:`True` on success.

    Source: https://core.telegram.org/bots/api#setchattitle
    """
    __returning__ = bool
    __api_method__ = 'setChatTitle'
    chat_id: Union[int, str]
    'Unique identifier for the target chat or username of the target channel (in the format :code:`@channelusername`)'
    title: str
    'New chat title, 1-128 characters'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, chat_id: Union[int, str], title: str, **__pydantic_kwargs: Any) -> None:
            if False:
                i = 10
                return i + 15
            super().__init__(chat_id=chat_id, title=title, **__pydantic_kwargs)