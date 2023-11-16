from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union
from .base import TelegramMethod

class UnbanChatSenderChat(TelegramMethod[bool]):
    """
    Use this method to unban a previously banned channel chat in a supergroup or channel. The bot must be an administrator for this to work and must have the appropriate administrator rights. Returns :code:`True` on success.

    Source: https://core.telegram.org/bots/api#unbanchatsenderchat
    """
    __returning__ = bool
    __api_method__ = 'unbanChatSenderChat'
    chat_id: Union[int, str]
    'Unique identifier for the target chat or username of the target channel (in the format :code:`@channelusername`)'
    sender_chat_id: int
    'Unique identifier of the target sender chat'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, chat_id: Union[int, str], sender_chat_id: int, **__pydantic_kwargs: Any) -> None:
            if False:
                print('Hello World!')
            super().__init__(chat_id=chat_id, sender_chat_id=sender_chat_id, **__pydantic_kwargs)