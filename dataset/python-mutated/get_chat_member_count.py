from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union
from .base import TelegramMethod

class GetChatMemberCount(TelegramMethod[int]):
    """
    Use this method to get the number of members in a chat. Returns *Int* on success.

    Source: https://core.telegram.org/bots/api#getchatmembercount
    """
    __returning__ = int
    __api_method__ = 'getChatMemberCount'
    chat_id: Union[int, str]
    'Unique identifier for the target chat or username of the target supergroup or channel (in the format :code:`@channelusername`)'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, chat_id: Union[int, str], **__pydantic_kwargs: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(chat_id=chat_id, **__pydantic_kwargs)