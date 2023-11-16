from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union
from .base import TelegramMethod

class HideGeneralForumTopic(TelegramMethod[bool]):
    """
    Use this method to hide the 'General' topic in a forum supergroup chat. The bot must be an administrator in the chat for this to work and must have the *can_manage_topics* administrator rights. The topic will be automatically closed if it was open. Returns :code:`True` on success.

    Source: https://core.telegram.org/bots/api#hidegeneralforumtopic
    """
    __returning__ = bool
    __api_method__ = 'hideGeneralForumTopic'
    chat_id: Union[int, str]
    'Unique identifier for the target chat or username of the target supergroup (in the format :code:`@supergroupusername`)'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, chat_id: Union[int, str], **__pydantic_kwargs: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(chat_id=chat_id, **__pydantic_kwargs)