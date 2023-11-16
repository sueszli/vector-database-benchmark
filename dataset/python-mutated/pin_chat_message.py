from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Union
from .base import TelegramMethod

class PinChatMessage(TelegramMethod[bool]):
    """
    Use this method to add a message to the list of pinned messages in a chat. If the chat is not a private chat, the bot must be an administrator in the chat for this to work and must have the 'can_pin_messages' administrator right in a supergroup or 'can_edit_messages' administrator right in a channel. Returns :code:`True` on success.

    Source: https://core.telegram.org/bots/api#pinchatmessage
    """
    __returning__ = bool
    __api_method__ = 'pinChatMessage'
    chat_id: Union[int, str]
    'Unique identifier for the target chat or username of the target channel (in the format :code:`@channelusername`)'
    message_id: int
    'Identifier of a message to pin'
    disable_notification: Optional[bool] = None
    'Pass :code:`True` if it is not necessary to send a notification to all chat members about the new pinned message. Notifications are always disabled in channels and private chats.'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, chat_id: Union[int, str], message_id: int, disable_notification: Optional[bool]=None, **__pydantic_kwargs: Any) -> None:
            if False:
                while True:
                    i = 10
            super().__init__(chat_id=chat_id, message_id=message_id, disable_notification=disable_notification, **__pydantic_kwargs)