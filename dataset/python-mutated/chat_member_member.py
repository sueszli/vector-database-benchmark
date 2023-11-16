from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..enums import ChatMemberStatus
from .chat_member import ChatMember
if TYPE_CHECKING:
    from .user import User

class ChatMemberMember(ChatMember):
    """
    Represents a `chat member <https://core.telegram.org/bots/api#chatmember>`_ that has no additional privileges or restrictions.

    Source: https://core.telegram.org/bots/api#chatmembermember
    """
    status: Literal[ChatMemberStatus.MEMBER] = ChatMemberStatus.MEMBER
    "The member's status in the chat, always 'member'"
    user: User
    'Information about the user'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, status: Literal[ChatMemberStatus.MEMBER]=ChatMemberStatus.MEMBER, user: User, **__pydantic_kwargs: Any) -> None:
            if False:
                print('Hello World!')
            super().__init__(status=status, user=user, **__pydantic_kwargs)