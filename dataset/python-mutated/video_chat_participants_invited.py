from __future__ import annotations
from typing import TYPE_CHECKING, Any, List
from .base import TelegramObject
if TYPE_CHECKING:
    from .user import User

class VideoChatParticipantsInvited(TelegramObject):
    """
    This object represents a service message about new members invited to a video chat.

    Source: https://core.telegram.org/bots/api#videochatparticipantsinvited
    """
    users: List[User]
    'New members that were invited to the video chat'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, users: List[User], **__pydantic_kwargs: Any) -> None:
            if False:
                return 10
            super().__init__(users=users, **__pydantic_kwargs)