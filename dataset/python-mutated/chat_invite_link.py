from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from .base import TelegramObject
from .custom import DateTime
if TYPE_CHECKING:
    from .user import User

class ChatInviteLink(TelegramObject):
    """
    Represents an invite link for a chat.

    Source: https://core.telegram.org/bots/api#chatinvitelink
    """
    invite_link: str
    "The invite link. If the link was created by another chat administrator, then the second part of the link will be replaced with '…'."
    creator: User
    'Creator of the link'
    creates_join_request: bool
    ':code:`True`, if users joining the chat via the link need to be approved by chat administrators'
    is_primary: bool
    ':code:`True`, if the link is primary'
    is_revoked: bool
    ':code:`True`, if the link is revoked'
    name: Optional[str] = None
    '*Optional*. Invite link name'
    expire_date: Optional[DateTime] = None
    '*Optional*. Point in time (Unix timestamp) when the link will expire or has been expired'
    member_limit: Optional[int] = None
    '*Optional*. The maximum number of users that can be members of the chat simultaneously after joining the chat via this invite link; 1-99999'
    pending_join_request_count: Optional[int] = None
    '*Optional*. Number of pending join requests created using this link'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, invite_link: str, creator: User, creates_join_request: bool, is_primary: bool, is_revoked: bool, name: Optional[str]=None, expire_date: Optional[DateTime]=None, member_limit: Optional[int]=None, pending_join_request_count: Optional[int]=None, **__pydantic_kwargs: Any) -> None:
            if False:
                print('Hello World!')
            super().__init__(invite_link=invite_link, creator=creator, creates_join_request=creates_join_request, is_primary=is_primary, is_revoked=is_revoked, name=name, expire_date=expire_date, member_limit=member_limit, pending_join_request_count=pending_join_request_count, **__pydantic_kwargs)