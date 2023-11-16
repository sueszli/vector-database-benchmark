"""This module contains an object that represents an invite link for a chat."""
import datetime
from typing import TYPE_CHECKING, Optional
from telegram._telegramobject import TelegramObject
from telegram._user import User
from telegram._utils.datetime import extract_tzinfo_from_defaults, from_timestamp
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot

class ChatInviteLink(TelegramObject):
    """This object represents an invite link for a chat.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`invite_link`, :attr:`creator`, :attr:`creates_join_request`,
    :attr:`is_primary` and :attr:`is_revoked` are equal.

    .. versionadded:: 13.4
    .. versionchanged:: 20.0

       * The argument & attribute :attr:`creates_join_request` is now required to comply with the
         Bot API.
       * Comparing objects of this class now also takes :attr:`creates_join_request` into account.

    Args:
        invite_link (:obj:`str`): The invite link.
        creator (:class:`telegram.User`): Creator of the link.
        creates_join_request (:obj:`bool`): :obj:`True`, if users joining the chat via
            the link need to be approved by chat administrators.

            .. versionadded:: 13.8
        is_primary (:obj:`bool`): :obj:`True`, if the link is primary.
        is_revoked (:obj:`bool`): :obj:`True`, if the link is revoked.
        expire_date (:class:`datetime.datetime`, optional): Date when the link will expire or
            has been expired.

            .. versionchanged:: 20.3
                |datetime_localization|
        member_limit (:obj:`int`, optional): Maximum number of users that can be members of the
            chat simultaneously after joining the chat via this invite link;
            :tg-const:`telegram.constants.ChatInviteLinkLimit.MIN_MEMBER_LIMIT`-
            :tg-const:`telegram.constants.ChatInviteLinkLimit.MAX_MEMBER_LIMIT`.
        name (:obj:`str`, optional): Invite link name.
            0-:tg-const:`telegram.constants.ChatInviteLinkLimit.NAME_LENGTH` characters.

            .. versionadded:: 13.8
        pending_join_request_count (:obj:`int`, optional): Number of pending join requests
            created using this link.

            .. versionadded:: 13.8
    Attributes:
        invite_link (:obj:`str`): The invite link. If the link was created by another chat
            administrator, then the second part of the link will be replaced with ``'…'``.
        creator (:class:`telegram.User`): Creator of the link.
        creates_join_request (:obj:`bool`): :obj:`True`, if users joining the chat via
            the link need to be approved by chat administrators.

            .. versionadded:: 13.8
        is_primary (:obj:`bool`): :obj:`True`, if the link is primary.
        is_revoked (:obj:`bool`): :obj:`True`, if the link is revoked.
        expire_date (:class:`datetime.datetime`): Optional. Date when the link will expire or
            has been expired.

            .. versionchanged:: 20.3
                |datetime_localization|
        member_limit (:obj:`int`): Optional. Maximum number of users that can be members
            of the chat simultaneously after joining the chat via this invite link;
            :tg-const:`telegram.constants.ChatInviteLinkLimit.MIN_MEMBER_LIMIT`-
            :tg-const:`telegram.constants.ChatInviteLinkLimit.MAX_MEMBER_LIMIT`.
        name (:obj:`str`): Optional. Invite link name.
            0-:tg-const:`telegram.constants.ChatInviteLinkLimit.NAME_LENGTH` characters.

            .. versionadded:: 13.8
        pending_join_request_count (:obj:`int`): Optional. Number of pending join requests
            created using this link.

            .. versionadded:: 13.8

    """
    __slots__ = ('invite_link', 'creator', 'is_primary', 'is_revoked', 'expire_date', 'member_limit', 'name', 'creates_join_request', 'pending_join_request_count')

    def __init__(self, invite_link: str, creator: User, creates_join_request: bool, is_primary: bool, is_revoked: bool, expire_date: Optional[datetime.datetime]=None, member_limit: Optional[int]=None, name: Optional[str]=None, pending_join_request_count: Optional[int]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self.invite_link: str = invite_link
        self.creator: User = creator
        self.creates_join_request: bool = creates_join_request
        self.is_primary: bool = is_primary
        self.is_revoked: bool = is_revoked
        self.expire_date: Optional[datetime.datetime] = expire_date
        self.member_limit: Optional[int] = member_limit
        self.name: Optional[str] = name
        self.pending_join_request_count: Optional[int] = int(pending_join_request_count) if pending_join_request_count is not None else None
        self._id_attrs = (self.invite_link, self.creates_join_request, self.creator, self.is_primary, self.is_revoked)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['ChatInviteLink']:
        if False:
            print('Hello World!')
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        loc_tzinfo = extract_tzinfo_from_defaults(bot)
        data['creator'] = User.de_json(data.get('creator'), bot)
        data['expire_date'] = from_timestamp(data.get('expire_date', None), tzinfo=loc_tzinfo)
        return super().de_json(data=data, bot=bot)