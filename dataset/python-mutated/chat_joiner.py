from datetime import datetime
from typing import Dict
import pyrogram
from pyrogram import raw, types, utils
from ..object import Object

class ChatJoiner(Object):
    """Contains information about a joiner member of a chat.

    Parameters:
        user (:obj:`~pyrogram.types.User`):
            Information about the user.

        date (:py:obj:`~datetime.datetime`):
            Date when the user joined.

        bio (``str``, *optional*):
            Bio of the user.

        pending (``bool``, *optional*):
            True in case the chat joiner has a pending request.

        approved_by (:obj:`~pyrogram.types.User`, *optional*):
            Administrator who approved this chat joiner.
    """

    def __init__(self, *, client: 'pyrogram.Client', user: 'types.User', date: datetime=None, bio: str=None, pending: bool=None, approved_by: 'types.User'=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(client)
        self.user = user
        self.date = date
        self.bio = bio
        self.pending = pending
        self.approved_by = approved_by

    @staticmethod
    def _parse(client: 'pyrogram.Client', joiner: 'raw.base.ChatInviteImporter', users: Dict[int, 'raw.base.User']) -> 'ChatJoiner':
        if False:
            while True:
                i = 10
        return ChatJoiner(user=types.User._parse(client, users[joiner.user_id]), date=utils.timestamp_to_datetime(joiner.date), pending=joiner.requested, bio=joiner.about, approved_by=types.User._parse(client, users[joiner.approved_by]) if joiner.approved_by else None, client=client)