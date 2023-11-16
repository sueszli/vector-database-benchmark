from typing import List, Dict
from pyrogram import raw, types
from ..object import Object

class VideoChatMembersInvited(Object):
    """A service message about new members invited to a voice chat.


    Parameters:
        users (List of :obj:`~pyrogram.types.User`):
            New members that were invited to the voice chat.
    """

    def __init__(self, *, users: List['types.User']):
        if False:
            while True:
                i = 10
        super().__init__()
        self.users = users

    @staticmethod
    def _parse(client, action: 'raw.types.MessageActionInviteToGroupCall', users: Dict[int, 'raw.types.User']) -> 'VideoChatMembersInvited':
        if False:
            while True:
                i = 10
        users = [types.User._parse(client, users[i]) for i in action.users]
        return VideoChatMembersInvited(users=users)