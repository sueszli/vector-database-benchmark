from datetime import datetime
from pyrogram import raw, utils
from pyrogram import types
from ..object import Object

class InviteLinkImporter(Object):
    """The date and user of when someone has joined with an invite link.

    Parameters:
        date (:py:obj:`~datetime.datetime`):
            The time of when this user used the given link

        user (:obj:`~pyrogram.types.User`):
            The user that has used the given invite link
    """

    def __init__(self, *, date: datetime, user: 'types.User'):
        if False:
            return 10
        super().__init__(None)
        self.date = date
        self.user = user

    @staticmethod
    def _parse(client, invite_importers: 'raw.types.messages.ChatInviteImporters'):
        if False:
            while True:
                i = 10
        importers = types.List()
        d = {i.id: i for i in invite_importers.users}
        for j in invite_importers.importers:
            importers.append(InviteLinkImporter(date=utils.timestamp_to_datetime(j.date), user=types.User._parse(client=None, user=d[j.user_id])))
        return importers