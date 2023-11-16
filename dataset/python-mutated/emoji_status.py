from datetime import datetime
from typing import Optional
import pyrogram
from pyrogram import raw
from pyrogram import utils
from ..object import Object

class EmojiStatus(Object):
    """A user emoji status.

    Parameters:
        custom_emoji_id (``int``):
            Custom emoji id.

        until_date (:py:obj:`~datetime.datetime`, *optional*):
            Valid until date.
    """

    def __init__(self, *, client: 'pyrogram.Client'=None, custom_emoji_id: int, until_date: Optional[datetime]=None):
        if False:
            print('Hello World!')
        super().__init__(client)
        self.custom_emoji_id = custom_emoji_id
        self.until_date = until_date

    @staticmethod
    def _parse(client, emoji_status: 'raw.base.EmojiStatus') -> Optional['EmojiStatus']:
        if False:
            while True:
                i = 10
        if isinstance(emoji_status, raw.types.EmojiStatus):
            return EmojiStatus(client=client, custom_emoji_id=emoji_status.document_id)
        if isinstance(emoji_status, raw.types.EmojiStatusUntil):
            return EmojiStatus(client=client, custom_emoji_id=emoji_status.document_id, until_date=utils.timestamp_to_datetime(emoji_status.until))
        return None

    def write(self):
        if False:
            i = 10
            return i + 15
        if self.until_date:
            return raw.types.EmojiStatusUntil(document_id=self.custom_emoji_id, until=utils.datetime_to_timestamp(self.until_date))
        return raw.types.EmojiStatus(document_id=self.custom_emoji_id)