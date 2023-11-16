import pyrogram
from pyrogram import raw
from ..object import Object

class StrippedThumbnail(Object):
    """A stripped thumbnail

    Parameters:
        data (``bytes``):
            Thumbnail data
    """

    def __init__(self, *, client: 'pyrogram.Client'=None, data: bytes):
        if False:
            while True:
                i = 10
        super().__init__(client)
        self.data = data

    @staticmethod
    def _parse(client, stripped_thumbnail: 'raw.types.PhotoStrippedSize') -> 'StrippedThumbnail':
        if False:
            i = 10
            return i + 15
        return StrippedThumbnail(data=stripped_thumbnail.bytes, client=client)