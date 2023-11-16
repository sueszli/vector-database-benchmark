"""This module contains an object that represents a Telegram UserProfilePhotos."""
from typing import TYPE_CHECKING, Optional, Sequence, Tuple
from telegram._files.photosize import PhotoSize
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot

class UserProfilePhotos(TelegramObject):
    """This object represents a user's profile pictures.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`total_count` and :attr:`photos` are equal.

    Args:
        total_count (:obj:`int`): Total number of profile pictures the target user has.
        photos (Sequence[Sequence[:class:`telegram.PhotoSize`]]): Requested profile pictures (in up
            to 4 sizes each).

            .. versionchanged:: 20.0
                |sequenceclassargs|

    Attributes:
        total_count (:obj:`int`): Total number of profile pictures.
        photos (Tuple[Tuple[:class:`telegram.PhotoSize`]]): Requested profile pictures (in up to 4
            sizes each).

            .. versionchanged:: 20.0
                |tupleclassattrs|

    """
    __slots__ = ('photos', 'total_count')

    def __init__(self, total_count: int, photos: Sequence[Sequence[PhotoSize]], *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.total_count: int = total_count
        self.photos: Tuple[Tuple[PhotoSize, ...], ...] = tuple((tuple(sizes) for sizes in photos))
        self._id_attrs = (self.total_count, self.photos)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['UserProfilePhotos']:
        if False:
            while True:
                i = 10
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        data['photos'] = [PhotoSize.de_list(photo, bot) for photo in data['photos']]
        return super().de_json(data=data, bot=bot)