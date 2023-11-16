"""This module contains an object that represents a location to which a chat is connected."""
from typing import TYPE_CHECKING, Final, Optional
from telegram import constants
from telegram._files.location import Location
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot

class ChatLocation(TelegramObject):
    """This object represents a location to which a chat is connected.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`location` is equal.

    Args:
        location (:class:`telegram.Location`): The location to which the supergroup is connected.
            Can't be a live location.
        address (:obj:`str`): Location address;
            :tg-const:`telegram.ChatLocation.MIN_ADDRESS`-
            :tg-const:`telegram.ChatLocation.MAX_ADDRESS` characters, as defined by the chat owner.
    Attributes:
        location (:class:`telegram.Location`): The location to which the supergroup is connected.
            Can't be a live location.
        address (:obj:`str`): Location address;
            :tg-const:`telegram.ChatLocation.MIN_ADDRESS`-
            :tg-const:`telegram.ChatLocation.MAX_ADDRESS` characters, as defined by the chat owner.

    """
    __slots__ = ('location', 'address')

    def __init__(self, location: Location, address: str, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.location: Location = location
        self.address: str = address
        self._id_attrs = (self.location,)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['ChatLocation']:
        if False:
            return 10
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        data['location'] = Location.de_json(data.get('location'), bot)
        return super().de_json(data=data, bot=bot)
    MIN_ADDRESS: Final[int] = constants.LocationLimit.MIN_CHAT_LOCATION_ADDRESS
    ':const:`telegram.constants.LocationLimit.MIN_CHAT_LOCATION_ADDRESS`\n\n    .. versionadded:: 20.0\n    '
    MAX_ADDRESS: Final[int] = constants.LocationLimit.MAX_CHAT_LOCATION_ADDRESS
    ':const:`telegram.constants.LocationLimit.MAX_CHAT_LOCATION_ADDRESS`\n\n    .. versionadded:: 20.0\n    '