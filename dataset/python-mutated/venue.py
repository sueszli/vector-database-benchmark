"""This module contains an object that represents a Telegram Venue."""
from typing import TYPE_CHECKING, Optional
from telegram._files.location import Location
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot

class Venue(TelegramObject):
    """This object represents a venue.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`location` and :attr:`title` are equal.

    Note:
      Foursquare details and Google Place details are mutually exclusive. However, this
      behaviour is undocumented and might be changed by Telegram.

    Args:
        location (:class:`telegram.Location`): Venue location.
        title (:obj:`str`): Name of the venue.
        address (:obj:`str`): Address of the venue.
        foursquare_id (:obj:`str`, optional): Foursquare identifier of the venue.
        foursquare_type (:obj:`str`, optional): Foursquare type of the venue. (For example,
            "arts_entertainment/default", "arts_entertainment/aquarium" or "food/icecream".)
        google_place_id (:obj:`str`, optional): Google Places identifier of the venue.
        google_place_type (:obj:`str`, optional): Google Places type of the venue. (See
            `supported types <https://developers.google.com/maps/documentation/places/web-service            /supported_types>`_.)

    Attributes:
        location (:class:`telegram.Location`): Venue location.
        title (:obj:`str`): Name of the venue.
        address (:obj:`str`): Address of the venue.
        foursquare_id (:obj:`str`): Optional. Foursquare identifier of the venue.
        foursquare_type (:obj:`str`): Optional. Foursquare type of the venue. (For example,
            "arts_entertainment/default", "arts_entertainment/aquarium" or "food/icecream".)
        google_place_id (:obj:`str`): Optional. Google Places identifier of the venue.
        google_place_type (:obj:`str`): Optional. Google Places type of the venue. (See
            `supported types <https://developers.google.com/maps/documentation/places/web-service            /supported_types>`_.)

    """
    __slots__ = ('address', 'location', 'foursquare_id', 'foursquare_type', 'google_place_id', 'google_place_type', 'title')

    def __init__(self, location: Location, title: str, address: str, foursquare_id: Optional[str]=None, foursquare_type: Optional[str]=None, google_place_id: Optional[str]=None, google_place_type: Optional[str]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            return 10
        super().__init__(api_kwargs=api_kwargs)
        self.location: Location = location
        self.title: str = title
        self.address: str = address
        self.foursquare_id: Optional[str] = foursquare_id
        self.foursquare_type: Optional[str] = foursquare_type
        self.google_place_id: Optional[str] = google_place_id
        self.google_place_type: Optional[str] = google_place_type
        self._id_attrs = (self.location, self.title)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['Venue']:
        if False:
            print('Hello World!')
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        data['location'] = Location.de_json(data.get('location'), bot)
        return super().de_json(data=data, bot=bot)