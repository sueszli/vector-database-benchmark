"""This module contains the classes that represent Telegram InputLocationMessageContent."""
from typing import Final, Optional
from telegram import constants
from telegram._inline.inputmessagecontent import InputMessageContent
from telegram._utils.types import JSONDict

class InputLocationMessageContent(InputMessageContent):
    """
    Represents the content of a location message to be sent as the result of an inline query.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`latitude` and :attr:`longitude` are equal.

    Args:
        latitude (:obj:`float`): Latitude of the location in degrees.
        longitude (:obj:`float`): Longitude of the location in degrees.
        horizontal_accuracy (:obj:`float`, optional): The radius of uncertainty for the location,
            measured in meters; 0-
            :tg-const:`telegram.InputLocationMessageContent.HORIZONTAL_ACCURACY`.
        live_period (:obj:`int`, optional): Period in seconds for which the location will be
            updated, should be between
            :tg-const:`telegram.InputLocationMessageContent.MIN_LIVE_PERIOD` and
            :tg-const:`telegram.InputLocationMessageContent.MAX_LIVE_PERIOD`.
        heading (:obj:`int`, optional): For live locations, a direction in which the user is
            moving, in degrees. Must be between
            :tg-const:`telegram.InputLocationMessageContent.MIN_HEADING` and
            :tg-const:`telegram.InputLocationMessageContent.MAX_HEADING` if specified.
        proximity_alert_radius (:obj:`int`, optional): For live locations, a maximum distance
            for proximity alerts about approaching another chat member, in meters. Must be
            between :tg-const:`telegram.InputLocationMessageContent.MIN_PROXIMITY_ALERT_RADIUS`
            and :tg-const:`telegram.InputLocationMessageContent.MAX_PROXIMITY_ALERT_RADIUS`
            if specified.

    Attributes:
        latitude (:obj:`float`): Latitude of the location in degrees.
        longitude (:obj:`float`): Longitude of the location in degrees.
        horizontal_accuracy (:obj:`float`): Optional. The radius of uncertainty for the location,
            measured in meters; 0-
            :tg-const:`telegram.InputLocationMessageContent.HORIZONTAL_ACCURACY`.
        live_period (:obj:`int`): Optional. Period in seconds for which the location can be
            updated, should be between
            :tg-const:`telegram.InputLocationMessageContent.MIN_LIVE_PERIOD` and
            :tg-const:`telegram.InputLocationMessageContent.MAX_LIVE_PERIOD`.
        heading (:obj:`int`): Optional. For live locations, a direction in which the user is
            moving, in degrees. Must be between
            :tg-const:`telegram.InputLocationMessageContent.MIN_HEADING` and
            :tg-const:`telegram.InputLocationMessageContent.MAX_HEADING` if specified.
        proximity_alert_radius (:obj:`int`): Optional. For live locations, a maximum distance
            for proximity alerts about approaching another chat member, in meters. Must be
            between :tg-const:`telegram.InputLocationMessageContent.MIN_PROXIMITY_ALERT_RADIUS`
            and :tg-const:`telegram.InputLocationMessageContent.MAX_PROXIMITY_ALERT_RADIUS`
            if specified.

    """
    __slots__ = ('longitude', 'horizontal_accuracy', 'proximity_alert_radius', 'live_period', 'latitude', 'heading')

    def __init__(self, latitude: float, longitude: float, live_period: Optional[int]=None, horizontal_accuracy: Optional[float]=None, heading: Optional[int]=None, proximity_alert_radius: Optional[int]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        with self._unfrozen():
            self.latitude: float = latitude
            self.longitude: float = longitude
            self.live_period: Optional[int] = live_period
            self.horizontal_accuracy: Optional[float] = horizontal_accuracy
            self.heading: Optional[int] = heading
            self.proximity_alert_radius: Optional[int] = int(proximity_alert_radius) if proximity_alert_radius else None
            self._id_attrs = (self.latitude, self.longitude)
    HORIZONTAL_ACCURACY: Final[int] = constants.LocationLimit.HORIZONTAL_ACCURACY
    ':const:`telegram.constants.LocationLimit.HORIZONTAL_ACCURACY`\n\n    .. versionadded:: 20.0\n    '
    MIN_HEADING: Final[int] = constants.LocationLimit.MIN_HEADING
    ':const:`telegram.constants.LocationLimit.MIN_HEADING`\n\n    .. versionadded:: 20.0\n    '
    MAX_HEADING: Final[int] = constants.LocationLimit.MAX_HEADING
    ':const:`telegram.constants.LocationLimit.MAX_HEADING`\n\n    .. versionadded:: 20.0\n    '
    MIN_LIVE_PERIOD: Final[int] = constants.LocationLimit.MIN_LIVE_PERIOD
    ':const:`telegram.constants.LocationLimit.MIN_LIVE_PERIOD`\n\n    .. versionadded:: 20.0\n    '
    MAX_LIVE_PERIOD: Final[int] = constants.LocationLimit.MAX_LIVE_PERIOD
    ':const:`telegram.constants.LocationLimit.MAX_LIVE_PERIOD`\n\n    .. versionadded:: 20.0\n    '
    MIN_PROXIMITY_ALERT_RADIUS: Final[int] = constants.LocationLimit.MIN_PROXIMITY_ALERT_RADIUS
    ':const:`telegram.constants.LocationLimit.MIN_PROXIMITY_ALERT_RADIUS`\n\n    .. versionadded:: 20.0\n    '
    MAX_PROXIMITY_ALERT_RADIUS: Final[int] = constants.LocationLimit.MAX_PROXIMITY_ALERT_RADIUS
    ':const:`telegram.constants.LocationLimit.MAX_PROXIMITY_ALERT_RADIUS`\n\n    .. versionadded:: 20.0\n    '