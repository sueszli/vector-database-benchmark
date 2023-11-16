"""This module contains an object that represents a Telegram ChosenInlineResult."""
from typing import TYPE_CHECKING, Optional
from telegram._files.location import Location
from telegram._telegramobject import TelegramObject
from telegram._user import User
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot

class ChosenInlineResult(TelegramObject):
    """
    Represents a result of an inline query that was chosen by the user and sent to their chat
    partner.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`result_id` is equal.

    Note:
        * In Python :keyword:`from` is a reserved word. Use :paramref:`from_user` instead.
        * It is necessary to enable inline feedback via `@Botfather <https://t.me/BotFather>`_ in
          order to receive these objects in updates.

    Args:
        result_id (:obj:`str`): The unique identifier for the result that was chosen.
        from_user (:class:`telegram.User`): The user that chose the result.
        location (:class:`telegram.Location`, optional): Sender location, only for bots that
            require user location.
        inline_message_id (:obj:`str`, optional): Identifier of the sent inline message. Available
            only if there is an inline keyboard attached to the message. Will be also received in
            callback queries and can be used to edit the message.
        query (:obj:`str`): The query that was used to obtain the result.

    Attributes:
        result_id (:obj:`str`): The unique identifier for the result that was chosen.
        from_user (:class:`telegram.User`): The user that chose the result.
        location (:class:`telegram.Location`): Optional. Sender location, only for bots that
            require user location.
        inline_message_id (:obj:`str`): Optional. Identifier of the sent inline message. Available
            only if there is an inline keyboard attached to the message. Will be also received in
            callback queries and can be used to edit the message.
        query (:obj:`str`): The query that was used to obtain the result.

    """
    __slots__ = ('location', 'result_id', 'from_user', 'inline_message_id', 'query')

    def __init__(self, result_id: str, from_user: User, query: str, location: Optional[Location]=None, inline_message_id: Optional[str]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self.result_id: str = result_id
        self.from_user: User = from_user
        self.query: str = query
        self.location: Optional[Location] = location
        self.inline_message_id: Optional[str] = inline_message_id
        self._id_attrs = (self.result_id,)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['ChosenInlineResult']:
        if False:
            i = 10
            return i + 15
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        data['from_user'] = User.de_json(data.pop('from', None), bot)
        data['location'] = Location.de_json(data.get('location'), bot)
        return super().de_json(data=data, bot=bot)