"""This module contains an object that represents a Telegram GameHighScore."""
from typing import TYPE_CHECKING, Optional
from telegram._telegramobject import TelegramObject
from telegram._user import User
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot

class GameHighScore(TelegramObject):
    """This object represents one row of the high scores table for a game.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`position`, :attr:`user` and :attr:`score` are equal.

    Args:
        position (:obj:`int`): Position in high score table for the game.
        user (:class:`telegram.User`): User.
        score (:obj:`int`): Score.

    Attributes:
        position (:obj:`int`): Position in high score table for the game.
        user (:class:`telegram.User`): User.
        score (:obj:`int`): Score.

    """
    __slots__ = ('position', 'user', 'score')

    def __init__(self, position: int, user: User, score: int, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.position: int = position
        self.user: User = user
        self.score: int = score
        self._id_attrs = (self.position, self.user, self.score)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['GameHighScore']:
        if False:
            return 10
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        data['user'] = User.de_json(data.get('user'), bot)
        return super().de_json(data=data, bot=bot)