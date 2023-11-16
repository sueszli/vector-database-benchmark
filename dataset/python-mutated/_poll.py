"""This module contains an object that represents a Telegram Poll."""
import datetime
from typing import TYPE_CHECKING, Dict, Final, List, Optional, Sequence, Tuple
from telegram import constants
from telegram._chat import Chat
from telegram._messageentity import MessageEntity
from telegram._telegramobject import TelegramObject
from telegram._user import User
from telegram._utils import enum
from telegram._utils.argumentparsing import parse_sequence_arg
from telegram._utils.datetime import extract_tzinfo_from_defaults, from_timestamp
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot

class PollOption(TelegramObject):
    """
    This object contains information about one answer option in a poll.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`text` and :attr:`voter_count` are equal.

    Args:
        text (:obj:`str`): Option text,
            :tg-const:`telegram.PollOption.MIN_LENGTH`-:tg-const:`telegram.PollOption.MAX_LENGTH`
            characters.
        voter_count (:obj:`int`): Number of users that voted for this option.

    Attributes:
        text (:obj:`str`): Option text,
            :tg-const:`telegram.PollOption.MIN_LENGTH`-:tg-const:`telegram.PollOption.MAX_LENGTH`
            characters.
        voter_count (:obj:`int`): Number of users that voted for this option.

    """
    __slots__ = ('voter_count', 'text')

    def __init__(self, text: str, voter_count: int, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self.text: str = text
        self.voter_count: int = voter_count
        self._id_attrs = (self.text, self.voter_count)
        self._freeze()
    MIN_LENGTH: Final[int] = constants.PollLimit.MIN_OPTION_LENGTH
    ':const:`telegram.constants.PollLimit.MIN_OPTION_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MAX_LENGTH: Final[int] = constants.PollLimit.MAX_OPTION_LENGTH
    ':const:`telegram.constants.PollLimit.MAX_OPTION_LENGTH`\n\n    .. versionadded:: 20.0\n    '

class PollAnswer(TelegramObject):
    """
    This object represents an answer of a user in a non-anonymous poll.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`poll_id`, :attr:`user` and :attr:`option_ids` are equal.

    .. versionchanged:: 20.5
        The order of :paramref:`option_ids` and :paramref:`user` is changed in
        20.5 as the latter one became optional.

    .. versionchanged:: 20.6
       Backward compatiblity for changed order of :paramref:`option_ids` and :paramref:`user`
       was removed.

    Args:
        poll_id (:obj:`str`): Unique poll identifier.
        option_ids (Sequence[:obj:`int`]): Identifiers of answer options, chosen by the user. May
            be empty if the user retracted their vote.

            .. versionchanged:: 20.0
                |sequenceclassargs|
        user (:class:`telegram.User`, optional): The user that changed the answer to the poll,
            if the voter isn't anonymous. If the voter is anonymous, this field will contain the
            user :tg-const:`telegram.constants.ChatID.FAKE_CHANNEL` for backwards compatibility.

            .. versionchanged:: 20.5
                :paramref:`user` became optional.
        voter_chat (:class:`telegram.Chat`, optional): The chat that changed the answer to the
            poll, if the voter is anonymous.

            .. versionadded:: 20.5

    Attributes:
        poll_id (:obj:`str`): Unique poll identifier.
        option_ids (Tuple[:obj:`int`]): Identifiers of answer options, chosen by the user. May
            be empty if the user retracted their vote.

            .. versionchanged:: 20.0
                |tupleclassattrs|
        user (:class:`telegram.User`): Optional. The user, who changed the answer to the
            poll, if the voter isn't anonymous. If the voter is anonymous, this field will contain
            the user :tg-const:`telegram.constants.ChatID.FAKE_CHANNEL` for backwards compatibility

            .. versionchanged:: 20.5
                :paramref:`user` became optional.
        voter_chat (:class:`telegram.Chat`): Optional. The chat that changed the answer to the
            poll, if the voter is anonymous.

            .. versionadded:: 20.5

    """
    __slots__ = ('option_ids', 'poll_id', 'user', 'voter_chat')

    def __init__(self, poll_id: str, option_ids: Sequence[int], user: Optional[User]=None, voter_chat: Optional[Chat]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.poll_id: str = poll_id
        self.voter_chat: Optional[Chat] = voter_chat
        self.option_ids: Tuple[int, ...] = parse_sequence_arg(option_ids)
        self.user: Optional[User] = user
        self._id_attrs = (self.poll_id, self.option_ids, self.user, self.voter_chat)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['PollAnswer']:
        if False:
            print('Hello World!')
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        data['user'] = User.de_json(data.get('user'), bot)
        data['voter_chat'] = Chat.de_json(data.get('voter_chat'), bot)
        return super().de_json(data=data, bot=bot)

class Poll(TelegramObject):
    """
    This object contains information about a poll.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`id` is equal.

    Examples:
        :any:`Poll Bot <examples.pollbot>`

    Args:
        id (:obj:`str`): Unique poll identifier.
        question (:obj:`str`): Poll question, :tg-const:`telegram.Poll.MIN_QUESTION_LENGTH`-
            :tg-const:`telegram.Poll.MAX_QUESTION_LENGTH` characters.
        options (Sequence[:class:`~telegram.PollOption`]): List of poll options.

            .. versionchanged:: 20.0
                |sequenceclassargs|
        is_closed (:obj:`bool`): :obj:`True`, if the poll is closed.
        is_anonymous (:obj:`bool`): :obj:`True`, if the poll is anonymous.
        type (:obj:`str`): Poll type, currently can be :attr:`REGULAR` or :attr:`QUIZ`.
        allows_multiple_answers (:obj:`bool`): :obj:`True`, if the poll allows multiple answers.
        correct_option_id (:obj:`int`, optional): A zero based identifier of the correct answer
            option. Available only for closed polls in the quiz mode, which were sent
            (not forwarded), by the bot or to a private chat with the bot.
        explanation (:obj:`str`, optional): Text that is shown when a user chooses an incorrect
            answer or taps on the lamp icon in a quiz-style poll,
            0-:tg-const:`telegram.Poll.MAX_EXPLANATION_LENGTH` characters.
        explanation_entities (Sequence[:class:`telegram.MessageEntity`], optional): Special
            entities like usernames, URLs, bot commands, etc. that appear in the
            :attr:`explanation`. This list is empty if the message does not contain explanation
            entities.

            .. versionchanged:: 20.0

               * This attribute is now always a (possibly empty) list and never :obj:`None`.
               * |sequenceclassargs|
        open_period (:obj:`int`, optional): Amount of time in seconds the poll will be active
            after creation.
        close_date (:obj:`datetime.datetime`, optional): Point in time (Unix timestamp) when the
            poll will be automatically closed. Converted to :obj:`datetime.datetime`.

            .. versionchanged:: 20.3
                |datetime_localization|

    Attributes:
        id (:obj:`str`): Unique poll identifier.
        question (:obj:`str`): Poll question, :tg-const:`telegram.Poll.MIN_QUESTION_LENGTH`-
            :tg-const:`telegram.Poll.MAX_QUESTION_LENGTH` characters.
        options (Tuple[:class:`~telegram.PollOption`]): List of poll options.

            .. versionchanged:: 20.0
                |tupleclassattrs|
        total_voter_count (:obj:`int`): Total number of users that voted in the poll.
        is_closed (:obj:`bool`): :obj:`True`, if the poll is closed.
        is_anonymous (:obj:`bool`): :obj:`True`, if the poll is anonymous.
        type (:obj:`str`): Poll type, currently can be :attr:`REGULAR` or :attr:`QUIZ`.
        allows_multiple_answers (:obj:`bool`): :obj:`True`, if the poll allows multiple answers.
        correct_option_id (:obj:`int`): Optional. A zero based identifier of the correct answer
            option. Available only for closed polls in the quiz mode, which were sent
            (not forwarded), by the bot or to a private chat with the bot.
        explanation (:obj:`str`): Optional. Text that is shown when a user chooses an incorrect
            answer or taps on the lamp icon in a quiz-style poll,
            0-:tg-const:`telegram.Poll.MAX_EXPLANATION_LENGTH` characters.
        explanation_entities (Tuple[:class:`telegram.MessageEntity`]): Special entities
            like usernames, URLs, bot commands, etc. that appear in the :attr:`explanation`.
            This list is empty if the message does not contain explanation entities.

            .. versionchanged:: 20.0
                |tupleclassattrs|

            .. versionchanged:: 20.0
               This attribute is now always a (possibly empty) list and never :obj:`None`.
        open_period (:obj:`int`): Optional. Amount of time in seconds the poll will be active
            after creation.
        close_date (:obj:`datetime.datetime`): Optional. Point in time when the poll will be
            automatically closed.

            .. versionchanged:: 20.3
                |datetime_localization|

    """
    __slots__ = ('total_voter_count', 'allows_multiple_answers', 'open_period', 'options', 'type', 'explanation_entities', 'is_anonymous', 'close_date', 'is_closed', 'id', 'explanation', 'question', 'correct_option_id')

    def __init__(self, id: str, question: str, options: Sequence[PollOption], total_voter_count: int, is_closed: bool, is_anonymous: bool, type: str, allows_multiple_answers: bool, correct_option_id: Optional[int]=None, explanation: Optional[str]=None, explanation_entities: Optional[Sequence[MessageEntity]]=None, open_period: Optional[int]=None, close_date: Optional[datetime.datetime]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self.id: str = id
        self.question: str = question
        self.options: Tuple[PollOption, ...] = parse_sequence_arg(options)
        self.total_voter_count: int = total_voter_count
        self.is_closed: bool = is_closed
        self.is_anonymous: bool = is_anonymous
        self.type: str = enum.get_member(constants.PollType, type, type)
        self.allows_multiple_answers: bool = allows_multiple_answers
        self.correct_option_id: Optional[int] = correct_option_id
        self.explanation: Optional[str] = explanation
        self.explanation_entities: Tuple[MessageEntity, ...] = parse_sequence_arg(explanation_entities)
        self.open_period: Optional[int] = open_period
        self.close_date: Optional[datetime.datetime] = close_date
        self._id_attrs = (self.id,)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['Poll']:
        if False:
            while True:
                i = 10
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        loc_tzinfo = extract_tzinfo_from_defaults(bot)
        data['options'] = [PollOption.de_json(option, bot) for option in data['options']]
        data['explanation_entities'] = MessageEntity.de_list(data.get('explanation_entities'), bot)
        data['close_date'] = from_timestamp(data.get('close_date'), tzinfo=loc_tzinfo)
        return super().de_json(data=data, bot=bot)

    def parse_explanation_entity(self, entity: MessageEntity) -> str:
        if False:
            for i in range(10):
                print('nop')
        "Returns the text from a given :class:`telegram.MessageEntity`.\n\n        Note:\n            This method is present because Telegram calculates the offset and length in\n            UTF-16 codepoint pairs, which some versions of Python don't handle automatically.\n            (That is, you can't just slice ``Message.text`` with the offset and length.)\n\n        Args:\n            entity (:class:`telegram.MessageEntity`): The entity to extract the text from. It must\n                be an entity that belongs to this message.\n\n        Returns:\n            :obj:`str`: The text of the given entity.\n\n        Raises:\n            RuntimeError: If the poll has no explanation.\n\n        "
        if not self.explanation:
            raise RuntimeError("This Poll has no 'explanation'.")
        entity_text = self.explanation.encode('utf-16-le')
        entity_text = entity_text[entity.offset * 2:(entity.offset + entity.length) * 2]
        return entity_text.decode('utf-16-le')

    def parse_explanation_entities(self, types: Optional[List[str]]=None) -> Dict[MessageEntity, str]:
        if False:
            print('Hello World!')
        '\n        Returns a :obj:`dict` that maps :class:`telegram.MessageEntity` to :obj:`str`.\n        It contains entities from this polls explanation filtered by their ``type`` attribute as\n        the key, and the text that each entity belongs to as the value of the :obj:`dict`.\n\n        Note:\n            This method should always be used instead of the :attr:`explanation_entities`\n            attribute, since it calculates the correct substring from the message text based on\n            UTF-16 codepoints. See :attr:`parse_explanation_entity` for more info.\n\n        Args:\n            types (List[:obj:`str`], optional): List of ``MessageEntity`` types as strings. If the\n                    ``type`` attribute of an entity is contained in this list, it will be returned.\n                    Defaults to :attr:`telegram.MessageEntity.ALL_TYPES`.\n\n        Returns:\n            Dict[:class:`telegram.MessageEntity`, :obj:`str`]: A dictionary of entities mapped to\n            the text that belongs to them, calculated based on UTF-16 codepoints.\n\n        '
        if types is None:
            types = MessageEntity.ALL_TYPES
        return {entity: self.parse_explanation_entity(entity) for entity in self.explanation_entities if entity.type in types}
    REGULAR: Final[str] = constants.PollType.REGULAR
    ':const:`telegram.constants.PollType.REGULAR`'
    QUIZ: Final[str] = constants.PollType.QUIZ
    ':const:`telegram.constants.PollType.QUIZ`'
    MAX_EXPLANATION_LENGTH: Final[int] = constants.PollLimit.MAX_EXPLANATION_LENGTH
    ':const:`telegram.constants.PollLimit.MAX_EXPLANATION_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MAX_EXPLANATION_LINE_FEEDS: Final[int] = constants.PollLimit.MAX_EXPLANATION_LINE_FEEDS
    ':const:`telegram.constants.PollLimit.MAX_EXPLANATION_LINE_FEEDS`\n\n    .. versionadded:: 20.0\n    '
    MIN_OPEN_PERIOD: Final[int] = constants.PollLimit.MIN_OPEN_PERIOD
    ':const:`telegram.constants.PollLimit.MIN_OPEN_PERIOD`\n\n    .. versionadded:: 20.0\n    '
    MAX_OPEN_PERIOD: Final[int] = constants.PollLimit.MAX_OPEN_PERIOD
    ':const:`telegram.constants.PollLimit.MAX_OPEN_PERIOD`\n\n    .. versionadded:: 20.0\n    '
    MIN_QUESTION_LENGTH: Final[int] = constants.PollLimit.MIN_QUESTION_LENGTH
    ':const:`telegram.constants.PollLimit.MIN_QUESTION_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MAX_QUESTION_LENGTH: Final[int] = constants.PollLimit.MAX_QUESTION_LENGTH
    ':const:`telegram.constants.PollLimit.MAX_QUESTION_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MIN_OPTION_LENGTH: Final[int] = constants.PollLimit.MIN_OPTION_LENGTH
    ':const:`telegram.constants.PollLimit.MIN_OPTION_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MAX_OPTION_LENGTH: Final[int] = constants.PollLimit.MAX_OPTION_LENGTH
    ':const:`telegram.constants.PollLimit.MAX_OPTION_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MIN_OPTION_NUMBER: Final[int] = constants.PollLimit.MIN_OPTION_NUMBER
    ':const:`telegram.constants.PollLimit.MIN_OPTION_NUMBER`\n\n    .. versionadded:: 20.0\n    '
    MAX_OPTION_NUMBER: Final[int] = constants.PollLimit.MAX_OPTION_NUMBER
    ':const:`telegram.constants.PollLimit.MAX_OPTION_NUMBER`\n\n    .. versionadded:: 20.0\n    '