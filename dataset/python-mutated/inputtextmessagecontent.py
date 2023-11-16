"""This module contains the classes that represent Telegram InputTextMessageContent."""
from typing import Optional, Sequence, Tuple
from telegram._inline.inputmessagecontent import InputMessageContent
from telegram._messageentity import MessageEntity
from telegram._utils.argumentparsing import parse_sequence_arg
from telegram._utils.defaultvalue import DEFAULT_NONE
from telegram._utils.types import JSONDict, ODVInput

class InputTextMessageContent(InputMessageContent):
    """
    Represents the content of a text message to be sent as the result of an inline query.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`message_text` is equal.

    Examples:
        :any:`Inline Bot <examples.inlinebot>`

    Args:
        message_text (:obj:`str`): Text of the message to be sent,
            :tg-const:`telegram.constants.MessageLimit.MIN_TEXT_LENGTH`-
            :tg-const:`telegram.constants.MessageLimit.MAX_TEXT_LENGTH` characters after entities
            parsing.
        parse_mode (:obj:`str`, optional): |parse_mode|
        entities (Sequence[:class:`telegram.MessageEntity`], optional): |caption_entities|

            .. versionchanged:: 20.0
                |sequenceclassargs|

        disable_web_page_preview (:obj:`bool`, optional): Disables link previews for links in the
            sent message.

    Attributes:
        message_text (:obj:`str`): Text of the message to be sent,
            :tg-const:`telegram.constants.MessageLimit.MIN_TEXT_LENGTH`-
            :tg-const:`telegram.constants.MessageLimit.MAX_TEXT_LENGTH` characters after entities
            parsing.
        parse_mode (:obj:`str`): Optional. |parse_mode|
        entities (Tuple[:class:`telegram.MessageEntity`]): Optional. |captionentitiesattr|

            .. versionchanged:: 20.0

                * |tupleclassattrs|
                * |alwaystuple|
        disable_web_page_preview (:obj:`bool`): Optional. Disables link previews for links in the
            sent message.

    """
    __slots__ = ('disable_web_page_preview', 'parse_mode', 'entities', 'message_text')

    def __init__(self, message_text: str, parse_mode: ODVInput[str]=DEFAULT_NONE, disable_web_page_preview: ODVInput[bool]=DEFAULT_NONE, entities: Optional[Sequence[MessageEntity]]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        with self._unfrozen():
            self.message_text: str = message_text
            self.parse_mode: ODVInput[str] = parse_mode
            self.entities: Tuple[MessageEntity, ...] = parse_sequence_arg(entities)
            self.disable_web_page_preview: ODVInput[bool] = disable_web_page_preview
            self._id_attrs = (self.message_text,)