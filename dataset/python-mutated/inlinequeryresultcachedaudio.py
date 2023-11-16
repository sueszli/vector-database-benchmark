"""This module contains the classes that represent Telegram InlineQueryResultCachedAudio."""
from typing import TYPE_CHECKING, Optional, Sequence, Tuple
from telegram._inline.inlinekeyboardmarkup import InlineKeyboardMarkup
from telegram._inline.inlinequeryresult import InlineQueryResult
from telegram._messageentity import MessageEntity
from telegram._utils.argumentparsing import parse_sequence_arg
from telegram._utils.defaultvalue import DEFAULT_NONE
from telegram._utils.types import JSONDict, ODVInput
from telegram.constants import InlineQueryResultType
if TYPE_CHECKING:
    from telegram import InputMessageContent

class InlineQueryResultCachedAudio(InlineQueryResult):
    """
    Represents a link to an mp3 audio file stored on the Telegram servers. By default, this audio
    file will be sent by the user. Alternatively, you can use :attr:`input_message_content` to
    send a message with the specified content instead of the audio.

    .. seealso:: :wiki:`Working with Files and Media <Working-with-Files-and-Media>`

    Args:
        id (:obj:`str`): Unique identifier for this result,
            :tg-const:`telegram.InlineQueryResult.MIN_ID_LENGTH`-
            :tg-const:`telegram.InlineQueryResult.MAX_ID_LENGTH` Bytes.
        audio_file_id (:obj:`str`): A valid file identifier for the audio file.
        caption (:obj:`str`, optional): Caption,
            0-:tg-const:`telegram.constants.MessageLimit.CAPTION_LENGTH` characters after entities
            parsing.
        parse_mode (:obj:`str`, optional): |parse_mode|
        caption_entities (Sequence[:class:`telegram.MessageEntity`], optional): |caption_entities|

            .. versionchanged:: 20.0
                |sequenceclassargs|

        reply_markup (:class:`telegram.InlineKeyboardMarkup`, optional): Inline keyboard attached
            to the message.
        input_message_content (:class:`telegram.InputMessageContent`, optional): Content of the
            message to be sent instead of the audio.

    Attributes:
        type (:obj:`str`): :tg-const:`telegram.constants.InlineQueryResultType.AUDIO`.
        id (:obj:`str`): Unique identifier for this result,
            :tg-const:`telegram.InlineQueryResult.MIN_ID_LENGTH`-
            :tg-const:`telegram.InlineQueryResult.MAX_ID_LENGTH` Bytes.
        audio_file_id (:obj:`str`): A valid file identifier for the audio file.
        caption (:obj:`str`): Optional. Caption,
            0-:tg-const:`telegram.constants.MessageLimit.CAPTION_LENGTH` characters after entities
            parsing.
        parse_mode (:obj:`str`): Optional. |parse_mode|
        caption_entities (Tuple[:class:`telegram.MessageEntity`]): Optional. |captionentitiesattr|

            .. versionchanged:: 20.0

                * |tupleclassattrs|
                * |alwaystuple|
        reply_markup (:class:`telegram.InlineKeyboardMarkup`): Optional. Inline keyboard attached
            to the message.
        input_message_content (:class:`telegram.InputMessageContent`): Optional. Content of the
            message to be sent instead of the audio.

    """
    __slots__ = ('reply_markup', 'caption_entities', 'caption', 'parse_mode', 'audio_file_id', 'input_message_content')

    def __init__(self, id: str, audio_file_id: str, caption: Optional[str]=None, reply_markup: Optional[InlineKeyboardMarkup]=None, input_message_content: Optional['InputMessageContent']=None, parse_mode: ODVInput[str]=DEFAULT_NONE, caption_entities: Optional[Sequence[MessageEntity]]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(InlineQueryResultType.AUDIO, id, api_kwargs=api_kwargs)
        with self._unfrozen():
            self.audio_file_id: str = audio_file_id
            self.caption: Optional[str] = caption
            self.parse_mode: ODVInput[str] = parse_mode
            self.caption_entities: Tuple[MessageEntity, ...] = parse_sequence_arg(caption_entities)
            self.reply_markup: Optional[InlineKeyboardMarkup] = reply_markup
            self.input_message_content: Optional[InputMessageContent] = input_message_content