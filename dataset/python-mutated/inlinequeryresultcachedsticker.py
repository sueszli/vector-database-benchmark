"""This module contains the classes that represent Telegram InlineQueryResultCachedSticker."""
from typing import TYPE_CHECKING, Optional
from telegram._inline.inlinekeyboardmarkup import InlineKeyboardMarkup
from telegram._inline.inlinequeryresult import InlineQueryResult
from telegram._utils.types import JSONDict
from telegram.constants import InlineQueryResultType
if TYPE_CHECKING:
    from telegram import InputMessageContent

class InlineQueryResultCachedSticker(InlineQueryResult):
    """
    Represents a link to a sticker stored on the Telegram servers. By default, this sticker will
    be sent by the user. Alternatively, you can use :attr:`input_message_content` to send a
    message with the specified content instead of the sticker.

    .. seealso:: :wiki:`Working with Files and Media <Working-with-Files-and-Media>`

    Args:
        id (:obj:`str`): Unique identifier for this result,
            :tg-const:`telegram.InlineQueryResult.MIN_ID_LENGTH`-
            :tg-const:`telegram.InlineQueryResult.MAX_ID_LENGTH` Bytes.
        sticker_file_id (:obj:`str`): A valid file identifier of the sticker.
        reply_markup (:class:`telegram.InlineKeyboardMarkup`, optional): Inline keyboard attached
            to the message.
        input_message_content (:class:`telegram.InputMessageContent`, optional): Content of the
            message to be sent instead of the sticker.

    Attributes:
        type (:obj:`str`): :tg-const:`telegram.constants.InlineQueryResultType.STICKER`.
        id (:obj:`str`): Unique identifier for this result,
            :tg-const:`telegram.InlineQueryResult.MIN_ID_LENGTH`-
            :tg-const:`telegram.InlineQueryResult.MAX_ID_LENGTH` Bytes.
        sticker_file_id (:obj:`str`): A valid file identifier of the sticker.
        reply_markup (:class:`telegram.InlineKeyboardMarkup`): Optional. Inline keyboard attached
            to the message.
        input_message_content (:class:`telegram.InputMessageContent`): Optional. Content of the
            message to be sent instead of the sticker.

    """
    __slots__ = ('reply_markup', 'input_message_content', 'sticker_file_id')

    def __init__(self, id: str, sticker_file_id: str, reply_markup: Optional[InlineKeyboardMarkup]=None, input_message_content: Optional['InputMessageContent']=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(InlineQueryResultType.STICKER, id, api_kwargs=api_kwargs)
        with self._unfrozen():
            self.sticker_file_id: str = sticker_file_id
            self.reply_markup: Optional[InlineKeyboardMarkup] = reply_markup
            self.input_message_content: Optional[InputMessageContent] = input_message_content