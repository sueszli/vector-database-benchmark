"""This module contains the classes that represent Telegram InlineQueryResultArticle."""
from typing import TYPE_CHECKING, Optional
from telegram._inline.inlinekeyboardmarkup import InlineKeyboardMarkup
from telegram._inline.inlinequeryresult import InlineQueryResult
from telegram._utils.types import JSONDict
from telegram.constants import InlineQueryResultType
if TYPE_CHECKING:
    from telegram import InputMessageContent

class InlineQueryResultArticle(InlineQueryResult):
    """This object represents a Telegram InlineQueryResultArticle.

    Examples:
        :any:`Inline Bot <examples.inlinebot>`

    .. versionchanged:: 20.5
      Removed the deprecated arguments and attributes ``thumb_*``.

    Args:
        id (:obj:`str`): Unique identifier for this result,
            :tg-const:`telegram.InlineQueryResult.MIN_ID_LENGTH`-
            :tg-const:`telegram.InlineQueryResult.MAX_ID_LENGTH` Bytes.
        title (:obj:`str`): Title of the result.
        input_message_content (:class:`telegram.InputMessageContent`): Content of the message to
            be sent.
        reply_markup (:class:`telegram.InlineKeyboardMarkup`, optional): Inline keyboard attached
            to the message.
        url (:obj:`str`, optional): URL of the result.
        hide_url (:obj:`bool`, optional): Pass :obj:`True`, if you don't want the URL to be shown
            in the message.
        description (:obj:`str`, optional): Short description of the result.
        thumbnail_url (:obj:`str`, optional): Url of the thumbnail for the result.

            .. versionadded:: 20.2
        thumbnail_width (:obj:`int`, optional): Thumbnail width.

            .. versionadded:: 20.2
        thumbnail_height (:obj:`int`, optional): Thumbnail height.

            .. versionadded:: 20.2

    Attributes:
        type (:obj:`str`): :tg-const:`telegram.constants.InlineQueryResultType.ARTICLE`.
        id (:obj:`str`): Unique identifier for this result,
            :tg-const:`telegram.InlineQueryResult.MIN_ID_LENGTH`-
            :tg-const:`telegram.InlineQueryResult.MAX_ID_LENGTH` Bytes.
        title (:obj:`str`): Title of the result.
        input_message_content (:class:`telegram.InputMessageContent`): Content of the message to
            be sent.
        reply_markup (:class:`telegram.InlineKeyboardMarkup`): Optional. Inline keyboard attached
            to the message.
        url (:obj:`str`): Optional. URL of the result.
        hide_url (:obj:`bool`): Optional. Pass :obj:`True`, if you don't want the URL to be shown
            in the message.
        description (:obj:`str`): Optional. Short description of the result.
        thumbnail_url (:obj:`str`): Optional. Url of the thumbnail for the result.

            .. versionadded:: 20.2
        thumbnail_width (:obj:`int`): Optional. Thumbnail width.

            .. versionadded:: 20.2
        thumbnail_height (:obj:`int`): Optional. Thumbnail height.

            .. versionadded:: 20.2

    """
    __slots__ = ('reply_markup', 'hide_url', 'url', 'title', 'description', 'input_message_content', 'thumbnail_width', 'thumbnail_height', 'thumbnail_url')

    def __init__(self, id: str, title: str, input_message_content: 'InputMessageContent', reply_markup: Optional[InlineKeyboardMarkup]=None, url: Optional[str]=None, hide_url: Optional[bool]=None, description: Optional[str]=None, thumbnail_url: Optional[str]=None, thumbnail_width: Optional[int]=None, thumbnail_height: Optional[int]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(InlineQueryResultType.ARTICLE, id, api_kwargs=api_kwargs)
        with self._unfrozen():
            self.title: str = title
            self.input_message_content: InputMessageContent = input_message_content
            self.reply_markup: Optional[InlineKeyboardMarkup] = reply_markup
            self.url: Optional[str] = url
            self.hide_url: Optional[bool] = hide_url
            self.description: Optional[str] = description
            self.thumbnail_url: Optional[str] = thumbnail_url
            self.thumbnail_width: Optional[int] = thumbnail_width
            self.thumbnail_height: Optional[int] = thumbnail_height