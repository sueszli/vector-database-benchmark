"""This module contains an object that represents a Telegram InputSticker."""
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union
from telegram._files.sticker import MaskPosition
from telegram._telegramobject import TelegramObject
from telegram._utils.argumentparsing import parse_sequence_arg
from telegram._utils.files import parse_file_input
from telegram._utils.types import FileInput, JSONDict
if TYPE_CHECKING:
    from telegram._files.inputfile import InputFile

class InputSticker(TelegramObject):
    """
    This object describes a sticker to be added to a sticker set.

    .. versionadded:: 20.2

    Args:
        sticker (:obj:`str` | :term:`file object` | :obj:`bytes` | :class:`pathlib.Path`): The
            added sticker. |uploadinputnopath| Animated and video stickers can't be uploaded via
            HTTP URL.
        emoji_list (Sequence[:obj:`str`]): Sequence of
            :tg-const:`telegram.constants.StickerLimit.MIN_STICKER_EMOJI` -
            :tg-const:`telegram.constants.StickerLimit.MAX_STICKER_EMOJI` emoji associated with the
            sticker.
        mask_position (:obj:`telegram.MaskPosition`, optional): Position where the mask should be
            placed on faces. For ":tg-const:`telegram.constants.StickerType.MASK`" stickers only.
        keywords (Sequence[:obj:`str`], optional): Sequence of
            0-:tg-const:`telegram.constants.StickerLimit.MAX_SEARCH_KEYWORDS` search keywords
            for the sticker with the total length of up to
            :tg-const:`telegram.constants.StickerLimit.MAX_KEYWORD_LENGTH` characters. For
            ":tg-const:`telegram.constants.StickerType.REGULAR`" and
            ":tg-const:`telegram.constants.StickerType.CUSTOM_EMOJI`" stickers only.

    Attributes:
        sticker (:obj:`str` | :class:`telegram.InputFile`): The added sticker.
        emoji_list (Tuple[:obj:`str`]): Tuple of
            :tg-const:`telegram.constants.StickerLimit.MIN_STICKER_EMOJI` -
            :tg-const:`telegram.constants.StickerLimit.MAX_STICKER_EMOJI` emoji associated with the
            sticker.
        mask_position (:obj:`telegram.MaskPosition`): Optional. Position where the mask should be
            placed on faces. For ":tg-const:`telegram.constants.StickerType.MASK`" stickers only.
        keywords (Tuple[:obj:`str`]): Optional. Tuple of
            0-:tg-const:`telegram.constants.StickerLimit.MAX_SEARCH_KEYWORDS` search keywords
            for the sticker with the total length of up to
            :tg-const:`telegram.constants.StickerLimit.MAX_KEYWORD_LENGTH` characters. For
            ":tg-const:`telegram.constants.StickerType.REGULAR`" and
            ":tg-const:`telegram.constants.StickerType.CUSTOM_EMOJI`" stickers only.

    """
    __slots__ = ('sticker', 'emoji_list', 'mask_position', 'keywords')

    def __init__(self, sticker: FileInput, emoji_list: Sequence[str], mask_position: Optional[MaskPosition]=None, keywords: Optional[Sequence[str]]=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.sticker: Union[str, InputFile] = parse_file_input(sticker, local_mode=True, attach=True)
        self.emoji_list: Tuple[str, ...] = parse_sequence_arg(emoji_list)
        self.mask_position: Optional[MaskPosition] = mask_position
        self.keywords: Tuple[str, ...] = parse_sequence_arg(keywords)
        self._freeze()