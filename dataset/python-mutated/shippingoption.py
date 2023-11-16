"""This module contains an object that represents a Telegram ShippingOption."""
from typing import TYPE_CHECKING, Optional, Sequence, Tuple
from telegram._telegramobject import TelegramObject
from telegram._utils.argumentparsing import parse_sequence_arg
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import LabeledPrice

class ShippingOption(TelegramObject):
    """This object represents one shipping option.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`id` is equal.

    Examples:
        :any:`Payment Bot <examples.paymentbot>`

    Args:
        id (:obj:`str`): Shipping option identifier.
        title (:obj:`str`): Option title.
        prices (Sequence[:class:`telegram.LabeledPrice`]): List of price portions.

            .. versionchanged:: 20.0
                |sequenceclassargs|

    Attributes:
        id (:obj:`str`): Shipping option identifier.
        title (:obj:`str`): Option title.
        prices (Tuple[:class:`telegram.LabeledPrice`]): List of price portions.

            .. versionchanged:: 20.0
                |tupleclassattrs|

    """
    __slots__ = ('prices', 'title', 'id')

    def __init__(self, id: str, title: str, prices: Sequence['LabeledPrice'], *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self.id: str = id
        self.title: str = title
        self.prices: Tuple[LabeledPrice, ...] = parse_sequence_arg(prices)
        self._id_attrs = (self.id,)
        self._freeze()