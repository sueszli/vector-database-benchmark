"""This module contains an object that represents a Telegram LabeledPrice."""
from typing import Optional
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class LabeledPrice(TelegramObject):
    """This object represents a portion of the price for goods or services.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`label` and :attr:`amount` are equal.

    Examples:
        :any:`Payment Bot <examples.paymentbot>`

    Args:
        label (:obj:`str`): Portion label.
        amount (:obj:`int`): Price of the product in the smallest units of the currency (integer,
            not float/double). For example, for a price of US$ 1.45 pass ``amount = 145``.
            See the ``exp`` parameter in
            `currencies.json <https://core.telegram.org/bots/payments/currencies.json>`_,
            it shows the number of digits past the decimal point for each currency
            (2 for the majority of currencies).

    Attributes:
        label (:obj:`str`): Portion label.
        amount (:obj:`int`): Price of the product in the smallest units of the currency (integer,
            not float/double). For example, for a price of US$ 1.45 ``amount`` is ``145``.
            See the ``exp`` parameter in
            `currencies.json <https://core.telegram.org/bots/payments/currencies.json>`_,
            it shows the number of digits past the decimal point for each currency
            (2 for the majority of currencies).

    """
    __slots__ = ('label', 'amount')

    def __init__(self, label: str, amount: int, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self.label: str = label
        self.amount: int = amount
        self._id_attrs = (self.label, self.amount)
        self._freeze()