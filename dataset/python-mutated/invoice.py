"""This module contains an object that represents a Telegram Invoice."""
from typing import Final, Optional
from telegram import constants
from telegram._telegramobject import TelegramObject
from telegram._utils.types import JSONDict

class Invoice(TelegramObject):
    """This object contains basic information about an invoice.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`title`, :attr:`description`, :paramref:`start_parameter`,
    :attr:`currency` and :attr:`total_amount` are equal.

    Args:
        title (:obj:`str`): Product name.
        description (:obj:`str`): Product description.
        start_parameter (:obj:`str`): Unique bot deep-linking parameter that can be used to
            generate this invoice.
        currency (:obj:`str`): Three-letter ISO 4217 currency code.
        total_amount (:obj:`int`): Total price in the smallest units of the currency (integer, not
            float/double). For example, for a price of US$ 1.45 pass ``amount = 145``. See the
            ``exp`` parameter in
            `currencies.json <https://core.telegram.org/bots/payments/currencies.json>`_,
            it shows the number of digits past the decimal point for each currency
            (2 for the majority of currencies).

    Attributes:
        title (:obj:`str`): Product name.
        description (:obj:`str`): Product description.
        start_parameter (:obj:`str`): Unique bot deep-linking parameter that can be used to
            generate this invoice.
        currency (:obj:`str`): Three-letter ISO 4217 currency code.
        total_amount (:obj:`int`): Total price in the smallest units of the currency (integer, not
            float/double). For example, for a price of US$ 1.45 ``amount`` is ``145``. See the
            ``exp`` parameter in
            `currencies.json <https://core.telegram.org/bots/payments/currencies.json>`_,
            it shows the number of digits past the decimal point for each currency
            (2 for the majority of currencies).

    """
    __slots__ = ('currency', 'start_parameter', 'title', 'description', 'total_amount')

    def __init__(self, title: str, description: str, start_parameter: str, currency: str, total_amount: int, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            while True:
                i = 10
        super().__init__(api_kwargs=api_kwargs)
        self.title: str = title
        self.description: str = description
        self.start_parameter: str = start_parameter
        self.currency: str = currency
        self.total_amount: int = total_amount
        self._id_attrs = (self.title, self.description, self.start_parameter, self.currency, self.total_amount)
        self._freeze()
    MIN_TITLE_LENGTH: Final[int] = constants.InvoiceLimit.MIN_TITLE_LENGTH
    ':const:`telegram.constants.InvoiceLimit.MIN_TITLE_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MAX_TITLE_LENGTH: Final[int] = constants.InvoiceLimit.MAX_TITLE_LENGTH
    ':const:`telegram.constants.InvoiceLimit.MAX_TITLE_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MIN_DESCRIPTION_LENGTH: Final[int] = constants.InvoiceLimit.MIN_DESCRIPTION_LENGTH
    ':const:`telegram.constants.InvoiceLimit.MIN_DESCRIPTION_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MAX_DESCRIPTION_LENGTH: Final[int] = constants.InvoiceLimit.MAX_DESCRIPTION_LENGTH
    ':const:`telegram.constants.InvoiceLimit.MAX_DESCRIPTION_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MIN_PAYLOAD_LENGTH: Final[int] = constants.InvoiceLimit.MIN_PAYLOAD_LENGTH
    ':const:`telegram.constants.InvoiceLimit.MIN_PAYLOAD_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MAX_PAYLOAD_LENGTH: Final[int] = constants.InvoiceLimit.MAX_PAYLOAD_LENGTH
    ':const:`telegram.constants.InvoiceLimit.MAX_PAYLOAD_LENGTH`\n\n    .. versionadded:: 20.0\n    '
    MAX_TIP_AMOUNTS: Final[int] = constants.InvoiceLimit.MAX_TIP_AMOUNTS
    ':const:`telegram.constants.InvoiceLimit.MAX_TIP_AMOUNTS`\n\n    .. versionadded:: 20.0\n    '