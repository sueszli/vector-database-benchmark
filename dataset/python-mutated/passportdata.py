"""Contains information about Telegram Passport data shared with the bot by the user."""
from typing import TYPE_CHECKING, Optional, Sequence, Tuple
from telegram._passport.credentials import EncryptedCredentials
from telegram._passport.encryptedpassportelement import EncryptedPassportElement
from telegram._telegramobject import TelegramObject
from telegram._utils.argumentparsing import parse_sequence_arg
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot, Credentials

class PassportData(TelegramObject):
    """Contains information about Telegram Passport data shared with the bot by the user.

    Note:
        To be able to decrypt this object, you must pass your ``private_key`` to either
        :class:`telegram.ext.Updater` or :class:`telegram.Bot`. Decrypted data is then found in
        :attr:`decrypted_data` and the payload can be found in :attr:`decrypted_credentials`'s
        attribute :attr:`telegram.Credentials.nonce`.

    Args:
        data (Sequence[:class:`telegram.EncryptedPassportElement`]): Array with encrypted
            information about documents and other Telegram Passport elements that was shared with
            the bot.

            .. versionchanged:: 20.0
                |sequenceclassargs|

        credentials (:class:`telegram.EncryptedCredentials`)): Encrypted credentials.

    Attributes:
        data (Tuple[:class:`telegram.EncryptedPassportElement`]): Array with encrypted
            information about documents and other Telegram Passport elements that was shared with
            the bot.

            .. versionchanged:: 20.0
                |tupleclassattrs|

        credentials (:class:`telegram.EncryptedCredentials`): Encrypted credentials.


    """
    __slots__ = ('credentials', 'data', '_decrypted_data')

    def __init__(self, data: Sequence[EncryptedPassportElement], credentials: EncryptedCredentials, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self.data: Tuple[EncryptedPassportElement, ...] = parse_sequence_arg(data)
        self.credentials: EncryptedCredentials = credentials
        self._decrypted_data: Optional[Tuple[EncryptedPassportElement]] = None
        self._id_attrs = tuple([x.type for x in data] + [credentials.hash])
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['PassportData']:
        if False:
            return 10
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        data['data'] = EncryptedPassportElement.de_list(data.get('data'), bot)
        data['credentials'] = EncryptedCredentials.de_json(data.get('credentials'), bot)
        return super().de_json(data=data, bot=bot)

    @property
    def decrypted_data(self) -> Tuple[EncryptedPassportElement, ...]:
        if False:
            print('Hello World!')
        '\n        Tuple[:class:`telegram.EncryptedPassportElement`]: Lazily decrypt and return information\n            about documents and other Telegram Passport elements which were shared with the bot.\n\n        .. versionchanged:: 20.0\n            Returns a tuple instead of a list.\n\n        Raises:\n            telegram.error.PassportDecryptionError: Decryption failed. Usually due to bad\n                private/public key but can also suggest malformed/tampered data.\n        '
        if self._decrypted_data is None:
            self._decrypted_data = tuple((EncryptedPassportElement.de_json_decrypted(element.to_dict(), self.get_bot(), self.decrypted_credentials) for element in self.data))
        return self._decrypted_data

    @property
    def decrypted_credentials(self) -> 'Credentials':
        if False:
            for i in range(10):
                print('nop')
        '\n        :class:`telegram.Credentials`: Lazily decrypt and return credentials that were used\n            to decrypt the data. This object also contains the user specified payload as\n            `decrypted_data.payload`.\n\n        Raises:\n            telegram.error.PassportDecryptionError: Decryption failed. Usually due to bad\n                private/public key but can also suggest malformed/tampered data.\n        '
        return self.credentials.decrypted_data