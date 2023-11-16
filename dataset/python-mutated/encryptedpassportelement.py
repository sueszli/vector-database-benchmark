"""This module contains an object that represents a Telegram EncryptedPassportElement."""
from base64 import b64decode
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union
from telegram._passport.credentials import decrypt_json
from telegram._passport.data import IdDocumentData, PersonalDetails, ResidentialAddress
from telegram._passport.passportfile import PassportFile
from telegram._telegramobject import TelegramObject
from telegram._utils.argumentparsing import parse_sequence_arg
from telegram._utils.types import JSONDict
if TYPE_CHECKING:
    from telegram import Bot, Credentials

class EncryptedPassportElement(TelegramObject):
    """
    Contains information about documents or other Telegram Passport elements shared with the bot
    by the user. The data has been automatically decrypted by python-telegram-bot.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`type`, :attr:`data`, :attr:`phone_number`, :attr:`email`,
    :attr:`files`, :attr:`front_side`, :attr:`reverse_side` and :attr:`selfie` are equal.

    Note:
        This object is decrypted only when originating from
        :obj:`telegram.PassportData.decrypted_data`.

    Args:
        type (:obj:`str`): Element type. One of "personal_details", "passport", "driver_license",
            "identity_card", "internal_passport", "address", "utility_bill", "bank_statement",
            "rental_agreement", "passport_registration", "temporary_registration", "phone_number",
            "email".
        hash (:obj:`str`): Base64-encoded element hash for using in
            :class:`telegram.PassportElementErrorUnspecified`.
        data (:class:`telegram.PersonalDetails` | :class:`telegram.IdDocumentData` |             :class:`telegram.ResidentialAddress` | :obj:`str`, optional):
            Decrypted or encrypted data, available for "personal_details", "passport",
            "driver_license", "identity_card", "internal_passport" and "address" types.
        phone_number (:obj:`str`, optional): User's verified phone number, available only for
            "phone_number" type.
        email (:obj:`str`, optional): User's verified email address, available only for "email"
            type.
        files (Sequence[:class:`telegram.PassportFile`], optional): Array of encrypted/decrypted
            files
            with documents provided by the user, available for "utility_bill", "bank_statement",
            "rental_agreement", "passport_registration" and "temporary_registration" types.

            .. versionchanged:: 20.0
                |sequenceclassargs|

        front_side (:class:`telegram.PassportFile`, optional): Encrypted/decrypted file with the
            front side of the document, provided by the user. Available for "passport",
            "driver_license", "identity_card" and "internal_passport".
        reverse_side (:class:`telegram.PassportFile`, optional): Encrypted/decrypted file with the
            reverse side of the document, provided by the user. Available for "driver_license" and
            "identity_card".
        selfie (:class:`telegram.PassportFile`, optional): Encrypted/decrypted file with the
            selfie of the user holding a document, provided by the user; available for "passport",
            "driver_license", "identity_card" and "internal_passport".
        translation (Sequence[:class:`telegram.PassportFile`], optional): Array of
            encrypted/decrypted
            files with translated versions of documents provided by the user. Available if
            requested for "passport", "driver_license", "identity_card", "internal_passport",
            "utility_bill", "bank_statement", "rental_agreement", "passport_registration" and
            "temporary_registration" types.

            .. versionchanged:: 20.0
                |sequenceclassargs|

    Attributes:
        type (:obj:`str`): Element type. One of "personal_details", "passport", "driver_license",
            "identity_card", "internal_passport", "address", "utility_bill", "bank_statement",
            "rental_agreement", "passport_registration", "temporary_registration", "phone_number",
            "email".
        hash (:obj:`str`): Base64-encoded element hash for using in
            :class:`telegram.PassportElementErrorUnspecified`.
        data (:class:`telegram.PersonalDetails` | :class:`telegram.IdDocumentData` |             :class:`telegram.ResidentialAddress` | :obj:`str`):
            Optional. Decrypted or encrypted data, available for "personal_details", "passport",
            "driver_license", "identity_card", "internal_passport" and "address" types.
        phone_number (:obj:`str`): Optional. User's verified phone number, available only for
            "phone_number" type.
        email (:obj:`str`): Optional. User's verified email address, available only for "email"
            type.
        files (Tuple[:class:`telegram.PassportFile`]): Optional. Array of encrypted/decrypted
            files
            with documents provided by the user, available for "utility_bill", "bank_statement",
            "rental_agreement", "passport_registration" and "temporary_registration" types.

            .. versionchanged:: 20.0

                * |tupleclassattrs|
                * |alwaystuple|

        front_side (:class:`telegram.PassportFile`): Optional. Encrypted/decrypted file with the
            front side of the document, provided by the user. Available for "passport",
            "driver_license", "identity_card" and "internal_passport".
        reverse_side (:class:`telegram.PassportFile`): Optional. Encrypted/decrypted file with the
            reverse side of the document, provided by the user. Available for "driver_license" and
            "identity_card".
        selfie (:class:`telegram.PassportFile`): Optional. Encrypted/decrypted file with the
            selfie of the user holding a document, provided by the user; available for "passport",
            "driver_license", "identity_card" and "internal_passport".
        translation (Tuple[:class:`telegram.PassportFile`]): Optional. Array of
            encrypted/decrypted
            files with translated versions of documents provided by the user. Available if
            requested for "passport", "driver_license", "identity_card", "internal_passport",
            "utility_bill", "bank_statement", "rental_agreement", "passport_registration" and
            "temporary_registration" types.

            .. versionchanged:: 20.0

                * |tupleclassattrs|
                * |alwaystuple|

    """
    __slots__ = ('selfie', 'files', 'type', 'translation', 'email', 'hash', 'phone_number', 'reverse_side', 'front_side', 'data')

    def __init__(self, type: str, hash: str, data: Optional[Union[PersonalDetails, IdDocumentData, ResidentialAddress]]=None, phone_number: Optional[str]=None, email: Optional[str]=None, files: Optional[Sequence[PassportFile]]=None, front_side: Optional[PassportFile]=None, reverse_side: Optional[PassportFile]=None, selfie: Optional[PassportFile]=None, translation: Optional[Sequence[PassportFile]]=None, credentials: Optional['Credentials']=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_kwargs=api_kwargs)
        self.type: str = type
        self.data: Optional[Union[PersonalDetails, IdDocumentData, ResidentialAddress]] = data
        self.phone_number: Optional[str] = phone_number
        self.email: Optional[str] = email
        self.files: Tuple[PassportFile, ...] = parse_sequence_arg(files)
        self.front_side: Optional[PassportFile] = front_side
        self.reverse_side: Optional[PassportFile] = reverse_side
        self.selfie: Optional[PassportFile] = selfie
        self.translation: Tuple[PassportFile, ...] = parse_sequence_arg(translation)
        self.hash: str = hash
        self._id_attrs = (self.type, self.data, self.phone_number, self.email, self.files, self.front_side, self.reverse_side, self.selfie)
        self._freeze()

    @classmethod
    def de_json(cls, data: Optional[JSONDict], bot: 'Bot') -> Optional['EncryptedPassportElement']:
        if False:
            print('Hello World!')
        'See :meth:`telegram.TelegramObject.de_json`.'
        data = cls._parse_data(data)
        if not data:
            return None
        data['files'] = PassportFile.de_list(data.get('files'), bot) or None
        data['front_side'] = PassportFile.de_json(data.get('front_side'), bot)
        data['reverse_side'] = PassportFile.de_json(data.get('reverse_side'), bot)
        data['selfie'] = PassportFile.de_json(data.get('selfie'), bot)
        data['translation'] = PassportFile.de_list(data.get('translation'), bot) or None
        return super().de_json(data=data, bot=bot)

    @classmethod
    def de_json_decrypted(cls, data: Optional[JSONDict], bot: 'Bot', credentials: 'Credentials') -> Optional['EncryptedPassportElement']:
        if False:
            return 10
        'Variant of :meth:`telegram.TelegramObject.de_json` that also takes into account\n        passport credentials.\n\n        Args:\n            data (Dict[:obj:`str`, ...]): The JSON data.\n            bot (:class:`telegram.Bot`): The bot associated with this object.\n            credentials (:class:`telegram.FileCredentials`): The credentials\n\n        Returns:\n            :class:`telegram.EncryptedPassportElement`:\n\n        '
        if not data:
            return None
        if data['type'] not in ('phone_number', 'email'):
            secure_data = getattr(credentials.secure_data, data['type'])
            if secure_data.data is not None:
                if not isinstance(data['data'], dict):
                    data['data'] = decrypt_json(b64decode(secure_data.data.secret), b64decode(secure_data.data.hash), b64decode(data['data']))
                if data['type'] == 'personal_details':
                    data['data'] = PersonalDetails.de_json(data['data'], bot=bot)
                elif data['type'] in ('passport', 'internal_passport', 'driver_license', 'identity_card'):
                    data['data'] = IdDocumentData.de_json(data['data'], bot=bot)
                elif data['type'] == 'address':
                    data['data'] = ResidentialAddress.de_json(data['data'], bot=bot)
            data['files'] = PassportFile.de_list_decrypted(data.get('files'), bot, secure_data.files) or None
            data['front_side'] = PassportFile.de_json_decrypted(data.get('front_side'), bot, secure_data.front_side)
            data['reverse_side'] = PassportFile.de_json_decrypted(data.get('reverse_side'), bot, secure_data.reverse_side)
            data['selfie'] = PassportFile.de_json_decrypted(data.get('selfie'), bot, secure_data.selfie)
            data['translation'] = PassportFile.de_list_decrypted(data.get('translation'), bot, secure_data.translation) or None
        return super().de_json(data=data, bot=bot)