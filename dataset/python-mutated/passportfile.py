"""This module contains an object that represents a Encrypted PassportFile."""
from typing import TYPE_CHECKING, List, Optional, Tuple
from telegram._telegramobject import TelegramObject
from telegram._utils.defaultvalue import DEFAULT_NONE
from telegram._utils.types import JSONDict, ODVInput
from telegram._utils.warnings import warn
from telegram.warnings import PTBDeprecationWarning
if TYPE_CHECKING:
    from telegram import Bot, File, FileCredentials

class PassportFile(TelegramObject):
    """
    This object represents a file uploaded to Telegram Passport. Currently all Telegram Passport
    files are in JPEG format when decrypted and don't exceed 10MB.

    Objects of this class are comparable in terms of equality. Two objects of this class are
    considered equal, if their :attr:`file_unique_id` is equal.

    Args:
        file_id (:obj:`str`): Identifier for this file, which can be used to download
            or reuse the file.
        file_unique_id (:obj:`str`): Unique identifier for this file, which
            is supposed to be the same over time and for different bots.
            Can't be used to download or reuse the file.
        file_size (:obj:`int`): File size in bytes.
        file_date (:obj:`int`): Unix time when the file was uploaded.

            .. deprecated:: 20.6
                This argument will only accept a datetime instead of an integer in future
                major versions.

    Attributes:
        file_id (:obj:`str`): Identifier for this file, which can be used to download
            or reuse the file.
        file_unique_id (:obj:`str`): Unique identifier for this file, which
            is supposed to be the same over time and for different bots.
            Can't be used to download or reuse the file.
        file_size (:obj:`int`): File size in bytes.
    """
    __slots__ = ('_file_date', 'file_id', 'file_size', '_credentials', 'file_unique_id')

    def __init__(self, file_id: str, file_unique_id: str, file_date: int, file_size: int, credentials: Optional['FileCredentials']=None, *, api_kwargs: Optional[JSONDict]=None):
        if False:
            print('Hello World!')
        super().__init__(api_kwargs=api_kwargs)
        self.file_id: str = file_id
        self.file_unique_id: str = file_unique_id
        self.file_size: int = file_size
        self._file_date: int = file_date
        self._credentials: Optional[FileCredentials] = credentials
        self._id_attrs = (self.file_unique_id,)
        self._freeze()

    def to_dict(self, recursive: bool=True) -> JSONDict:
        if False:
            i = 10
            return i + 15
        'See :meth:`telegram.TelegramObject.to_dict` for details.'
        data = super().to_dict(recursive)
        data['file_date'] = self._file_date
        return data

    @property
    def file_date(self) -> int:
        if False:
            return 10
        ':obj:`int`: Unix time when the file was uploaded.\n\n        .. deprecated:: 20.6\n            This attribute will return a datetime instead of a integer in future major versions.\n        '
        warn('The attribute `file_date` will return a datetime instead of an integer in future major versions.', PTBDeprecationWarning, stacklevel=2)
        return self._file_date

    @classmethod
    def de_json_decrypted(cls, data: Optional[JSONDict], bot: 'Bot', credentials: 'FileCredentials') -> Optional['PassportFile']:
        if False:
            i = 10
            return i + 15
        'Variant of :meth:`telegram.TelegramObject.de_json` that also takes into account\n        passport credentials.\n\n        Args:\n            data (Dict[:obj:`str`, ...]): The JSON data.\n            bot (:class:`telegram.Bot`): The bot associated with this object.\n            credentials (:class:`telegram.FileCredentials`): The credentials\n\n        Returns:\n            :class:`telegram.PassportFile`:\n\n        '
        data = cls._parse_data(data)
        if not data:
            return None
        data['credentials'] = credentials
        return super().de_json(data=data, bot=bot)

    @classmethod
    def de_list_decrypted(cls, data: Optional[List[JSONDict]], bot: 'Bot', credentials: List['FileCredentials']) -> Tuple[Optional['PassportFile'], ...]:
        if False:
            print('Hello World!')
        'Variant of :meth:`telegram.TelegramObject.de_list` that also takes into account\n        passport credentials.\n\n        .. versionchanged:: 20.0\n\n           * Returns a tuple instead of a list.\n           * Filters out any :obj:`None` values\n\n        Args:\n            data (List[Dict[:obj:`str`, ...]]): The JSON data.\n            bot (:class:`telegram.Bot`): The bot associated with these objects.\n            credentials (:class:`telegram.FileCredentials`): The credentials\n\n        Returns:\n            Tuple[:class:`telegram.PassportFile`]:\n\n        '
        if not data:
            return ()
        return tuple((obj for obj in (cls.de_json_decrypted(passport_file, bot, credentials[i]) for (i, passport_file) in enumerate(data)) if obj is not None))

    async def get_file(self, *, read_timeout: ODVInput[float]=DEFAULT_NONE, write_timeout: ODVInput[float]=DEFAULT_NONE, connect_timeout: ODVInput[float]=DEFAULT_NONE, pool_timeout: ODVInput[float]=DEFAULT_NONE, api_kwargs: Optional[JSONDict]=None) -> 'File':
        """
        Wrapper over :meth:`telegram.Bot.get_file`. Will automatically assign the correct
        credentials to the returned :class:`telegram.File` if originating from
        :obj:`telegram.PassportData.decrypted_data`.

        For the documentation of the arguments, please see :meth:`telegram.Bot.get_file`.

        Returns:
            :class:`telegram.File`

        Raises:
            :class:`telegram.error.TelegramError`

        """
        file = await self.get_bot().get_file(file_id=self.file_id, read_timeout=read_timeout, write_timeout=write_timeout, connect_timeout=connect_timeout, pool_timeout=pool_timeout, api_kwargs=api_kwargs)
        file.set_credentials(self._credentials)
        return file