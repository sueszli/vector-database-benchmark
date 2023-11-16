from __future__ import annotations
from typing import TYPE_CHECKING, Any
from ..types import File
from .base import TelegramMethod

class GetFile(TelegramMethod[File]):
    """
    Use this method to get basic information about a file and prepare it for downloading. For the moment, bots can download files of up to 20MB in size. On success, a :class:`aiogram.types.file.File` object is returned. The file can then be downloaded via the link :code:`https://api.telegram.org/file/bot<token>/<file_path>`, where :code:`<file_path>` is taken from the response. It is guaranteed that the link will be valid for at least 1 hour. When the link expires, a new one can be requested by calling :class:`aiogram.methods.get_file.GetFile` again.
    **Note:** This function may not preserve the original file name and MIME type. You should save the file's MIME type and name (if available) when the File object is received.

    Source: https://core.telegram.org/bots/api#getfile
    """
    __returning__ = File
    __api_method__ = 'getFile'
    file_id: str
    'File identifier to get information about'
    if TYPE_CHECKING:

        def __init__(__pydantic__self__, *, file_id: str, **__pydantic_kwargs: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(file_id=file_id, **__pydantic_kwargs)