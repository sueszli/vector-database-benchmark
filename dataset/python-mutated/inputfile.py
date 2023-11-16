"""This module contains an object that represents a Telegram InputFile."""
import mimetypes
from typing import IO, Optional, Union
from uuid import uuid4
from telegram._utils.files import load_file
from telegram._utils.types import FieldTuple
_DEFAULT_MIME_TYPE = 'application/octet-stream'

class InputFile:
    """This object represents a Telegram InputFile.

    .. versionchanged:: 20.0

        * The former attribute ``attach`` was renamed to :attr:`attach_name`.
        * Method ``is_image`` was removed. If you pass :obj:`bytes` to :paramref:`obj` and would
          like to have the mime type automatically guessed, please pass :paramref:`filename`
          in addition.

    Args:
        obj (:term:`file object` | :obj:`bytes` | :obj:`str`): An open file descriptor or the files
            content as bytes or string.

            Note:
                If :paramref:`obj` is a string, it will be encoded as bytes via
                :external:obj:`obj.encode('utf-8') <str.encode>`.

            .. versionchanged:: 20.0
                Accept string input.
        filename (:obj:`str`, optional): Filename for this InputFile.
        attach (:obj:`bool`, optional): Pass :obj:`True` if the parameter this file belongs to in
            the request to Telegram should point to the multipart data via an ``attach://`` URI.
            Defaults to `False`.

    Attributes:
        input_file_content (:obj:`bytes`): The binary content of the file to send.
        attach_name (:obj:`str`): Optional. If present, the parameter this file belongs to in
            the request to Telegram should point to the multipart data via a an URI of the form
            ``attach://<attach_name>`` URI.
        filename (:obj:`str`): Filename for the file to be sent.
        mimetype (:obj:`str`): The mimetype inferred from the file to be sent.

    """
    __slots__ = ('filename', 'attach_name', 'input_file_content', 'mimetype')

    def __init__(self, obj: Union[IO[bytes], bytes, str], filename: Optional[str]=None, attach: bool=False):
        if False:
            print('Hello World!')
        if isinstance(obj, bytes):
            self.input_file_content: bytes = obj
        elif isinstance(obj, str):
            self.input_file_content = obj.encode('utf-8')
        else:
            (reported_filename, self.input_file_content) = load_file(obj)
            filename = filename or reported_filename
        self.attach_name: Optional[str] = 'attached' + uuid4().hex if attach else None
        if filename:
            self.mimetype: str = mimetypes.guess_type(filename, strict=False)[0] or _DEFAULT_MIME_TYPE
        else:
            self.mimetype = _DEFAULT_MIME_TYPE
        self.filename: str = filename or self.mimetype.replace('/', '.')

    @property
    def field_tuple(self) -> FieldTuple:
        if False:
            for i in range(10):
                print('nop')
        'Field tuple representing the contents of the file for upload to the Telegram servers.\n\n        Returns:\n            Tuple[:obj:`str`, :obj:`bytes`, :obj:`str`]:\n        '
        return (self.filename, self.input_file_content, self.mimetype)

    @property
    def attach_uri(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        'URI to insert into the JSON data for uploading the file. Returns :obj:`None`, if\n        :attr:`attach_name` is :obj:`None`.\n        '
        return f'attach://{self.attach_name}' if self.attach_name else None