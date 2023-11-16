"""This module contains a class that describes a single parameter of a request to the Bot API."""
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple, final
from telegram._files.inputfile import InputFile
from telegram._files.inputmedia import InputMedia
from telegram._files.inputsticker import InputSticker
from telegram._telegramobject import TelegramObject
from telegram._utils.datetime import to_timestamp
from telegram._utils.enum import StringEnum
from telegram._utils.types import UploadFileDict

@final
@dataclass(repr=True, eq=False, order=False, frozen=True)
class RequestParameter:
    """Instances of this class represent a single parameter to be sent along with a request to
    the Bot API.

    .. versionadded:: 20.0

    Warning:
        This class intended is to be used internally by the library and *not* by the user. Changes
        to this class are not considered breaking changes and may not be documented in the
        changelog.

    Args:
        name (:obj:`str`): The name of the parameter.
        value (:obj:`object` | :obj:`None`): The value of the parameter. Must be JSON-dumpable.
        input_files (List[:class:`telegram.InputFile`], optional): A list of files that should be
            uploaded along with this parameter.

    Attributes:
        name (:obj:`str`): The name of the parameter.
        value (:obj:`object` | :obj:`None`): The value of the parameter.
        input_files (List[:class:`telegram.InputFile` | :obj:`None`): A list of files that should
            be uploaded along with this parameter.
    """
    __slots__ = ('name', 'value', 'input_files')
    name: str
    value: object
    input_files: Optional[List[InputFile]]

    @property
    def json_value(self) -> Optional[str]:
        if False:
            return 10
        'The JSON dumped :attr:`value` or :obj:`None` if :attr:`value` is :obj:`None`.\n        The latter can currently only happen if :attr:`input_files` has exactly one element that\n        must not be uploaded via an attach:// URI.\n        '
        if isinstance(self.value, str):
            return self.value
        if self.value is None:
            return None
        return json.dumps(self.value)

    @property
    def multipart_data(self) -> Optional[UploadFileDict]:
        if False:
            return 10
        'A dict with the file data to upload, if any.'
        if not self.input_files:
            return None
        return {input_file.attach_name or self.name: input_file.field_tuple for input_file in self.input_files}

    @staticmethod
    def _value_and_input_files_from_input(value: object) -> Tuple[object, List[InputFile]]:
        if False:
            for i in range(10):
                print('nop')
        "Converts `value` into something that we can json-dump. Returns two values:\n        1. the JSON-dumpable value. Maybe be `None` in case the value is an InputFile which must\n           not be uploaded via an attach:// URI\n        2. A list of InputFiles that should be uploaded for this value\n\n        Note that we handle files differently depending on whether attaching them via an URI of the\n        form attach://<name> is documented to be allowed or not.\n        There was some confusion whether this worked for all files, so that we stick to the\n        documented ways for now.\n        See https://github.com/tdlib/telegram-bot-api/issues/167 and\n        https://github.com/tdlib/telegram-bot-api/issues/259\n\n        This method only does some special casing for our own helper class StringEnum, but not\n        for general enums. This is because:\n        * tg.constants currently only uses IntEnum as second enum type and json dumping that\n          is no problem\n        * if a user passes a custom enum, it's unlikely that we can actually properly handle it\n          even with some special casing.\n        "
        if isinstance(value, datetime):
            return (to_timestamp(value), [])
        if isinstance(value, StringEnum):
            return (value.value, [])
        if isinstance(value, InputFile):
            if value.attach_uri:
                return (value.attach_uri, [value])
            return (None, [value])
        if isinstance(value, InputMedia) and isinstance(value.media, InputFile):
            data = value.to_dict()
            if value.media.attach_uri:
                data['media'] = value.media.attach_uri
            else:
                data.pop('media', None)
            thumbnail = data.get('thumbnail', None)
            if isinstance(thumbnail, InputFile):
                if thumbnail.attach_uri:
                    data['thumbnail'] = thumbnail.attach_uri
                else:
                    data.pop('thumbnail', None)
                return (data, [value.media, thumbnail])
            return (data, [value.media])
        if isinstance(value, InputSticker) and isinstance(value.sticker, InputFile):
            data = value.to_dict()
            data['sticker'] = value.sticker.attach_uri
            return (data, [value.sticker])
        if isinstance(value, TelegramObject):
            return (value.to_dict(), [])
        return (value, [])

    @classmethod
    def from_input(cls, key: str, value: object) -> 'RequestParameter':
        if False:
            while True:
                i = 10
        'Builds an instance of this class for a given key-value pair that represents the raw\n        input as passed along from a method of :class:`telegram.Bot`.\n        '
        if not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
            param_values = []
            input_files = []
            for obj in value:
                (param_value, input_file) = cls._value_and_input_files_from_input(obj)
                if param_value is not None:
                    param_values.append(param_value)
                input_files.extend(input_file)
            return RequestParameter(name=key, value=param_values, input_files=input_files if input_files else None)
        (param_value, input_files) = cls._value_and_input_files_from_input(value)
        return RequestParameter(name=key, value=param_value, input_files=input_files if input_files else None)