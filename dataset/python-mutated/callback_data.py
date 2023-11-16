from __future__ import annotations
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Literal, Optional, Type, TypeVar, Union
from uuid import UUID
from magic_filter import MagicFilter
from pydantic import BaseModel
from aiogram.filters.base import Filter
from aiogram.types import CallbackQuery
T = TypeVar('T', bound='CallbackData')
MAX_CALLBACK_LENGTH: int = 64

class CallbackDataException(Exception):
    pass

class CallbackData(BaseModel):
    """
    Base class for callback data wrapper

    This class should be used as super-class of user-defined callbacks.

    The class-keyword :code:`prefix` is required to define prefix
    and also the argument :code:`sep` can be passed to define separator (default is :code:`:`).
    """
    if TYPE_CHECKING:
        __separator__: ClassVar[str]
        'Data separator (default is :code:`:`)'
        __prefix__: ClassVar[str]
        'Callback prefix'

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        if 'prefix' not in kwargs:
            raise ValueError(f"prefix required, usage example: `class {cls.__name__}(CallbackData, prefix='my_callback'): ...`")
        cls.__separator__ = kwargs.pop('sep', ':')
        cls.__prefix__ = kwargs.pop('prefix')
        if cls.__separator__ in cls.__prefix__:
            raise ValueError(f'Separator symbol {cls.__separator__!r} can not be used inside prefix {cls.__prefix__!r}')
        super().__init_subclass__(**kwargs)

    def _encode_value(self, key: str, value: Any) -> str:
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            return ''
        if isinstance(value, Enum):
            return str(value.value)
        if isinstance(value, UUID):
            return value.hex
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, (int, str, float, Decimal, Fraction)):
            return str(value)
        raise ValueError(f'Attribute {key}={value!r} of type {type(value).__name__!r} can not be packed to callback data')

    def pack(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Generate callback data string\n\n        :return: valid callback data for Telegram Bot API\n        '
        result = [self.__prefix__]
        for (key, value) in self.model_dump(mode='json').items():
            encoded = self._encode_value(key, value)
            if self.__separator__ in encoded:
                raise ValueError(f'Separator symbol {self.__separator__!r} can not be used in value {key}={encoded!r}')
            result.append(encoded)
        callback_data = self.__separator__.join(result)
        if len(callback_data.encode()) > MAX_CALLBACK_LENGTH:
            raise ValueError(f'Resulted callback data is too long! len({callback_data!r}.encode()) > {MAX_CALLBACK_LENGTH}')
        return callback_data

    @classmethod
    def unpack(cls: Type[T], value: str) -> T:
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse callback data string\n\n        :param value: value from Telegram\n        :return: instance of CallbackData\n        '
        (prefix, *parts) = value.split(cls.__separator__)
        names = cls.model_fields.keys()
        if len(parts) != len(names):
            raise TypeError(f'Callback data {cls.__name__!r} takes {len(names)} arguments but {len(parts)} were given')
        if prefix != cls.__prefix__:
            raise ValueError(f'Bad prefix ({prefix!r} != {cls.__prefix__!r})')
        payload = {}
        for (k, v) in zip(names, parts):
            if (field := cls.model_fields.get(k)):
                if v == '' and (not field.is_required()):
                    v = None
            payload[k] = v
        return cls(**payload)

    @classmethod
    def filter(cls, rule: Optional[MagicFilter]=None) -> CallbackQueryFilter:
        if False:
            return 10
        '\n        Generates a filter for callback query with rule\n\n        :param rule: magic rule\n        :return: instance of filter\n        '
        return CallbackQueryFilter(callback_data=cls, rule=rule)

class CallbackQueryFilter(Filter):
    """
    This filter helps to handle callback query.

    Should not be used directly, you should create the instance of this filter
    via callback data instance
    """
    __slots__ = ('callback_data', 'rule')

    def __init__(self, *, callback_data: Type[CallbackData], rule: Optional[MagicFilter]=None):
        if False:
            i = 10
            return i + 15
        '\n        :param callback_data: Expected type of callback data\n        :param rule: Magic rule\n        '
        self.callback_data = callback_data
        self.rule = rule

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._signature_to_string(callback_data=self.callback_data, rule=self.rule)

    async def __call__(self, query: CallbackQuery) -> Union[Literal[False], Dict[str, Any]]:
        if not isinstance(query, CallbackQuery) or not query.data:
            return False
        try:
            callback_data = self.callback_data.unpack(query.data)
        except (TypeError, ValueError):
            return False
        if self.rule is None or self.rule.resolve(callback_data):
            return {'callback_data': callback_data}
        return False