"""This module contains the CallbackDataCache class."""
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, MutableMapping, Optional, Tuple, Union, cast
from uuid import uuid4
try:
    from cachetools import LRUCache
    CACHE_TOOLS_AVAILABLE = True
except ImportError:
    CACHE_TOOLS_AVAILABLE = False
from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, User
from telegram._utils.datetime import to_float_timestamp
from telegram.error import TelegramError
from telegram.ext._utils.types import CDCData
if TYPE_CHECKING:
    from telegram.ext import ExtBot

class InvalidCallbackData(TelegramError):
    """
    Raised when the received callback data has been tampered with or deleted from cache.

    Examples:
        :any:`Arbitrary Callback Data Bot <examples.arbitrarycallbackdatabot>`

    .. seealso:: :wiki:`Arbitrary callback_data <Arbitrary-callback_data>`

    .. versionadded:: 13.6

    Args:
        callback_data (:obj:`int`, optional): The button data of which the callback data could not
            be found.

    Attributes:
        callback_data (:obj:`int`): Optional. The button data of which the callback data could not
            be found.
    """
    __slots__ = ('callback_data',)

    def __init__(self, callback_data: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__('The object belonging to this callback_data was deleted or the callback_data was manipulated.')
        self.callback_data: Optional[str] = callback_data

    def __reduce__(self) -> Tuple[type, Tuple[Optional[str]]]:
        if False:
            print('Hello World!')
        return (self.__class__, (self.callback_data,))

class _KeyboardData:
    __slots__ = ('keyboard_uuid', 'button_data', 'access_time')

    def __init__(self, keyboard_uuid: str, access_time: Optional[float]=None, button_data: Optional[Dict[str, object]]=None):
        if False:
            for i in range(10):
                print('nop')
        self.keyboard_uuid = keyboard_uuid
        self.button_data = button_data or {}
        self.access_time = access_time or time.time()

    def update_access_time(self) -> None:
        if False:
            return 10
        'Updates the access time with the current time.'
        self.access_time = time.time()

    def to_tuple(self) -> Tuple[str, float, Dict[str, object]]:
        if False:
            print('Hello World!')
        'Gives a tuple representation consisting of the keyboard uuid, the access time and the\n        button data.\n        '
        return (self.keyboard_uuid, self.access_time, self.button_data)

class CallbackDataCache:
    """A custom cache for storing the callback data of a :class:`telegram.ext.ExtBot`. Internally,
    it keeps two mappings with fixed maximum size:

    * One for mapping the data received in callback queries to the cached objects
    * One for mapping the IDs of received callback queries to the cached objects

    The second mapping allows to manually drop data that has been cached for keyboards of messages
    sent via inline mode.
    If necessary, will drop the least recently used items.

    Important:
        If you want to use this class, you must install PTB with the optional requirement
        ``callback-data``, i.e.

        .. code-block:: bash

           pip install "python-telegram-bot[callback-data]"

    Examples:
        :any:`Arbitrary Callback Data Bot <examples.arbitrarycallbackdatabot>`

    .. seealso:: :wiki:`Architecture Overview <Architecture>`,
        :wiki:`Arbitrary callback_data <Arbitrary-callback_data>`

    .. versionadded:: 13.6

    .. versionchanged:: 20.0
        To use this class, PTB must be installed via
        ``pip install "python-telegram-bot[callback-data]"``.

    Args:
        bot (:class:`telegram.ext.ExtBot`): The bot this cache is for.
        maxsize (:obj:`int`, optional): Maximum number of items in each of the internal mappings.
            Defaults to ``1024``.

        persistent_data (Tuple[List[Tuple[:obj:`str`, :obj:`float`,         Dict[:obj:`str`, :class:`object`]]], Dict[:obj:`str`, :obj:`str`]], optional):         Data to initialize the cache with, as returned by         :meth:`telegram.ext.BasePersistence.get_callback_data`.

    Attributes:
        bot (:class:`telegram.ext.ExtBot`): The bot this cache is for.

    """
    __slots__ = ('bot', '_maxsize', '_keyboard_data', '_callback_queries')

    def __init__(self, bot: 'ExtBot[Any]', maxsize: int=1024, persistent_data: Optional[CDCData]=None):
        if False:
            while True:
                i = 10
        if not CACHE_TOOLS_AVAILABLE:
            raise RuntimeError('To use `CallbackDataCache`, PTB must be installed via `pip install "python-telegram-bot[callback-data]"`.')
        self.bot: ExtBot[Any] = bot
        self._maxsize: int = maxsize
        self._keyboard_data: MutableMapping[str, _KeyboardData] = LRUCache(maxsize=maxsize)
        self._callback_queries: MutableMapping[str, str] = LRUCache(maxsize=maxsize)
        if persistent_data:
            self.load_persistence_data(persistent_data)

    def load_persistence_data(self, persistent_data: CDCData) -> None:
        if False:
            i = 10
            return i + 15
        'Loads data into the cache.\n\n        Warning:\n            This method is not intended to be called by users directly.\n\n        .. versionadded:: 20.0\n\n        Args:\n            persistent_data (Tuple[List[Tuple[:obj:`str`, :obj:`float`,             Dict[:obj:`str`, :class:`object`]]], Dict[:obj:`str`, :obj:`str`]], optional):             Data to load, as returned by             :meth:`telegram.ext.BasePersistence.get_callback_data`.\n        '
        (keyboard_data, callback_queries) = persistent_data
        for (key, value) in callback_queries.items():
            self._callback_queries[key] = value
        for (uuid, access_time, data) in keyboard_data:
            self._keyboard_data[uuid] = _KeyboardData(keyboard_uuid=uuid, access_time=access_time, button_data=data)

    @property
    def maxsize(self) -> int:
        if False:
            while True:
                i = 10
        ':obj:`int`: The maximum size of the cache.\n\n        .. versionchanged:: 20.0\n           This property is now read-only.\n        '
        return self._maxsize

    @property
    def persistence_data(self) -> CDCData:
        if False:
            for i in range(10):
                print('nop')
        'Tuple[List[Tuple[:obj:`str`, :obj:`float`, Dict[:obj:`str`, :class:`object`]]],\n        Dict[:obj:`str`, :obj:`str`]]: The data that needs to be persisted to allow\n        caching callback data across bot reboots.\n        '
        return ([data.to_tuple() for data in self._keyboard_data.values()], dict(self._callback_queries.items()))

    def process_keyboard(self, reply_markup: InlineKeyboardMarkup) -> InlineKeyboardMarkup:
        if False:
            for i in range(10):
                print('nop')
        'Registers the reply markup to the cache. If any of the buttons have\n        :attr:`~telegram.InlineKeyboardButton.callback_data`, stores that data and builds a new\n        keyboard with the correspondingly replaced buttons. Otherwise, does nothing and returns\n        the original reply markup.\n\n        Args:\n            reply_markup (:class:`telegram.InlineKeyboardMarkup`): The keyboard.\n\n        Returns:\n            :class:`telegram.InlineKeyboardMarkup`: The keyboard to be passed to Telegram.\n\n        '
        keyboard_uuid = uuid4().hex
        keyboard_data = _KeyboardData(keyboard_uuid)
        buttons = [[InlineKeyboardButton(btn.text, callback_data=self.__put_button(btn.callback_data, keyboard_data)) if btn.callback_data else btn for btn in column] for column in reply_markup.inline_keyboard]
        if not keyboard_data.button_data:
            return reply_markup
        self._keyboard_data[keyboard_uuid] = keyboard_data
        return InlineKeyboardMarkup(buttons)

    @staticmethod
    def __put_button(callback_data: object, keyboard_data: _KeyboardData) -> str:
        if False:
            return 10
        'Stores the data for a single button in :attr:`keyboard_data`.\n        Returns the string that should be passed instead of the callback_data, which is\n        ``keyboard_uuid + button_uuids``.\n        '
        uuid = uuid4().hex
        keyboard_data.button_data[uuid] = callback_data
        return f'{keyboard_data.keyboard_uuid}{uuid}'

    def __get_keyboard_uuid_and_button_data(self, callback_data: str) -> Union[Tuple[str, object], Tuple[None, InvalidCallbackData]]:
        if False:
            i = 10
            return i + 15
        (keyboard, button) = self.extract_uuids(callback_data)
        try:
            keyboard_data = self._keyboard_data[keyboard]
            button_data = keyboard_data.button_data[button]
            keyboard_data.update_access_time()
            return (keyboard, button_data)
        except KeyError:
            return (None, InvalidCallbackData(callback_data))

    @staticmethod
    def extract_uuids(callback_data: str) -> Tuple[str, str]:
        if False:
            i = 10
            return i + 15
        'Extracts the keyboard uuid and the button uuid from the given :paramref:`callback_data`.\n\n        Args:\n            callback_data (:obj:`str`): The\n                :paramref:`~telegram.InlineKeyboardButton.callback_data` as present in the button.\n\n        Returns:\n            (:obj:`str`, :obj:`str`): Tuple of keyboard and button uuid\n\n        '
        return (callback_data[:32], callback_data[32:])

    def process_message(self, message: Message) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Replaces the data in the inline keyboard attached to the message with the cached\n        objects, if necessary. If the data could not be found,\n        :class:`telegram.ext.InvalidCallbackData` will be inserted.\n\n        Note:\n            Checks :attr:`telegram.Message.via_bot` and :attr:`telegram.Message.from_user` to check\n            if the reply markup (if any) was actually sent by this cache's bot. If it was not, the\n            message will be returned unchanged.\n\n            Note that this will fail for channel posts, as :attr:`telegram.Message.from_user` is\n            :obj:`None` for those! In the corresponding reply markups the callback data will be\n            replaced by :class:`telegram.ext.InvalidCallbackData`.\n\n        Warning:\n            * Does *not* consider :attr:`telegram.Message.reply_to_message` and\n              :attr:`telegram.Message.pinned_message`. Pass them to this method separately.\n            * *In place*, i.e. the passed :class:`telegram.Message` will be changed!\n\n        Args:\n            message (:class:`telegram.Message`): The message.\n\n        "
        self.__process_message(message)

    def __process_message(self, message: Message) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'As documented in process_message, but returns the uuid of the attached keyboard, if any,\n        which is relevant for process_callback_query.\n\n        **IN PLACE**\n        '
        if not message.reply_markup:
            return None
        if message.via_bot:
            sender: Optional[User] = message.via_bot
        elif message.from_user:
            sender = message.from_user
        else:
            sender = None
        if sender is not None and sender != self.bot.bot:
            return None
        keyboard_uuid = None
        for row in message.reply_markup.inline_keyboard:
            for button in row:
                if button.callback_data:
                    button_data = cast(str, button.callback_data)
                    (keyboard_id, callback_data) = self.__get_keyboard_uuid_and_button_data(button_data)
                    button.update_callback_data(callback_data)
                    if not keyboard_uuid and (not isinstance(callback_data, InvalidCallbackData)):
                        keyboard_uuid = keyboard_id
        return keyboard_uuid

    def process_callback_query(self, callback_query: CallbackQuery) -> None:
        if False:
            print('Hello World!')
        'Replaces the data in the callback query and the attached messages keyboard with the\n        cached objects, if necessary. If the data could not be found,\n        :class:`telegram.ext.InvalidCallbackData` will be inserted.\n        If :attr:`telegram.CallbackQuery.data` or :attr:`telegram.CallbackQuery.message` is\n        present, this also saves the callback queries ID in order to be able to resolve it to the\n        stored data.\n\n        Note:\n            Also considers inserts data into the buttons of\n            :attr:`telegram.Message.reply_to_message` and :attr:`telegram.Message.pinned_message`\n            if necessary.\n\n        Warning:\n            *In place*, i.e. the passed :class:`telegram.CallbackQuery` will be changed!\n\n        Args:\n            callback_query (:class:`telegram.CallbackQuery`): The callback query.\n\n        '
        mapped = False
        if callback_query.data:
            data = callback_query.data
            (keyboard_uuid, button_data) = self.__get_keyboard_uuid_and_button_data(data)
            with callback_query._unfrozen():
                callback_query.data = button_data
            if not mapped and (not isinstance(button_data, InvalidCallbackData)):
                self._callback_queries[callback_query.id] = keyboard_uuid
                mapped = True
        if callback_query.message:
            self.__process_message(callback_query.message)
            for message in (callback_query.message.pinned_message, callback_query.message.reply_to_message):
                if message:
                    self.__process_message(message)

    def drop_data(self, callback_query: CallbackQuery) -> None:
        if False:
            while True:
                i = 10
        'Deletes the data for the specified callback query.\n\n        Note:\n            Will *not* raise exceptions in case the callback data is not found in the cache.\n            *Will* raise :exc:`KeyError` in case the callback query can not be found in the\n            cache.\n\n        Args:\n            callback_query (:class:`telegram.CallbackQuery`): The callback query.\n\n        Raises:\n            KeyError: If the callback query can not be found in the cache\n        '
        try:
            keyboard_uuid = self._callback_queries.pop(callback_query.id)
            self.__drop_keyboard(keyboard_uuid)
        except KeyError as exc:
            raise KeyError('CallbackQuery was not found in cache.') from exc

    def __drop_keyboard(self, keyboard_uuid: str) -> None:
        if False:
            print('Hello World!')
        try:
            self._keyboard_data.pop(keyboard_uuid)
        except KeyError:
            return

    def clear_callback_data(self, time_cutoff: Optional[Union[float, datetime]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Clears the stored callback data.\n\n        Args:\n            time_cutoff (:obj:`float` | :obj:`datetime.datetime`, optional): Pass a UNIX timestamp\n                or a :obj:`datetime.datetime` to clear only entries which are older.\n                For timezone naive :obj:`datetime.datetime` objects, the default timezone of the\n                bot will be used, which is UTC unless :attr:`telegram.ext.Defaults.tzinfo` is\n                used.\n\n        '
        self.__clear(self._keyboard_data, time_cutoff=time_cutoff)

    def clear_callback_queries(self) -> None:
        if False:
            return 10
        'Clears the stored callback query IDs.'
        self.__clear(self._callback_queries)

    def __clear(self, mapping: MutableMapping, time_cutoff: Optional[Union[float, datetime]]=None) -> None:
        if False:
            while True:
                i = 10
        if not time_cutoff:
            mapping.clear()
            return
        if isinstance(time_cutoff, datetime):
            effective_cutoff = to_float_timestamp(time_cutoff, tzinfo=self.bot.defaults.tzinfo if self.bot.defaults else None)
        else:
            effective_cutoff = time_cutoff
        to_drop = [key for (key, data) in mapping.items() if data.access_time < effective_cutoff]
        for key in to_drop:
            mapping.pop(key)