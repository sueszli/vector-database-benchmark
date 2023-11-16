"""This module contains the Application class."""
import asyncio
import contextlib
import inspect
import itertools
import platform
import signal
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, AsyncContextManager, Awaitable, Callable, Coroutine, DefaultDict, Dict, Generator, Generic, List, Mapping, NoReturn, Optional, Sequence, Set, Tuple, Type, TypeVar, Union
from telegram._update import Update
from telegram._utils.defaultvalue import DEFAULT_NONE, DEFAULT_TRUE, DefaultValue
from telegram._utils.logging import get_logger
from telegram._utils.repr import build_repr_with_selected_attrs
from telegram._utils.types import SCT, DVType, ODVInput
from telegram._utils.warnings import warn
from telegram.error import TelegramError
from telegram.ext._basehandler import BaseHandler
from telegram.ext._basepersistence import BasePersistence
from telegram.ext._contexttypes import ContextTypes
from telegram.ext._extbot import ExtBot
from telegram.ext._updater import Updater
from telegram.ext._utils.stack import was_called_by
from telegram.ext._utils.trackingdict import TrackingDict
from telegram.ext._utils.types import BD, BT, CCT, CD, JQ, RT, UD, ConversationKey, HandlerCallback
from telegram.warnings import PTBDeprecationWarning
if TYPE_CHECKING:
    from telegram import Message
    from telegram.ext import ConversationHandler, JobQueue
    from telegram.ext._applicationbuilder import InitApplicationBuilder
    from telegram.ext._baseupdateprocessor import BaseUpdateProcessor
    from telegram.ext._jobqueue import Job
DEFAULT_GROUP: int = 0
_AppType = TypeVar('_AppType', bound='Application')
_STOP_SIGNAL = object()
_DEFAULT_0 = DefaultValue(0)
if sys.version_info >= (3, 12):
    _CoroType = Awaitable[RT]
else:
    _CoroType = Union[Generator['asyncio.Future[object]', None, RT], Awaitable[RT]]
_ErrorCoroType = Optional[_CoroType[RT]]
_LOGGER = get_logger(__name__)

class ApplicationHandlerStop(Exception):
    """
    Raise this in a handler or an error handler to prevent execution of any other handler (even in
    different groups).

    In order to use this exception in a :class:`telegram.ext.ConversationHandler`, pass the
    optional :paramref:`state` parameter instead of returning the next state:

    .. code-block:: python

        async def conversation_callback(update, context):
            ...
            raise ApplicationHandlerStop(next_state)

    Note:
        Has no effect, if the handler or error handler is run in a non-blocking way.

    Args:
        state (:obj:`object`, optional): The next state of the conversation.

    Attributes:
        state (:obj:`object`): Optional. The next state of the conversation.
    """
    __slots__ = ('state',)

    def __init__(self, state: Optional[object]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.state: Optional[object] = state

class Application(Generic[BT, CCT, UD, CD, BD, JQ], AsyncContextManager['Application']):
    """This class dispatches all kinds of updates to its registered handlers, and is the entry
    point to a PTB application.

    Tip:
         This class may not be initialized directly. Use :class:`telegram.ext.ApplicationBuilder`
         or :meth:`builder` (for convenience).

    Instances of this class can be used as asyncio context managers, where

    .. code:: python

        async with application:
            # code

    is roughly equivalent to

    .. code:: python

        try:
            await application.initialize()
            # code
        finally:
            await application.shutdown()

    .. seealso:: :meth:`__aenter__` and :meth:`__aexit__`.

    Examples:
        :any:`Echo Bot <examples.echobot>`

    .. seealso:: :wiki:`Your First Bot <Extensions---Your-first-Bot>`,
        :wiki:`Architecture Overview <Architecture>`

    .. versionchanged:: 20.0

        * Initialization is now done through the :class:`telegram.ext.ApplicationBuilder`.
        * Removed the attribute ``groups``.

    Attributes:
        bot (:class:`telegram.Bot`): The bot object that should be passed to the handlers.
        update_queue (:class:`asyncio.Queue`): The synchronized queue that will contain the
            updates.
        updater (:class:`telegram.ext.Updater`): Optional. The updater used by this application.
        chat_data (:obj:`types.MappingProxyType`): A dictionary handlers can use to store data for
            the chat. For each integer chat id, the corresponding value of this mapping is
            available as :attr:`telegram.ext.CallbackContext.chat_data` in handler callbacks for
            updates from that chat.

            .. versionchanged:: 20.0
                :attr:`chat_data` is now read-only. Note that the values of the mapping are still
                mutable, i.e. editing ``context.chat_data`` within a handler callback is possible
                (and encouraged), but editing the mapping ``application.chat_data`` itself is not.

            .. tip::

                * Manually modifying :attr:`chat_data` is almost never needed and unadvisable.
                * Entries are never deleted automatically from this mapping. If you want to delete
                  the data associated with a specific chat, e.g. if the bot got removed from that
                  chat, please use :meth:`drop_chat_data`.

        user_data (:obj:`types.MappingProxyType`): A dictionary handlers can use to store data for
            the user. For each integer user id, the corresponding value of this mapping is
            available as :attr:`telegram.ext.CallbackContext.user_data` in handler callbacks for
            updates from that user.

            .. versionchanged:: 20.0
                :attr:`user_data` is now read-only. Note that the values of the mapping are still
                mutable, i.e. editing ``context.user_data`` within a handler callback is possible
                (and encouraged), but editing the mapping ``application.user_data`` itself is not.

            .. tip::

               * Manually modifying :attr:`user_data` is almost never needed and unadvisable.
               * Entries are never deleted automatically from this mapping. If you want to delete
                 the data associated with a specific user, e.g. if that user blocked the bot,
                 please use :meth:`drop_user_data`.

        bot_data (:obj:`dict`): A dictionary handlers can use to store data for the bot.
        persistence (:class:`telegram.ext.BasePersistence`): The persistence class to
            store data that should be persistent over restarts.
        handlers (Dict[:obj:`int`, List[:class:`telegram.ext.BaseHandler`]]): A dictionary mapping
            each handler group to the list of handlers registered to that group.

            .. seealso::
                :meth:`add_handler`, :meth:`add_handlers`.
        error_handlers (Dict[:term:`coroutine function`, :obj:`bool`]): A dictionary where the keys
            are error handlers and the values indicate whether they are to be run blocking.

            .. seealso::
                :meth:`add_error_handler`
        context_types (:class:`telegram.ext.ContextTypes`): Specifies the types used by this
            dispatcher for the ``context`` argument of handler and job callbacks.
        post_init (:term:`coroutine function`): Optional. A callback that will be executed by
            :meth:`Application.run_polling` and :meth:`Application.run_webhook` after initializing
            the application via :meth:`initialize`.
        post_shutdown (:term:`coroutine function`): Optional. A callback that will be executed by
            :meth:`Application.run_polling` and :meth:`Application.run_webhook` after shutting down
            the application via :meth:`shutdown`.
        post_stop (:term:`coroutine function`): Optional. A callback that will be executed by
            :meth:`Application.run_polling` and :meth:`Application.run_webhook` after stopping
            the application via :meth:`stop`.

            .. versionadded:: 20.1

    """
    __slots__ = ('__create_task_tasks', '__update_fetcher_task', '__update_persistence_event', '__update_persistence_lock', '__update_persistence_task', '_chat_data', '_chat_ids_to_be_deleted_in_persistence', '_chat_ids_to_be_updated_in_persistence', '_conversation_handler_conversations', '_initialized', '_job_queue', '_running', '_update_processor', '_user_data', '_user_ids_to_be_deleted_in_persistence', '_user_ids_to_be_updated_in_persistence', 'bot', 'bot_data', 'chat_data', 'context_types', 'error_handlers', 'handlers', 'persistence', 'post_init', 'post_shutdown', 'post_stop', 'update_queue', 'updater', 'user_data')

    def __init__(self: 'Application[BT, CCT, UD, CD, BD, JQ]', *, bot: BT, update_queue: 'asyncio.Queue[object]', updater: Optional[Updater], job_queue: JQ, update_processor: 'BaseUpdateProcessor', persistence: Optional[BasePersistence[UD, CD, BD]], context_types: ContextTypes[CCT, UD, CD, BD], post_init: Optional[Callable[['Application[BT, CCT, UD, CD, BD, JQ]'], Coroutine[Any, Any, None]]], post_shutdown: Optional[Callable[['Application[BT, CCT, UD, CD, BD, JQ]'], Coroutine[Any, Any, None]]], post_stop: Optional[Callable[['Application[BT, CCT, UD, CD, BD, JQ]'], Coroutine[Any, Any, None]]]):
        if False:
            while True:
                i = 10
        if not was_called_by(inspect.currentframe(), Path(__file__).parent.resolve() / '_applicationbuilder.py'):
            warn('`Application` instances should be built via the `ApplicationBuilder`.', stacklevel=2)
        self.bot: BT = bot
        self.update_queue: asyncio.Queue[object] = update_queue
        self.context_types: ContextTypes[CCT, UD, CD, BD] = context_types
        self.updater: Optional[Updater] = updater
        self.handlers: Dict[int, List[BaseHandler[Any, CCT]]] = {}
        self.error_handlers: Dict[HandlerCallback[object, CCT, None], Union[bool, DefaultValue[bool]]] = {}
        self.post_init: Optional[Callable[[Application[BT, CCT, UD, CD, BD, JQ]], Coroutine[Any, Any, None]]] = post_init
        self.post_shutdown: Optional[Callable[[Application[BT, CCT, UD, CD, BD, JQ]], Coroutine[Any, Any, None]]] = post_shutdown
        self.post_stop: Optional[Callable[[Application[BT, CCT, UD, CD, BD, JQ]], Coroutine[Any, Any, None]]] = post_stop
        self._update_processor = update_processor
        self.bot_data: BD = self.context_types.bot_data()
        self._user_data: DefaultDict[int, UD] = defaultdict(self.context_types.user_data)
        self._chat_data: DefaultDict[int, CD] = defaultdict(self.context_types.chat_data)
        self.user_data: Mapping[int, UD] = MappingProxyType(self._user_data)
        self.chat_data: Mapping[int, CD] = MappingProxyType(self._chat_data)
        self.persistence: Optional[BasePersistence[UD, CD, BD]] = None
        if persistence and (not isinstance(persistence, BasePersistence)):
            raise TypeError('persistence must be based on telegram.ext.BasePersistence')
        self.persistence = persistence
        self._chat_ids_to_be_updated_in_persistence: Set[int] = set()
        self._user_ids_to_be_updated_in_persistence: Set[int] = set()
        self._chat_ids_to_be_deleted_in_persistence: Set[int] = set()
        self._user_ids_to_be_deleted_in_persistence: Set[int] = set()
        self._conversation_handler_conversations: Dict[str, TrackingDict[ConversationKey, object]] = {}
        self._initialized = False
        self._running = False
        self._job_queue: JQ = job_queue
        self.__update_fetcher_task: Optional[asyncio.Task] = None
        self.__update_persistence_task: Optional[asyncio.Task] = None
        self.__update_persistence_event = asyncio.Event()
        self.__update_persistence_lock = asyncio.Lock()
        self.__create_task_tasks: Set[asyncio.Task] = set()

    async def __aenter__(self: _AppType) -> _AppType:
        """|async_context_manager| :meth:`initializes <initialize>` the App.

        Returns:
            The initialized App instance.

        Raises:
            :exc:`Exception`: If an exception is raised during initialization, :meth:`shutdown`
                is called in this case.
        """
        try:
            await self.initialize()
            return self
        except Exception as exc:
            await self.shutdown()
            raise exc

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        """|async_context_manager| :meth:`shuts down <shutdown>` the App."""
        await self.shutdown()

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        "Give a string representation of the application in the form ``Application[bot=...]``.\n\n        As this class doesn't implement :meth:`object.__str__`, the default implementation\n        will be used, which is equivalent to :meth:`__repr__`.\n\n        Returns:\n            :obj:`str`\n        "
        return build_repr_with_selected_attrs(self, bot=self.bot)

    @property
    def running(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ':obj:`bool`: Indicates if this application is running.\n\n        .. seealso::\n            :meth:`start`, :meth:`stop`\n        '
        return self._running

    @property
    def concurrent_updates(self) -> int:
        if False:
            while True:
                i = 10
        ':obj:`int`: The number of concurrent updates that will be processed in parallel. A\n        value of ``0`` indicates updates are *not* being processed concurrently.\n\n        .. versionchanged:: 20.4\n            This is now just a shortcut to :attr:`update_processor.max_concurrent_updates\n            <telegram.ext.BaseUpdateProcessor.max_concurrent_updates>`.\n\n        .. seealso:: :wiki:`Concurrency`\n        '
        return self._update_processor.max_concurrent_updates

    @property
    def job_queue(self) -> Optional['JobQueue[CCT]']:
        if False:
            for i in range(10):
                print('nop')
        '\n        :class:`telegram.ext.JobQueue`: The :class:`JobQueue` used by the\n            :class:`telegram.ext.Application`.\n\n        .. seealso:: :wiki:`Job Queue <Extensions---JobQueue>`\n        '
        if self._job_queue is None:
            warn('No `JobQueue` set up. To use `JobQueue`, you must install PTB via `pip install "python-telegram-bot[job-queue]"`.', stacklevel=2)
        return self._job_queue

    @property
    def update_processor(self) -> 'BaseUpdateProcessor':
        if False:
            i = 10
            return i + 15
        ':class:`telegram.ext.BaseUpdateProcessor`: The update processor used by this\n        application.\n\n        .. seealso:: :wiki:`Concurrency`\n\n        .. versionadded:: 20.4\n        '
        return self._update_processor

    @staticmethod
    def _raise_system_exit() -> NoReturn:
        if False:
            return 10
        raise SystemExit

    @staticmethod
    def builder() -> 'InitApplicationBuilder':
        if False:
            print('Hello World!')
        'Convenience method. Returns a new :class:`telegram.ext.ApplicationBuilder`.\n\n        .. versionadded:: 20.0\n        '
        from telegram.ext import ApplicationBuilder
        return ApplicationBuilder()

    def _check_initialized(self) -> None:
        if False:
            while True:
                i = 10
        if not self._initialized:
            raise RuntimeError('This Application was not initialized via `Application.initialize`!')

    async def initialize(self) -> None:
        """Initializes the Application by initializing:

        * The :attr:`bot`, by calling :meth:`telegram.Bot.initialize`.
        * The :attr:`updater`, by calling :meth:`telegram.ext.Updater.initialize`.
        * The :attr:`persistence`, by loading persistent conversations and data.
        * The :attr:`update_processor` by calling
          :meth:`telegram.ext.BaseUpdateProcessor.initialize`.

        Does *not* call :attr:`post_init` - that is only done by :meth:`run_polling` and
        :meth:`run_webhook`.

        .. seealso::
            :meth:`shutdown`
        """
        if self._initialized:
            _LOGGER.debug('This Application is already initialized.')
            return
        await self.bot.initialize()
        await self._update_processor.initialize()
        if self.updater:
            await self.updater.initialize()
        if not self.persistence:
            self._initialized = True
            return
        await self._initialize_persistence()
        from telegram.ext._conversationhandler import ConversationHandler
        for handler in itertools.chain.from_iterable(self.handlers.values()):
            if isinstance(handler, ConversationHandler) and handler.persistent and handler.name:
                await self._add_ch_to_persistence(handler)
        self._initialized = True

    async def _add_ch_to_persistence(self, handler: 'ConversationHandler') -> None:
        self._conversation_handler_conversations.update(await handler._initialize_persistence(self))

    async def shutdown(self) -> None:
        """Shuts down the Application by shutting down:

        * :attr:`bot` by calling :meth:`telegram.Bot.shutdown`
        * :attr:`updater` by calling :meth:`telegram.ext.Updater.shutdown`
        * :attr:`persistence` by calling :meth:`update_persistence` and
          :meth:`BasePersistence.flush`
        * :attr:`update_processor` by calling :meth:`telegram.ext.BaseUpdateProcessor.shutdown`

        Does *not* call :attr:`post_shutdown` - that is only done by :meth:`run_polling` and
        :meth:`run_webhook`.

        .. seealso::
            :meth:`initialize`

        Raises:
            :exc:`RuntimeError`: If the application is still :attr:`running`.
        """
        if self.running:
            raise RuntimeError('This Application is still running!')
        if not self._initialized:
            _LOGGER.debug('This Application is already shut down. Returning.')
            return
        await self.bot.shutdown()
        await self._update_processor.shutdown()
        if self.updater:
            await self.updater.shutdown()
        if self.persistence:
            _LOGGER.debug('Updating & flushing persistence before shutdown')
            await self.update_persistence()
            await self.persistence.flush()
            _LOGGER.debug('Updated and flushed persistence')
        self._initialized = False

    async def _initialize_persistence(self) -> None:
        """This method basically just loads all the data by awaiting the BP methods"""
        if not self.persistence:
            return
        if self.persistence.store_data.user_data:
            self._user_data.update(await self.persistence.get_user_data())
        if self.persistence.store_data.chat_data:
            self._chat_data.update(await self.persistence.get_chat_data())
        if self.persistence.store_data.bot_data:
            self.bot_data = await self.persistence.get_bot_data()
            if not isinstance(self.bot_data, self.context_types.bot_data):
                raise ValueError(f'bot_data must be of type {self.context_types.bot_data.__name__}')
        if self.persistence.store_data.callback_data and self.bot.callback_data_cache is not None:
            persistent_data = await self.persistence.get_callback_data()
            if persistent_data is not None:
                if not isinstance(persistent_data, tuple) or len(persistent_data) != 2:
                    raise ValueError('callback_data must be a tuple of length 2')
                self.bot.callback_data_cache.load_persistence_data(persistent_data)

    async def start(self) -> None:
        """Starts

        * a background task that fetches updates from :attr:`update_queue` and processes them via
          :meth:`process_update`.
        * :attr:`job_queue`, if set.
        * a background task that calls :meth:`update_persistence` in regular intervals, if
          :attr:`persistence` is set.

        Note:
            This does *not* start fetching updates from Telegram. To fetch updates, you need to
            either start :attr:`updater` manually or use one of :meth:`run_polling` or
            :meth:`run_webhook`.

        Tip:
            When using a custom logic for startup and shutdown of the application, eventual
            cancellation of pending tasks should happen only `after` :meth:`stop` has been called
            in order to ensure that the tasks mentioned above are not cancelled prematurely.

        .. seealso::
            :meth:`stop`

        Raises:
            :exc:`RuntimeError`: If the application is already running or was not initialized.
        """
        if self.running:
            raise RuntimeError('This Application is already running!')
        self._check_initialized()
        self._running = True
        self.__update_persistence_event.clear()
        try:
            if self.persistence:
                self.__update_persistence_task = asyncio.create_task(self._persistence_updater(), name=f'Application:{self.bot.id}:persistence_updater')
                _LOGGER.debug('Loop for updating persistence started')
            if self._job_queue:
                await self._job_queue.start()
                _LOGGER.debug('JobQueue started')
            self.__update_fetcher_task = asyncio.create_task(self._update_fetcher(), name=f'Application:{self.bot.id}:update_fetcher')
            _LOGGER.info('Application started')
        except Exception as exc:
            self._running = False
            raise exc

    async def stop(self) -> None:
        """Stops the process after processing any pending updates or tasks created by
        :meth:`create_task`. Also stops :attr:`job_queue`, if set.
        Finally, calls :meth:`update_persistence` and :meth:`BasePersistence.flush` on
        :attr:`persistence`, if set.

        Warning:
            Once this method is called, no more updates will be fetched from :attr:`update_queue`,
            even if it's not empty.

        .. seealso::
            :meth:`start`

        Note:
            * This does *not* stop :attr:`updater`. You need to either manually call
              :meth:`telegram.ext.Updater.stop` or use one of :meth:`run_polling` or
              :meth:`run_webhook`.
            * Does *not* call :attr:`post_stop` - that is only done by
              :meth:`run_polling` and :meth:`run_webhook`.

        Raises:
            :exc:`RuntimeError`: If the application is not running.
        """
        if not self.running:
            raise RuntimeError('This Application is not running!')
        self._running = False
        _LOGGER.info('Application is stopping. This might take a moment.')
        await self.update_queue.put(_STOP_SIGNAL)
        _LOGGER.debug('Waiting for update_queue to join')
        await self.update_queue.join()
        if self.__update_fetcher_task:
            await self.__update_fetcher_task
        _LOGGER.debug('Application stopped fetching of updates.')
        if self._job_queue:
            _LOGGER.debug('Waiting for running jobs to finish')
            await self._job_queue.stop(wait=True)
            _LOGGER.debug('JobQueue stopped')
        _LOGGER.debug('Waiting for `create_task` calls to be processed')
        await asyncio.gather(*self.__create_task_tasks, return_exceptions=True)
        if self.persistence and self.__update_persistence_task:
            _LOGGER.debug('Waiting for persistence loop to finish')
            self.__update_persistence_event.set()
            await self.__update_persistence_task
            self.__update_persistence_event.clear()
        _LOGGER.info('Application.stop() complete')

    def stop_running(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'This method can be used to stop the execution of :meth:`run_polling` or\n        :meth:`run_webhook` from within a handler, job or error callback. This allows a graceful\n        shutdown of the application, i.e. the methods listed in :attr:`run_polling` and\n        :attr:`run_webhook` will still be executed.\n\n        Note:\n            If the application is not running, this method does nothing.\n\n        .. versionadded:: 20.5\n        '
        if self.running:
            asyncio.get_running_loop().stop()
        else:
            _LOGGER.debug('Application is not running, stop_running() does nothing.')

    def run_polling(self, poll_interval: float=0.0, timeout: int=10, bootstrap_retries: int=-1, read_timeout: float=2, write_timeout: ODVInput[float]=DEFAULT_NONE, connect_timeout: ODVInput[float]=DEFAULT_NONE, pool_timeout: ODVInput[float]=DEFAULT_NONE, allowed_updates: Optional[List[str]]=None, drop_pending_updates: Optional[bool]=None, close_loop: bool=True, stop_signals: ODVInput[Sequence[int]]=DEFAULT_NONE) -> None:
        if False:
            print('Hello World!')
        'Convenience method that takes care of initializing and starting the app,\n        polling updates from Telegram using :meth:`telegram.ext.Updater.start_polling` and\n        a graceful shutdown of the app on exit.\n\n        The app will shut down when :exc:`KeyboardInterrupt` or :exc:`SystemExit` is raised.\n        On unix, the app will also shut down on receiving the signals specified by\n        :paramref:`stop_signals`.\n\n        The order of execution by :meth:`run_polling` is roughly as follows:\n\n        - :meth:`initialize`\n        - :meth:`post_init`\n        - :meth:`telegram.ext.Updater.start_polling`\n        - :meth:`start`\n        - Run the application until the users stops it\n        - :meth:`telegram.ext.Updater.stop`\n        - :meth:`stop`\n        - :meth:`post_stop`\n        - :meth:`shutdown`\n        - :meth:`post_shutdown`\n\n        .. include:: inclusions/application_run_tip.rst\n\n        Args:\n            poll_interval (:obj:`float`, optional): Time to wait between polling updates from\n                Telegram in seconds. Default is ``0.0``.\n            timeout (:obj:`int`, optional): Passed to\n                :paramref:`telegram.Bot.get_updates.timeout`. Default is ``10`` seconds.\n            bootstrap_retries (:obj:`int`, optional): Whether the bootstrapping phase of the\n                :class:`telegram.ext.Updater` will retry on failures on the Telegram server.\n\n                * < 0 - retry indefinitely (default)\n                *   0 - no retries\n                * > 0 - retry up to X times\n\n            read_timeout (:obj:`float`, optional): Value to pass to\n                :paramref:`telegram.Bot.get_updates.read_timeout`. Defaults to ``2``.\n            write_timeout (:obj:`float` | :obj:`None`, optional): Value to pass to\n                :paramref:`telegram.Bot.get_updates.write_timeout`. Defaults to\n                :attr:`~telegram.request.BaseRequest.DEFAULT_NONE`.\n            connect_timeout (:obj:`float` | :obj:`None`, optional): Value to pass to\n                :paramref:`telegram.Bot.get_updates.connect_timeout`. Defaults to\n                :attr:`~telegram.request.BaseRequest.DEFAULT_NONE`.\n            pool_timeout (:obj:`float` | :obj:`None`, optional): Value to pass to\n                :paramref:`telegram.Bot.get_updates.pool_timeout`. Defaults to\n                :attr:`~telegram.request.BaseRequest.DEFAULT_NONE`.\n            drop_pending_updates (:obj:`bool`, optional): Whether to clean any pending updates on\n                Telegram servers before actually starting to poll. Default is :obj:`False`.\n            allowed_updates (List[:obj:`str`], optional): Passed to\n                :meth:`telegram.Bot.get_updates`.\n            close_loop (:obj:`bool`, optional): If :obj:`True`, the current event loop will be\n                closed upon shutdown. Defaults to :obj:`True`.\n\n                .. seealso::\n                    :meth:`asyncio.loop.close`\n            stop_signals (Sequence[:obj:`int`] | :obj:`None`, optional): Signals that will shut\n                down the app. Pass :obj:`None` to not use stop signals.\n                Defaults to :data:`signal.SIGINT`, :data:`signal.SIGTERM` and\n                :data:`signal.SIGABRT` on non Windows platforms.\n\n                Caution:\n                    Not every :class:`asyncio.AbstractEventLoop` implements\n                    :meth:`asyncio.loop.add_signal_handler`. Most notably, the standard event loop\n                    on Windows, :class:`asyncio.ProactorEventLoop`, does not implement this method.\n                    If this method is not available, stop signals can not be set.\n\n        Raises:\n            :exc:`RuntimeError`: If the Application does not have an :class:`telegram.ext.Updater`.\n        '
        if not self.updater:
            raise RuntimeError('Application.run_polling is only available if the application has an Updater.')

        def error_callback(exc: TelegramError) -> None:
            if False:
                return 10
            self.create_task(self.process_error(error=exc, update=None))
        return self.__run(updater_coroutine=self.updater.start_polling(poll_interval=poll_interval, timeout=timeout, bootstrap_retries=bootstrap_retries, read_timeout=read_timeout, write_timeout=write_timeout, connect_timeout=connect_timeout, pool_timeout=pool_timeout, allowed_updates=allowed_updates, drop_pending_updates=drop_pending_updates, error_callback=error_callback), close_loop=close_loop, stop_signals=stop_signals)

    def run_webhook(self, listen: str='127.0.0.1', port: int=80, url_path: str='', cert: Optional[Union[str, Path]]=None, key: Optional[Union[str, Path]]=None, bootstrap_retries: int=0, webhook_url: Optional[str]=None, allowed_updates: Optional[List[str]]=None, drop_pending_updates: Optional[bool]=None, ip_address: Optional[str]=None, max_connections: int=40, close_loop: bool=True, stop_signals: ODVInput[Sequence[int]]=DEFAULT_NONE, secret_token: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Convenience method that takes care of initializing and starting the app,\n        listening for updates from Telegram using :meth:`telegram.ext.Updater.start_webhook` and\n        a graceful shutdown of the app on exit.\n\n        The app will shut down when :exc:`KeyboardInterrupt` or :exc:`SystemExit` is raised.\n        On unix, the app will also shut down on receiving the signals specified by\n        :paramref:`stop_signals`.\n\n        If :paramref:`cert`\n        and :paramref:`key` are not provided, the webhook will be started directly on\n        ``http://listen:port/url_path``, so SSL can be handled by another\n        application. Else, the webhook will be started on\n        ``https://listen:port/url_path``. Also calls :meth:`telegram.Bot.set_webhook` as\n        required.\n\n        The order of execution by :meth:`run_webhook` is roughly as follows:\n\n        - :meth:`initialize`\n        - :meth:`post_init`\n        - :meth:`telegram.ext.Updater.start_webhook`\n        - :meth:`start`\n        - Run the application until the users stops it\n        - :meth:`telegram.ext.Updater.stop`\n        - :meth:`stop`\n        - :meth:`post_stop`\n        - :meth:`shutdown`\n        - :meth:`post_shutdown`\n\n        Important:\n            If you want to use this method, you must install PTB with the optional requirement\n            ``webhooks``, i.e.\n\n            .. code-block:: bash\n\n               pip install "python-telegram-bot[webhooks]"\n\n        .. include:: inclusions/application_run_tip.rst\n\n        .. seealso::\n            :wiki:`Webhooks`\n\n        Args:\n            listen (:obj:`str`, optional): IP-Address to listen on. Defaults to\n                `127.0.0.1 <https://en.wikipedia.org/wiki/Localhost>`_.\n            port (:obj:`int`, optional): Port the bot should be listening on. Must be one of\n                :attr:`telegram.constants.SUPPORTED_WEBHOOK_PORTS` unless the bot is running\n                behind a proxy. Defaults to ``80``.\n            url_path (:obj:`str`, optional): Path inside url. Defaults to `` \'\' ``\n            cert (:class:`pathlib.Path` | :obj:`str`, optional): Path to the SSL certificate file.\n            key (:class:`pathlib.Path` | :obj:`str`, optional): Path to the SSL key file.\n            bootstrap_retries (:obj:`int`, optional): Whether the bootstrapping phase of the\n                :class:`telegram.ext.Updater` will retry on failures on the Telegram server.\n\n                * < 0 - retry indefinitely\n                *   0 - no retries (default)\n                * > 0 - retry up to X times\n            webhook_url (:obj:`str`, optional): Explicitly specify the webhook url. Useful behind\n                NAT, reverse proxy, etc. Default is derived from :paramref:`listen`,\n                :paramref:`port`, :paramref:`url_path`, :paramref:`cert`, and :paramref:`key`.\n            allowed_updates (List[:obj:`str`], optional): Passed to\n                :meth:`telegram.Bot.set_webhook`.\n            drop_pending_updates (:obj:`bool`, optional): Whether to clean any pending updates on\n                Telegram servers before actually starting to poll. Default is :obj:`False`.\n            ip_address (:obj:`str`, optional): Passed to :meth:`telegram.Bot.set_webhook`.\n            max_connections (:obj:`int`, optional): Passed to\n                :meth:`telegram.Bot.set_webhook`. Defaults to ``40``.\n            close_loop (:obj:`bool`, optional): If :obj:`True`, the current event loop will be\n                closed upon shutdown. Defaults to :obj:`True`.\n\n                .. seealso::\n                    :meth:`asyncio.loop.close`\n            stop_signals (Sequence[:obj:`int`] | :obj:`None`, optional): Signals that will shut\n                down the app. Pass :obj:`None` to not use stop signals.\n                Defaults to :data:`signal.SIGINT`, :data:`signal.SIGTERM` and\n                :data:`signal.SIGABRT`.\n\n                Caution:\n                    Not every :class:`asyncio.AbstractEventLoop` implements\n                    :meth:`asyncio.loop.add_signal_handler`. Most notably, the standard event loop\n                    on Windows, :class:`asyncio.ProactorEventLoop`, does not implement this method.\n                    If this method is not available, stop signals can not be set.\n            secret_token (:obj:`str`, optional): Secret token to ensure webhook requests originate\n                from Telegram. See :paramref:`telegram.Bot.set_webhook.secret_token` for more\n                details.\n\n                When added, the web server started by this call will expect the token to be set in\n                the ``X-Telegram-Bot-Api-Secret-Token`` header of an incoming request and will\n                raise a :class:`http.HTTPStatus.FORBIDDEN <http.HTTPStatus>` error if either the\n                header isn\'t set or it is set to a wrong token.\n\n                .. versionadded:: 20.0\n        '
        if not self.updater:
            raise RuntimeError('Application.run_webhook is only available if the application has an Updater.')
        return self.__run(updater_coroutine=self.updater.start_webhook(listen=listen, port=port, url_path=url_path, cert=cert, key=key, bootstrap_retries=bootstrap_retries, drop_pending_updates=drop_pending_updates, webhook_url=webhook_url, allowed_updates=allowed_updates, ip_address=ip_address, max_connections=max_connections, secret_token=secret_token), close_loop=close_loop, stop_signals=stop_signals)

    def __run(self, updater_coroutine: Coroutine, stop_signals: ODVInput[Sequence[int]], close_loop: bool=True) -> None:
        if False:
            return 10
        loop = asyncio.get_event_loop()
        if stop_signals is DEFAULT_NONE and platform.system() != 'Windows':
            stop_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGABRT)
        try:
            if not isinstance(stop_signals, DefaultValue):
                for sig in stop_signals or []:
                    loop.add_signal_handler(sig, self._raise_system_exit)
        except NotImplementedError as exc:
            warn(f'Could not add signal handlers for the stop signals {stop_signals} due to exception `{exc!r}`. If your event loop does not implement `add_signal_handler`, please pass `stop_signals=None`.', stacklevel=3)
        try:
            loop.run_until_complete(self.initialize())
            if self.post_init:
                loop.run_until_complete(self.post_init(self))
            loop.run_until_complete(updater_coroutine)
            loop.run_until_complete(self.start())
            loop.run_forever()
        except (KeyboardInterrupt, SystemExit):
            _LOGGER.debug('Application received stop signal. Shutting down.')
        except Exception as exc:
            updater_coroutine.close()
            raise exc
        finally:
            try:
                if self.updater.running:
                    loop.run_until_complete(self.updater.stop())
                if self.running:
                    loop.run_until_complete(self.stop())
                if self.post_stop:
                    loop.run_until_complete(self.post_stop(self))
                loop.run_until_complete(self.shutdown())
                if self.post_shutdown:
                    loop.run_until_complete(self.post_shutdown(self))
            finally:
                if close_loop:
                    loop.close()

    def create_task(self, coroutine: _CoroType[RT], update: Optional[object]=None, *, name: Optional[str]=None) -> 'asyncio.Task[RT]':
        if False:
            i = 10
            return i + 15
        "Thin wrapper around :func:`asyncio.create_task` that handles exceptions raised by\n        the :paramref:`coroutine` with :meth:`process_error`.\n\n        Note:\n            * If :paramref:`coroutine` raises an exception, it will be set on the task created by\n              this method even though it's handled by :meth:`process_error`.\n            * If the application is currently running, tasks created by this method will be\n              awaited with :meth:`stop`.\n\n        .. seealso:: :wiki:`Concurrency`\n\n        Args:\n            coroutine (:term:`awaitable`): The awaitable to run as task.\n\n                .. versionchanged:: 20.2\n                    Accepts :class:`asyncio.Future` and generator-based coroutine functions.\n                .. deprecated:: 20.4\n                    Since Python 3.12, generator-based coroutine functions are no longer accepted.\n            update (:obj:`object`, optional): If set, will be passed to :meth:`process_error`\n                as additional information for the error handlers. Moreover, the corresponding\n                :attr:`chat_data` and :attr:`user_data` entries will be updated in the next run of\n                :meth:`update_persistence` after the :paramref:`coroutine` is finished.\n\n        Keyword Args:\n            name (:obj:`str`, optional): The name of the task.\n\n                .. versionadded:: 20.4\n\n        Returns:\n            :class:`asyncio.Task`: The created task.\n        "
        return self.__create_task(coroutine=coroutine, update=update, name=name)

    def __create_task(self, coroutine: _CoroType[RT], update: Optional[object]=None, is_error_handler: bool=False, name: Optional[str]=None) -> 'asyncio.Task[RT]':
        if False:
            print('Hello World!')
        task: asyncio.Task[RT] = asyncio.create_task(self.__create_task_callback(coroutine=coroutine, update=update, is_error_handler=is_error_handler), name=name)
        if self.running:
            self.__create_task_tasks.add(task)
            task.add_done_callback(self.__create_task_done_callback)
        else:
            warn("Tasks created via `Application.create_task` while the application is not running won't be automatically awaited!", stacklevel=3)
        return task

    def __create_task_done_callback(self, task: asyncio.Task) -> None:
        if False:
            i = 10
            return i + 15
        self.__create_task_tasks.discard(task)
        with contextlib.suppress(asyncio.CancelledError, asyncio.InvalidStateError):
            task.exception()

    async def __create_task_callback(self, coroutine: _CoroType[RT], update: Optional[object]=None, is_error_handler: bool=False) -> RT:
        try:
            if sys.version_info < (3, 12) and isinstance(coroutine, Generator):
                warn('Generator-based coroutines are deprecated in create_task and will not work in Python 3.12+', category=PTBDeprecationWarning)
                return await asyncio.create_task(coroutine)
            return await coroutine
        except Exception as exception:
            if isinstance(exception, ApplicationHandlerStop):
                warn('ApplicationHandlerStop is not supported with handlers running non-blocking.', stacklevel=1)
            elif is_error_handler:
                _LOGGER.exception('An error was raised and an uncaught error was raised while handling the error with an error_handler.', exc_info=exception)
            else:
                await self.process_error(update=update, error=exception, coroutine=coroutine)
            raise exception
        finally:
            self._mark_for_persistence_update(update=update)

    async def _update_fetcher(self) -> None:
        while True:
            try:
                update = await self.update_queue.get()
                if update is _STOP_SIGNAL:
                    _LOGGER.debug('Dropping pending updates')
                    while not self.update_queue.empty():
                        self.update_queue.task_done()
                    self.update_queue.task_done()
                    return
                _LOGGER.debug('Processing update %s', update)
                if self._update_processor.max_concurrent_updates > 1:
                    self.create_task(self.__process_update_wrapper(update), update=update, name=f'Application:{self.bot.id}:process_concurrent_update')
                else:
                    await self.__process_update_wrapper(update)
            except asyncio.CancelledError:
                _LOGGER.warning('Fetching updates got a asyncio.CancelledError. Ignoring as this task may onlybe closed via `Application.stop`.')

    async def __process_update_wrapper(self, update: object) -> None:
        await self._update_processor.process_update(update, self.process_update(update))
        self.update_queue.task_done()

    async def process_update(self, update: object) -> None:
        """Processes a single update and marks the update to be updated by the persistence later.
        Exceptions raised by handler callbacks will be processed by :meth:`process_error`.

        .. seealso:: :wiki:`Concurrency`

        .. versionchanged:: 20.0
            Persistence is now updated in an interval set by
            :attr:`telegram.ext.BasePersistence.update_interval`.

        Args:
            update (:class:`telegram.Update` | :obj:`object` |                 :class:`telegram.error.TelegramError`): The update to process.

        Raises:
            :exc:`RuntimeError`: If the application was not initialized.
        """
        self._check_initialized()
        context = None
        any_blocking = False
        for handlers in self.handlers.values():
            try:
                for handler in handlers:
                    check = handler.check_update(update)
                    if not (check is None or check is False):
                        if not context:
                            context = self.context_types.context.from_update(update, self)
                            await context.refresh_data()
                        coroutine: Coroutine = handler.handle_update(update, self, check, context)
                        if not handler.block or (handler.block is DEFAULT_TRUE and isinstance(self.bot, ExtBot) and self.bot.defaults and (not self.bot.defaults.block)):
                            self.create_task(coroutine, update=update, name=f'Application:{self.bot.id}:process_update_non_blocking:{handler}')
                        else:
                            any_blocking = True
                            await coroutine
                        break
            except ApplicationHandlerStop:
                _LOGGER.debug('Stopping further handlers due to ApplicationHandlerStop')
                break
            except Exception as exc:
                if await self.process_error(update=update, error=exc):
                    _LOGGER.debug('Error handler stopped further handlers.')
                    break
        if any_blocking:
            self._mark_for_persistence_update(update=update)

    def add_handler(self, handler: BaseHandler[Any, CCT], group: int=DEFAULT_GROUP) -> None:
        if False:
            while True:
                i = 10
        'Register a handler.\n\n        TL;DR: Order and priority counts. 0 or 1 handlers per group will be used. End handling of\n        update with :class:`telegram.ext.ApplicationHandlerStop`.\n\n        A handler must be an instance of a subclass of :class:`telegram.ext.BaseHandler`. All\n        handlers\n        are organized in groups with a numeric value. The default group is 0. All groups will be\n        evaluated for handling an update, but only 0 or 1 handler per group will be used. If\n        :class:`telegram.ext.ApplicationHandlerStop` is raised from one of the handlers, no further\n        handlers (regardless of the group) will be called.\n\n        The priority/order of handlers is determined as follows:\n\n          * Priority of the group (lower group number == higher priority)\n          * The first handler in a group which can handle an update (see\n            :attr:`telegram.ext.BaseHandler.check_update`) will be used. Other handlers from the\n            group will not be used. The order in which handlers were added to the group defines the\n            priority.\n\n        Warning:\n            Adding persistent :class:`telegram.ext.ConversationHandler` after the application has\n            been initialized is discouraged. This is because the persisted conversation states need\n            to be loaded into memory while the application is already processing updates, which\n            might lead to race conditions and undesired behavior. In particular, current\n            conversation states may be overridden by the loaded data.\n\n        Args:\n            handler (:class:`telegram.ext.BaseHandler`): A BaseHandler instance.\n            group (:obj:`int`, optional): The group identifier. Default is ``0``.\n\n        '
        from telegram.ext._conversationhandler import ConversationHandler
        if not isinstance(handler, BaseHandler):
            raise TypeError(f'handler is not an instance of {BaseHandler.__name__}')
        if not isinstance(group, int):
            raise TypeError('group is not int')
        if isinstance(handler, ConversationHandler) and handler.persistent and handler.name:
            if not self.persistence:
                raise ValueError(f'ConversationHandler {handler.name} can not be persistent if application has no persistence')
            if self._initialized:
                self.create_task(self._add_ch_to_persistence(handler), name=f'Application:{self.bot.id}:add_handler:conversation_handler_after_init')
                warn('A persistent `ConversationHandler` was passed to `add_handler`, after `Application.initialize` was called. This is discouraged.See the docs of `Application.add_handler` for details.', stacklevel=2)
        if group not in self.handlers:
            self.handlers[group] = []
            self.handlers = dict(sorted(self.handlers.items()))
        self.handlers[group].append(handler)

    def add_handlers(self, handlers: Union[Union[List[BaseHandler[Any, CCT]], Tuple[BaseHandler[Any, CCT]]], Dict[int, Union[List[BaseHandler[Any, CCT]], Tuple[BaseHandler[Any, CCT]]]]], group: Union[int, DefaultValue[int]]=_DEFAULT_0) -> None:
        if False:
            return 10
        'Registers multiple handlers at once. The order of the handlers in the passed\n        sequence(s) matters. See :meth:`add_handler` for details.\n\n        .. versionadded:: 20.0\n\n        Args:\n            handlers (List[:class:`telegram.ext.BaseHandler`] |                 Dict[int, List[:class:`telegram.ext.BaseHandler`]]):                 Specify a sequence of handlers *or* a dictionary where the keys are groups and\n                values are handlers.\n            group (:obj:`int`, optional): Specify which group the sequence of :paramref:`handlers`\n                should be added to. Defaults to ``0``.\n\n        Example::\n\n            app.add_handlers(handlers={\n                -1: [MessageHandler(...)],\n                1: [CallbackQueryHandler(...), CommandHandler(...)]\n            }\n\n        '
        if isinstance(handlers, dict) and (not isinstance(group, DefaultValue)):
            raise ValueError('The `group` argument can only be used with a sequence of handlers.')
        if isinstance(handlers, dict):
            for (handler_group, grp_handlers) in handlers.items():
                if not isinstance(grp_handlers, (list, tuple)):
                    raise ValueError(f'Handlers for group {handler_group} must be a list or tuple')
                for handler in grp_handlers:
                    self.add_handler(handler, handler_group)
        elif isinstance(handlers, (list, tuple)):
            for handler in handlers:
                self.add_handler(handler, DefaultValue.get_value(group))
        else:
            raise ValueError('The `handlers` argument must be a sequence of handlers or a dictionary where the keys are groups and values are sequences of handlers.')

    def remove_handler(self, handler: BaseHandler[Any, CCT], group: int=DEFAULT_GROUP) -> None:
        if False:
            while True:
                i = 10
        'Remove a handler from the specified group.\n\n        Args:\n            handler (:class:`telegram.ext.BaseHandler`): A :class:`telegram.ext.BaseHandler`\n                instance.\n            group (:obj:`object`, optional): The group identifier. Default is ``0``.\n\n        '
        if handler in self.handlers[group]:
            self.handlers[group].remove(handler)
            if not self.handlers[group]:
                del self.handlers[group]

    def drop_chat_data(self, chat_id: int) -> None:
        if False:
            print('Hello World!')
        'Drops the corresponding entry from the :attr:`chat_data`. Will also be deleted from\n        the persistence on the next run of :meth:`update_persistence`, if applicable.\n\n        Warning:\n            When using :attr:`concurrent_updates` or the :attr:`job_queue`,\n            :meth:`process_update` or :meth:`telegram.ext.Job.run` may re-create this entry due to\n            the asynchronous nature of these features. Please make sure that your program can\n            avoid or handle such situations.\n\n        .. versionadded:: 20.0\n\n        Args:\n            chat_id (:obj:`int`): The chat id to delete. The entry will be deleted even if it is\n                not empty.\n        '
        self._chat_data.pop(chat_id, None)
        self._chat_ids_to_be_deleted_in_persistence.add(chat_id)

    def drop_user_data(self, user_id: int) -> None:
        if False:
            while True:
                i = 10
        'Drops the corresponding entry from the :attr:`user_data`. Will also be deleted from\n        the persistence on the next run of :meth:`update_persistence`, if applicable.\n\n        Warning:\n            When using :attr:`concurrent_updates` or the :attr:`job_queue`,\n            :meth:`process_update` or :meth:`telegram.ext.Job.run` may re-create this entry due to\n            the asynchronous nature of these features. Please make sure that your program can\n            avoid or handle such situations.\n\n        .. versionadded:: 20.0\n\n        Args:\n            user_id (:obj:`int`): The user id to delete. The entry will be deleted even if it is\n                not empty.\n        '
        self._user_data.pop(user_id, None)
        self._user_ids_to_be_deleted_in_persistence.add(user_id)

    def migrate_chat_data(self, message: Optional['Message']=None, old_chat_id: Optional[int]=None, new_chat_id: Optional[int]=None) -> None:
        if False:
            return 10
        'Moves the contents of :attr:`chat_data` at key :paramref:`old_chat_id` to the key\n        :paramref:`new_chat_id`. Also marks the entries to be updated accordingly in the next run\n        of :meth:`update_persistence`.\n\n        Warning:\n            * Any data stored in :attr:`chat_data` at key :paramref:`new_chat_id` will be\n              overridden\n            * The key :paramref:`old_chat_id` of :attr:`chat_data` will be deleted\n            * This does not update the :attr:`~telegram.ext.Job.chat_id` attribute of any scheduled\n              :class:`telegram.ext.Job`.\n\n            When using :attr:`concurrent_updates` or the :attr:`job_queue`,\n            :meth:`process_update` or :meth:`telegram.ext.Job.run` may re-create the old entry due\n            to the asynchronous nature of these features. Please make sure that your program can\n            avoid or handle such situations.\n\n        .. seealso:: :wiki:`Storing Bot, User and Chat Related Data            <Storing-bot%2C-user-and-chat-related-data>`\n\n        Args:\n            message (:class:`telegram.Message`, optional): A message with either\n                :attr:`~telegram.Message.migrate_from_chat_id` or\n                :attr:`~telegram.Message.migrate_to_chat_id`.\n                Mutually exclusive with passing :paramref:`old_chat_id` and\n                :paramref:`new_chat_id`.\n\n                .. seealso::\n                    :attr:`telegram.ext.filters.StatusUpdate.MIGRATE`\n\n            old_chat_id (:obj:`int`, optional): The old chat ID.\n                Mutually exclusive with passing :paramref:`message`\n            new_chat_id (:obj:`int`, optional): The new chat ID.\n                Mutually exclusive with passing :paramref:`message`\n\n        Raises:\n            ValueError: Raised if the input is invalid.\n        '
        if message and (old_chat_id or new_chat_id):
            raise ValueError('Message and chat_id pair are mutually exclusive')
        if not any((message, old_chat_id, new_chat_id)):
            raise ValueError('chat_id pair or message must be passed')
        if message:
            if message.migrate_from_chat_id is None and message.migrate_to_chat_id is None:
                raise ValueError('Invalid message instance. The message must have either `Message.migrate_from_chat_id` or `Message.migrate_to_chat_id`.')
            old_chat_id = message.migrate_from_chat_id or message.chat.id
            new_chat_id = message.migrate_to_chat_id or message.chat.id
        elif not (isinstance(old_chat_id, int) and isinstance(new_chat_id, int)):
            raise ValueError('old_chat_id and new_chat_id must be integers')
        self._chat_data[new_chat_id] = self._chat_data[old_chat_id]
        self.drop_chat_data(old_chat_id)
        self._chat_ids_to_be_updated_in_persistence.add(new_chat_id)

    def _mark_for_persistence_update(self, *, update: Optional[object]=None, job: Optional['Job']=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(update, Update):
            if update.effective_chat:
                self._chat_ids_to_be_updated_in_persistence.add(update.effective_chat.id)
            if update.effective_user:
                self._user_ids_to_be_updated_in_persistence.add(update.effective_user.id)
        if job:
            if job.chat_id:
                self._chat_ids_to_be_updated_in_persistence.add(job.chat_id)
            if job.user_id:
                self._user_ids_to_be_updated_in_persistence.add(job.user_id)

    def mark_data_for_update_persistence(self, chat_ids: Optional[SCT[int]]=None, user_ids: Optional[SCT[int]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Mark entries of :attr:`chat_data` and :attr:`user_data` to be updated on the next\n        run of :meth:`update_persistence`.\n\n        Tip:\n            Use this method sparingly. If you have to use this method, it likely means that you\n            access and modify ``context.application.chat/user_data[some_id]`` within a callback.\n            Note that for data which should be available globally in all handler callbacks\n            independent of the chat/user, it is recommended to use :attr:`bot_data` instead.\n\n        .. versionadded:: 20.3\n\n        Args:\n            chat_ids (:obj:`int` | Collection[:obj:`int`], optional): Chat IDs to mark.\n            user_ids (:obj:`int` | Collection[:obj:`int`], optional): User IDs to mark.\n\n        '
        if chat_ids:
            if isinstance(chat_ids, int):
                self._chat_ids_to_be_updated_in_persistence.add(chat_ids)
            else:
                self._chat_ids_to_be_updated_in_persistence.update(chat_ids)
        if user_ids:
            if isinstance(user_ids, int):
                self._user_ids_to_be_updated_in_persistence.add(user_ids)
            else:
                self._user_ids_to_be_updated_in_persistence.update(user_ids)

    async def _persistence_updater(self) -> None:
        while not self.__update_persistence_event.is_set():
            if not self.persistence:
                return
            try:
                await asyncio.wait_for(self.__update_persistence_event.wait(), timeout=self.persistence.update_interval)
                return
            except asyncio.TimeoutError:
                pass
            await self.update_persistence()

    async def update_persistence(self) -> None:
        """Updates :attr:`user_data`, :attr:`chat_data`, :attr:`bot_data` in :attr:`persistence`
        along with :attr:`~telegram.ext.ExtBot.callback_data_cache` and the conversation states of
        any persistent :class:`~telegram.ext.ConversationHandler` registered for this application.

        For :attr:`user_data` and :attr:`chat_data`, only those entries are updated which either
        were used or have been manually marked via :meth:`mark_data_for_update_persistence` since
        the last run of this method.

        Tip:
            This method will be called in regular intervals by the application. There is usually
            no need to call it manually.

        Note:
            Any data is deep copied with :func:`copy.deepcopy` before handing it over to the
            persistence in order to avoid race conditions, so all persisted data must be copyable.

        .. seealso:: :attr:`telegram.ext.BasePersistence.update_interval`,
            :meth:`mark_data_for_update_persistence`
        """
        async with self.__update_persistence_lock:
            await self.__update_persistence()

    async def __update_persistence(self) -> None:
        if not self.persistence:
            return
        _LOGGER.debug('Starting next run of updating the persistence.')
        coroutines: Set[Coroutine] = set()
        if self.persistence.store_data.callback_data and self.bot.callback_data_cache is not None:
            coroutines.add(self.persistence.update_callback_data(deepcopy(self.bot.callback_data_cache.persistence_data)))
        if self.persistence.store_data.bot_data:
            coroutines.add(self.persistence.update_bot_data(deepcopy(self.bot_data)))
        if self.persistence.store_data.chat_data:
            update_ids = self._chat_ids_to_be_updated_in_persistence
            self._chat_ids_to_be_updated_in_persistence = set()
            delete_ids = self._chat_ids_to_be_deleted_in_persistence
            self._chat_ids_to_be_deleted_in_persistence = set()
            update_ids -= delete_ids
            for chat_id in update_ids:
                coroutines.add(self.persistence.update_chat_data(chat_id, deepcopy(self.chat_data[chat_id])))
            for chat_id in delete_ids:
                coroutines.add(self.persistence.drop_chat_data(chat_id))
        if self.persistence.store_data.user_data:
            update_ids = self._user_ids_to_be_updated_in_persistence
            self._user_ids_to_be_updated_in_persistence = set()
            delete_ids = self._user_ids_to_be_deleted_in_persistence
            self._user_ids_to_be_deleted_in_persistence = set()
            update_ids -= delete_ids
            for user_id in update_ids:
                coroutines.add(self.persistence.update_user_data(user_id, deepcopy(self.user_data[user_id])))
            for user_id in delete_ids:
                coroutines.add(self.persistence.drop_user_data(user_id))
        from telegram.ext._conversationhandler import PendingState
        for (name, (key, new_state)) in itertools.chain.from_iterable((zip(itertools.repeat(name), states_dict.pop_accessed_write_items()) for (name, states_dict) in self._conversation_handler_conversations.items())):
            if isinstance(new_state, PendingState):
                if not new_state.done():
                    if self.running:
                        _LOGGER.debug('A ConversationHandlers state was not yet resolved. Updating the persistence with the current state. Will check again on next run of Application.update_persistence.')
                    else:
                        _LOGGER.warning('A ConversationHandlers state was not yet resolved. Updating the persistence with the current state.')
                    result = new_state.old_state
                    self._conversation_handler_conversations[name].mark_as_accessed(key)
                else:
                    result = new_state.resolve()
            else:
                result = new_state
            effective_new_state = None if result is TrackingDict.DELETED else result
            coroutines.add(self.persistence.update_conversation(name=name, key=key, new_state=effective_new_state))
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        _LOGGER.debug('Finished updating persistence.')
        await asyncio.gather(*(self.process_error(error=result, update=None) for result in results if isinstance(result, Exception)))

    def add_error_handler(self, callback: HandlerCallback[object, CCT, None], block: DVType[bool]=DEFAULT_TRUE) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Registers an error handler in the Application. This handler will receive every error\n        which happens in your bot. See the docs of :meth:`process_error` for more details on how\n        errors are handled.\n\n        Note:\n            Attempts to add the same callback multiple times will be ignored.\n\n        Examples:\n            :any:`Errorhandler Bot <examples.errorhandlerbot>`\n\n        .. seealso:: :wiki:`Exceptions, Warnings and Logging <Exceptions%2C-Warnings-and-Logging>`\n\n        Args:\n            callback (:term:`coroutine function`): The callback function for this error handler.\n                Will be called when an error is raised. Callback signature::\n\n                    async def callback(update: Optional[object], context: CallbackContext)\n\n                The error that happened will be present in\n                :attr:`telegram.ext.CallbackContext.error`.\n            block (:obj:`bool`, optional): Determines whether the return value of the callback\n                should be awaited before processing the next error handler in\n                :meth:`process_error`. Defaults to :obj:`True`.\n        '
        if callback in self.error_handlers:
            _LOGGER.warning('The callback is already registered as an error handler. Ignoring.')
            return
        self.error_handlers[callback] = block

    def remove_error_handler(self, callback: HandlerCallback[object, CCT, None]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Removes an error handler.\n\n        Args:\n            callback (:term:`coroutine function`): The error handler to remove.\n\n        '
        self.error_handlers.pop(callback, None)

    async def process_error(self, update: Optional[object], error: Exception, job: Optional['Job[CCT]']=None, coroutine: _ErrorCoroType[RT]=None) -> bool:
        """Processes an error by passing it to all error handlers registered with
        :meth:`add_error_handler`. If one of the error handlers raises
        :class:`telegram.ext.ApplicationHandlerStop`, the error will not be handled by other error
        handlers. Raising :class:`telegram.ext.ApplicationHandlerStop` also stops processing of
        the update when this method is called by :meth:`process_update`, i.e. no further handlers
        (even in other groups) will handle the update. All other exceptions raised by an error
        handler will just be logged.

        .. versionchanged:: 20.0

            * ``dispatch_error`` was renamed to :meth:`process_error`.
            * Exceptions raised by error handlers are now properly logged.
            * :class:`telegram.ext.ApplicationHandlerStop` is no longer reraised but converted into
              the return value.

        Args:
            update (:obj:`object` | :class:`telegram.Update`): The update that caused the error.
            error (:obj:`Exception`): The error that was raised.
            job (:class:`telegram.ext.Job`, optional): The job that caused the error.

                .. versionadded:: 20.0
            coroutine (:term:`coroutine function`, optional): The coroutine that caused the error.

        Returns:
            :obj:`bool`: :obj:`True`, if one of the error handlers raised
            :class:`telegram.ext.ApplicationHandlerStop`. :obj:`False`, otherwise.
        """
        if self.error_handlers:
            for (callback, block) in self.error_handlers.items():
                context = self.context_types.context.from_error(update=update, error=error, application=self, job=job, coroutine=coroutine)
                if not block or (block is DEFAULT_TRUE and isinstance(self.bot, ExtBot) and self.bot.defaults and (not self.bot.defaults.block)):
                    self.__create_task(callback(update, context), update=update, is_error_handler=True, name=f'Application:{self.bot.id}:process_error:non_blocking')
                else:
                    try:
                        await callback(update, context)
                    except ApplicationHandlerStop:
                        return True
                    except Exception as exc:
                        _LOGGER.exception('An error was raised and an uncaught error was raised while handling the error with an error_handler.', exc_info=exc)
            return False
        _LOGGER.exception('No error handlers are registered, logging exception.', exc_info=error)
        return False