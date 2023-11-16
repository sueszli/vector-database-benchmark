"""This module contains the Builder classes for the telegram.ext module."""
from asyncio import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Collection, Coroutine, Dict, Generic, Optional, Type, TypeVar, Union
import httpx
from telegram._bot import Bot
from telegram._utils.defaultvalue import DEFAULT_FALSE, DEFAULT_NONE, DefaultValue
from telegram._utils.types import DVInput, DVType, FilePathInput, HTTPVersion, ODVInput, SocketOpt
from telegram._utils.warnings import warn
from telegram.ext._application import Application
from telegram.ext._baseupdateprocessor import BaseUpdateProcessor, SimpleUpdateProcessor
from telegram.ext._contexttypes import ContextTypes
from telegram.ext._extbot import ExtBot
from telegram.ext._jobqueue import JobQueue
from telegram.ext._updater import Updater
from telegram.ext._utils.types import BD, BT, CCT, CD, JQ, UD
from telegram.request import BaseRequest
from telegram.request._httpxrequest import HTTPXRequest
from telegram.warnings import PTBDeprecationWarning
if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import BasePersistence, BaseRateLimiter, CallbackContext, Defaults
    from telegram.ext._utils.types import RLARGS
InBT = TypeVar('InBT', bound=Bot)
InJQ = TypeVar('InJQ', bound=Union[None, JobQueue])
InCCT = TypeVar('InCCT', bound='CallbackContext')
InUD = TypeVar('InUD')
InCD = TypeVar('InCD')
InBD = TypeVar('InBD')
BuilderType = TypeVar('BuilderType', bound='ApplicationBuilder')
_BOT_CHECKS = [('request', 'request instance'), ('get_updates_request', 'get_updates_request instance'), ('connection_pool_size', 'connection_pool_size'), ('proxy', 'proxy'), ('socket_options', 'socket_options'), ('pool_timeout', 'pool_timeout'), ('connect_timeout', 'connect_timeout'), ('read_timeout', 'read_timeout'), ('write_timeout', 'write_timeout'), ('http_version', 'http_version'), ('get_updates_connection_pool_size', 'get_updates_connection_pool_size'), ('get_updates_proxy', 'get_updates_proxy'), ('get_updates_socket_options', 'get_updates_socket_options'), ('get_updates_pool_timeout', 'get_updates_pool_timeout'), ('get_updates_connect_timeout', 'get_updates_connect_timeout'), ('get_updates_read_timeout', 'get_updates_read_timeout'), ('get_updates_write_timeout', 'get_updates_write_timeout'), ('get_updates_http_version', 'get_updates_http_version'), ('base_file_url', 'base_file_url'), ('base_url', 'base_url'), ('token', 'token'), ('defaults', 'defaults'), ('arbitrary_callback_data', 'arbitrary_callback_data'), ('private_key', 'private_key'), ('rate_limiter', 'rate_limiter instance'), ('local_mode', 'local_mode setting')]
_TWO_ARGS_REQ = 'The parameter `{}` may only be set, if no {} was set.'

class ApplicationBuilder(Generic[BT, CCT, UD, CD, BD, JQ]):
    """This class serves as initializer for :class:`telegram.ext.Application` via the so called
    `builder pattern`_. To build a :class:`telegram.ext.Application`, one first initializes an
    instance of this class. Arguments for the :class:`telegram.ext.Application` to build are then
    added by subsequently calling the methods of the builder. Finally, the
    :class:`telegram.ext.Application` is built by calling :meth:`build`. In the simplest case this
    can look like the following example.

    Example:
        .. code:: python

            application = ApplicationBuilder().token("TOKEN").build()

    Please see the description of the individual methods for information on which arguments can be
    set and what the defaults are when not called. When no default is mentioned, the argument will
    not be used by default.

    Note:
        * Some arguments are mutually exclusive. E.g. after calling :meth:`token`, you can't set
          a custom bot with :meth:`bot` and vice versa.
        * Unless a custom :class:`telegram.Bot` instance is set via :meth:`bot`, :meth:`build` will
          use :class:`telegram.ext.ExtBot` for the bot.

    .. seealso:: :wiki:`Your First Bot <Extensions---Your-first-Bot>`,
        :wiki:`Builder Pattern <Builder-Pattern>`

    .. _`builder pattern`: https://en.wikipedia.org/wiki/Builder_pattern
    """
    __slots__ = ('_application_class', '_application_kwargs', '_arbitrary_callback_data', '_base_file_url', '_base_url', '_bot', '_update_processor', '_connect_timeout', '_connection_pool_size', '_context_types', '_defaults', '_get_updates_connect_timeout', '_get_updates_connection_pool_size', '_get_updates_pool_timeout', '_get_updates_proxy', '_get_updates_read_timeout', '_get_updates_request', '_get_updates_socket_options', '_get_updates_write_timeout', '_get_updates_http_version', '_job_queue', '_persistence', '_pool_timeout', '_post_init', '_post_shutdown', '_post_stop', '_private_key', '_private_key_password', '_proxy', '_rate_limiter', '_read_timeout', '_request', '_socket_options', '_token', '_update_queue', '_updater', '_write_timeout', '_local_mode', '_http_version')

    def __init__(self: 'InitApplicationBuilder'):
        if False:
            for i in range(10):
                print('nop')
        self._token: DVType[str] = DefaultValue('')
        self._base_url: DVType[str] = DefaultValue('https://api.telegram.org/bot')
        self._base_file_url: DVType[str] = DefaultValue('https://api.telegram.org/file/bot')
        self._connection_pool_size: DVInput[int] = DEFAULT_NONE
        self._proxy: DVInput[Union[str, httpx.Proxy, httpx.URL]] = DEFAULT_NONE
        self._socket_options: DVInput[Collection[SocketOpt]] = DEFAULT_NONE
        self._connect_timeout: ODVInput[float] = DEFAULT_NONE
        self._read_timeout: ODVInput[float] = DEFAULT_NONE
        self._write_timeout: ODVInput[float] = DEFAULT_NONE
        self._pool_timeout: ODVInput[float] = DEFAULT_NONE
        self._request: DVInput[BaseRequest] = DEFAULT_NONE
        self._get_updates_connection_pool_size: DVInput[int] = DEFAULT_NONE
        self._get_updates_proxy: DVInput[Union[str, httpx.Proxy, httpx.URL]] = DEFAULT_NONE
        self._get_updates_socket_options: DVInput[Collection[SocketOpt]] = DEFAULT_NONE
        self._get_updates_connect_timeout: ODVInput[float] = DEFAULT_NONE
        self._get_updates_read_timeout: ODVInput[float] = DEFAULT_NONE
        self._get_updates_write_timeout: ODVInput[float] = DEFAULT_NONE
        self._get_updates_pool_timeout: ODVInput[float] = DEFAULT_NONE
        self._get_updates_request: DVInput[BaseRequest] = DEFAULT_NONE
        self._get_updates_http_version: DVInput[str] = DefaultValue('1.1')
        self._private_key: ODVInput[bytes] = DEFAULT_NONE
        self._private_key_password: ODVInput[bytes] = DEFAULT_NONE
        self._defaults: ODVInput[Defaults] = DEFAULT_NONE
        self._arbitrary_callback_data: Union[DefaultValue[bool], int] = DEFAULT_FALSE
        self._local_mode: DVType[bool] = DEFAULT_FALSE
        self._bot: DVInput[Bot] = DEFAULT_NONE
        self._update_queue: DVType[Queue[Union[Update, object]]] = DefaultValue(Queue())
        try:
            self._job_queue: ODVInput[JobQueue] = DefaultValue(JobQueue())
        except RuntimeError as exc:
            if 'PTB must be installed via' not in str(exc):
                raise exc
            self._job_queue = DEFAULT_NONE
        self._persistence: ODVInput[BasePersistence] = DEFAULT_NONE
        self._context_types: DVType[ContextTypes] = DefaultValue(ContextTypes())
        self._application_class: DVType[Type[Application]] = DefaultValue(Application)
        self._application_kwargs: Dict[str, object] = {}
        self._update_processor: BaseUpdateProcessor = SimpleUpdateProcessor(max_concurrent_updates=1)
        self._updater: ODVInput[Updater] = DEFAULT_NONE
        self._post_init: Optional[Callable[[Application], Coroutine[Any, Any, None]]] = None
        self._post_shutdown: Optional[Callable[[Application], Coroutine[Any, Any, None]]] = None
        self._post_stop: Optional[Callable[[Application], Coroutine[Any, Any, None]]] = None
        self._rate_limiter: ODVInput[BaseRateLimiter] = DEFAULT_NONE
        self._http_version: DVInput[str] = DefaultValue('1.1')

    def _build_request(self, get_updates: bool) -> BaseRequest:
        if False:
            return 10
        prefix = '_get_updates_' if get_updates else '_'
        if not isinstance(getattr(self, f'{prefix}request'), DefaultValue):
            return getattr(self, f'{prefix}request')
        proxy = DefaultValue.get_value(getattr(self, f'{prefix}proxy'))
        socket_options = DefaultValue.get_value(getattr(self, f'{prefix}socket_options'))
        if get_updates:
            connection_pool_size = DefaultValue.get_value(getattr(self, f'{prefix}connection_pool_size')) or 1
        else:
            connection_pool_size = DefaultValue.get_value(getattr(self, f'{prefix}connection_pool_size')) or 256
        timeouts = {'connect_timeout': getattr(self, f'{prefix}connect_timeout'), 'read_timeout': getattr(self, f'{prefix}read_timeout'), 'write_timeout': getattr(self, f'{prefix}write_timeout'), 'pool_timeout': getattr(self, f'{prefix}pool_timeout')}
        effective_timeouts = {key: value for (key, value) in timeouts.items() if not isinstance(value, DefaultValue)}
        http_version = DefaultValue.get_value(getattr(self, f'{prefix}http_version')) or '1.1'
        return HTTPXRequest(connection_pool_size=connection_pool_size, proxy=proxy, http_version=http_version, socket_options=socket_options, **effective_timeouts)

    def _build_ext_bot(self) -> ExtBot:
        if False:
            i = 10
            return i + 15
        if isinstance(self._token, DefaultValue):
            raise RuntimeError('No bot token was set.')
        return ExtBot(token=self._token, base_url=DefaultValue.get_value(self._base_url), base_file_url=DefaultValue.get_value(self._base_file_url), private_key=DefaultValue.get_value(self._private_key), private_key_password=DefaultValue.get_value(self._private_key_password), defaults=DefaultValue.get_value(self._defaults), arbitrary_callback_data=DefaultValue.get_value(self._arbitrary_callback_data), request=self._build_request(get_updates=False), get_updates_request=self._build_request(get_updates=True), rate_limiter=DefaultValue.get_value(self._rate_limiter), local_mode=DefaultValue.get_value(self._local_mode))

    def _bot_check(self, name: str) -> None:
        if False:
            return 10
        if self._bot is not DEFAULT_NONE:
            raise RuntimeError(_TWO_ARGS_REQ.format(name, 'bot instance'))

    def _updater_check(self, name: str) -> None:
        if False:
            return 10
        if self._updater not in (DEFAULT_NONE, None):
            raise RuntimeError(_TWO_ARGS_REQ.format(name, 'updater'))

    def build(self: 'ApplicationBuilder[BT, CCT, UD, CD, BD, JQ]') -> Application[BT, CCT, UD, CD, BD, JQ]:
        if False:
            return 10
        'Builds a :class:`telegram.ext.Application` with the provided arguments.\n\n        Calls :meth:`telegram.ext.JobQueue.set_application` and\n        :meth:`telegram.ext.BasePersistence.set_bot` if appropriate.\n\n        Returns:\n            :class:`telegram.ext.Application`\n        '
        job_queue = DefaultValue.get_value(self._job_queue)
        persistence = DefaultValue.get_value(self._persistence)
        if isinstance(self._updater, DefaultValue) or self._updater is None:
            if isinstance(self._bot, DefaultValue):
                bot: Bot = self._build_ext_bot()
            else:
                bot = self._bot
            update_queue = DefaultValue.get_value(self._update_queue)
            if self._updater is None:
                updater = None
            else:
                updater = Updater(bot=bot, update_queue=update_queue)
        else:
            updater = self._updater
            bot = self._updater.bot
            update_queue = self._updater.update_queue
        application: Application[BT, CCT, UD, CD, BD, JQ] = DefaultValue.get_value(self._application_class)(bot=bot, update_queue=update_queue, updater=updater, update_processor=self._update_processor, job_queue=job_queue, persistence=persistence, context_types=DefaultValue.get_value(self._context_types), post_init=self._post_init, post_shutdown=self._post_shutdown, post_stop=self._post_stop, **self._application_kwargs)
        if job_queue is not None:
            job_queue.set_application(application)
        if persistence is not None:
            persistence.set_bot(bot)
        return application

    def application_class(self: BuilderType, application_class: Type[Application[Any, Any, Any, Any, Any, Any]], kwargs: Optional[Dict[str, object]]=None) -> BuilderType:
        if False:
            for i in range(10):
                print('nop')
        "Sets a custom subclass instead of :class:`telegram.ext.Application`. The\n        subclass's ``__init__`` should look like this\n\n        .. code:: python\n\n            def __init__(self, custom_arg_1, custom_arg_2, ..., **kwargs):\n                super().__init__(**kwargs)\n                self.custom_arg_1 = custom_arg_1\n                self.custom_arg_2 = custom_arg_2\n\n        Args:\n            application_class (:obj:`type`): A subclass of :class:`telegram.ext.Application`\n            kwargs (Dict[:obj:`str`, :obj:`object`], optional): Keyword arguments for the\n                initialization. Defaults to an empty dict.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        "
        self._application_class = application_class
        self._application_kwargs = kwargs or {}
        return self

    def token(self: BuilderType, token: str) -> BuilderType:
        if False:
            print('Hello World!')
        'Sets the token for :attr:`telegram.ext.Application.bot`.\n\n        Args:\n            token (:obj:`str`): The token.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._bot_check('token')
        self._updater_check('token')
        self._token = token
        return self

    def base_url(self: BuilderType, base_url: str) -> BuilderType:
        if False:
            print('Hello World!')
        "Sets the base URL for :attr:`telegram.ext.Application.bot`. If not called,\n        will default to ``'https://api.telegram.org/bot'``.\n\n        .. seealso:: :paramref:`telegram.Bot.base_url`,\n            :wiki:`Local Bot API Server <Local-Bot-API-Server>`, :meth:`base_file_url`\n\n        Args:\n            base_url (:obj:`str`): The URL.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        "
        self._bot_check('base_url')
        self._updater_check('base_url')
        self._base_url = base_url
        return self

    def base_file_url(self: BuilderType, base_file_url: str) -> BuilderType:
        if False:
            while True:
                i = 10
        "Sets the base file URL for :attr:`telegram.ext.Application.bot`. If not\n        called, will default to ``'https://api.telegram.org/file/bot'``.\n\n        .. seealso:: :paramref:`telegram.Bot.base_file_url`,\n            :wiki:`Local Bot API Server <Local-Bot-API-Server>`, :meth:`base_url`\n\n        Args:\n            base_file_url (:obj:`str`): The URL.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        "
        self._bot_check('base_file_url')
        self._updater_check('base_file_url')
        self._base_file_url = base_file_url
        return self

    def _request_check(self, get_updates: bool) -> None:
        if False:
            print('Hello World!')
        prefix = 'get_updates_' if get_updates else ''
        name = prefix + 'request'
        for attr in ('connect_timeout', 'read_timeout', 'write_timeout', 'pool_timeout'):
            if not isinstance(getattr(self, f'_{prefix}{attr}'), DefaultValue):
                raise RuntimeError(_TWO_ARGS_REQ.format(name, attr))
        if not isinstance(getattr(self, f'_{prefix}connection_pool_size'), DefaultValue):
            raise RuntimeError(_TWO_ARGS_REQ.format(name, 'connection_pool_size'))
        if not isinstance(getattr(self, f'_{prefix}proxy'), DefaultValue):
            raise RuntimeError(_TWO_ARGS_REQ.format(name, 'proxy'))
        if not isinstance(getattr(self, f'_{prefix}socket_options'), DefaultValue):
            raise RuntimeError(_TWO_ARGS_REQ.format(name, 'socket_options'))
        if not isinstance(getattr(self, f'_{prefix}http_version'), DefaultValue):
            raise RuntimeError(_TWO_ARGS_REQ.format(name, 'http_version'))
        self._bot_check(name)
        if self._updater not in (DEFAULT_NONE, None):
            raise RuntimeError(_TWO_ARGS_REQ.format(name, 'updater instance'))

    def _request_param_check(self, name: str, get_updates: bool) -> None:
        if False:
            while True:
                i = 10
        if get_updates and self._get_updates_request is not DEFAULT_NONE:
            raise RuntimeError(_TWO_ARGS_REQ.format(f'get_updates_{name}', 'get_updates_request instance'))
        if self._request is not DEFAULT_NONE:
            raise RuntimeError(_TWO_ARGS_REQ.format(name, 'request instance'))
        if self._bot is not DEFAULT_NONE:
            raise RuntimeError(_TWO_ARGS_REQ.format(f'get_updates_{name}' if get_updates else name, 'bot instance'))
        if self._updater not in (DEFAULT_NONE, None):
            raise RuntimeError(_TWO_ARGS_REQ.format(f'get_updates_{name}' if get_updates else name, 'updater'))

    def request(self: BuilderType, request: BaseRequest) -> BuilderType:
        if False:
            i = 10
            return i + 15
        'Sets a :class:`telegram.request.BaseRequest` instance for the\n        :paramref:`telegram.Bot.request` parameter of :attr:`telegram.ext.Application.bot`.\n\n        .. seealso:: :meth:`get_updates_request`\n\n        Args:\n            request (:class:`telegram.request.BaseRequest`): The request instance.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_check(get_updates=False)
        self._request = request
        return self

    def connection_pool_size(self: BuilderType, connection_pool_size: int) -> BuilderType:
        if False:
            i = 10
            return i + 15
        'Sets the size of the connection pool for the\n        :paramref:`~telegram.request.HTTPXRequest.connection_pool_size` parameter of\n        :attr:`telegram.Bot.request`. Defaults to ``256``.\n\n        .. include:: inclusions/pool_size_tip.rst\n\n        .. seealso:: :meth:`get_updates_connection_pool_size`\n\n        Args:\n            connection_pool_size (:obj:`int`): The size of the connection pool.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='connection_pool_size', get_updates=False)
        self._connection_pool_size = connection_pool_size
        return self

    def proxy_url(self: BuilderType, proxy_url: str) -> BuilderType:
        if False:
            for i in range(10):
                print('nop')
        'Legacy name for :meth:`proxy`, kept for backward compatibility.\n\n        .. seealso:: :meth:`get_updates_proxy`\n\n        .. deprecated:: NEXT.VERSION\n\n        Args:\n            proxy_url (:obj:`str` | ``httpx.Proxy`` | ``httpx.URL``): See\n                :paramref:`telegram.ext.ApplicationBuilder.proxy.proxy`.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        warn('`ApplicationBuilder.proxy_url` is deprecated since version NEXT.VERSION. Use `ApplicationBuilder.proxy` instead.', PTBDeprecationWarning, stacklevel=2)
        return self.proxy(proxy_url)

    def proxy(self: BuilderType, proxy: Union[str, httpx.Proxy, httpx.URL]) -> BuilderType:
        if False:
            i = 10
            return i + 15
        'Sets the proxy for the :paramref:`~telegram.request.HTTPXRequest.proxy`\n        parameter of :attr:`telegram.Bot.request`. Defaults to :obj:`None`.\n\n        .. seealso:: :meth:`get_updates_proxy`\n\n        .. versionadded:: NEXT.VERSION\n\n        Args:\n            proxy (:obj:`str` | ``httpx.Proxy`` | ``httpx.URL``): The URL to a proxy\n                server, a ``httpx.Proxy`` object or a ``httpx.URL`` object. See\n                :paramref:`telegram.request.HTTPXRequest.proxy` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='proxy', get_updates=False)
        self._proxy = proxy
        return self

    def socket_options(self: BuilderType, socket_options: Collection[SocketOpt]) -> BuilderType:
        if False:
            print('Hello World!')
        'Sets the options for the :paramref:`~telegram.request.HTTPXRequest.socket_options`\n        parameter of :attr:`telegram.Bot.request`. Defaults to :obj:`None`.\n\n        .. seealso:: :meth:`get_updates_socket_options`\n\n        .. versionadded:: NEXT.VERSION\n\n        Args:\n            socket_options (Collection[:obj:`tuple`], optional): Socket options. See\n                :paramref:`telegram.request.HTTPXRequest.socket_options` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='socket_options', get_updates=False)
        self._socket_options = socket_options
        return self

    def connect_timeout(self: BuilderType, connect_timeout: Optional[float]) -> BuilderType:
        if False:
            return 10
        'Sets the connection attempt timeout for the\n        :paramref:`~telegram.request.HTTPXRequest.connect_timeout` parameter of\n        :attr:`telegram.Bot.request`. Defaults to ``5.0``.\n\n        .. seealso:: :meth:`get_updates_connect_timeout`\n\n        Args:\n            connect_timeout (:obj:`float`): See\n                :paramref:`telegram.request.HTTPXRequest.connect_timeout` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='connect_timeout', get_updates=False)
        self._connect_timeout = connect_timeout
        return self

    def read_timeout(self: BuilderType, read_timeout: Optional[float]) -> BuilderType:
        if False:
            i = 10
            return i + 15
        'Sets the waiting timeout for the\n        :paramref:`~telegram.request.HTTPXRequest.read_timeout` parameter of\n        :attr:`telegram.Bot.request`. Defaults to ``5.0``.\n\n        .. seealso:: :meth:`get_updates_read_timeout`\n\n        Args:\n            read_timeout (:obj:`float`): See\n                :paramref:`telegram.request.HTTPXRequest.read_timeout` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='read_timeout', get_updates=False)
        self._read_timeout = read_timeout
        return self

    def write_timeout(self: BuilderType, write_timeout: Optional[float]) -> BuilderType:
        if False:
            print('Hello World!')
        'Sets the write operation timeout for the\n        :paramref:`~telegram.request.HTTPXRequest.write_timeout` parameter of\n        :attr:`telegram.Bot.request`. Defaults to ``5.0``.\n\n        .. seealso:: :meth:`get_updates_write_timeout`\n\n        Args:\n            write_timeout (:obj:`float`): See\n                :paramref:`telegram.request.HTTPXRequest.write_timeout` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='write_timeout', get_updates=False)
        self._write_timeout = write_timeout
        return self

    def pool_timeout(self: BuilderType, pool_timeout: Optional[float]) -> BuilderType:
        if False:
            while True:
                i = 10
        "Sets the connection pool's connection freeing timeout for the\n        :paramref:`~telegram.request.HTTPXRequest.pool_timeout` parameter of\n        :attr:`telegram.Bot.request`. Defaults to ``1.0``.\n\n        .. include:: inclusions/pool_size_tip.rst\n\n        .. seealso:: :meth:`get_updates_pool_timeout`\n\n        Args:\n            pool_timeout (:obj:`float`): See\n                :paramref:`telegram.request.HTTPXRequest.pool_timeout` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        "
        self._request_param_check(name='pool_timeout', get_updates=False)
        self._pool_timeout = pool_timeout
        return self

    def http_version(self: BuilderType, http_version: HTTPVersion) -> BuilderType:
        if False:
            return 10
        'Sets the HTTP protocol version which is used for the\n        :paramref:`~telegram.request.HTTPXRequest.http_version` parameter of\n        :attr:`telegram.Bot.request`. By default, HTTP/1.1 is used.\n\n        .. seealso:: :meth:`get_updates_http_version`\n\n        Note:\n            Users have observed stability issues with HTTP/2, which happen due to how the `h2\n            library handles <https://github.com/python-hyper/h2/issues/1181>`_ cancellations of\n            keepalive connections. See `#3556 <https://github.com/python-telegram-bot/\n            python-telegram-bot/issues/3556>`_ for a discussion.\n\n            If you want to use HTTP/2, you must install PTB with the optional requirement\n            ``http2``, i.e.\n\n            .. code-block:: bash\n\n               pip install "python-telegram-bot[http2]"\n\n            Keep in mind that the HTTP/1.1 implementation may be considered the `"more\n            robust option at this time" <https://www.python-httpx.org/http2#enabling-http2>`_.\n\n        .. versionadded:: 20.1\n        .. versionchanged:: 20.2\n            Reset the default version to 1.1.\n\n        Args:\n            http_version (:obj:`str`): Pass ``"2"`` or ``"2.0"`` if you\'d like to use HTTP/2 for\n                making requests to Telegram. Defaults to ``"1.1"``, in which case HTTP/1.1 is used.\n\n                .. versionchanged:: 20.5\n                    Accept ``"2"`` as a valid value.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='http_version', get_updates=False)
        self._http_version = http_version
        return self

    def get_updates_request(self: BuilderType, get_updates_request: BaseRequest) -> BuilderType:
        if False:
            print('Hello World!')
        'Sets a :class:`telegram.request.BaseRequest` instance for the\n        :paramref:`~telegram.Bot.get_updates_request` parameter of\n        :attr:`telegram.ext.Application.bot`.\n\n        .. seealso:: :meth:`request`\n\n        Args:\n            get_updates_request (:class:`telegram.request.BaseRequest`): The request instance.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_check(get_updates=True)
        self._get_updates_request = get_updates_request
        return self

    def get_updates_connection_pool_size(self: BuilderType, get_updates_connection_pool_size: int) -> BuilderType:
        if False:
            i = 10
            return i + 15
        'Sets the size of the connection pool for the\n        :paramref:`telegram.request.HTTPXRequest.connection_pool_size` parameter which is used\n        for the :meth:`telegram.Bot.get_updates` request. Defaults to ``1``.\n\n        .. seealso:: :meth:`connection_pool_size`\n\n        Args:\n            get_updates_connection_pool_size (:obj:`int`): The size of the connection pool.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='connection_pool_size', get_updates=True)
        self._get_updates_connection_pool_size = get_updates_connection_pool_size
        return self

    def get_updates_proxy_url(self: BuilderType, get_updates_proxy_url: str) -> BuilderType:
        if False:
            i = 10
            return i + 15
        'Legacy name for :meth:`get_updates_proxy`, kept for backward compatibility.\n\n        .. seealso:: :meth:`proxy`\n\n        .. deprecated:: NEXT.VERSION\n\n        Args:\n            get_updates_proxy_url (:obj:`str` | ``httpx.Proxy`` | ``httpx.URL``): See\n                :paramref:`telegram.ext.ApplicationBuilder.get_updates_proxy.get_updates_proxy`.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        warn('`ApplicationBuilder.get_updates_proxy_url` is deprecated since version NEXT.VERSION. Use `ApplicationBuilder.get_updates_proxy` instead.', PTBDeprecationWarning, stacklevel=2)
        return self.get_updates_proxy(get_updates_proxy_url)

    def get_updates_proxy(self: BuilderType, get_updates_proxy: Union[str, httpx.Proxy, httpx.URL]) -> BuilderType:
        if False:
            while True:
                i = 10
        'Sets the proxy for the :paramref:`telegram.request.HTTPXRequest.proxy`\n        parameter which is used for :meth:`telegram.Bot.get_updates`. Defaults to :obj:`None`.\n\n        .. seealso:: :meth:`proxy`\n\n        .. versionadded:: NEXT.VERSION\n\n        Args:\n            proxy (:obj:`str` | ``httpx.Proxy`` | ``httpx.URL``): The URL to a proxy server,\n                a ``httpx.Proxy`` object or a ``httpx.URL`` object. See\n                :paramref:`telegram.request.HTTPXRequest.proxy` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='proxy', get_updates=True)
        self._get_updates_proxy = get_updates_proxy
        return self

    def get_updates_socket_options(self: BuilderType, get_updates_socket_options: Collection[SocketOpt]) -> BuilderType:
        if False:
            while True:
                i = 10
        'Sets the options for the :paramref:`~telegram.request.HTTPXRequest.socket_options`\n        parameter of :paramref:`telegram.Bot.get_updates_request`. Defaults to :obj:`None`.\n\n        .. seealso:: :meth:`socket_options`\n\n        .. versionadded:: NEXT.VERSION\n\n        Args:\n            get_updates_socket_options (Collection[:obj:`tuple`], optional): Socket options. See\n                :paramref:`telegram.request.HTTPXRequest.socket_options` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='socket_options', get_updates=True)
        self._get_updates_socket_options = get_updates_socket_options
        return self

    def get_updates_connect_timeout(self: BuilderType, get_updates_connect_timeout: Optional[float]) -> BuilderType:
        if False:
            print('Hello World!')
        'Sets the connection attempt timeout for the\n        :paramref:`telegram.request.HTTPXRequest.connect_timeout` parameter which is used for\n        the :meth:`telegram.Bot.get_updates` request. Defaults to ``5.0``.\n\n        .. seealso:: :meth:`connect_timeout`\n\n        Args:\n            get_updates_connect_timeout (:obj:`float`): See\n                :paramref:`telegram.request.HTTPXRequest.connect_timeout` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='connect_timeout', get_updates=True)
        self._get_updates_connect_timeout = get_updates_connect_timeout
        return self

    def get_updates_read_timeout(self: BuilderType, get_updates_read_timeout: Optional[float]) -> BuilderType:
        if False:
            return 10
        'Sets the waiting timeout for the\n        :paramref:`telegram.request.HTTPXRequest.read_timeout` parameter which is used for the\n        :meth:`telegram.Bot.get_updates` request. Defaults to ``5.0``.\n\n        .. seealso:: :meth:`read_timeout`\n\n        Args:\n            get_updates_read_timeout (:obj:`float`): See\n                :paramref:`telegram.request.HTTPXRequest.read_timeout` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='read_timeout', get_updates=True)
        self._get_updates_read_timeout = get_updates_read_timeout
        return self

    def get_updates_write_timeout(self: BuilderType, get_updates_write_timeout: Optional[float]) -> BuilderType:
        if False:
            i = 10
            return i + 15
        'Sets the write operation timeout for the\n        :paramref:`telegram.request.HTTPXRequest.write_timeout` parameter which is used for\n        the :meth:`telegram.Bot.get_updates` request. Defaults to ``5.0``.\n\n        .. seealso:: :meth:`write_timeout`\n\n        Args:\n            get_updates_write_timeout (:obj:`float`): See\n                :paramref:`telegram.request.HTTPXRequest.write_timeout` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='write_timeout', get_updates=True)
        self._get_updates_write_timeout = get_updates_write_timeout
        return self

    def get_updates_pool_timeout(self: BuilderType, get_updates_pool_timeout: Optional[float]) -> BuilderType:
        if False:
            i = 10
            return i + 15
        "Sets the connection pool's connection freeing timeout for the\n        :paramref:`~telegram.request.HTTPXRequest.pool_timeout` parameter which is used for the\n        :meth:`telegram.Bot.get_updates` request. Defaults to ``1.0``.\n\n        .. seealso:: :meth:`pool_timeout`\n\n        Args:\n            get_updates_pool_timeout (:obj:`float`): See\n                :paramref:`telegram.request.HTTPXRequest.pool_timeout` for more information.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        "
        self._request_param_check(name='pool_timeout', get_updates=True)
        self._get_updates_pool_timeout = get_updates_pool_timeout
        return self

    def get_updates_http_version(self: BuilderType, get_updates_http_version: HTTPVersion) -> BuilderType:
        if False:
            for i in range(10):
                print('nop')
        'Sets the HTTP protocol version which is used for the\n        :paramref:`~telegram.request.HTTPXRequest.http_version` parameter which is used in the\n        :meth:`telegram.Bot.get_updates` request. By default, HTTP/1.1 is used.\n\n        .. seealso:: :meth:`http_version`\n\n        Note:\n            Users have observed stability issues with HTTP/2, which happen due to how the `h2\n            library handles <https://github.com/python-hyper/h2/issues/1181>`_ cancellations of\n            keepalive connections. See `#3556 <https://github.com/python-telegram-bot/\n            python-telegram-bot/issues/3556>`_ for a discussion.\n\n            You will also need to install the http2 dependency. Keep in mind that the HTTP/1.1\n            implementation may be considered the `"more robust option at this time"\n            <https://www.python-httpx.org/http2#enabling-http2>`_.\n\n            .. code-block:: bash\n\n               pip install httpx[http2]\n\n        .. versionadded:: 20.1\n        .. versionchanged:: 20.2\n            Reset the default version to 1.1.\n\n        Args:\n            get_updates_http_version (:obj:`str`): Pass ``"2"`` or ``"2.0"`` if you\'d like to use\n                HTTP/2 for making requests to Telegram. Defaults to ``"1.1"``, in which case\n                HTTP/1.1 is used.\n\n                .. versionchanged:: 20.5\n                    Accept ``"2"`` as a valid value.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._request_param_check(name='http_version', get_updates=True)
        self._get_updates_http_version = get_updates_http_version
        return self

    def private_key(self: BuilderType, private_key: Union[bytes, FilePathInput], password: Optional[Union[bytes, FilePathInput]]=None) -> BuilderType:
        if False:
            for i in range(10):
                print('nop')
        "Sets the private key and corresponding password for decryption of telegram passport data\n        for :attr:`telegram.ext.Application.bot`.\n\n        Examples:\n            :any:`Passport Bot <examples.passportbot>`\n\n        .. seealso:: :wiki:`Telegram Passports <Telegram-Passport>`\n\n        Args:\n            private_key (:obj:`bytes` | :obj:`str` | :obj:`pathlib.Path`): The private key or the\n                file path of a file that contains the key. In the latter case, the file's content\n                will be read automatically.\n            password (:obj:`bytes` | :obj:`str` | :obj:`pathlib.Path`, optional): The corresponding\n                password or the file path of a file that contains the password. In the latter case,\n                the file's content will be read automatically.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        "
        self._bot_check('private_key')
        self._updater_check('private_key')
        self._private_key = private_key if isinstance(private_key, bytes) else Path(private_key).read_bytes()
        if password is None or isinstance(password, bytes):
            self._private_key_password = password
        else:
            self._private_key_password = Path(password).read_bytes()
        return self

    def defaults(self: BuilderType, defaults: 'Defaults') -> BuilderType:
        if False:
            return 10
        'Sets the :class:`telegram.ext.Defaults` instance for\n        :attr:`telegram.ext.Application.bot`.\n\n        .. seealso:: :wiki:`Adding Defaults to Your Bot <Adding-defaults-to-your-bot>`\n\n        Args:\n            defaults (:class:`telegram.ext.Defaults`): The defaults instance.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._bot_check('defaults')
        self._updater_check('defaults')
        self._defaults = defaults
        return self

    def arbitrary_callback_data(self: BuilderType, arbitrary_callback_data: Union[bool, int]) -> BuilderType:
        if False:
            return 10
        'Specifies whether :attr:`telegram.ext.Application.bot` should allow arbitrary objects as\n        callback data for :class:`telegram.InlineKeyboardButton` and how many keyboards should be\n        cached in memory. If not called, only strings can be used as callback data and no data will\n        be stored in memory.\n\n        Important:\n            If you want to use this feature, you must install PTB with the optional requirement\n            ``callback-data``, i.e.\n\n            .. code-block:: bash\n\n               pip install "python-telegram-bot[callback-data]"\n\n        Examples:\n            :any:`Arbitrary callback_data Bot <examples.arbitrarycallbackdatabot>`\n\n        .. seealso:: :wiki:`Arbitrary callback_data <Arbitrary-callback_data>`\n\n        Args:\n            arbitrary_callback_data (:obj:`bool` | :obj:`int`): If :obj:`True` is passed, the\n                default cache size of ``1024`` will be used. Pass an integer to specify a different\n                cache size.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._bot_check('arbitrary_callback_data')
        self._updater_check('arbitrary_callback_data')
        self._arbitrary_callback_data = arbitrary_callback_data
        return self

    def local_mode(self: BuilderType, local_mode: bool) -> BuilderType:
        if False:
            i = 10
            return i + 15
        'Specifies the value for :paramref:`~telegram.Bot.local_mode` for the\n        :attr:`telegram.ext.Application.bot`.\n        If not called, will default to :obj:`False`.\n\n        .. seealso:: :wiki:`Local Bot API Server <Local-Bot-API-Server>`\n\n        Args:\n            local_mode (:obj:`bool`): Whether the bot should run in local mode.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._bot_check('local_mode')
        self._updater_check('local_mode')
        self._local_mode = local_mode
        return self

    def bot(self: 'ApplicationBuilder[BT, CCT, UD, CD, BD, JQ]', bot: InBT) -> 'ApplicationBuilder[InBT, CCT, UD, CD, BD, JQ]':
        if False:
            i = 10
            return i + 15
        'Sets a :class:`telegram.Bot` instance for\n        :attr:`telegram.ext.Application.bot`. Instances of subclasses like\n        :class:`telegram.ext.ExtBot` are also valid.\n\n        Args:\n            bot (:class:`telegram.Bot`): The bot.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._updater_check('bot')
        for (attr, error) in _BOT_CHECKS:
            if not isinstance(getattr(self, f'_{attr}'), DefaultValue):
                raise RuntimeError(_TWO_ARGS_REQ.format('bot', error))
        self._bot = bot
        return self

    def update_queue(self: BuilderType, update_queue: 'Queue[object]') -> BuilderType:
        if False:
            i = 10
            return i + 15
        'Sets a :class:`asyncio.Queue` instance for\n        :attr:`telegram.ext.Application.update_queue`, i.e. the queue that the application will\n        fetch updates from. Will also be used for the :attr:`telegram.ext.Application.updater`.\n        If not called, a queue will be instantiated.\n\n        .. seealso:: :attr:`telegram.ext.Updater.update_queue`\n\n        Args:\n            update_queue (:class:`asyncio.Queue`): The queue.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        if self._updater not in (DEFAULT_NONE, None):
            raise RuntimeError(_TWO_ARGS_REQ.format('update_queue', 'updater instance'))
        self._update_queue = update_queue
        return self

    def concurrent_updates(self: BuilderType, concurrent_updates: Union[bool, int, 'BaseUpdateProcessor']) -> BuilderType:
        if False:
            while True:
                i = 10
        'Specifies if and how many updates may be processed concurrently instead of one by one.\n        If not called, updates will be processed one by one.\n\n        Warning:\n            Processing updates concurrently is not recommended when stateful handlers like\n            :class:`telegram.ext.ConversationHandler` are used. Only use this if you are sure\n            that your bot does not (explicitly or implicitly) rely on updates being processed\n            sequentially.\n\n        .. include:: inclusions/pool_size_tip.rst\n\n        .. seealso:: :attr:`telegram.ext.Application.concurrent_updates`\n\n        Args:\n            concurrent_updates (:obj:`bool` | :obj:`int` | :class:`BaseUpdateProcessor`): Passing\n                :obj:`True` will allow for ``256`` updates to be processed concurrently using\n                :class:`telegram.ext.SimpleUpdateProcessor`. Pass an integer to specify a different\n                number of updates that may be processed concurrently. Pass an instance of\n                :class:`telegram.ext.BaseUpdateProcessor` to use that instance for handling updates\n                concurrently.\n\n                .. versionchanged:: 20.4\n                    Now accepts :class:`BaseUpdateProcessor` instances.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        if concurrent_updates is True:
            concurrent_updates = 256
        elif concurrent_updates is False:
            concurrent_updates = 1
        if isinstance(concurrent_updates, int):
            concurrent_updates = SimpleUpdateProcessor(concurrent_updates)
        self._update_processor: BaseUpdateProcessor = concurrent_updates
        return self

    def job_queue(self: 'ApplicationBuilder[BT, CCT, UD, CD, BD, JQ]', job_queue: InJQ) -> 'ApplicationBuilder[BT, CCT, UD, CD, BD, InJQ]':
        if False:
            while True:
                i = 10
        "Sets a :class:`telegram.ext.JobQueue` instance for\n        :attr:`telegram.ext.Application.job_queue`. If not called, a job queue will be\n        instantiated if the requirements of :class:`telegram.ext.JobQueue` are installed.\n\n        Examples:\n            :any:`Timer Bot <examples.timerbot>`\n\n        .. seealso:: :wiki:`Job Queue <Extensions---JobQueue>`\n\n        Note:\n            * :meth:`telegram.ext.JobQueue.set_application` will be called automatically by\n              :meth:`build`.\n            * The job queue will be automatically started and stopped by\n              :meth:`telegram.ext.Application.start` and :meth:`telegram.ext.Application.stop`,\n              respectively.\n            * When passing :obj:`None` or when the requirements of :class:`telegram.ext.JobQueue`\n              are not installed, :attr:`telegram.ext.ConversationHandler.conversation_timeout`\n              can not be used, as this uses :attr:`telegram.ext.Application.job_queue` internally.\n\n        Args:\n            job_queue (:class:`telegram.ext.JobQueue`): The job queue. Pass :obj:`None` if you\n                don't want to use a job queue.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        "
        self._job_queue = job_queue
        return self

    def persistence(self: BuilderType, persistence: 'BasePersistence[Any, Any, Any]') -> BuilderType:
        if False:
            return 10
        'Sets a :class:`telegram.ext.BasePersistence` instance for\n        :attr:`telegram.ext.Application.persistence`.\n\n        Note:\n            When using a persistence, note that all\n            data stored in :attr:`context.user_data <telegram.ext.CallbackContext.user_data>`,\n            :attr:`context.chat_data <telegram.ext.CallbackContext.chat_data>`,\n            :attr:`context.bot_data <telegram.ext.CallbackContext.bot_data>` and\n            in :attr:`telegram.ext.ExtBot.callback_data_cache` must be copyable with\n            :func:`copy.deepcopy`. This is due to the data being deep copied before handing it over\n            to the persistence in order to avoid race conditions.\n\n        Examples:\n            :any:`Persistent Conversation Bot <examples.persistentconversationbot>`\n\n        .. seealso:: :wiki:`Making Your Bot Persistent <Making-your-bot-persistent>`\n\n        Warning:\n            If a :class:`telegram.ext.ContextTypes` instance is set via :meth:`context_types`,\n            the persistence instance must use the same types!\n\n        Args:\n            persistence (:class:`telegram.ext.BasePersistence`): The persistence instance.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._persistence = persistence
        return self

    def context_types(self: 'ApplicationBuilder[BT, CCT, UD, CD, BD, JQ]', context_types: 'ContextTypes[InCCT, InUD, InCD, InBD]') -> 'ApplicationBuilder[BT, InCCT, InUD, InCD, InBD, JQ]':
        if False:
            for i in range(10):
                print('nop')
        'Sets a :class:`telegram.ext.ContextTypes` instance for\n        :attr:`telegram.ext.Application.context_types`.\n\n        Examples:\n            :any:`Context Types Bot <examples.contexttypesbot>`\n\n        Args:\n            context_types (:class:`telegram.ext.ContextTypes`): The context types.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._context_types = context_types
        return self

    def updater(self: BuilderType, updater: Optional[Updater]) -> BuilderType:
        if False:
            return 10
        'Sets a :class:`telegram.ext.Updater` instance for\n        :attr:`telegram.ext.Application.updater`. The :attr:`telegram.ext.Updater.bot` and\n        :attr:`telegram.ext.Updater.update_queue` will be used for\n        :attr:`telegram.ext.Application.bot` and :attr:`telegram.ext.Application.update_queue`,\n        respectively.\n\n        Args:\n            updater (:class:`telegram.ext.Updater` | :obj:`None`): The updater instance or\n                :obj:`None` if no updater should be used.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        if updater is None:
            self._updater = updater
            return self
        for (attr, error) in ((self._bot, 'bot instance'), (self._update_queue, 'update_queue')):
            if not isinstance(attr, DefaultValue):
                raise RuntimeError(_TWO_ARGS_REQ.format('updater', error))
        for (attr_name, error) in _BOT_CHECKS:
            if not isinstance(getattr(self, f'_{attr_name}'), DefaultValue):
                raise RuntimeError(_TWO_ARGS_REQ.format('updater', error))
        self._updater = updater
        return self

    def post_init(self: BuilderType, post_init: Callable[[Application], Coroutine[Any, Any, None]]) -> BuilderType:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets a callback to be executed by :meth:`Application.run_polling` and\n        :meth:`Application.run_webhook` *after* executing :meth:`Application.initialize` but\n        *before* executing :meth:`Updater.start_polling` or :meth:`Updater.start_webhook`,\n        respectively.\n\n        Tip:\n            This can be used for custom startup logic that requires to await coroutines, e.g.\n            setting up the bots commands via :meth:`~telegram.Bot.set_my_commands`.\n\n        Example:\n            .. code::\n\n                async def post_init(application: Application) -> None:\n                    await application.bot.set_my_commands([(\'start\', \'Starts the bot\')])\n\n                application = Application.builder().token("TOKEN").post_init(post_init).build()\n\n        Note:\n            |post_methods_note|\n\n        .. seealso:: :meth:`post_stop`, :meth:`post_shutdown`\n\n        Args:\n            post_init (:term:`coroutine function`): The custom callback. Must be a\n                :term:`coroutine function` and must accept exactly one positional argument, which\n                is the :class:`~telegram.ext.Application`::\n\n                    async def post_init(application: Application) -> None:\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._post_init = post_init
        return self

    def post_shutdown(self: BuilderType, post_shutdown: Callable[[Application], Coroutine[Any, Any, None]]) -> BuilderType:
        if False:
            return 10
        '\n        Sets a callback to be executed by :meth:`Application.run_polling` and\n        :meth:`Application.run_webhook` *after* executing :meth:`Updater.shutdown`\n        and :meth:`Application.shutdown`.\n\n        Tip:\n            This can be used for custom shutdown logic that requires to await coroutines, e.g.\n            closing a database connection\n\n        Example:\n            .. code::\n\n                async def post_shutdown(application: Application) -> None:\n                    await application.bot_data[\'database\'].close()\n\n                application = Application.builder()\n                                        .token("TOKEN")\n                                        .post_shutdown(post_shutdown)\n                                        .build()\n\n        Note:\n            |post_methods_note|\n\n        .. seealso:: :meth:`post_init`, :meth:`post_stop`\n\n        Args:\n            post_shutdown (:term:`coroutine function`): The custom callback. Must be a\n                :term:`coroutine function` and must accept exactly one positional argument, which\n                is the :class:`~telegram.ext.Application`::\n\n                    async def post_shutdown(application: Application) -> None:\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._post_shutdown = post_shutdown
        return self

    def post_stop(self: BuilderType, post_stop: Callable[[Application], Coroutine[Any, Any, None]]) -> BuilderType:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets a callback to be executed by :meth:`Application.run_polling` and\n        :meth:`Application.run_webhook` *after* executing :meth:`Updater.stop`\n        and :meth:`Application.stop`.\n\n        .. versionadded:: 20.1\n\n        Tip:\n            This can be used for custom stop logic that requires to await coroutines, e.g.\n            sending message to a chat before shutting down the bot\n\n        Example:\n            .. code::\n\n                async def post_stop(application: Application) -> None:\n                    await application.bot.send_message(123456, "Shutting down...")\n\n                application = Application.builder()\n                                        .token("TOKEN")\n                                        .post_stop(post_stop)\n                                        .build()\n\n        Note:\n            |post_methods_note|\n\n        .. seealso:: :meth:`post_init`, :meth:`post_shutdown`\n\n        Args:\n            post_stop (:term:`coroutine function`): The custom callback. Must be a\n                :term:`coroutine function` and must accept exactly one positional argument, which\n                is the :class:`~telegram.ext.Application`::\n\n                    async def post_stop(application: Application) -> None:\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._post_stop = post_stop
        return self

    def rate_limiter(self: 'ApplicationBuilder[BT, CCT, UD, CD, BD, JQ]', rate_limiter: 'BaseRateLimiter[RLARGS]') -> 'ApplicationBuilder[ExtBot[RLARGS], CCT, UD, CD, BD, JQ]':
        if False:
            i = 10
            return i + 15
        'Sets a :class:`telegram.ext.BaseRateLimiter` instance for the\n        :paramref:`telegram.ext.ExtBot.rate_limiter` parameter of\n        :attr:`telegram.ext.Application.bot`.\n\n        Args:\n            rate_limiter (:class:`telegram.ext.BaseRateLimiter`): The rate limiter.\n\n        Returns:\n            :class:`ApplicationBuilder`: The same builder with the updated argument.\n        '
        self._bot_check('rate_limiter')
        self._updater_check('rate_limiter')
        self._rate_limiter = rate_limiter
        return self
InitApplicationBuilder = ApplicationBuilder[ExtBot[None], ContextTypes.DEFAULT_TYPE, Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], JobQueue[ContextTypes.DEFAULT_TYPE]]