import asyncio
import json
from http import HTTPStatus
from ssl import SSLContext
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Type
import tornado.web
from tornado.httpserver import HTTPServer
from telegram import Update
from telegram._utils.logging import get_logger
from telegram.ext._extbot import ExtBot
if TYPE_CHECKING:
    from telegram import Bot
_LOGGER = get_logger(__name__, class_name='Updater')

class WebhookServer:
    """Thin wrapper around ``tornado.httpserver.HTTPServer``."""
    __slots__ = ('_http_server', 'listen', 'port', 'is_running', '_server_lock', '_shutdown_lock')

    def __init__(self, listen: str, port: int, webhook_app: 'WebhookAppClass', ssl_ctx: Optional[SSLContext]):
        if False:
            i = 10
            return i + 15
        self._http_server = HTTPServer(webhook_app, ssl_options=ssl_ctx)
        self.listen = listen
        self.port = port
        self.is_running = False
        self._server_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()

    async def serve_forever(self, ready: Optional[asyncio.Event]=None) -> None:
        async with self._server_lock:
            self._http_server.listen(self.port, address=self.listen)
            self.is_running = True
            if ready is not None:
                ready.set()
            _LOGGER.debug('Webhook Server started.')

    async def shutdown(self) -> None:
        async with self._shutdown_lock:
            if not self.is_running:
                _LOGGER.debug('Webhook Server is already shut down. Returning')
                return
            self.is_running = False
            self._http_server.stop()
            await self._http_server.close_all_connections()
            _LOGGER.debug('Webhook Server stopped')

class WebhookAppClass(tornado.web.Application):
    """Application used in the Webserver"""

    def __init__(self, webhook_path: str, bot: 'Bot', update_queue: asyncio.Queue, secret_token: Optional[str]=None):
        if False:
            print('Hello World!')
        self.shared_objects = {'bot': bot, 'update_queue': update_queue, 'secret_token': secret_token}
        handlers = [(f'{webhook_path}/?', TelegramHandler, self.shared_objects)]
        tornado.web.Application.__init__(self, handlers)

    def log_request(self, handler: tornado.web.RequestHandler) -> None:
        if False:
            while True:
                i = 10
        'Overrides the default implementation since we have our own logging setup.'

class TelegramHandler(tornado.web.RequestHandler):
    """BaseHandler that processes incoming requests from Telegram"""
    __slots__ = ('bot', 'update_queue', 'secret_token')
    SUPPORTED_METHODS = ('POST',)

    def initialize(self, bot: 'Bot', update_queue: asyncio.Queue, secret_token: str) -> None:
        if False:
            i = 10
            return i + 15
        "Initialize for each request - that's the interface provided by tornado"
        self.bot = bot
        self.update_queue = update_queue
        self.secret_token = secret_token
        if secret_token:
            _LOGGER.debug('The webhook server has a secret token, expecting it in incoming requests now')

    def set_default_headers(self) -> None:
        if False:
            return 10
        'Sets default headers'
        self.set_header('Content-Type', 'application/json; charset="utf-8"')

    async def post(self) -> None:
        """Handle incoming POST request"""
        _LOGGER.debug('Webhook triggered')
        self._validate_post()
        json_string = self.request.body.decode()
        data = json.loads(json_string)
        self.set_status(HTTPStatus.OK)
        _LOGGER.debug('Webhook received data: %s', json_string)
        try:
            update = Update.de_json(data, self.bot)
        except Exception as exc:
            _LOGGER.critical('Something went wrong processing the data received from Telegram. Received data was *not* processed!', exc_info=exc)
        if update:
            _LOGGER.debug('Received Update with ID %d on Webhook', update.update_id)
            if isinstance(self.bot, ExtBot):
                self.bot.insert_callback_data(update)
            await self.update_queue.put(update)

    def _validate_post(self) -> None:
        if False:
            i = 10
            return i + 15
        'Only accept requests with content type JSON'
        ct_header = self.request.headers.get('Content-Type', None)
        if ct_header != 'application/json':
            raise tornado.web.HTTPError(HTTPStatus.FORBIDDEN)
        if self.secret_token is not None:
            token = self.request.headers.get('X-Telegram-Bot-Api-Secret-Token')
            if not token:
                _LOGGER.debug('Request did not include the secret token')
                raise tornado.web.HTTPError(HTTPStatus.FORBIDDEN, reason='Request did not include the secret token')
            if token != self.secret_token:
                _LOGGER.debug('Request had the wrong secret token: %s', token)
                raise tornado.web.HTTPError(HTTPStatus.FORBIDDEN, reason='Request had the wrong secret token')

    def log_exception(self, typ: Optional[Type[BaseException]], value: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        if False:
            i = 10
            return i + 15
        'Override the default logging and instead use our custom logging.'
        _LOGGER.debug('%s - %s', self.request.remote_ip, 'Exception in TelegramHandler', exc_info=(typ, value, tb) if typ and value and tb else value)