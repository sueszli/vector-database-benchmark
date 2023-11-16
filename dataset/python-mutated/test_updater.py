import asyncio
import logging
from collections import defaultdict
from http import HTTPStatus
from pathlib import Path
from random import randrange
import pytest
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram._utils.defaultvalue import DEFAULT_NONE
from telegram.error import InvalidToken, RetryAfter, TelegramError, TimedOut
from telegram.ext import ExtBot, InvalidCallbackData, Updater
from telegram.request import HTTPXRequest
from tests.auxil.build_messages import make_message, make_message_update
from tests.auxil.envvars import TEST_WITH_OPT_DEPS
from tests.auxil.files import data_file
from tests.auxil.networking import send_webhook_message
from tests.auxil.pytest_classes import PytestBot, make_bot
from tests.auxil.slots import mro_slots
if TEST_WITH_OPT_DEPS:
    from telegram.ext._utils.webhookhandler import WebhookServer

@pytest.mark.skipif(TEST_WITH_OPT_DEPS, reason='Only relevant if the optional dependency is not installed')
class TestNoWebhooks:

    async def test_no_webhooks(self, bot):
        async with Updater(bot=bot, update_queue=asyncio.Queue()) as updater:
            with pytest.raises(RuntimeError, match='python-telegram-bot\\[webhooks\\]'):
                await updater.start_webhook()

@pytest.mark.skipif(not TEST_WITH_OPT_DEPS, reason='Only relevant if the optional dependency is installed')
class TestUpdater:
    message_count = 0
    received = None
    attempts = 0
    err_handler_called = None
    cb_handler_called = None
    offset = 0
    test_flag = False

    @pytest.fixture(autouse=True)
    def _reset(self):
        if False:
            i = 10
            return i + 15
        self.message_count = 0
        self.received = None
        self.attempts = 0
        self.err_handler_called = None
        self.cb_handler_called = None
        self.test_flag = False

    def error_callback(self, error):
        if False:
            print('Hello World!')
        self.received = error
        self.err_handler_called.set()

    def callback(self, update, context):
        if False:
            while True:
                i = 10
        self.received = update.message.text
        self.cb_handler_called.set()

    async def test_slot_behaviour(self, updater):
        async with updater:
            for at in updater.__slots__:
                attr = f'_Updater{at}' if at.startswith('__') and (not at.endswith('__')) else at
                assert getattr(updater, attr, 'err') != 'err', f"got extra slot '{attr}'"
            assert len(mro_slots(updater)) == len(set(mro_slots(updater))), 'duplicate slot'

    def test_init(self, bot):
        if False:
            for i in range(10):
                print('nop')
        queue = asyncio.Queue()
        updater = Updater(bot=bot, update_queue=queue)
        assert updater.bot is bot
        assert updater.update_queue is queue

    def test_repr(self, bot):
        if False:
            for i in range(10):
                print('nop')
        queue = asyncio.Queue()
        updater = Updater(bot=bot, update_queue=queue)
        assert repr(updater) == f'Updater[bot={updater.bot!r}]'

    async def test_initialize(self, bot, monkeypatch):

        async def initialize_bot(*args, **kwargs):
            self.test_flag = True
        async with make_bot(token=bot.token) as test_bot:
            monkeypatch.setattr(test_bot, 'initialize', initialize_bot)
            updater = Updater(bot=test_bot, update_queue=asyncio.Queue())
            await updater.initialize()
        assert self.test_flag

    async def test_shutdown(self, bot, monkeypatch):

        async def shutdown_bot(*args, **kwargs):
            self.test_flag = True
        async with make_bot(token=bot.token) as test_bot:
            monkeypatch.setattr(test_bot, 'shutdown', shutdown_bot)
            updater = Updater(bot=test_bot, update_queue=asyncio.Queue())
            await updater.initialize()
            await updater.shutdown()
        assert self.test_flag

    async def test_multiple_inits_and_shutdowns(self, updater, monkeypatch):
        self.test_flag = defaultdict(int)

        async def initialize(*args, **kargs):
            self.test_flag['init'] += 1

        async def shutdown(*args, **kwargs):
            self.test_flag['shutdown'] += 1
        monkeypatch.setattr(updater.bot, 'initialize', initialize)
        monkeypatch.setattr(updater.bot, 'shutdown', shutdown)
        await updater.initialize()
        await updater.initialize()
        await updater.initialize()
        await updater.shutdown()
        await updater.shutdown()
        await updater.shutdown()
        assert self.test_flag['init'] == 1
        assert self.test_flag['shutdown'] == 1

    async def test_multiple_init_cycles(self, updater):
        async with updater:
            await updater.bot.get_me()
        async with updater:
            await updater.bot.get_me()

    @pytest.mark.parametrize('method', ['start_polling', 'start_webhook'])
    async def test_start_without_initialize(self, updater, method):
        with pytest.raises(RuntimeError, match='not initialized'):
            await getattr(updater, method)()

    @pytest.mark.parametrize('method', ['start_polling', 'start_webhook'])
    async def test_shutdown_while_running(self, updater, method, monkeypatch):

        async def set_webhook(*args, **kwargs):
            return True
        monkeypatch.setattr(updater.bot, 'set_webhook', set_webhook)
        ip = '127.0.0.1'
        port = randrange(1024, 49152)
        async with updater:
            if 'webhook' in method:
                await getattr(updater, method)(ip_address=ip, port=port)
            else:
                await getattr(updater, method)()
            with pytest.raises(RuntimeError, match='still running'):
                await updater.shutdown()
            await updater.stop()

    async def test_context_manager(self, monkeypatch, updater):

        async def initialize(*args, **kwargs):
            self.test_flag = ['initialize']

        async def shutdown(*args, **kwargs):
            self.test_flag.append('stop')
        monkeypatch.setattr(Updater, 'initialize', initialize)
        monkeypatch.setattr(Updater, 'shutdown', shutdown)
        async with updater:
            pass
        assert self.test_flag == ['initialize', 'stop']

    async def test_context_manager_exception_on_init(self, monkeypatch, updater):

        async def initialize(*args, **kwargs):
            raise RuntimeError('initialize')

        async def shutdown(*args):
            self.test_flag = 'stop'
        monkeypatch.setattr(Updater, 'initialize', initialize)
        monkeypatch.setattr(Updater, 'shutdown', shutdown)
        with pytest.raises(RuntimeError, match='initialize'):
            async with updater:
                pass
        assert self.test_flag == 'stop'

    @pytest.mark.parametrize('drop_pending_updates', [True, False])
    async def test_polling_basic(self, monkeypatch, updater, drop_pending_updates):
        updates = asyncio.Queue()
        await updates.put(Update(update_id=1))
        await updates.put(Update(update_id=2))

        async def get_updates(*args, **kwargs):
            if not updates.empty():
                next_update = await updates.get()
                updates.task_done()
                return [next_update]
            await asyncio.sleep(0)
            return []
        orig_del_webhook = updater.bot.delete_webhook

        async def delete_webhook(*args, **kwargs):
            if kwargs.get('drop_pending_updates'):
                self.message_count += 1
            return await orig_del_webhook(*args, **kwargs)
        monkeypatch.setattr(updater.bot, 'get_updates', get_updates)
        monkeypatch.setattr(updater.bot, 'delete_webhook', delete_webhook)
        async with updater:
            return_value = await updater.start_polling(drop_pending_updates=drop_pending_updates)
            assert return_value is updater.update_queue
            assert updater.running
            await updates.join()
            await updater.stop()
            assert not updater.running
            assert not (await updater.bot.get_webhook_info()).url
            if drop_pending_updates:
                assert self.message_count == 1
            else:
                assert self.message_count == 0
            await updates.put(Update(update_id=3))
            await updates.put(Update(update_id=4))
            await updater.start_polling(drop_pending_updates=drop_pending_updates)
            assert updater.running
            tasks = asyncio.all_tasks()
            assert any(('Updater:start_polling:polling_task' in t.get_name() for t in tasks))
            await updates.join()
            await updater.stop()
            assert not updater.running
            assert not (await updater.bot.get_webhook_info()).url
        self.received = []
        self.message_count = 0
        while not updater.update_queue.empty():
            update = updater.update_queue.get_nowait()
            self.message_count += 1
            self.received.append(update.update_id)
        assert self.message_count == 4
        assert self.received == [1, 2, 3, 4]

    async def test_polling_mark_updates_as_read(self, monkeypatch, updater, caplog):
        updates = asyncio.Queue()
        max_update_id = 3
        for i in range(1, max_update_id + 1):
            await updates.put(Update(update_id=i))
        tracking_flag = False
        received_kwargs = {}
        expected_kwargs = {'timeout': 0, 'read_timeout': 'read_timeout', 'connect_timeout': 'connect_timeout', 'write_timeout': 'write_timeout', 'pool_timeout': 'pool_timeout', 'allowed_updates': 'allowed_updates'}

        async def get_updates(*args, **kwargs):
            if tracking_flag:
                received_kwargs.update(kwargs)
            if not updates.empty():
                next_update = await updates.get()
                updates.task_done()
                return [next_update]
            await asyncio.sleep(0)
            return []
        monkeypatch.setattr(updater.bot, 'get_updates', get_updates)
        async with updater:
            await updater.start_polling(**expected_kwargs)
            await updates.join()
            assert not received_kwargs
            tracking_flag = True
            with caplog.at_level(logging.DEBUG):
                await updater.stop()
        assert received_kwargs['offset'] == max_update_id + 1
        for (name, value) in expected_kwargs.items():
            assert received_kwargs[name] == value
        assert len(caplog.records) >= 1
        log_found = False
        for record in caplog.records:
            if not record.getMessage().startswith('Calling `get_updates` one more time'):
                continue
            assert record.name == 'telegram.ext.Updater'
            assert record.levelno == logging.DEBUG
            log_found = True
            break
        assert log_found

    async def test_polling_mark_updates_as_read_failure(self, monkeypatch, updater, caplog):

        async def get_updates(*args, **kwargs):
            await asyncio.sleep(0)
            return []
        monkeypatch.setattr(updater.bot, 'get_updates', get_updates)
        async with updater:
            await updater.start_polling()
            updater._Updater__polling_cleanup_cb = None
            with caplog.at_level(logging.DEBUG):
                await updater.stop()
        assert len(caplog.records) >= 1
        log_found = False
        for record in caplog.records:
            if not record.getMessage().startswith('No polling cleanup callback defined'):
                continue
            assert record.name == 'telegram.ext.Updater'
            assert record.levelno == logging.WARNING
            log_found = True
            break
        assert log_found

    async def test_start_polling_already_running(self, updater):
        async with updater:
            await updater.start_polling()
            task = asyncio.create_task(updater.start_polling())
            with pytest.raises(RuntimeError, match='already running'):
                await task
            await updater.stop()
            with pytest.raises(RuntimeError, match='not running'):
                await updater.stop()

    async def test_start_polling_get_updates_parameters(self, updater, monkeypatch):
        update_queue = asyncio.Queue()
        await update_queue.put(Update(update_id=1))
        on_stop_flag = False
        expected = {'timeout': 10, 'read_timeout': 2, 'write_timeout': DEFAULT_NONE, 'connect_timeout': DEFAULT_NONE, 'pool_timeout': DEFAULT_NONE, 'allowed_updates': None, 'api_kwargs': None}

        async def get_updates(*args, **kwargs):
            if on_stop_flag:
                await asyncio.sleep(0)
                return []
            for (key, value) in expected.items():
                assert kwargs.pop(key, None) == value
            offset = kwargs.pop('offset', None)
            assert kwargs == {}
            if offset is not None and self.message_count != 0:
                assert offset == self.message_count + 1, 'get_updates got wrong `offset` parameter'
            if not update_queue.empty():
                update = await update_queue.get()
                self.message_count = update.update_id
                update_queue.task_done()
                return [update]
            await asyncio.sleep(0)
            return []
        monkeypatch.setattr(updater.bot, 'get_updates', get_updates)
        async with updater:
            await updater.start_polling()
            await update_queue.join()
            on_stop_flag = True
            await updater.stop()
            on_stop_flag = False
            expected = {'timeout': 42, 'read_timeout': 43, 'write_timeout': 44, 'connect_timeout': 45, 'pool_timeout': 46, 'allowed_updates': ['message'], 'api_kwargs': None}
            await update_queue.put(Update(update_id=2))
            await updater.start_polling(timeout=42, read_timeout=43, write_timeout=44, connect_timeout=45, pool_timeout=46, allowed_updates=['message'])
            await update_queue.join()
            on_stop_flag = True
            await updater.stop()

    @pytest.mark.parametrize('exception_class', [InvalidToken, TelegramError])
    @pytest.mark.parametrize('retries', [3, 0])
    async def test_start_polling_bootstrap_retries(self, updater, monkeypatch, exception_class, retries):

        async def do_request(*args, **kwargs):
            self.message_count += 1
            raise exception_class(str(self.message_count))
        async with updater:
            monkeypatch.setattr(HTTPXRequest, 'do_request', do_request)
            if exception_class == InvalidToken:
                with pytest.raises(InvalidToken, match='1'):
                    await updater.start_polling(bootstrap_retries=retries)
            else:
                with pytest.raises(TelegramError, match=str(retries + 1)):
                    await updater.start_polling(bootstrap_retries=retries)

    @pytest.mark.parametrize(('error', 'callback_should_be_called'), argvalues=[(TelegramError('TestMessage'), True), (RetryAfter(1), False), (TimedOut('TestMessage'), False)], ids=('TelegramError', 'RetryAfter', 'TimedOut'))
    @pytest.mark.parametrize('custom_error_callback', [True, False])
    async def test_start_polling_exceptions_and_error_callback(self, monkeypatch, updater, error, callback_should_be_called, custom_error_callback, caplog):
        raise_exception = True
        get_updates_event = asyncio.Event()

        async def get_updates(*args, **kwargs):
            await asyncio.sleep(0)
            if not raise_exception:
                return []
            get_updates_event.set()
            raise error
        monkeypatch.setattr(updater.bot, 'get_updates', get_updates)
        monkeypatch.setattr(updater.bot, 'set_webhook', lambda *args, **kwargs: True)
        with pytest.raises(TypeError, match='`error_callback` must not be a coroutine function'):
            await updater.start_polling(error_callback=get_updates)
        async with updater:
            self.err_handler_called = asyncio.Event()
            with caplog.at_level(logging.ERROR):
                if custom_error_callback:
                    await updater.start_polling(error_callback=self.error_callback)
                else:
                    await updater.start_polling()
                await get_updates_event.wait()
                if callback_should_be_called:
                    if custom_error_callback:
                        assert self.received == error
                    else:
                        assert len(caplog.records) > 0
                        assert any(('Error while getting Updates: TestMessage' in record.getMessage() and record.name == 'telegram.ext.Updater' for record in caplog.records))
                assert get_updates_event.is_set()
            self.err_handler_called.clear()
            get_updates_event.clear()
            caplog.clear()
            await get_updates_event.wait()
            if callback_should_be_called:
                if custom_error_callback:
                    assert self.received == error
                else:
                    assert len(caplog.records) > 0
                    assert any(('Error while getting Updates: TestMessage' in record.getMessage() and record.name == 'telegram.ext.Updater' for record in caplog.records))
            raise_exception = False
            await updater.stop()

    async def test_start_polling_unexpected_shutdown(self, updater, monkeypatch, caplog):
        update_queue = asyncio.Queue()
        await update_queue.put(Update(update_id=1))
        await update_queue.put(Update(update_id=2))
        first_update_event = asyncio.Event()
        second_update_event = asyncio.Event()

        async def get_updates(*args, **kwargs):
            self.message_count = kwargs.get('offset')
            update = await update_queue.get()
            if update.update_id == 1:
                first_update_event.set()
            else:
                await second_update_event.wait()
            return [update]
        monkeypatch.setattr(updater.bot, 'get_updates', get_updates)
        async with updater:
            with caplog.at_level(logging.ERROR):
                await updater.start_polling()
                await first_update_event.wait()
                updater._running = False
                second_update_event.set()
                await asyncio.sleep(0.1)
                assert caplog.records
                assert any(('Updater stopped unexpectedly.' in record.getMessage() and record.name == 'telegram.ext.Updater' for record in caplog.records))
        assert self.message_count == 2

    async def test_start_polling_not_running_after_failure(self, updater, monkeypatch):

        async def _start_polling(*args, **kwargs):
            raise Exception('Test Exception')
        monkeypatch.setattr(Updater, '_start_polling', _start_polling)
        async with updater:
            with pytest.raises(Exception, match='Test Exception'):
                await updater.start_polling()
            assert updater.running is False

    async def test_polling_update_de_json_fails(self, monkeypatch, updater, caplog):
        updates = asyncio.Queue()
        raise_exception = True
        await updates.put(Update(update_id=1))

        async def get_updates(*args, **kwargs):
            if raise_exception:
                await asyncio.sleep(0.01)
                raise TypeError('Invalid Data')
            if not updates.empty():
                next_update = await updates.get()
                updates.task_done()
                return [next_update]
            await asyncio.sleep(0)
            return []
        orig_del_webhook = updater.bot.delete_webhook

        async def delete_webhook(*args, **kwargs):
            if kwargs.get('drop_pending_updates'):
                self.message_count += 1
            return await orig_del_webhook(*args, **kwargs)
        monkeypatch.setattr(updater.bot, 'get_updates', get_updates)
        monkeypatch.setattr(updater.bot, 'delete_webhook', delete_webhook)
        async with updater:
            with caplog.at_level(logging.CRITICAL):
                await updater.start_polling()
                assert updater.running
                await asyncio.sleep(1)
            assert len(caplog.records) > 0
            for record in caplog.records:
                assert record.getMessage().startswith('Something went wrong processing')
                assert record.name == 'telegram.ext.Updater'
            raise_exception = False
            await asyncio.sleep(0.5)
            caplog.clear()
            with caplog.at_level(logging.CRITICAL):
                await updates.join()
            assert len(caplog.records) == 0
            await updater.stop()
            assert not updater.running

    @pytest.mark.parametrize('ext_bot', [True, False])
    @pytest.mark.parametrize('drop_pending_updates', [True, False])
    @pytest.mark.parametrize('secret_token', ['SecretToken', None])
    async def test_webhook_basic(self, monkeypatch, updater, drop_pending_updates, ext_bot, secret_token):
        if ext_bot and (not isinstance(updater.bot, ExtBot)):
            updater.bot = ExtBot(updater.bot.token)
        if not ext_bot and type(updater.bot) is not Bot:
            updater.bot = PytestBot(updater.bot.token)

        async def delete_webhook(*args, **kwargs):
            if kwargs.get('drop_pending_updates'):
                self.message_count += 1
            return True

        async def set_webhook(*args, **kwargs):
            return True
        monkeypatch.setattr(updater.bot, 'set_webhook', set_webhook)
        monkeypatch.setattr(updater.bot, 'delete_webhook', delete_webhook)
        ip = '127.0.0.1'
        port = randrange(1024, 49152)
        async with updater:
            return_value = await updater.start_webhook(drop_pending_updates=drop_pending_updates, ip_address=ip, port=port, url_path='TOKEN', secret_token=secret_token)
            assert return_value is updater.update_queue
            assert updater.running
            update = make_message_update('Webhook')
            await send_webhook_message(ip, port, update.to_json(), 'TOKEN', secret_token=secret_token)
            assert (await updater.update_queue.get()).to_dict() == update.to_dict()
            response = await send_webhook_message(ip, port, '123456', 'webhook_handler.py')
            assert response.status_code == HTTPStatus.NOT_FOUND
            response = await send_webhook_message(ip, port, None, 'TOKEN', get_method='HEAD')
            assert response.status_code == HTTPStatus.METHOD_NOT_ALLOWED
            if secret_token:
                response_text = '<html><title>403: {0}</title><body>403: {0}</body></html>'
                response = await send_webhook_message(ip, port, update.to_json(), 'TOKEN')
                assert response.status_code == HTTPStatus.FORBIDDEN
                assert response.text == response_text.format('Request did not include the secret token')
                response = await send_webhook_message(ip, port, update.to_json(), 'TOKEN', secret_token='NotTheSecretToken')
                assert response.status_code == HTTPStatus.FORBIDDEN
                assert response.text == response_text.format('Request had the wrong secret token')
            await updater.stop()
            assert not updater.running
            if drop_pending_updates:
                assert self.message_count == 1
            else:
                assert self.message_count == 0
            await updater.start_webhook(drop_pending_updates=drop_pending_updates, ip_address=ip, port=port, url_path='TOKEN')
            assert updater.running
            update = make_message_update('Webhook')
            await send_webhook_message(ip, port, update.to_json(), 'TOKEN')
            assert (await updater.update_queue.get()).to_dict() == update.to_dict()
            await updater.stop()
            assert not updater.running

    async def test_start_webhook_already_running(self, updater, monkeypatch):

        async def return_true(*args, **kwargs):
            return True
        monkeypatch.setattr(updater.bot, 'set_webhook', return_true)
        monkeypatch.setattr(updater.bot, 'delete_webhook', return_true)
        ip = '127.0.0.1'
        port = randrange(1024, 49152)
        async with updater:
            await updater.start_webhook(ip, port, url_path='TOKEN')
            task = asyncio.create_task(updater.start_webhook(ip, port, url_path='TOKEN'))
            with pytest.raises(RuntimeError, match='already running'):
                await task
            await updater.stop()
            with pytest.raises(RuntimeError, match='not running'):
                await updater.stop()

    async def test_start_webhook_parameters_passing(self, updater, monkeypatch):
        expected_delete_webhook = {'drop_pending_updates': None}
        expected_set_webhook = dict(certificate=None, max_connections=40, allowed_updates=None, ip_address=None, secret_token=None, **expected_delete_webhook)

        async def set_webhook(*args, **kwargs):
            for (key, value) in expected_set_webhook.items():
                assert kwargs.pop(key, None) == value, f'set, {key}, {value}'
            assert kwargs in ({'url': 'http://127.0.0.1:80/'}, {'url': 'http://listen:80/'}, {'url': 'https://listen-ssl:42/ssl-path'})
            return True

        async def delete_webhook(*args, **kwargs):
            for (key, value) in expected_delete_webhook.items():
                assert kwargs.pop(key, None) == value, f'delete, {key}, {value}'
            assert kwargs == {}
            return True

        async def serve_forever(*args, **kwargs):
            kwargs.get('ready').set()
        monkeypatch.setattr(updater.bot, 'set_webhook', set_webhook)
        monkeypatch.setattr(updater.bot, 'delete_webhook', delete_webhook)
        monkeypatch.setattr(WebhookServer, 'serve_forever', serve_forever)
        async with updater:
            await updater.start_webhook()
            await updater.stop()
            expected_delete_webhook = {'drop_pending_updates': True, 'api_kwargs': None}
            expected_set_webhook = dict(certificate=data_file('sslcert.pem').read_bytes(), max_connections=47, allowed_updates=['message'], ip_address='123.456.789', secret_token=None, **expected_delete_webhook)
            await updater.start_webhook(listen='listen', allowed_updates=['message'], drop_pending_updates=True, ip_address='123.456.789', max_connections=47, cert=str(data_file('sslcert.pem').resolve()))
            await updater.stop()
            await updater.start_webhook(listen='listen-ssl', port=42, url_path='ssl-path', allowed_updates=['message'], drop_pending_updates=True, ip_address='123.456.789', max_connections=47, cert=data_file('sslcert.pem'), key=data_file('sslcert.key'))
            await updater.stop()

    @pytest.mark.parametrize('invalid_data', [True, False], ids=('invalid data', 'valid data'))
    async def test_webhook_arbitrary_callback_data(self, monkeypatch, cdc_bot, invalid_data, chat_id):
        """Here we only test one simple setup. telegram.ext.ExtBot.insert_callback_data is tested
        extensively in test_bot.py in conjunction with get_updates."""
        updater = Updater(bot=cdc_bot, update_queue=asyncio.Queue())

        async def return_true(*args, **kwargs):
            return True
        try:
            monkeypatch.setattr(updater.bot, 'set_webhook', return_true)
            monkeypatch.setattr(updater.bot, 'delete_webhook', return_true)
            ip = '127.0.0.1'
            port = randrange(1024, 49152)
            async with updater:
                await updater.start_webhook(ip, port, url_path='TOKEN')
                reply_markup = InlineKeyboardMarkup.from_button(InlineKeyboardButton(text='text', callback_data='callback_data'))
                if not invalid_data:
                    reply_markup = updater.bot.callback_data_cache.process_keyboard(reply_markup)
                update = make_message_update(message='test_webhook_arbitrary_callback_data', message_factory=make_message, reply_markup=reply_markup, user=updater.bot.bot)
                await send_webhook_message(ip, port, update.to_json(), 'TOKEN')
                received_update = await updater.update_queue.get()
                assert received_update.update_id == update.update_id
                message_dict = update.message.to_dict()
                received_dict = received_update.message.to_dict()
                message_dict.pop('reply_markup')
                received_dict.pop('reply_markup')
                assert message_dict == received_dict
                button = received_update.message.reply_markup.inline_keyboard[0][0]
                if invalid_data:
                    assert isinstance(button.callback_data, InvalidCallbackData)
                else:
                    assert button.callback_data == 'callback_data'
                await updater.stop()
        finally:
            updater.bot.callback_data_cache.clear_callback_data()
            updater.bot.callback_data_cache.clear_callback_queries()

    async def test_webhook_invalid_ssl(self, monkeypatch, updater):

        async def return_true(*args, **kwargs):
            return True
        monkeypatch.setattr(updater.bot, 'set_webhook', return_true)
        monkeypatch.setattr(updater.bot, 'delete_webhook', return_true)
        ip = '127.0.0.1'
        port = randrange(1024, 49152)
        async with updater:
            with pytest.raises(TelegramError, match='Invalid SSL'):
                await updater.start_webhook(ip, port, url_path='TOKEN', cert=Path(__file__).as_posix(), key=Path(__file__).as_posix(), bootstrap_retries=0, drop_pending_updates=False, webhook_url=None, allowed_updates=None)
            assert updater.running is False

    async def test_webhook_ssl_just_for_telegram(self, monkeypatch, updater):
        """Here we just test that the SSL info is pased to Telegram, but __not__ to the the
        webhook server"""

        async def set_webhook(**kwargs):
            self.test_flag.append(bool(kwargs.get('certificate')))
            return True

        async def return_true(*args, **kwargs):
            return True
        orig_wh_server_init = WebhookServer.__init__

        def webhook_server_init(*args, **kwargs):
            if False:
                print('Hello World!')
            self.test_flag = [kwargs.get('ssl_ctx') is None]
            orig_wh_server_init(*args, **kwargs)
        monkeypatch.setattr(updater.bot, 'set_webhook', set_webhook)
        monkeypatch.setattr(updater.bot, 'delete_webhook', return_true)
        monkeypatch.setattr('telegram.ext._utils.webhookhandler.WebhookServer.__init__', webhook_server_init)
        ip = '127.0.0.1'
        port = randrange(1024, 49152)
        async with updater:
            await updater.start_webhook(ip, port, webhook_url=None, cert=Path(__file__).as_posix())
            update = make_message_update(message='test_message')
            await send_webhook_message(ip, port, update.to_json())
            assert (await updater.update_queue.get()).to_dict() == update.to_dict()
            assert self.test_flag == [True, True]
            await updater.stop()

    @pytest.mark.parametrize('exception_class', [InvalidToken, TelegramError])
    @pytest.mark.parametrize('retries', [3, 0])
    async def test_start_webhook_bootstrap_retries(self, updater, monkeypatch, exception_class, retries):

        async def do_request(*args, **kwargs):
            self.message_count += 1
            raise exception_class(str(self.message_count))
        async with updater:
            monkeypatch.setattr(HTTPXRequest, 'do_request', do_request)
            if exception_class == InvalidToken:
                with pytest.raises(InvalidToken, match='1'):
                    await updater.start_webhook(bootstrap_retries=retries)
            else:
                with pytest.raises(TelegramError, match=str(retries + 1)):
                    await updater.start_webhook(bootstrap_retries=retries)

    async def test_webhook_invalid_posts(self, updater, monkeypatch):

        async def return_true(*args, **kwargs):
            return True
        monkeypatch.setattr(updater.bot, 'set_webhook', return_true)
        monkeypatch.setattr(updater.bot, 'delete_webhook', return_true)
        ip = '127.0.0.1'
        port = randrange(1024, 49152)
        async with updater:
            await updater.start_webhook(listen=ip, port=port)
            response = await send_webhook_message(ip, port, None, content_type='invalid')
            assert response.status_code == HTTPStatus.FORBIDDEN
            response = await send_webhook_message(ip, port, payload_str='<root><bla>data</bla></root>', content_type='application/xml')
            assert response.status_code == HTTPStatus.FORBIDDEN
            response = await send_webhook_message(ip, port, 'dummy-payload', content_len=None)
            assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
            await updater.stop()

    async def test_webhook_update_de_json_fails(self, monkeypatch, updater, caplog):

        async def delete_webhook(*args, **kwargs):
            return True

        async def set_webhook(*args, **kwargs):
            return True

        def de_json_fails(*args, **kwargs):
            if False:
                while True:
                    i = 10
            raise TypeError('Invalid input')
        monkeypatch.setattr(updater.bot, 'set_webhook', set_webhook)
        monkeypatch.setattr(updater.bot, 'delete_webhook', delete_webhook)
        orig_de_json = Update.de_json
        monkeypatch.setattr(Update, 'de_json', de_json_fails)
        ip = '127.0.0.1'
        port = randrange(1024, 49152)
        async with updater:
            return_value = await updater.start_webhook(ip_address=ip, port=port, url_path='TOKEN')
            assert return_value is updater.update_queue
            assert updater.running
            update = make_message_update('Webhook')
            with caplog.at_level(logging.CRITICAL):
                await send_webhook_message(ip, port, update.to_json(), 'TOKEN')
            assert len(caplog.records) == 1
            assert caplog.records[-1].getMessage().startswith('Something went wrong processing')
            assert caplog.records[-1].name == 'telegram.ext.Updater'
            caplog.clear()
            with caplog.at_level(logging.CRITICAL):
                monkeypatch.setattr(Update, 'de_json', orig_de_json)
                await send_webhook_message(ip, port, update.to_json(), 'TOKEN')
                assert (await updater.update_queue.get()).to_dict() == update.to_dict()
            assert len(caplog.records) == 0
            await updater.stop()
            assert not updater.running