import asyncio
import inspect
from dataclasses import dataclass
import httpx
import pytest
from telegram import Bot
from telegram.ext import AIORateLimiter, Application, ApplicationBuilder, CallbackDataCache, ContextTypes, Defaults, ExtBot, JobQueue, PicklePersistence, Updater
from telegram.ext._applicationbuilder import _BOT_CHECKS
from telegram.ext._baseupdateprocessor import SimpleUpdateProcessor
from telegram.request import HTTPXRequest
from telegram.warnings import PTBDeprecationWarning
from tests.auxil.constants import PRIVATE_KEY
from tests.auxil.envvars import TEST_WITH_OPT_DEPS
from tests.auxil.files import data_file
from tests.auxil.slots import mro_slots

@pytest.fixture()
def builder():
    if False:
        while True:
            i = 10
    return ApplicationBuilder()

@pytest.mark.skipif(TEST_WITH_OPT_DEPS, reason='Optional dependencies are installed')
class TestApplicationBuilderNoOptDeps:

    @pytest.mark.filterwarnings('ignore::telegram.warnings.PTBUserWarning')
    def test_init(self, builder):
        if False:
            while True:
                i = 10
        builder.token('token')
        app = builder.build()
        assert app.job_queue is None

@pytest.mark.skipif(not TEST_WITH_OPT_DEPS, reason='Optional dependencies not installed')
class TestApplicationBuilder:

    def test_slot_behaviour(self, builder):
        if False:
            return 10
        for attr in builder.__slots__:
            assert getattr(builder, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(builder)) == len(set(mro_slots(builder))), 'duplicate slot'

    @pytest.mark.parametrize('get_updates', [True, False])
    def test_all_methods_request(self, builder, get_updates):
        if False:
            i = 10
            return i + 15
        arguments = inspect.signature(HTTPXRequest.__init__).parameters.keys()
        prefix = 'get_updates_' if get_updates else ''
        for argument in arguments:
            if argument == 'self':
                continue
            assert hasattr(builder, prefix + argument), f'missing method {prefix}{argument}'

    @pytest.mark.parametrize('bot_class', [Bot, ExtBot])
    def test_all_methods_bot(self, builder, bot_class):
        if False:
            i = 10
            return i + 15
        arguments = inspect.signature(bot_class.__init__).parameters.keys()
        for argument in arguments:
            if argument == 'self':
                continue
            if argument == 'private_key_password':
                argument = 'private_key'
            assert hasattr(builder, argument), f'missing method {argument}'

    def test_all_methods_application(self, builder):
        if False:
            return 10
        arguments = inspect.signature(Application.__init__).parameters.keys()
        for argument in arguments:
            if argument == 'self':
                continue
            if argument == 'update_processor':
                argument = 'concurrent_updates'
            assert hasattr(builder, argument), f'missing method {argument}'

    def test_job_queue_init_exception(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')

        def init_raises_runtime_error(*args, **kwargs):
            if False:
                print('Hello World!')
            raise RuntimeError('RuntimeError')
        monkeypatch.setattr(JobQueue, '__init__', init_raises_runtime_error)
        with pytest.raises(RuntimeError, match='RuntimeError'):
            ApplicationBuilder()

    def test_build_without_token(self, builder):
        if False:
            return 10
        with pytest.raises(RuntimeError, match='No bot token was set.'):
            builder.build()

    def test_build_custom_bot(self, builder, bot):
        if False:
            print('Hello World!')
        builder.bot(bot)
        app = builder.build()
        assert app.bot is bot
        assert app.updater.bot is bot

    def test_default_values(self, bot, monkeypatch, builder):
        if False:
            print('Hello World!')

        @dataclass
        class Client:
            timeout: object
            proxies: object
            limits: object
            http1: object
            http2: object
            transport: object = None
        monkeypatch.setattr(httpx, 'AsyncClient', Client)
        app = builder.token(bot.token).build()
        assert isinstance(app, Application)
        assert isinstance(app.update_processor, SimpleUpdateProcessor)
        assert app.update_processor.max_concurrent_updates == 1
        assert isinstance(app.bot, ExtBot)
        assert isinstance(app.bot.request, HTTPXRequest)
        assert 'api.telegram.org' in app.bot.base_url
        assert bot.token in app.bot.base_url
        assert 'api.telegram.org' in app.bot.base_file_url
        assert bot.token in app.bot.base_file_url
        assert app.bot.private_key is None
        assert app.bot.callback_data_cache is None
        assert app.bot.defaults is None
        assert app.bot.rate_limiter is None
        assert app.bot.local_mode is False
        get_updates_client = app.bot._request[0]._client
        assert get_updates_client.limits == httpx.Limits(max_connections=1, max_keepalive_connections=1)
        assert get_updates_client.proxies is None
        assert get_updates_client.timeout == httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=1.0)
        assert get_updates_client.http1 is True
        assert not get_updates_client.http2
        client = app.bot.request._client
        assert client.limits == httpx.Limits(max_connections=256, max_keepalive_connections=256)
        assert client.proxies is None
        assert client.timeout == httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=1.0)
        assert client.http1 is True
        assert not client.http2
        assert isinstance(app.update_queue, asyncio.Queue)
        assert isinstance(app.updater, Updater)
        assert app.updater.bot is app.bot
        assert app.updater.update_queue is app.update_queue
        assert isinstance(app.job_queue, JobQueue)
        assert app.job_queue.application is app
        assert app.persistence is None
        assert app.post_init is None
        assert app.post_shutdown is None
        assert app.post_stop is None

    @pytest.mark.parametrize(('method', 'description'), _BOT_CHECKS, ids=[entry[0] for entry in _BOT_CHECKS])
    def test_mutually_exclusive_for_bot(self, builder, method, description):
        if False:
            print('Hello World!')
        getattr(builder, method)(data_file('private.key'))
        with pytest.raises(RuntimeError, match=f'`bot` may only be set, if no {description}'):
            builder.bot(None)
        builder = builder.__class__()
        builder.bot(None)
        with pytest.raises(RuntimeError, match=f'`{method}` may only be set, if no bot instance'):
            getattr(builder, method)(data_file('private.key'))

    @pytest.mark.parametrize('method', ['connection_pool_size', 'connect_timeout', 'pool_timeout', 'read_timeout', 'write_timeout', 'proxy', 'proxy_url', 'socket_options', 'bot', 'updater', 'http_version'])
    def test_mutually_exclusive_for_request(self, builder, method):
        if False:
            return 10
        builder.request(1)
        method_name = method.replace('proxy_url', 'proxy')
        with pytest.raises(RuntimeError, match=f'`{method_name}` may only be set, if no request instance'):
            getattr(builder, method)(data_file('private.key'))
        builder = ApplicationBuilder()
        getattr(builder, method)(1)
        with pytest.raises(RuntimeError, match='`request` may only be set, if no'):
            builder.request(1)

    @pytest.mark.parametrize('method', ['get_updates_connection_pool_size', 'get_updates_connect_timeout', 'get_updates_pool_timeout', 'get_updates_read_timeout', 'get_updates_write_timeout', 'get_updates_proxy', 'get_updates_proxy_url', 'get_updates_socket_options', 'get_updates_http_version', 'bot', 'updater'])
    def test_mutually_exclusive_for_get_updates_request(self, builder, method):
        if False:
            print('Hello World!')
        builder.get_updates_request(1)
        method_name = method.replace('proxy_url', 'proxy')
        with pytest.raises(RuntimeError, match=f'`{method_name}` may only be set, if no get_updates_request instance'):
            getattr(builder, method)(data_file('private.key'))
        builder = ApplicationBuilder()
        getattr(builder, method)(1)
        with pytest.raises(RuntimeError, match='`get_updates_request` may only be set, if no'):
            builder.get_updates_request(1)

    @pytest.mark.parametrize('method', ['get_updates_connection_pool_size', 'get_updates_connect_timeout', 'get_updates_pool_timeout', 'get_updates_read_timeout', 'get_updates_write_timeout', 'get_updates_proxy_url', 'get_updates_proxy', 'get_updates_socket_options', 'get_updates_http_version', 'connection_pool_size', 'connect_timeout', 'pool_timeout', 'read_timeout', 'write_timeout', 'proxy', 'proxy_url', 'socket_options', 'http_version', 'bot', 'update_queue', 'rate_limiter'] + [entry[0] for entry in _BOT_CHECKS])
    def test_mutually_exclusive_for_updater(self, builder, method):
        if False:
            while True:
                i = 10
        builder.updater(1)
        method_name = method.replace('proxy_url', 'proxy')
        with pytest.raises(RuntimeError, match=f'`{method_name}` may only be set, if no updater'):
            getattr(builder, method)(data_file('private.key'))
        builder = ApplicationBuilder()
        getattr(builder, method)(data_file('private.key'))
        method = method.replace('proxy_url', 'proxy')
        with pytest.raises(RuntimeError, match=f'`updater` may only be set, if no {method}'):
            builder.updater(1)

    @pytest.mark.parametrize('method', ['get_updates_connection_pool_size', 'get_updates_connect_timeout', 'get_updates_pool_timeout', 'get_updates_read_timeout', 'get_updates_write_timeout', 'get_updates_proxy', 'get_updates_proxy_url', 'get_updates_socket_options', 'get_updates_http_version', 'connection_pool_size', 'connect_timeout', 'pool_timeout', 'read_timeout', 'write_timeout', 'proxy', 'proxy_url', 'socket_options', 'bot', 'http_version'] + [entry[0] for entry in _BOT_CHECKS])
    def test_mutually_non_exclusive_for_updater(self, builder, method):
        if False:
            print('Hello World!')
        builder.updater(None)
        getattr(builder, method)(data_file('private.key'))
        builder = ApplicationBuilder()
        getattr(builder, method)(data_file('private.key'))
        builder.updater(None)

    @pytest.mark.parametrize(('proxy_method', 'get_updates_proxy_method'), [('proxy', 'get_updates_proxy'), ('proxy_url', 'get_updates_proxy_url')], ids=['new', 'legacy'])
    def test_all_bot_args_custom(self, builder, bot, monkeypatch, proxy_method, get_updates_proxy_method):
        if False:
            return 10
        defaults = Defaults()
        request = HTTPXRequest()
        get_updates_request = HTTPXRequest()
        rate_limiter = AIORateLimiter()
        builder.token(bot.token).base_url('base_url').base_file_url('base_file_url').private_key(PRIVATE_KEY).defaults(defaults).arbitrary_callback_data(42).request(request).get_updates_request(get_updates_request).rate_limiter(rate_limiter).local_mode(True)
        built_bot = builder.build().bot
        assert built_bot.token == bot.token
        assert built_bot.base_url == 'base_url' + bot.token
        assert built_bot.base_file_url == 'base_file_url' + bot.token
        assert built_bot.defaults is defaults
        assert built_bot.request is request
        assert built_bot._request[0] is get_updates_request
        assert built_bot.callback_data_cache.maxsize == 42
        assert built_bot.private_key
        assert built_bot.rate_limiter is rate_limiter
        assert built_bot.local_mode is True

        @dataclass
        class Client:
            timeout: object
            proxies: object
            limits: object
            http1: object
            http2: object
            transport: object = None
        monkeypatch.setattr(httpx, 'AsyncClient', Client)
        builder = ApplicationBuilder().token(bot.token)
        builder.connection_pool_size(1).connect_timeout(2).pool_timeout(3).read_timeout(4).write_timeout(5).http_version('1.1')
        getattr(builder, proxy_method)('proxy')
        app = builder.build()
        client = app.bot.request._client
        assert client.timeout == httpx.Timeout(pool=3, connect=2, read=4, write=5)
        assert client.limits == httpx.Limits(max_connections=1, max_keepalive_connections=1)
        assert client.proxies == 'proxy'
        assert client.http1 is True
        assert client.http2 is False
        builder = ApplicationBuilder().token(bot.token)
        builder.get_updates_connection_pool_size(1).get_updates_connect_timeout(2).get_updates_pool_timeout(3).get_updates_read_timeout(4).get_updates_write_timeout(5).get_updates_http_version('1.1')
        getattr(builder, get_updates_proxy_method)('get_updates_proxy')
        app = builder.build()
        client = app.bot._request[0]._client
        assert client.timeout == httpx.Timeout(pool=3, connect=2, read=4, write=5)
        assert client.limits == httpx.Limits(max_connections=1, max_keepalive_connections=1)
        assert client.proxies == 'get_updates_proxy'
        assert client.http1 is True
        assert client.http2 is False

    def test_custom_socket_options(self, builder, monkeypatch, bot):
        if False:
            i = 10
            return i + 15
        httpx_request_kwargs = []
        httpx_request_init = HTTPXRequest.__init__

        def init_transport(*args, **kwargs):
            if False:
                while True:
                    i = 10
            nonlocal httpx_request_kwargs
            httpx_request_kwargs.append(kwargs.copy())
            httpx_request_init(*args, **kwargs)
        monkeypatch.setattr(HTTPXRequest, '__init__', init_transport)
        builder.token(bot.token).build()
        assert httpx_request_kwargs[0].get('socket_options') is None
        assert httpx_request_kwargs[1].get('socket_options') is None
        httpx_request_kwargs = []
        ApplicationBuilder().token(bot.token).socket_options(((1, 2, 3),)).connection_pool_size('request').get_updates_socket_options(((4, 5, 6),)).get_updates_connection_pool_size('get_updates').build()
        for kwargs in httpx_request_kwargs:
            if kwargs.get('connection_pool_size') == 'request':
                assert kwargs.get('socket_options') == ((1, 2, 3),)
            else:
                assert kwargs.get('socket_options') == ((4, 5, 6),)

    def test_custom_application_class(self, bot, builder):
        if False:
            while True:
                i = 10

        class CustomApplication(Application):

            def __init__(self, arg, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__(**kwargs)
                self.arg = arg
        builder.application_class(CustomApplication, kwargs={'arg': 2}).token(bot.token)
        app = builder.build()
        assert isinstance(app, CustomApplication)
        assert app.arg == 2

    @pytest.mark.parametrize(('concurrent_updates', 'expected'), [(4, SimpleUpdateProcessor(4)), (False, SimpleUpdateProcessor(1)), (True, SimpleUpdateProcessor(256))])
    def test_all_application_args_custom(self, builder, bot, monkeypatch, concurrent_updates, expected):
        if False:
            print('Hello World!')
        job_queue = JobQueue()
        persistence = PicklePersistence('file_path')
        update_queue = asyncio.Queue()
        context_types = ContextTypes()

        async def post_init(app: Application) -> None:
            pass

        async def post_shutdown(app: Application) -> None:
            pass

        async def post_stop(app: Application) -> None:
            pass
        app = builder.token(bot.token).job_queue(job_queue).persistence(persistence).update_queue(update_queue).context_types(context_types).concurrent_updates(concurrent_updates).post_init(post_init).post_shutdown(post_shutdown).post_stop(post_stop).arbitrary_callback_data(True).build()
        assert app.job_queue is job_queue
        assert app.job_queue.application is app
        assert app.persistence is persistence
        assert app.persistence.bot is app.bot
        assert app.update_queue is update_queue
        assert app.updater.update_queue is update_queue
        assert app.updater.bot is app.bot
        assert app.context_types is context_types
        assert isinstance(app.update_processor, SimpleUpdateProcessor)
        assert app.update_processor.max_concurrent_updates == expected.max_concurrent_updates
        assert app.concurrent_updates == app.update_processor.max_concurrent_updates
        assert app.post_init is post_init
        assert app.post_shutdown is post_shutdown
        assert app.post_stop is post_stop
        assert isinstance(app.bot.callback_data_cache, CallbackDataCache)
        updater = Updater(bot=bot, update_queue=update_queue)
        app = ApplicationBuilder().updater(updater).build()
        assert app.updater is updater
        assert app.bot is updater.bot
        assert app.update_queue is updater.update_queue
        app = builder.token(bot.token).job_queue(job_queue).persistence(persistence).update_queue(update_queue).context_types(context_types).concurrent_updates(expected).post_init(post_init).post_shutdown(post_shutdown).post_stop(post_stop).arbitrary_callback_data(True).build()
        assert app.update_processor is expected

    @pytest.mark.parametrize('input_type', ['bytes', 'str', 'Path'])
    def test_all_private_key_input_types(self, builder, bot, input_type):
        if False:
            while True:
                i = 10
        private_key = data_file('private.key')
        password = data_file('private_key.password')
        if input_type == 'bytes':
            private_key = private_key.read_bytes()
            password = password.read_bytes()
        if input_type == 'str':
            private_key = str(private_key)
            password = str(password)
        builder.token(bot.token).private_key(private_key=private_key, password=password)
        bot = builder.build().bot
        assert bot.private_key

    def test_no_updater(self, bot, builder):
        if False:
            i = 10
            return i + 15
        app = builder.token(bot.token).updater(None).build()
        assert app.bot.token == bot.token
        assert app.updater is None
        assert isinstance(app.update_queue, asyncio.Queue)
        assert isinstance(app.job_queue, JobQueue)
        assert app.job_queue.application is app

    @pytest.mark.filterwarnings('ignore::telegram.warnings.PTBUserWarning')
    def test_no_job_queue(self, bot, builder):
        if False:
            print('Hello World!')
        app = builder.token(bot.token).job_queue(None).build()
        assert app.bot.token == bot.token
        assert app.job_queue is None
        assert isinstance(app.update_queue, asyncio.Queue)
        assert isinstance(app.updater, Updater)

    def test_proxy_url_deprecation_warning(self, bot, builder, recwarn):
        if False:
            print('Hello World!')
        builder.token(bot.token).proxy_url('proxy_url')
        assert len(recwarn) == 1
        assert '`ApplicationBuilder.proxy_url` is deprecated' in str(recwarn[0].message)
        assert recwarn[0].category is PTBDeprecationWarning
        assert recwarn[0].filename == __file__, 'wrong stacklevel'

    def test_get_updates_proxy_url_deprecation_warning(self, bot, builder, recwarn):
        if False:
            return 10
        builder.token(bot.token).get_updates_proxy_url('get_updates_proxy_url')
        assert len(recwarn) == 1
        assert '`ApplicationBuilder.get_updates_proxy_url` is deprecated' in str(recwarn[0].message)
        assert recwarn[0].category is PTBDeprecationWarning
        assert recwarn[0].filename == __file__, 'wrong stacklevel'