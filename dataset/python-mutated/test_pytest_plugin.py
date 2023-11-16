import socket
import time
from functools import partial
import pytest
from six.moves import queue
from nameko.constants import WEB_SERVER_CONFIG_KEY
from nameko.extensions import DependencyProvider
from nameko.rpc import RpcProxy, rpc
from nameko.standalone.rpc import ServiceRpcProxy
from nameko.testing import rabbit
from nameko.testing.utils import get_rabbit_connections
from nameko.web.handlers import http
from nameko.web.server import parse_address
from nameko.web.websocket import rpc as wsrpc
pytest_plugins = 'pytester'

@pytest.fixture
def plugin_options(request):
    if False:
        for i in range(10):
            print('nop')
    ' Get the options pytest may have been invoked with so we can pass\n    them into subprocess pytests created by the pytester plugin.\n    '
    options = ('--rabbit-amqp-uri', '--rabbit-api-uri')
    args = ['{}={}'.format(opt, request.config.getoption(opt)) for opt in options]
    return args

class TestOptions(object):

    def test_options(self, testdir):
        if False:
            return 10
        options = (('--amqp-uri', 'amqp://localhost:5672/vhost'), ('--rabbit-api-uri', 'http://localhost:15672'), ('--amqp-ssl-port', '1234'))
        testdir.makepyfile("\n            import re\n\n            def test_option(request):\n                assert request.config.getoption('RABBIT_AMQP_URI') == (\n                    'amqp://localhost:5672/vhost'\n                )\n                assert request.config.getoption('RABBIT_API_URI') == (\n                    'http://localhost:15672'\n                )\n                assert request.config.getoption('AMQP_SSL_PORT') == '1234'\n            ")
        args = []
        for (option, value) in options:
            args.extend((option, value))
        result = testdir.runpytest(*args)
        assert result.ret == 0

    def test_ssl_options(self, testdir):
        if False:
            return 10
        options = (('certfile', 'path/cert.pem'), ('string', 'string'), ('number', 1), ('list', '[1, 2, 3]'), ('map', '{"foo": "bar"}'))
        testdir.makepyfile("\n            import re\n            import ssl\n\n            def test_ssl_options(request, rabbit_ssl_config):\n                assert request.config.getoption('AMQP_SSL_OPTIONS') == [\n                    # defaults\n                    ('ca_certs', 'certs/cacert.pem'),\n                    ('certfile', 'certs/clientcert.pem'),\n                    ('keyfile', 'certs/clientkey.pem'),\n                    ('cert_reqs', ssl.CERT_REQUIRED),\n                    # additions\n                    ('certfile', 'path/cert.pem'),\n                    ('string', 'string'),\n                    ('number', 1),\n                    ('list', [1, 2, 3]),\n                    ('map', {'foo': 'bar'}),\n                    ('keyonly', True),\n                ]\n\n                expected_ssl_options = {\n                    'certfile': 'path/cert.pem',  # default overridden\n                    'ca_certs': 'certs/cacert.pem',\n                    'keyfile': 'certs/clientkey.pem',\n                    'cert_reqs': ssl.CERT_REQUIRED,\n                    'string': 'string',\n                    'number': 1,\n                    'list': [1, 2, 3],\n                    'map': {'foo': 'bar'},\n                    'keyonly': True,\n                }\n                assert rabbit_ssl_config['AMQP_SSL'] == expected_ssl_options\n            ")
        args = []
        for (key, value) in options:
            args.extend(['--amqp-ssl-option', '{}={}'.format(key, value)])
        args.extend(['--amqp-ssl-option', 'keyonly'])
        result = testdir.runpytest(*args)
        assert result.ret == 0

@pytest.mark.filterwarnings('ignore:For versions after 2.13.0:UserWarning')
class TestMonkeyPatchWarning:

    @pytest.fixture
    def pytest_without_monkeypatch(self, testdir):
        if False:
            i = 10
            return i + 15
        return partial(testdir.runpytest_subprocess, '-p', 'no:pytest_eventlet')

    @pytest.mark.parametrize('suppress_warning', [True, False])
    def test_warning(self, suppress_warning, testdir, pytest_without_monkeypatch):
        if False:
            while True:
                i = 10
        args = []
        if suppress_warning:
            args.append('--suppress-nameko-eventlet-notification')
        testdir.makepyfile('\n            import eventlet\n\n            def test_warning(request):\n                assert not eventlet.patcher.is_monkey_patched("socket")\n                assert not eventlet.patcher.is_monkey_patched("os")\n                assert not eventlet.patcher.is_monkey_patched("select")\n                assert not eventlet.patcher.is_monkey_patched("thread")\n                assert not eventlet.patcher.is_monkey_patched("time")\n            ')
        result = pytest_without_monkeypatch(*args)
        assert result.ret == 0
        stderr = '\n'.join(result.stderr.lines)
        if not suppress_warning:
            assert 'pytest-eventlet' in stderr
        else:
            assert 'pytest-eventlet' not in stderr

    def test_monkeypatch_applied_in_conftest(self, testdir, pytest_without_monkeypatch):
        if False:
            return 10
        testdir.makeconftest('\n            import eventlet\n\n            eventlet.monkey_patch()\n            ')
        testdir.makepyfile('\n            import eventlet\n\n            def test_no_warning(request):\n                assert eventlet.patcher.is_monkey_patched("socket")\n                assert eventlet.patcher.is_monkey_patched("os")\n                assert eventlet.patcher.is_monkey_patched("select")\n                assert eventlet.patcher.is_monkey_patched("thread")\n                assert eventlet.patcher.is_monkey_patched("time")\n            ')
        result = pytest_without_monkeypatch()
        assert result.ret == 0
        stderr = '\n'.join(result.stderr.lines)
        assert 'pytest-eventlet' not in stderr

def test_empty_config(empty_config):
    if False:
        i = 10
        return i + 15
    assert empty_config == {}

def test_rabbit_manager(rabbit_manager):
    if False:
        i = 10
        return i + 15
    assert isinstance(rabbit_manager, rabbit.Client)
    assert '/' in [vhost['name'] for vhost in rabbit_manager.get_all_vhosts()]

def test_amqp_uri(testdir):
    if False:
        while True:
            i = 10
    amqp_uri = 'amqp://user:pass@host:5672/vhost'
    testdir.makeconftest("\n        import pytest\n\n        @pytest.fixture\n        def rabbit_config():\n            return dict(AMQP_URI='{}')\n        ".format(amqp_uri))
    testdir.makepyfile("\n        import re\n\n        def test_amqp_uri(amqp_uri):\n            assert amqp_uri == '{}'\n        ".format(amqp_uri))
    result = testdir.runpytest('--amqp-uri', amqp_uri)
    assert result.ret == 0

class TestGetMessageFromQueue(object):

    @pytest.fixture
    def queue_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'queue'

    @pytest.fixture
    def publish_message(self, rabbit_manager, rabbit_config, queue_name):
        if False:
            for i in range(10):
                print('nop')
        vhost = rabbit_config['vhost']
        rabbit_manager.create_queue(vhost, queue_name, durable=True)

        def publish(payload, **properties):
            if False:
                return 10
            rabbit_manager.publish(vhost, 'amq.default', queue_name, payload, properties)
        return publish

    def test_get_message(self, publish_message, get_message_from_queue, queue_name, rabbit_manager, rabbit_config):
        if False:
            for i in range(10):
                print('nop')
        payload = 'payload'
        publish_message(payload)
        message = get_message_from_queue(queue_name)
        assert message.payload == payload
        vhost = rabbit_config['vhost']
        assert rabbit_manager.get_queue(vhost, queue_name)['messages'] == 0

    def test_requeue(self, publish_message, get_message_from_queue, queue_name, rabbit_manager, rabbit_config):
        if False:
            print('Hello World!')
        payload = 'payload'
        publish_message(payload)
        message = get_message_from_queue(queue_name, ack=False)
        assert message.payload == payload
        time.sleep(1)
        vhost = rabbit_config['vhost']
        assert rabbit_manager.get_queue(vhost, queue_name)['messages'] == 1

    def test_non_blocking(self, publish_message, get_message_from_queue, queue_name):
        if False:
            i = 10
            return i + 15
        with pytest.raises(queue.Empty):
            get_message_from_queue(queue_name, block=False)

    def test_timeout(self, publish_message, get_message_from_queue, queue_name):
        if False:
            print('Hello World!')
        with pytest.raises(queue.Empty):
            get_message_from_queue(queue_name, timeout=0.01)

    def test_accept(self, publish_message, get_message_from_queue, queue_name):
        if False:
            return 10
        payload = 'payload'
        content_type = 'application/x-special'
        publish_message(payload, content_type=content_type)
        message = get_message_from_queue(queue_name, accept=content_type)
        assert message.properties['content_type'] == content_type
        assert message.payload == payload

class TestFastTeardown(object):

    def test_order(self, testdir):
        if False:
            while True:
                i = 10
        testdir.makeconftest('\n            from mock import Mock\n            import pytest\n\n            @pytest.fixture(scope=\'session\')\n            def tracker():\n                return Mock()\n\n            @pytest.fixture\n            def rabbit_config(tracker):\n                tracker("rabbit_config", "up")\n                yield\n                tracker("rabbit_config", "down")\n\n            @pytest.fixture\n            def container_factory(tracker):\n                tracker("container_factory", "up")\n                yield\n                tracker("container_factory", "down")\n            ')
        testdir.makepyfile('\n            from mock import call\n\n            def test_foo(container_factory, rabbit_config):\n                pass  # factory first\n\n            def test_bar(rabbit_config, container_factory):\n                pass  # rabbit first\n\n            def test_check(tracker):\n                assert tracker.call_args_list == [\n                    # test_foo\n                    call("container_factory", "up"),\n                    call("rabbit_config", "up"),\n                    call("rabbit_config", "down"),\n                    call("container_factory", "down"),\n                    # test_bar\n                    call("container_factory", "up"),\n                    call("rabbit_config", "up"),\n                    call("rabbit_config", "down"),\n                    call("container_factory", "down"),\n                ]\n            ')
        result = testdir.runpytest()
        assert result.ret == 0

    def test_only_affects_used_fixtures(self, testdir):
        if False:
            return 10
        testdir.makeconftest('\n            from mock import Mock\n            import pytest\n\n            @pytest.fixture(scope=\'session\')\n            def tracker():\n                return Mock()\n\n            @pytest.fixture\n            def rabbit_config(tracker):\n                tracker("rabbit_config", "up")\n                yield\n                tracker("rabbit_config", "down")\n\n            @pytest.fixture\n            def container_factory(tracker):\n                tracker("container_factory", "up")\n                yield\n                tracker("container_factory", "down")\n            ')
        testdir.makepyfile('\n            from mock import call\n\n            def test_no_rabbit(container_factory):\n                pass  # factory first\n\n            def test_check(tracker):\n                assert tracker.call_args_list == [\n                    call("container_factory", "up"),\n                    call("container_factory", "down"),\n                ]\n            ')
        result = testdir.runpytest()
        assert result.ret == 0

    def test_consumer_mixin_patch(self, testdir):
        if False:
            print('Hello World!')
        testdir.makeconftest("\n            from kombu.mixins import ConsumerMixin\n            import pytest\n\n            consumers = []\n\n            @pytest.fixture(autouse=True)\n            def fast_teardown(patch_checker, fast_teardown):\n                ''' Shadow the fast_teardown fixture to set fixture order:\n\n                Setup:\n\n                    1. patch_checker\n                    2. original fast_teardown (applies monkeypatch)\n                    3. this fixture (creates consumer)\n\n                Teardown:\n\n                    1. this fixture (creates consumer)\n                    2. original fast_teardown (removes patch; sets attribute)\n                    3. patch_checker (verifies consumer was stopped)\n                '''\n                consumers.append(ConsumerMixin())\n\n            @pytest.fixture\n            def patch_checker():\n                yield\n                assert consumers[0].should_stop is True\n            ")
        testdir.makepyfile('\n            def test_mixin_patch(patch_checker):\n                pass\n            ')
        result = testdir.runpytest()
        assert result.ret == 0

def test_container_factory(testdir, rabbit_config, rabbit_manager, plugin_options):
    if False:
        for i in range(10):
            print('nop')
    testdir.makepyfile('\n        from nameko.rpc import rpc\n        from nameko.standalone.rpc import ServiceRpcProxy\n\n        class ServiceX(object):\n            name = "x"\n\n            @rpc\n            def method(self):\n                return "OK"\n\n        def test_container_factory(container_factory, rabbit_config):\n            container = container_factory(ServiceX, rabbit_config)\n            container.start()\n\n            with ServiceRpcProxy("x", rabbit_config) as proxy:\n                assert proxy.method() == "OK"\n        ')
    result = testdir.runpytest(*plugin_options)
    assert result.ret == 0
    vhost = rabbit_config['vhost']
    assert get_rabbit_connections(vhost, rabbit_manager) == []

def test_container_factory_with_custom_container_cls(testdir, plugin_options):
    if False:
        print('Hello World!')
    testdir.makepyfile(container_module='\n        from nameko.containers import ServiceContainer\n\n        class ServiceContainerX(ServiceContainer):\n            pass\n    ')
    testdir.makepyfile('\n        from nameko.rpc import rpc\n        from nameko.standalone.rpc import ServiceRpcProxy\n\n        from container_module import ServiceContainerX\n\n        class ServiceX(object):\n            name = "x"\n\n            @rpc\n            def method(self):\n                return "OK"\n\n        def test_container_factory(\n            container_factory, rabbit_config\n        ):\n            rabbit_config[\'SERVICE_CONTAINER_CLS\'] = (\n                "container_module.ServiceContainerX"\n            )\n\n            container = container_factory(ServiceX, rabbit_config)\n            container.start()\n\n            assert isinstance(container, ServiceContainerX)\n\n            with ServiceRpcProxy("x", rabbit_config) as proxy:\n                assert proxy.method() == "OK"\n        ')
    result = testdir.runpytest(*plugin_options)
    assert result.ret == 0

def test_runner_factory(testdir, plugin_options, rabbit_config, rabbit_manager):
    if False:
        i = 10
        return i + 15
    testdir.makepyfile('\n        from nameko.rpc import rpc\n        from nameko.standalone.rpc import ServiceRpcProxy\n\n        class ServiceX(object):\n            name = "x"\n\n            @rpc\n            def method(self):\n                return "OK"\n\n        def test_runner(runner_factory, rabbit_config):\n            runner = runner_factory(rabbit_config, ServiceX)\n            runner.start()\n\n            with ServiceRpcProxy("x", rabbit_config) as proxy:\n                assert proxy.method() == "OK"\n        ')
    result = testdir.runpytest(*plugin_options)
    assert result.ret == 0
    vhost = rabbit_config['vhost']
    assert get_rabbit_connections(vhost, rabbit_manager) == []

@pytest.mark.usefixtures('predictable_call_ids')
def test_predictable_call_ids(runner_factory, rabbit_config):
    if False:
        i = 10
        return i + 15
    worker_contexts = []

    class CaptureWorkerContext(DependencyProvider):

        def worker_setup(self, worker_ctx):
            if False:
                i = 10
                return i + 15
            worker_contexts.append(worker_ctx)

    class ServiceX(object):
        name = 'x'
        capture = CaptureWorkerContext()
        service_y = RpcProxy('y')

        @rpc
        def method(self):
            if False:
                return 10
            self.service_y.method()

    class ServiceY(object):
        name = 'y'
        capture = CaptureWorkerContext()

        @rpc
        def method(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
    runner = runner_factory(rabbit_config, ServiceX, ServiceY)
    runner.start()
    with ServiceRpcProxy('x', rabbit_config) as service_x:
        service_x.method()
    call_ids = [worker_ctx.call_id for worker_ctx in worker_contexts]
    assert call_ids == ['x.method.1', 'y.method.2']

def test_web_config(web_config):
    if False:
        print('Hello World!')
    assert WEB_SERVER_CONFIG_KEY in web_config
    bind_address = parse_address(web_config[WEB_SERVER_CONFIG_KEY])
    sock = socket.socket()
    sock.bind(bind_address)

def test_web_session(web_config, container_factory, web_session):
    if False:
        for i in range(10):
            print('nop')

    class Service(object):
        name = 'web'

        @http('GET', '/foo')
        def method(self, request):
            if False:
                for i in range(10):
                    print('nop')
            return 'OK'
    container = container_factory(Service, web_config)
    container.start()
    assert web_session.get('/foo').status_code == 200

def test_websocket(web_config, container_factory, websocket):
    if False:
        while True:
            i = 10

    class Service(object):
        name = 'ws'

        @wsrpc
        def uppercase(self, socket_id, arg):
            if False:
                i = 10
                return i + 15
            return arg.upper()
    container = container_factory(Service, web_config)
    container.start()
    ws = websocket()
    assert ws.rpc('uppercase', arg='foo') == 'FOO'