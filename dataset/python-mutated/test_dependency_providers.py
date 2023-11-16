import pytest
from nameko.dependency_providers import Config
from nameko.rpc import rpc
from nameko.testing.services import entrypoint_hook

class TestConfig:

    @pytest.fixture
    def service_cls(self):
        if False:
            i = 10
            return i + 15

        class Service:
            name = 'test_x'
            config = Config()

            @rpc
            def foo(self):
                if False:
                    while True:
                        i = 10
                return self.config.get('FOO')
        return Service

    def test_get_config_value(self, rabbit_config, container_factory, service_cls):
        if False:
            i = 10
            return i + 15
        config = rabbit_config
        config['FOO'] = 'bar'
        container = container_factory(service_cls, config)
        container.start()
        with entrypoint_hook(container, 'foo') as foo:
            assert foo() == 'bar'