import pytest
from mock import Mock
from nameko.constants import AUTH_TOKEN_CONTEXT_KEY, LANGUAGE_CONTEXT_KEY, USER_AGENT_CONTEXT_KEY, USER_ID_CONTEXT_KEY
from nameko.containers import ServiceContainer, WorkerContext
from nameko.contextdata import AuthToken, ContextDataProvider, Language, UserAgent, UserId
from nameko.testing.utils import get_extension
CUSTOM_CONTEXT_KEY = 'custom'

class CustomValue(ContextDataProvider):
    context_key = CUSTOM_CONTEXT_KEY

class Service(object):
    name = 'service'
    auth_token = AuthToken()
    language = Language()
    user_id = UserId()
    user_agent = UserAgent()
    custom_value = CustomValue()

@pytest.fixture
def container():
    if False:
        while True:
            i = 10
    return ServiceContainer(Service, {})

def test_get_custom_context_value(container):
    if False:
        for i in range(10):
            print('nop')
    dependency = get_extension(container, ContextDataProvider, attr_name='custom_value')
    worker_ctx = WorkerContext(container, 'service', Mock(), data={CUSTOM_CONTEXT_KEY: 'hello'})
    assert dependency.get_dependency(worker_ctx) == 'hello'

def test_get_unset_value(container):
    if False:
        while True:
            i = 10
    dependency = get_extension(container, ContextDataProvider, attr_name='custom_value')
    worker_ctx = WorkerContext(container, 'service', Mock(), data={})
    assert dependency.get_dependency(worker_ctx) is None

@pytest.mark.parametrize('attr_name, context_key', [('auth_token', AUTH_TOKEN_CONTEXT_KEY), ('language', LANGUAGE_CONTEXT_KEY), ('user_id', USER_ID_CONTEXT_KEY), ('user_agent', USER_AGENT_CONTEXT_KEY)])
def test_get_builtin_dependencies(attr_name, context_key, container):
    if False:
        print('Hello World!')
    dependency = get_extension(container, ContextDataProvider, attr_name=attr_name)
    worker_ctx = WorkerContext(container, 'service', Mock(), data={context_key: 'value'})
    assert dependency.get_dependency(worker_ctx) == 'value'