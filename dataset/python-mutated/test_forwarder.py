import pytest
from localstack.aws.api import RequestContext, ServiceException, ServiceRequest, ServiceResponse, handler
from localstack.aws.forwarder import ForwardingFallbackDispatcher, NotImplementedAvoidFallbackError

def test_forwarding_fallback_dispatcher():
    if False:
        for i in range(10):
            print('nop')

    class TestProvider:

        @handler(operation='TestOperation')
        def test_method(self, context):
            if False:
                i = 10
                return i + 15
            raise NotImplementedError
    test_provider = TestProvider()

    def test_request_forwarder(_, __) -> ServiceResponse:
        if False:
            print('Hello World!')
        return 'fallback-result'
    dispatcher = ForwardingFallbackDispatcher(test_provider, test_request_forwarder)
    assert dispatcher['TestOperation'](RequestContext(), ServiceRequest()) == 'fallback-result'

def test_forwarding_fallback_dispatcher_avoid_fallback():
    if False:
        for i in range(10):
            print('nop')

    class TestProvider:

        @handler(operation='TestOperation')
        def test_method(self, context):
            if False:
                i = 10
                return i + 15
            raise NotImplementedAvoidFallbackError
    test_provider = TestProvider()

    def test_request_forwarder(_, __) -> ServiceResponse:
        if False:
            i = 10
            return i + 15
        raise ServiceException
    dispatcher = ForwardingFallbackDispatcher(test_provider, test_request_forwarder)
    with pytest.raises(NotImplementedError):
        dispatcher['TestOperation'](RequestContext(), ServiceRequest())