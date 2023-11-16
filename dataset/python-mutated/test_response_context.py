import pytest
import falcon
import falcon.asgi

@pytest.fixture(params=[True, False], ids=['asgi.Response', 'Response'])
def resp_type(request):
    if False:
        i = 10
        return i + 15
    if request.param:
        return falcon.asgi.Response
    return falcon.Response

class TestResponseContext:

    def test_default_response_context(self, resp_type):
        if False:
            return 10
        resp = resp_type()
        resp.context.hello = 'World!'
        assert resp.context.hello == 'World!'
        assert resp.context['hello'] == 'World!'
        resp.context['note'] = 'Default Response.context_type used to be dict.'
        assert 'note' in resp.context
        assert hasattr(resp.context, 'note')
        assert resp.context.get('note') == resp.context['note']

    def test_custom_response_context(self, resp_type):
        if False:
            for i in range(10):
                print('nop')

        class MyCustomContextType:
            pass

        class MyCustomResponse(resp_type):
            context_type = MyCustomContextType
        resp = MyCustomResponse()
        assert isinstance(resp.context, MyCustomContextType)

    def test_custom_response_context_failure(self, resp_type):
        if False:
            while True:
                i = 10

        class MyCustomResponse(resp_type):
            context_type = False
        with pytest.raises(TypeError):
            MyCustomResponse()

    def test_custom_response_context_factory(self, resp_type):
        if False:
            print('Hello World!')

        def create_context(resp):
            if False:
                print('Hello World!')
            return {'resp': resp}

        class MyCustomResponse(resp_type):
            context_type = create_context
        resp = MyCustomResponse()
        assert isinstance(resp.context, dict)
        assert resp.context['resp'] is resp