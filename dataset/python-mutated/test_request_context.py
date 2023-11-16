import pytest
from falcon.request import Request
import falcon.testing as testing

class TestRequestContext:

    def test_default_request_context(self):
        if False:
            i = 10
            return i + 15
        req = testing.create_req()
        req.context.hello = 'World'
        assert req.context.hello == 'World'
        assert req.context['hello'] == 'World'
        req.context['note'] = 'Default Request.context_type used to be dict.'
        assert 'note' in req.context
        assert hasattr(req.context, 'note')
        assert req.context.get('note') == req.context['note']

    def test_custom_request_context(self):
        if False:
            return 10

        class MyCustomContextType:
            pass

        class MyCustomRequest(Request):
            context_type = MyCustomContextType
        env = testing.create_environ()
        req = MyCustomRequest(env)
        assert isinstance(req.context, MyCustomContextType)

    def test_custom_request_context_failure(self):
        if False:
            for i in range(10):
                print('nop')

        class MyCustomRequest(Request):
            context_type = False
        env = testing.create_environ()
        with pytest.raises(TypeError):
            MyCustomRequest(env)

    def test_custom_request_context_request_access(self):
        if False:
            i = 10
            return i + 15

        def create_context(req):
            if False:
                print('Hello World!')
            return {'uri': req.uri}

        class MyCustomRequest(Request):
            context_type = create_context
        env = testing.create_environ()
        req = MyCustomRequest(env)
        assert isinstance(req.context, dict)
        assert req.context['uri'] == req.uri