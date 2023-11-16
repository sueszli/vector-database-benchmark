import json
import falcon
from falcon import testing

class TypeResource(testing.SimpleTestResource):
    """A simple resource to return the posted request body."""

    @falcon.before(testing.capture_responder_args)
    def on_post(self, req, resp, **kwargs):
        if False:
            i = 10
            return i + 15
        resp.status = falcon.HTTP_200
        resp.text = json.dumps({'data': req.bounded_stream.read().decode('utf-8')})

class TestWsgiRefInputWrapper:

    def test_resources_can_read_request_stream_during_tests(self):
        if False:
            print('Hello World!')
        'Make sure we can perform a simple request during testing.\n\n        Originally, testing would fail after performing a request because no\n        size was specified when calling `wsgiref.validate.InputWrapper.read()`\n        via `req.stream.read()`'
        app = falcon.App()
        type_route = '/type'
        app.add_route(type_route, TypeResource())
        client = testing.TestClient(app)
        result = client.simulate_post(path=type_route, body='hello')
        assert result.status == falcon.HTTP_200
        assert result.json == {'data': 'hello'}