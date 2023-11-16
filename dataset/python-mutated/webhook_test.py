"""Test webhook"""
import flask
import pytest
from webhook import handle_webhook
request = {'fulfillmentInfo': {'tag': 'Default Welcome Intent'}}

@pytest.fixture(scope='module')
def app():
    if False:
        while True:
            i = 10
    return flask.Flask(__name__)

def test_handle_webhook(app):
    if False:
        for i in range(10):
            print('nop')
    with app.test_request_context(json=request):
        res = handle_webhook(flask.request)
        assert 'Hello from a GCF Webhook' in str(res)