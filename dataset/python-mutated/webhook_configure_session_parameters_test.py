"""Test configure new session parameters"""
import flask
import pytest
from webhook_configure_session_parameters import configure_session_params

@pytest.fixture(name='app', scope='module')
def fixture_app():
    if False:
        for i in range(10):
            print('nop')
    'Flask fixture to pass a flask.Request to the test function.'
    return flask.Flask(__name__)

def test_validate_parameter(app):
    if False:
        i = 10
        return i + 15
    'Test for configure new session parameters.'
    request = {'fulfillmentInfo': {'tag': 'configure-session-parameter'}}
    with app.test_request_context(json=request):
        res = configure_session_params(flask.request)
        assert 'orderNumber' in res['sessionInfo']['parameters']