"""Test validate form parameter webhook snippet."""
import flask
import pytest
from webhook_validate_form_parameter import validate_parameter

@pytest.fixture(name='app', scope='module')
def fixture_app():
    if False:
        for i in range(10):
            print('nop')
    'Flask fixture to pass a flask.Request to the test function'
    return flask.Flask(__name__)

def test_validate_parameter(app):
    if False:
        i = 10
        return i + 15
    'Parameterized test for validate form parameter webhook snippet.'
    request = {'pageInfo': {'formInfo': {'parameterInfo': [{'value': 123}]}}}
    with app.test_request_context(json=request):
        res = validate_parameter(flask.request)
        assert res['page_info']['form_info']['parameter_info'][0]['state'] == 'INVALID'