"""Test validate form webhook request session ID snippet."""
import flask
import pytest
from webhook_log_session_info import log_session_id_for_troubleshooting

@pytest.fixture(name='app', scope='module')
def fixture_app():
    if False:
        return 10
    'Flask fixture to pass a flask.Request to the test function'
    return flask.Flask(__name__)

@pytest.fixture
def session_id():
    if False:
        i = 10
        return i + 15
    return 'd0bdaa0c-0d00-0000-b0eb-b00b0db000b0'

@pytest.fixture
def session_prefix():
    if False:
        print('Hello World!')
    agent_id = '000000f0-f000-00b0-0000-af00d0e00000'
    return f'projects/test_project/locations/us-central1/agents/{agent_id}'

@pytest.fixture
def session(session_prefix, session_id):
    if False:
        while True:
            i = 10
    'Session string without environment path'
    return f'{session_prefix}/sessions/{session_id}'

@pytest.fixture
def env_session(session_prefix, session_id):
    if False:
        for i in range(10):
            print('nop')
    'Session string with environment path'
    environment = '0d0000f0-0aac-0d0c-0a00-b00b0000a000'
    return f'{session_prefix}/environments/{environment}/sessions/{session_id}'

def test_logging_session_id(app, session, session_id):
    if False:
        print('Hello World!')
    'Parameterized test for regular session string.'
    request = {'sessionInfo': {'session': session}}
    with app.test_request_context(json=request):
        res = log_session_id_for_troubleshooting(flask.request)
        assert session_id in str(res)

def test_logging_session_id_with_env_path(app, env_session, session_id):
    if False:
        i = 10
        return i + 15
    'Parameterized test for session string with environment path.'
    request = {'sessionInfo': {'session': env_session}}
    with app.test_request_context(json=request):
        res = log_session_id_for_troubleshooting(flask.request)
        assert session_id in str(res)