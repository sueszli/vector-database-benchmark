import pytest
pytestmark = [pytest.mark.skip_on_windows]

def test_salt_api(api_request):
    if False:
        return 10
    '\n    Test running a command against the salt api\n    '
    ret = api_request.post('/run', data={'client': 'local', 'tgt': '*', 'fun': 'test.arg', 'arg': ['foo', 'bar']})
    assert ret['args'] == ['foo', 'bar']