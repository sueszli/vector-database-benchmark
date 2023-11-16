from unittest import mock
import pytest
import requests
from apprise.plugins.NotifySendGrid import NotifySendGrid
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
UUID4 = '8b799edf-6f98-4d3a-9be7-2862fb4e5752'
apprise_url_tests = (('sendgrid://', {'instance': None}), ('sendgrid://:@/', {'instance': None}), ('sendgrid://abcd', {'instance': None}), ('sendgrid://abcd@host', {'instance': None}), ('sendgrid://invalid-api-key+*-d:user@example.com', {'instance': TypeError}), ('sendgrid://abcd:user@example.com', {'instance': NotifySendGrid}), ('sendgrid://abcd:user@example.com/newuser@example.com', {'instance': NotifySendGrid}), ('sendgrid://abcd:user@example.com/newuser@example.com?bcc=l2g@nuxref.com', {'instance': NotifySendGrid}), ('sendgrid://abcd:user@example.com/newuser@example.com?cc=l2g@nuxref.com', {'instance': NotifySendGrid}), ('sendgrid://abcd:user@example.com/newuser@example.com?to=l2g@nuxref.com', {'instance': NotifySendGrid}), ('sendgrid://abcd:user@example.com/newuser@example.com?template={}'.format(UUID4), {'instance': NotifySendGrid}), ('sendgrid://abcd:user@example.com/newuser@example.com?template={}&+sub=value&+sub2=value2'.format(UUID4), {'instance': NotifySendGrid, 'privacy_url': 'sendgrid://a...d:user@example.com/'}), ('sendgrid://abcd:user@example.ca/newuser@example.ca', {'instance': NotifySendGrid, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('sendgrid://abcd:user@example.uk/newuser@example.uk', {'instance': NotifySendGrid, 'response': False, 'requests_response_code': 999}), ('sendgrid://abcd:user@example.au/newuser@example.au', {'instance': NotifySendGrid, 'test_requests_exceptions': True}))

def test_plugin_sendgrid_urls():
    if False:
        while True:
            i = 10
    '\n    NotifySendGrid() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_sendgrid_edge_cases(mock_post, mock_get):
    if False:
        return 10
    '\n    NotifySendGrid() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifySendGrid(apikey=None, from_email='user@example.com')
    with pytest.raises(TypeError):
        NotifySendGrid(apikey='abcd', from_email='!invalid')
    with pytest.raises(TypeError):
        NotifySendGrid(apikey='abcd', from_email=None)
    NotifySendGrid(apikey='abcd', from_email='user@example.com', targets='!invalid')
    assert isinstance(NotifySendGrid(apikey='abcd', from_email='l2g@example.com', bcc=('abc@def.com', '!invalid'), cc=('abc@test.org', '!invalid')), NotifySendGrid)