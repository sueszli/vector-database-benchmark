import pytest
import requests
from apprise.plugins.NotifyZulip import NotifyZulip
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('zulip://', {'instance': TypeError}), ('zulip://:@/', {'instance': TypeError}), ('zulip://apprise', {'instance': TypeError}), ('zulip://botname@apprise', {'instance': TypeError}), ('zulip://botname@apprise/{}'.format('a' * 24), {'instance': TypeError}), ('zulip://....@apprise/{}'.format('a' * 32), {'instance': TypeError}), ('zulip://bot-name@apprise/{}'.format('a' * 32), {'instance': NotifyZulip, 'privacy_url': 'zulip://bot-name@apprise/a...a/'}), ('zulip://botname@apprise/{}'.format('a' * 32), {'instance': NotifyZulip, 'privacy_url': 'zulip://botname@apprise/a...a/'}), ('zulip://botname@apprise.zulipchat.com/{}'.format('a' * 32), {'instance': NotifyZulip}), ('zulip://botname@apprise/{}/channel1/channel2'.format('a' * 32), {'instance': NotifyZulip}), ('zulip://botname@apprise/{}/?to=channel1/channel2'.format('a' * 32), {'instance': NotifyZulip}), ('zulip://botname@apprise/{}/user@example.com/user2@example.com'.format('a' * 32), {'instance': NotifyZulip}), ('zulip://botname@apprise/{}'.format('a' * 32), {'instance': NotifyZulip, 'include_image': False}), ('zulip://botname@apprise/{}'.format('a' * 32), {'instance': NotifyZulip, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('zulip://botname@apprise/{}'.format('a' * 32), {'instance': NotifyZulip, 'response': False, 'requests_response_code': 999}), ('zulip://botname@apprise/{}'.format('a' * 32), {'instance': NotifyZulip, 'test_requests_exceptions': True}))

def test_plugin_zulip_urls():
    if False:
        while True:
            i = 10
    '\n    NotifyZulip() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_zulip_edge_cases():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyZulip() Edge Cases\n\n    '
    token = 'a' * 32
    with pytest.raises(TypeError):
        NotifyZulip(botname='test', organization='#', token=token)