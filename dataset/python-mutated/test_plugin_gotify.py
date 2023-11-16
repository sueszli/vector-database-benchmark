from unittest import mock
import pytest
import requests
import apprise
from apprise.plugins.NotifyGotify import GotifyPriority, NotifyGotify
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('gotify://', {'instance': None}), ('gotify://hostname', {'instance': TypeError}), ('gotify://hostname/%s' % ('t' * 16), {'instance': NotifyGotify, 'privacy_url': 'gotify://hostname/t...t'}), ('gotify://hostname/a/path/ending/in/a/slash/%s' % ('u' * 16), {'instance': NotifyGotify, 'privacy_url': 'gotify://hostname/a/path/ending/in/a/slash/u...u/'}), ('gotify://hostname/%s?format=markdown' % ('t' * 16), {'instance': NotifyGotify}), ('gotify://hostname/a/path/not/ending/in/a/slash/%s' % ('v' * 16), {'instance': NotifyGotify, 'privacy_url': 'gotify://hostname/a/path/not/ending/in/a/slash/v...v/'}), ('gotify://hostname/%s?priority=high' % ('i' * 16), {'instance': NotifyGotify}), ('gotify://hostname:8008/%s?priority=invalid' % ('i' * 16), {'instance': NotifyGotify}), ('gotify://:@/', {'instance': None}), ('gotify://hostname/%s/' % ('t' * 16), {'instance': NotifyGotify, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('gotifys://localhost/%s/' % ('t' * 16), {'instance': NotifyGotify, 'response': False, 'requests_response_code': 999}), ('gotify://localhost/%s/' % ('t' * 16), {'instance': NotifyGotify, 'test_requests_exceptions': True}))

def test_plugin_gotify_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyGotify() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_gotify_edge_cases():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyGotify() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyGotify(token=None)
    with pytest.raises(TypeError):
        NotifyGotify(token='   ')

@mock.patch('requests.post')
def test_plugin_gotify_config_files(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifyGotify() Config File Cases\n    '
    content = '\n    urls:\n      - gotify://hostname/%s:\n          - priority: 0\n            tag: gotify_int low\n          - priority: "0"\n            tag: gotify_str_int low\n          # We want to make sure our \'1\' does not match the \'10\' entry\n          - priority: "1"\n            tag: gotify_str_int low\n          - priority: low\n            tag: gotify_str low\n\n          # This will take on moderate (default) priority\n          - priority: invalid\n            tag: gotify_invalid\n\n      - gotify://hostname/%s:\n          - priority: 10\n            tag: gotify_int emerg\n          - priority: "10"\n            tag: gotify_str_int emerg\n          - priority: emergency\n            tag: gotify_str emerg\n    ' % ('a' * 16, 'b' * 16)
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    ac = apprise.AppriseConfig()
    assert ac.add_config(content=content) is True
    aobj = apprise.Apprise()
    aobj.add(ac)
    assert len(ac.servers()) == 8
    assert len(aobj) == 8
    assert len([x for x in aobj.find(tag='low')]) == 4
    for s in aobj.find(tag='low'):
        assert s.priority == GotifyPriority.LOW
    assert len([x for x in aobj.find(tag='emerg')]) == 3
    for s in aobj.find(tag='emerg'):
        assert s.priority == GotifyPriority.EMERGENCY
    assert len([x for x in aobj.find(tag='gotify_str')]) == 2
    assert len([x for x in aobj.find(tag='gotify_str_int')]) == 3
    assert len([x for x in aobj.find(tag='gotify_int')]) == 2
    assert len([x for x in aobj.find(tag='gotify_invalid')]) == 1
    assert next(aobj.find(tag='gotify_invalid')).priority == GotifyPriority.NORMAL