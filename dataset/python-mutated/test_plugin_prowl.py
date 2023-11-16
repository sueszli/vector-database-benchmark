from unittest import mock
import pytest
import requests
import apprise
from apprise.plugins.NotifyProwl import NotifyProwl, ProwlPriority
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('prowl://', {'instance': TypeError}), ('prowl://:@/', {'instance': TypeError}), ('prowl://%s' % ('a' * 20), {'instance': TypeError}), ('prowl://%s/%s' % ('a' * 40, 'b' * 40), {'instance': NotifyProwl}), ('prowl://%s/%s' % ('a' * 40, 'b' * 20), {'instance': TypeError}), ('prowl://%s' % ('a' * 40), {'instance': NotifyProwl}), ('prowl://%s' % ('a' * 40), {'instance': NotifyProwl, 'include_image': False}), ('prowl://%s?priority=high' % ('a' * 40), {'instance': NotifyProwl}), ('prowl://%s?priority=invalid' % ('a' * 40), {'instance': NotifyProwl}), ('prowl://%s?priority=' % ('a' * 40), {'instance': NotifyProwl}), ('prowl://%s///' % ('w' * 40), {'instance': NotifyProwl, 'privacy_url': 'prowl://w...w/'}), ('prowl://%s/%s' % ('a' * 40, 'b' * 40), {'instance': NotifyProwl, 'privacy_url': 'prowl://a...a/b...b'}), ('prowl://%s' % ('a' * 40), {'instance': NotifyProwl}), ('prowl://%s' % ('a' * 40), {'instance': NotifyProwl, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('prowl://%s' % ('a' * 40), {'instance': NotifyProwl, 'response': False, 'requests_response_code': 999}), ('prowl://%s' % ('a' * 40), {'instance': NotifyProwl, 'test_requests_exceptions': True}))

def test_plugin_prowl():
    if False:
        i = 10
        return i + 15
    '\n    NotifyProwl() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_prowl_edge_cases():
    if False:
        i = 10
        return i + 15
    '\n    NotifyProwl() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyProwl(apikey=None)
    with pytest.raises(TypeError):
        NotifyProwl(apikey='  ')
    with pytest.raises(TypeError):
        NotifyProwl(apikey='abcd', providerkey=object())
    with pytest.raises(TypeError):
        NotifyProwl(apikey='abcd', providerkey='  ')

@mock.patch('requests.post')
def test_plugin_prowl_config_files(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyProwl() Config File Cases\n    '
    content = '\n    urls:\n      - prowl://%s:\n          - priority: -2\n            tag: prowl_int low\n          - priority: "-2"\n            tag: prowl_str_int low\n          - priority: low\n            tag: prowl_str low\n\n          # This will take on moderate (default) priority\n          - priority: invalid\n            tag: prowl_invalid\n\n      - prowl://%s:\n          - priority: 2\n            tag: prowl_int emerg\n          - priority: "2"\n            tag: prowl_str_int emerg\n          - priority: emergency\n            tag: prowl_str emerg\n    ' % ('a' * 40, 'b' * 40)
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    ac = apprise.AppriseConfig()
    assert ac.add_config(content=content) is True
    aobj = apprise.Apprise()
    aobj.add(ac)
    assert len(ac.servers()) == 7
    assert len(aobj) == 7
    assert len([x for x in aobj.find(tag='low')]) == 3
    for s in aobj.find(tag='low'):
        assert s.priority == ProwlPriority.LOW
    assert len([x for x in aobj.find(tag='emerg')]) == 3
    for s in aobj.find(tag='emerg'):
        assert s.priority == ProwlPriority.EMERGENCY
    assert len([x for x in aobj.find(tag='prowl_str')]) == 2
    assert len([x for x in aobj.find(tag='prowl_str_int')]) == 2
    assert len([x for x in aobj.find(tag='prowl_int')]) == 2
    assert len([x for x in aobj.find(tag='prowl_invalid')]) == 1
    assert next(aobj.find(tag='prowl_invalid')).priority == ProwlPriority.NORMAL