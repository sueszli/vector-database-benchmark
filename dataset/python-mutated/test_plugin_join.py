from unittest import mock
import pytest
import requests
import apprise
from apprise import NotifyType
from helpers import AppriseURLTester
from apprise.plugins.NotifyJoin import JoinPriority, NotifyJoin
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('join://', {'instance': TypeError}), ('join://:@/', {'instance': TypeError}), ('join://%s' % ('a' * 32), {'instance': NotifyJoin}), ('join://%s?to=%s' % ('a' * 32, 'd' * 32), {'instance': NotifyJoin, 'privacy_url': 'join://a...a/'}), ('join://%s?priority=high' % ('a' * 32), {'instance': NotifyJoin}), ('join://%s?priority=invalid' % ('a' * 32), {'instance': NotifyJoin}), ('join://%s?priority=' % ('a' * 32), {'instance': NotifyJoin}), ('join://%s@%s?image=True' % ('a' * 32, 'd' * 32), {'instance': NotifyJoin}), ('join://%s@%s?image=False' % ('a' * 32, 'd' * 32), {'instance': NotifyJoin}), ('join://%s/%s' % ('a' * 32, 'My Device'), {'instance': NotifyJoin}), ('join://%s/%s' % ('a' * 32, 'd' * 32), {'instance': NotifyJoin, 'include_image': False}), ('join://%s/%s/%s' % ('a' * 32, 'd' * 32, 'e' * 32), {'instance': NotifyJoin, 'include_image': False}), ('join://%s/%s/%s' % ('a' * 32, 'd' * 32, 'group.chrome'), {'instance': NotifyJoin}), ('join://%s' % ('a' * 32), {'instance': NotifyJoin, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('join://%s' % ('a' * 32), {'instance': NotifyJoin, 'response': False, 'requests_response_code': 999}), ('join://%s' % ('a' * 32), {'instance': NotifyJoin, 'test_requests_exceptions': True}))

def test_plugin_join_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyJoin() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_join_edge_cases(mock_post, mock_get):
    if False:
        i = 10
        return i + 15
    '\n    NotifyJoin() Edge Cases\n\n    '
    device = 'A' * 32
    group = 'group.chrome'
    apikey = 'a' * 32
    NotifyJoin(apikey=apikey, targets=group)
    NotifyJoin(apikey=apikey, targets=None)
    with pytest.raises(TypeError):
        NotifyJoin(apikey=None)
    with pytest.raises(TypeError):
        NotifyJoin(apikey='   ')
    p = NotifyJoin(apikey=apikey, targets=[group, device])
    req = requests.Request()
    req.status_code = requests.codes.created
    req.content = ''
    mock_get.return_value = req
    mock_post.return_value = req
    p.notify(body=None, title=None, notify_type=NotifyType.INFO) is False

@mock.patch('requests.post')
def test_plugin_join_config_files(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyJoin() Config File Cases\n    '
    content = '\n    urls:\n      - join://%s@%s:\n          - priority: -2\n            tag: join_int low\n          - priority: "-2"\n            tag: join_str_int low\n          - priority: low\n            tag: join_str low\n\n          # This will take on normal (default) priority\n          - priority: invalid\n            tag: join_invalid\n\n      - join://%s@%s:\n          - priority: 2\n            tag: join_int emerg\n          - priority: "2"\n            tag: join_str_int emerg\n          - priority: emergency\n            tag: join_str emerg\n    ' % ('a' * 32, 'b' * 32, 'c' * 32, 'd' * 32)
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
        assert s.priority == JoinPriority.LOW
    assert len([x for x in aobj.find(tag='emerg')]) == 3
    for s in aobj.find(tag='emerg'):
        assert s.priority == JoinPriority.EMERGENCY
    assert len([x for x in aobj.find(tag='join_str')]) == 2
    assert len([x for x in aobj.find(tag='join_str_int')]) == 2
    assert len([x for x in aobj.find(tag='join_int')]) == 2
    assert len([x for x in aobj.find(tag='join_invalid')]) == 1
    assert next(aobj.find(tag='join_invalid')).priority == JoinPriority.NORMAL
    assert aobj.notify(title='title', body='body') is True