import requests
from unittest import mock
import apprise
from apprise.plugins.NotifyDapnet import DapnetPriority, NotifyDapnet
from helpers import AppriseURLTester
from apprise import NotifyType
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('dapnet://', {'instance': TypeError}), ('dapnet://:@/', {'instance': TypeError}), ('dapnet://user:pass', {'instance': TypeError}), ('dapnet://user@host', {'instance': TypeError}), ('dapnet://user:pass@{}'.format('DF1ABC'), {'instance': NotifyDapnet, 'requests_response_code': requests.codes.created}), ('dapnet://user:pass@{}/{}'.format('DF1ABC', 'DF1DEF'), {'instance': NotifyDapnet, 'requests_response_code': requests.codes.created}), ('dapnet://user:pass@DF1ABC-1/DF1ABC/DF1ABC-15', {'instance': NotifyDapnet, 'requests_response_code': requests.codes.created, 'privacy_url': 'dapnet://user:****@D...C?'}), ('dapnet://user:pass@?to={},{}'.format('DF1ABC', 'DF1DEF'), {'instance': NotifyDapnet, 'requests_response_code': requests.codes.created}), ('dapnet://user:pass@{}?priority=normal'.format('DF1ABC'), {'instance': NotifyDapnet, 'requests_response_code': requests.codes.created}), ('dapnet://user:pass@{}?priority=em&batch=false'.format('/'.join(['DF1ABC', '0A1DEF'])), {'instance': NotifyDapnet, 'requests_response_code': requests.codes.created}), ('dapnet://user:pass@{}?priority=invalid'.format('DF1ABC'), {'instance': NotifyDapnet, 'requests_response_code': requests.codes.created}), ('dapnet://user:pass@{}?txgroups=dl-all,all'.format('DF1ABC'), {'instance': NotifyDapnet, 'requests_response_code': requests.codes.created}), ('dapnet://user:pass@{}?txgroups=invalid'.format('DF1ABC'), {'instance': NotifyDapnet, 'requests_response_code': requests.codes.created}), ('dapnet://user:pass@{}/{}'.format('abcdefghi', 'a'), {'instance': NotifyDapnet, 'notify_response': False}), ('dapnet://user:pass@{}'.format('DF1ABC'), {'instance': NotifyDapnet, 'response': False, 'requests_response_code': 999}), ('dapnet://user:pass@{}'.format('DF1ABC'), {'instance': NotifyDapnet, 'test_requests_exceptions': True}))

def test_plugin_dapnet_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyDapnet() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_dapnet_edge_cases(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyDapnet() Edge Cases\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.created
    obj = apprise.Apprise.instantiate('dapnet://user:pass@{}?batch=yes'.format('/'.join(['DF1ABC', 'DF1DEF'])))
    assert isinstance(obj, NotifyDapnet)
    assert len(obj) == 1
    obj.default_batch_size = 1
    assert len(obj) == 2
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 2

@mock.patch('requests.post')
def test_plugin_dapnet_config_files(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyDapnet() Config File Cases\n    '
    content = '\n    urls:\n      - dapnet://user:pass@DF1ABC:\n          - priority: 0\n            tag: dapnet_int normal\n          - priority: "0"\n            tag: dapnet_str_int normal\n          - priority: normal\n            tag: dapnet_str normal\n\n          # This will take on normal (default) priority\n          - priority: invalid\n            tag: dapnet_invalid\n\n      - dapnet://user1:pass2@DF1ABC:\n          - priority: 1\n            tag: dapnet_int emerg\n          - priority: "1"\n            tag: dapnet_str_int emerg\n          - priority: emergency\n            tag: dapnet_str emerg\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.created
    ac = apprise.AppriseConfig()
    assert ac.add_config(content=content) is True
    aobj = apprise.Apprise()
    aobj.add(ac)
    assert len(ac.servers()) == 7
    assert len(aobj) == 7
    assert len([x for x in aobj.find(tag='normal')]) == 3
    for s in aobj.find(tag='normal'):
        assert s.priority == DapnetPriority.NORMAL
    assert len([x for x in aobj.find(tag='emerg')]) == 3
    for s in aobj.find(tag='emerg'):
        assert s.priority == DapnetPriority.EMERGENCY
    assert len([x for x in aobj.find(tag='dapnet_str')]) == 2
    assert len([x for x in aobj.find(tag='dapnet_str_int')]) == 2
    assert len([x for x in aobj.find(tag='dapnet_int')]) == 2
    assert len([x for x in aobj.find(tag='dapnet_invalid')]) == 1
    assert next(aobj.find(tag='dapnet_invalid')).priority == DapnetPriority.NORMAL
    assert aobj.notify(title='title', body='body') is True