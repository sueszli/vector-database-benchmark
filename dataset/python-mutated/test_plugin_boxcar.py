import pytest
from unittest import mock
from apprise.plugins.NotifyBoxcar import NotifyBoxcar
from helpers import AppriseURLTester
from apprise import NotifyType
import requests
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('boxcar://', {'instance': TypeError}), ('boxcar://:@/', {'instance': TypeError}), ('boxcar://%s' % ('a' * 64), {'instance': TypeError}), ('boxcar://%%20/%s' % ('a' * 64), {'instance': TypeError}), ('boxcar://%s/%%20' % ('a' * 64), {'instance': TypeError}), ('boxcar://%s/%s' % ('a' * 64, 'b' * 64), {'instance': NotifyBoxcar, 'requests_response_code': requests.codes.created, 'privacy_url': 'boxcar://a...a/****/'}), ('boxcar://%s/%s?image=True' % ('a' * 64, 'b' * 64), {'instance': NotifyBoxcar, 'requests_response_code': requests.codes.created, 'include_image': False}), ('boxcar://%s/%s?image=False' % ('a' * 64, 'b' * 64), {'instance': NotifyBoxcar, 'requests_response_code': requests.codes.created}), ('boxcar://%s/%s/@tag1/tag2///%s/?to=tag3' % ('a' * 64, 'b' * 64, 'd' * 64), {'instance': NotifyBoxcar, 'requests_response_code': requests.codes.created}), ('boxcar://?access=%s&secret=%s&to=tag5' % ('d' * 64, 'b' * 64), {'privacy_url': 'boxcar://d...d/****/', 'instance': NotifyBoxcar, 'requests_response_code': requests.codes.created}), ('boxcar://%s/%s/@%s' % ('a' * 64, 'b' * 64, 't' * 64), {'instance': NotifyBoxcar, 'requests_response_code': requests.codes.created}), ('boxcar://%s/%s/' % ('a' * 64, 'b' * 64), {'instance': NotifyBoxcar, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('boxcar://%s/%s/' % ('a' * 64, 'b' * 64), {'instance': NotifyBoxcar, 'response': False, 'requests_response_code': 999}), ('boxcar://%s/%s/' % ('a' * 64, 'b' * 64), {'instance': NotifyBoxcar, 'test_requests_exceptions': True}))

def test_plugin_boxcar_urls():
    if False:
        print('Hello World!')
    '\n    NotifyBoxcar() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_boxcar_edge_cases(mock_post, mock_get):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyBoxcar() Edge Cases\n\n    '
    device = 'A' * 64
    tag = '@B' * 63
    access = '-' * 64
    secret = '_' * 64
    NotifyBoxcar(access=access, secret=secret, targets=None)
    with pytest.raises(TypeError):
        NotifyBoxcar(access=None, secret=secret, targets=None)
    with pytest.raises(TypeError):
        NotifyBoxcar(access=access, secret=None, targets=None)
    NotifyBoxcar(access=access, secret=secret, targets=[device, tag])
    mock_get.return_value = requests.Request()
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.created
    mock_get.return_value.status_code = requests.codes.created
    p = NotifyBoxcar(access=access, secret=secret, targets=None)
    assert p.notify(body=None, title=None, notify_type=NotifyType.INFO) is False
    assert p.notify(body='Test', title=None, notify_type=NotifyType.INFO) is True
    device = 'a' * 64
    p = NotifyBoxcar(access=access, secret=secret, targets=','.join([device, device, device]))
    assert len(p.device_tokens) == 1
    p = NotifyBoxcar(access=access, secret=secret, targets=','.join(['a' * 64, 'b' * 64, 'c' * 64]))
    assert len(p.device_tokens) == 3