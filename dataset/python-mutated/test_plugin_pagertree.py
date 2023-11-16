import requests
from unittest import mock
import pytest
from apprise.plugins.NotifyPagerTree import NotifyPagerTree
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
INTEGRATION_ID = 'int_xxxxxxxxxxx'
apprise_url_tests = (('pagertree://', {'instance': TypeError}), ('pagertree://%s' % ('+' * 24), {'instance': TypeError}), ('pagertree://%s' % INTEGRATION_ID, {'instance': NotifyPagerTree, 'privacy_url': 'pagertree://i...x?'}), ('pagertree://%s?integration=int_yyyyyyyyyy' % INTEGRATION_ID, {'instance': NotifyPagerTree, 'privacy_url': 'pagertree://i...y?'}), ('pagertree://%s?id=int_zzzzzzzzzz' % INTEGRATION_ID, {'instance': NotifyPagerTree, 'privacy_url': 'pagertree://i...z?'}), ('pagertree://:@/', {'instance': TypeError}), ('pagertree://%s' % INTEGRATION_ID, {'instance': NotifyPagerTree, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('pagertree://%s' % INTEGRATION_ID, {'instance': NotifyPagerTree, 'response': False, 'requests_response_code': 999}), ('pagertree://%s' % INTEGRATION_ID, {'instance': NotifyPagerTree, 'test_requests_exceptions': True}), ('pagertree://%s?urgency=low' % INTEGRATION_ID, {'instance': NotifyPagerTree}), ('pagertree://?id=%s&urgency=low' % INTEGRATION_ID, {'instance': NotifyPagerTree}), ('pagertree://%s?tags=production,web' % INTEGRATION_ID, {'instance': NotifyPagerTree}), ('pagertree://%s?action=resolve&thirdparty=123' % INTEGRATION_ID, {'instance': NotifyPagerTree}), ('pagertree://%s?+pagertree-token=123&:env=prod&-m=v' % INTEGRATION_ID, {'instance': NotifyPagerTree}))

def test_plugin_pagertree_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyPagerTree() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_pagertree_general(mock_post):
    if False:
        return 10
    '\n    NotifyPagerTree() General Checks\n\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    with pytest.raises(TypeError):
        NotifyPagerTree(integration=INTEGRATION_ID, thirdparty='   ')