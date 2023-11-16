from apprise.plugins.NotifyDingTalk import NotifyDingTalk
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('dingtalk://', {'instance': TypeError}), ('dingtalk://a_bd_/', {'instance': TypeError}), ('dingtalk://12345678', {'instance': NotifyDingTalk, 'privacy_url': 'dingtalk://1...8'}), ('dingtalk://{}/{}'.format('a' * 8, '1' * 14), {'instance': NotifyDingTalk}), ('dingtalk://{}/{}/invalid'.format('a' * 8, '1' * 3), {'instance': NotifyDingTalk}), ('dingtalk://{}/?to={}'.format('a' * 8, '1' * 14), {'instance': NotifyDingTalk}), ('dingtalk://secret@{}/?to={}'.format('a' * 8, '1' * 14), {'instance': NotifyDingTalk, 'privacy_url': 'dingtalk://****@a...a'}), ('dingtalk://?token={}&to={}&secret={}'.format('b' * 8, '1' * 14, 'a' * 15), {'instance': NotifyDingTalk, 'privacy_url': 'dingtalk://****@b...b'}), ('dingtalk://{}/?to={}&secret=_'.format('a' * 8, '1' * 14), {'instance': TypeError}), ('dingtalk://{}?format=markdown'.format('a' * 8), {'instance': NotifyDingTalk}), ('dingtalk://{}'.format('a' * 8), {'instance': NotifyDingTalk, 'response': False, 'requests_response_code': 999}), ('dingtalk://{}'.format('a' * 8), {'instance': NotifyDingTalk, 'test_requests_exceptions': True}))

def test_plugin_dingtalk_urls():
    if False:
        return 10
    '\n    NotifyDingTalk() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()