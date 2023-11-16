import requests
from apprise.plugins.NotifyNotica import NotifyNotica
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('notica://', {'instance': TypeError}), ('notica://:@/', {'instance': TypeError}), ('https://notica.us/?%s' % ('z' * 6), {'instance': NotifyNotica, 'privacy_url': 'notica://z...z/'}), ('https://notica.us/?%s&overflow=upstream' % ('z' * 6), {'instance': NotifyNotica, 'privacy_url': 'notica://z...z/'}), ('notica://%s' % ('a' * 6), {'instance': NotifyNotica, 'privacy_url': 'notica://a...a/'}), ('notica://localhost/%s' % ('b' * 6), {'instance': NotifyNotica}), ('notica://user@localhost/%s' % ('c' * 6), {'instance': NotifyNotica}), ('notica://user:pass@localhost/%s/' % ('d' * 6), {'instance': NotifyNotica, 'privacy_url': 'notica://user:****@localhost/d...d'}), ('notica://user:pass@localhost/a/path/%s/' % ('r' * 6), {'instance': NotifyNotica, 'privacy_url': 'notica://user:****@localhost/a/path/r...r'}), ('notica://localhost:8080/%s' % ('a' * 6), {'instance': NotifyNotica}), ('notica://user:pass@localhost:8080/%s' % ('b' * 6), {'instance': NotifyNotica}), ('noticas://localhost/%s' % ('j' * 6), {'instance': NotifyNotica, 'privacy_url': 'noticas://localhost/j...j'}), ('noticas://user:pass@localhost/%s' % ('e' * 6), {'instance': NotifyNotica, 'privacy_url': 'noticas://user:****@localhost/e...e'}), ('noticas://localhost:8080/path/%s' % ('5' * 6), {'instance': NotifyNotica, 'privacy_url': 'noticas://localhost:8080/path/5...5'}), ('noticas://user:pass@localhost:8080/%s' % ('6' * 6), {'instance': NotifyNotica}), ('notica://%s' % ('b' * 6), {'instance': NotifyNotica, 'include_image': False}), ('notica://localhost:8080//%s/?+HeaderKey=HeaderValue' % ('7' * 6), {'instance': NotifyNotica}), ('notica://%s' % ('c' * 6), {'instance': NotifyNotica, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('notica://%s' % ('d' * 7), {'instance': NotifyNotica, 'response': False, 'requests_response_code': 999}), ('notica://%s' % ('e' * 8), {'instance': NotifyNotica, 'test_requests_exceptions': True}))

def test_plugin_notica_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyNotica() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()