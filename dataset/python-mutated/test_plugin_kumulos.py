import pytest
import requests
from apprise.plugins.NotifyKumulos import NotifyKumulos
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
UUID4 = '8b799edf-6f98-4d3a-9be7-2862fb4e5752'
apprise_url_tests = (('kumulos://', {'instance': TypeError}), ('kumulos://:@/', {'instance': TypeError}), ('kumulos://{}/'.format(UUID4), {'instance': TypeError}), ('kumulos://{}/{}/'.format(UUID4, 'w' * 36), {'instance': NotifyKumulos, 'privacy_url': 'kumulos://8...2/w...w/'}), ('kumulos://{}/{}/'.format(UUID4, 'x' * 36), {'instance': NotifyKumulos, 'response': False, 'requests_response_code': requests.codes.internal_server_error, 'privacy_url': 'kumulos://8...2/x...x/'}), ('kumulos://{}/{}/'.format(UUID4, 'y' * 36), {'instance': NotifyKumulos, 'response': False, 'requests_response_code': 999, 'privacy_url': 'kumulos://8...2/y...y/'}), ('kumulos://{}/{}/'.format(UUID4, 'z' * 36), {'instance': NotifyKumulos, 'test_requests_exceptions': True}))

def test_plugin_kumulos_urls():
    if False:
        while True:
            i = 10
    '\n    NotifyKumulos() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_kumulos_edge_cases():
    if False:
        while True:
            i = 10
    '\n    NotifyKumulos() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyKumulos(None, None)
    with pytest.raises(TypeError):
        NotifyKumulos('     ', None)
    with pytest.raises(TypeError):
        NotifyKumulos('abcd', None)
    with pytest.raises(TypeError):
        NotifyKumulos('abcd', '       ')