import requests
from apprise.plugins.NotifyPopcornNotify import NotifyPopcornNotify
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('popcorn://', {'instance': TypeError}), ('popcorn://{}/18001231234'.format('_' * 9), {'instance': TypeError}), ('popcorn://{}/1232348923489234923489234289-32423'.format('a' * 9), {'instance': NotifyPopcornNotify, 'notify_response': False}), ('popcorn://{}/abc'.format('b' * 9), {'instance': NotifyPopcornNotify, 'notify_response': False}), ('popcorn://{}/15551232000/user@example.com'.format('c' * 9), {'instance': NotifyPopcornNotify}), ('popcorn://{}/15551232000/user@example.com?batch=yes'.format('w' * 9), {'instance': NotifyPopcornNotify}), ('popcorn://{}/?to=15551232000'.format('w' * 9), {'instance': NotifyPopcornNotify}), ('popcorn://{}/15551232000'.format('x' * 9), {'instance': NotifyPopcornNotify, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('popcorn://{}/15551232000'.format('y' * 9), {'instance': NotifyPopcornNotify, 'response': False, 'requests_response_code': 999}), ('popcorn://{}/15551232000'.format('z' * 9), {'instance': NotifyPopcornNotify, 'test_requests_exceptions': True}))

def test_plugin_popcorn_notify_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyPopcornNotify() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()