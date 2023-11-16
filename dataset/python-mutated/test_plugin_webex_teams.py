import requests
from apprise.plugins.NotifyWebexTeams import NotifyWebexTeams
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('wxteams://', {'instance': TypeError}), ('wxteams://:@/', {'instance': TypeError}), ('wxteams://{}'.format('a' * 80), {'instance': NotifyWebexTeams, 'privacy_url': 'wxteams://a...a/'}), ('webex://{}'.format('a' * 140), {'instance': NotifyWebexTeams, 'privacy_url': 'wxteams://a...a/'}), ('https://api.ciscospark.com/v1/webhooks/incoming/{}'.format('a' * 80), {'instance': NotifyWebexTeams}), ('https://webexapis.com/v1/webhooks/incoming/{}'.format('a' * 100), {'instance': NotifyWebexTeams}), ('https://api.ciscospark.com/v1/webhooks/incoming/{}?format=text'.format('a' * 80), {'instance': NotifyWebexTeams}), ('wxteams://{}'.format('a' * 80), {'instance': NotifyWebexTeams, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('wxteams://{}'.format('a' * 80), {'instance': NotifyWebexTeams, 'response': False, 'requests_response_code': 999}), ('wxteams://{}'.format('a' * 80), {'instance': NotifyWebexTeams, 'test_requests_exceptions': True}))

def test_plugin_webex_teams_urls():
    if False:
        while True:
            i = 10
    '\n    NotifyWebexTeams() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()