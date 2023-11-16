import requests
from apprise.plugins.NotifyNotifico import NotifyNotifico
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('notifico://', {'instance': TypeError}), ('notifico://:@/', {'instance': TypeError}), ('notifico://1234', {'instance': TypeError}), ('notifico://abcd/ckhrjW8w672m6HG', {'instance': TypeError}), ('notifico://1234/ckhrjW8w672m6HG', {'instance': NotifyNotifico}), ('notifico://1234/ckhrjW8w672m6HG?prefix=no', {'instance': NotifyNotifico}), ('notifico://1234/ckhrjW8w672m6HG?color=yes', {'instance': NotifyNotifico, 'notify_type': 'info'}), ('notifico://1234/ckhrjW8w672m6HG?color=yes', {'instance': NotifyNotifico, 'notify_type': 'success'}), ('notifico://1234/ckhrjW8w672m6HG?color=yes', {'instance': NotifyNotifico, 'notify_type': 'warning'}), ('notifico://1234/ckhrjW8w672m6HG?color=yes', {'instance': NotifyNotifico, 'notify_type': 'failure'}), ('notifico://1234/ckhrjW8w672m6HG?color=yes', {'instance': NotifyNotifico, 'notify_type': 'invalid'}), ('notifico://1234/ckhrjW8w672m6HG?color=no', {'instance': NotifyNotifico, 'privacy_url': 'notifico://1...4/c...G'}), ('https://n.tkte.ch/h/2144/uJmKaBW9WFk42miB146ci3Kj', {'instance': NotifyNotifico}), ('notifico://1234/ckhrjW8w672m6HG', {'instance': NotifyNotifico, 'include_image': False}), ('notifico://1234/ckhrjW8w672m6HG', {'instance': NotifyNotifico, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('notifico://1234/ckhrjW8w672m6HG', {'instance': NotifyNotifico, 'response': False, 'requests_response_code': 999}), ('notifico://1234/ckhrjW8w672m6HG', {'instance': NotifyNotifico, 'test_requests_exceptions': True}))

def test_plugin_notifico_urls():
    if False:
        return 10
    '\n    NotifyNotifico() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()