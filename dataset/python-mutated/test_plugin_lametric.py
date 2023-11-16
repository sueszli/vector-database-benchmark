import pytest
import requests
from apprise.plugins.NotifyLametric import NotifyLametric
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
UUID4 = '8b799edf-6f98-4d3a-9be7-2862fb4e5752'
apprise_url_tests = (('lametric://', {'instance': TypeError}), ('lametric://:@/', {'instance': TypeError}), ('lametric://{}/'.format('com.lametric.941c51dff3135bd87aa72db9d855dd50'), {'instance': TypeError}), ('lametric://root:{}@192.168.0.5:8080/'.format(UUID4), {'instance': NotifyLametric, 'privacy_url': 'lametric://root:8...2@192.168.0.5/'}), ('lametric://{}@192.168.0.4:8000/'.format(UUID4), {'instance': NotifyLametric, 'privacy_url': 'lametric://8...2@192.168.0.4:8000/'}), ('lametric://{}@192.168.0.5/'.format(UUID4), {'instance': NotifyLametric, 'privacy_url': 'lametric://8...2@192.168.0.5/'}), ('lametrics://{}@192.168.0.6/?mode=device'.format(UUID4), {'instance': NotifyLametric, 'privacy_url': 'lametrics://8...2@192.168.0.6/'}), ('https://developer.lametric.com/api/v1/dev/widget/update/com.lametric.ABCD123/1?token={}=='.format('D' * 88), {'instance': NotifyLametric, 'privacy_url': 'lametric://D...=@A...3/1/'}), ('lametric://192.168.2.8/?mode=device&apikey=abc123', {'instance': NotifyLametric, 'privacy_url': 'lametric://a...3@192.168.2.8/'}), ('lametrics://{}==@com.lametric.941c51dff3135bd87aa72db9d855dd50/?mode=cloud&app_ver=2'.format('A' * 88), {'instance': NotifyLametric, 'privacy_url': 'lametric://A...=@9...0/'}), ('lametrics://{}==@com.lametric.941c51dff3135bd87aa72db9d855dd50/?app_ver=invalid'.format('A' * 88), {'instance': TypeError}), ('lametric://?app=com.lametric.941c51dff3135bd87aa72db9d855dd50&token={}==&mode=cloud'.format('B' * 88), {'instance': NotifyLametric, 'privacy_url': 'lametric://B...=@9...0/'}), ('lametrics://{}==@abcd/?mode=cloud&sound=knock&icon_type=info&priority=critical&cycles=10'.format('C' * 88), {'instance': NotifyLametric, 'privacy_url': 'lametric://C...=@a...d/'}), ('lametrics://{}@192.168.0.7/?mode=invalid'.format(UUID4), {'instance': TypeError}), ('lametrics://{}@192.168.0.6/?sound=alarm1'.format(UUID4), {'instance': NotifyLametric}), ('lametrics://{}@192.168.0.7/?sound=bike'.format(UUID4), {'instance': NotifyLametric, 'url_matches': 'sound=bicycle'}), ('lametrics://{}@192.168.0.8/?sound=invalid!'.format(UUID4), {'instance': NotifyLametric}), ('lametrics://{}@192.168.0.9/?icon_type=alert'.format(UUID4), {'instance': NotifyLametric, 'url_matches': 'icon_type=alert'}), ('lametrics://{}@192.168.0.10/?icon_type=invalid'.format(UUID4), {'instance': NotifyLametric}), ('lametric://{}@192.168.1.1/?priority=warning'.format(UUID4), {'instance': NotifyLametric}), ('lametrics://{}@192.168.1.2/?priority=invalid'.format(UUID4), {'instance': NotifyLametric}), ('lametric://{}@192.168.1.2/?icon=230'.format(UUID4), {'instance': NotifyLametric}), ('lametrics://{}@192.168.1.2/?icon=#230'.format(UUID4), {'instance': NotifyLametric}), ('lametric://{}@192.168.1.2/?icon=Heart'.format(UUID4), {'instance': NotifyLametric}), ('lametric://{}@192.168.1.2/?icon=#'.format(UUID4), {'instance': NotifyLametric}), ('lametric://{}@192.168.1.2/?icon=#%20%20%20'.format(UUID4), {'instance': NotifyLametric}), ('lametric://{}@192.168.1.3/?cycles=2'.format(UUID4), {'instance': NotifyLametric}), ('lametric://{}@192.168.1.4/?cycles=-1'.format(UUID4), {'instance': NotifyLametric}), ('lametrics://{}@192.168.1.5/?cycles=invalid'.format(UUID4), {'instance': NotifyLametric}), ('lametric://{}@example.com/'.format(UUID4), {'instance': NotifyLametric, 'response': False, 'requests_response_code': requests.codes.internal_server_error, 'privacy_url': 'lametric://8...2@example.com/'}), ('lametrics://{}@example.ca/'.format(UUID4), {'instance': NotifyLametric, 'response': False, 'requests_response_code': 999, 'privacy_url': 'lametrics://8...2@example.ca/'}), ('lametrics://{}@example.net/'.format(UUID4), {'instance': NotifyLametric, 'test_requests_exceptions': True}))

def test_plugin_lametric_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyLametric() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_lametric_edge_cases():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyLametric() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyLametric(apikey=None, mode='device')
    with pytest.raises(TypeError):
        NotifyLametric(client_id='valid', secret=None, mode='cloud')