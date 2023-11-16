import pytest
import requests
from apprise.plugins.NotifyMattermost import NotifyMattermost
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('mmost://', {'instance': None}), ('mmosts://', {'instance': None}), ('mmost://:@/', {'instance': None}), ('mmosts://localhost', {'instance': TypeError}), ('mmost://localhost/3ccdd113474722377935511fc85d3dd4', {'instance': NotifyMattermost}), ('mmost://user@localhost/3ccdd113474722377935511fc85d3dd4?channel=test', {'instance': NotifyMattermost}), ('mmost://user@localhost/3ccdd113474722377935511fc85d3dd4?to=test', {'instance': NotifyMattermost, 'privacy_url': 'mmost://user@localhost/3...4/'}), ('mmost://localhost/3ccdd113474722377935511fc85d3dd4?to=test&image=True', {'instance': NotifyMattermost}), ('mmost://localhost/3ccdd113474722377935511fc85d3dd4?to=test&image=False', {'instance': NotifyMattermost}), ('mmost://localhost/3ccdd113474722377935511fc85d3dd4?to=test&image=True', {'instance': NotifyMattermost, 'include_image': False}), ('mmost://localhost:8080/3ccdd113474722377935511fc85d3dd4', {'instance': NotifyMattermost, 'privacy_url': 'mmost://localhost:8080/3...4/'}), ('mmost://localhost:8080/3ccdd113474722377935511fc85d3dd4', {'instance': NotifyMattermost}), ('mmost://localhost:invalid-port/3ccdd113474722377935511fc85d3dd4', {'instance': None}), ('mmosts://localhost/3ccdd113474722377935511fc85d3dd4', {'instance': NotifyMattermost}), ('mmosts://localhost/a/path/3ccdd113474722377935511fc85d3dd4', {'instance': NotifyMattermost}), ('mmosts://localhost/////3ccdd113474722377935511fc85d3dd4///', {'instance': NotifyMattermost}), ('mmost://localhost/3ccdd113474722377935511fc85d3dd4', {'instance': NotifyMattermost, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('mmost://localhost/3ccdd113474722377935511fc85d3dd4', {'instance': NotifyMattermost, 'response': False, 'requests_response_code': 999}), ('mmost://localhost/3ccdd113474722377935511fc85d3dd4', {'instance': NotifyMattermost, 'test_requests_exceptions': True}))

def test_plugin_mattermost_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyMattermost() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

def test_plugin_mattermost_edge_cases():
    if False:
        return 10
    '\n    NotifyMattermost() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyMattermost(None)
    with pytest.raises(TypeError):
        NotifyMattermost('     ')