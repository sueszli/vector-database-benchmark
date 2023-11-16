from unittest import mock
import requests
import apprise
from apprise.plugins.NotifyOpsgenie import NotifyOpsgenie, OpsgeniePriority
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
UUID4 = '8b799edf-6f98-4d3a-9be7-2862fb4e5752'
apprise_url_tests = (('opsgenie://', {'instance': TypeError}), ('opsgenie://:@/', {'instance': TypeError}), ('opsgenie://%20%20/', {'instance': TypeError}), ('opsgenie://apikey/user/?region=xx', {'instance': TypeError}), ('opsgenie://apikey/', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/user', {'instance': NotifyOpsgenie, 'privacy_url': 'opsgenie://a...y/%40user'}), ('opsgenie://apikey/@user?region=eu', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/@user?entity=A%20Entity', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/@user?alias=An%20Alias', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/@user?priority=p3', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/?tags=comma,separated', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/@user?priority=invalid', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/user@email.com/#team/*sche/^esc/%20/a', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/@{}/#{}/*{}/^{}/'.format(UUID4, UUID4, UUID4, UUID4), {'instance': NotifyOpsgenie}), ('opsgenie://apikey/{}/#{}/*{}/^{}/'.format(UUID4, UUID4, UUID4, UUID4), {'instance': NotifyOpsgenie}), ('opsgenie://apikey?to=#team,user&+key=value&+type=override', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/#team/@user/?batch=yes', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/#team/@user/?batch=no', {'instance': NotifyOpsgenie}), ('opsgenie://?apikey=abc&to=user', {'instance': NotifyOpsgenie}), ('opsgenie://apikey/#team/user/', {'instance': NotifyOpsgenie, 'response': False, 'requests_response_code': 999}), ('opsgenie://apikey/#topic1/device/', {'instance': NotifyOpsgenie, 'test_requests_exceptions': True}))

def test_plugin_opsgenie_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyOpsgenie() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_opsgenie_config_files(mock_post):
    if False:
        return 10
    '\n    NotifyOpsgenie() Config File Cases\n    '
    content = '\n    urls:\n      - opsgenie://apikey/user:\n          - priority: 1\n            tag: opsgenie_int low\n          - priority: "1"\n            tag: opsgenie_str_int low\n          - priority: "p1"\n            tag: opsgenie_pstr_int low\n          - priority: low\n            tag: opsgenie_str low\n\n          # This will take on moderate (default) priority\n          - priority: invalid\n            tag: opsgenie_invalid\n\n      - opsgenie://apikey2/user2:\n          - priority: 5\n            tag: opsgenie_int emerg\n          - priority: "5"\n            tag: opsgenie_str_int emerg\n          - priority: "p5"\n            tag: opsgenie_pstr_int emerg\n          - priority: emergency\n            tag: opsgenie_str emerg\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    ac = apprise.AppriseConfig()
    assert ac.add_config(content=content) is True
    aobj = apprise.Apprise()
    aobj.add(ac)
    assert len(ac.servers()) == 9
    assert len(aobj) == 9
    assert len([x for x in aobj.find(tag='low')]) == 4
    for s in aobj.find(tag='low'):
        assert s.priority == OpsgeniePriority.LOW
    assert len([x for x in aobj.find(tag='emerg')]) == 4
    for s in aobj.find(tag='emerg'):
        assert s.priority == OpsgeniePriority.EMERGENCY
    assert len([x for x in aobj.find(tag='opsgenie_str')]) == 2
    assert len([x for x in aobj.find(tag='opsgenie_str_int')]) == 2
    assert len([x for x in aobj.find(tag='opsgenie_pstr_int')]) == 2
    assert len([x for x in aobj.find(tag='opsgenie_int')]) == 2
    assert len([x for x in aobj.find(tag='opsgenie_invalid')]) == 1
    assert next(aobj.find(tag='opsgenie_invalid')).priority == OpsgeniePriority.NORMAL