import pytest
from unittest import mock
import requests
from apprise import NotifyType
from apprise.plugins.NotifyIFTTT import NotifyIFTTT
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('ifttt://', {'instance': TypeError}), ('ifttt://:@/', {'instance': TypeError}), ('ifttt://EventID/', {'instance': TypeError}), ('ifttt://WebHookID@EventID/?+TemplateKey=TemplateVal', {'instance': NotifyIFTTT, 'privacy_url': 'ifttt://W...D'}), ('ifttt://WebHookID?to=EventID,EventID2', {'instance': NotifyIFTTT}), ('ifttt://WebHookID@EventID/?-Value1=&-Value2', {'instance': NotifyIFTTT}), ('ifttt://WebHookID@EventID/EventID2/', {'instance': NotifyIFTTT}), ('https://maker.ifttt.com/use/WebHookID/', {'instance': TypeError}), ('https://maker.ifttt.com/use/WebHookID/EventID/', {'instance': NotifyIFTTT}), ('https://maker.ifttt.com/use/WebHookID/EventID/?-Value1=', {'instance': NotifyIFTTT}), ('ifttt://WebHookID@EventID', {'instance': NotifyIFTTT, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('ifttt://WebHookID@EventID', {'instance': NotifyIFTTT, 'response': False, 'requests_response_code': 999}), ('ifttt://WebHookID@EventID', {'instance': NotifyIFTTT, 'test_requests_exceptions': True}))

def test_plugin_ifttt_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyIFTTT() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_ifttt_edge_cases(mock_post, mock_get):
    if False:
        return 10
    '\n    NotifyIFTTT() Edge Cases\n\n    '
    webhook_id = 'webhook_id'
    events = ['event1', 'event2']
    mock_get.return_value = requests.Request()
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_get.return_value.status_code = requests.codes.ok
    mock_get.return_value.content = '{}'
    mock_post.return_value.content = '{}'
    with pytest.raises(TypeError):
        NotifyIFTTT(webhook_id=None, events=None)
    with pytest.raises(TypeError):
        NotifyIFTTT(webhook_id=None, events=events)
    with pytest.raises(TypeError):
        NotifyIFTTT(webhook_id='   ', events=events)
    with pytest.raises(TypeError):
        NotifyIFTTT(webhook_id=webhook_id, events=None)
    obj = NotifyIFTTT(webhook_id=webhook_id, events=events)
    assert isinstance(obj, NotifyIFTTT) is True
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    obj = NotifyIFTTT(webhook_id=webhook_id, events=events, add_tokens={'Test': 'ValueA', 'Test2': 'ValueB'})
    assert isinstance(obj, NotifyIFTTT) is True
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    with pytest.raises(TypeError):
        NotifyIFTTT(webhook_id=webhook_id, events=events, del_tokens=NotifyIFTTT.ifttt_default_title_key)
    assert isinstance(obj, NotifyIFTTT) is True
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    obj = NotifyIFTTT(webhook_id=webhook_id, events=events, add_tokens={'MyKey': 'MyValue'}, del_tokens=(NotifyIFTTT.ifttt_default_title_key, NotifyIFTTT.ifttt_default_body_key, NotifyIFTTT.ifttt_default_type_key))
    assert isinstance(obj, NotifyIFTTT) is True
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    obj = NotifyIFTTT(webhook_id=webhook_id, events=events, add_tokens={'MyKey': 'MyValue'}, del_tokens={NotifyIFTTT.ifttt_default_title_key: None, NotifyIFTTT.ifttt_default_body_key: None, NotifyIFTTT.ifttt_default_type_key: None})
    assert isinstance(obj, NotifyIFTTT) is True