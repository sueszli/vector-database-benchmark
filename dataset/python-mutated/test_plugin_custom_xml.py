import os
import re
from unittest import mock
import requests
from apprise import Apprise
from apprise import AppriseAttachment
from apprise import NotifyType
from apprise.plugins.NotifyXML import NotifyXML
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('xml://:@/', {'instance': None}), ('xml://', {'instance': None}), ('xmls://', {'instance': None}), ('xml://localhost', {'instance': NotifyXML}), ('xml://user@localhost', {'instance': NotifyXML}), ('xml://user@localhost?method=invalid', {'instance': TypeError}), ('xml://user:pass@localhost', {'instance': NotifyXML, 'privacy_url': 'xml://user:****@localhost'}), ('xml://user@localhost?method=put', {'instance': NotifyXML}), ('xml://user@localhost?method=get', {'instance': NotifyXML}), ('xml://user@localhost?method=post', {'instance': NotifyXML}), ('xml://user@localhost?method=head', {'instance': NotifyXML}), ('xml://user@localhost?method=delete', {'instance': NotifyXML}), ('xml://user@localhost?method=patch', {'instance': NotifyXML}), ('xml://localhost:8080', {'instance': NotifyXML}), ('xml://user:pass@localhost:8080', {'instance': NotifyXML}), ('xmls://localhost', {'instance': NotifyXML}), ('xmls://user:pass@localhost', {'instance': NotifyXML}), ('xml://localhost:8080', {'instance': NotifyXML}), ('xml://user:pass@localhost:8080', {'instance': NotifyXML}), ('xml://localhost', {'instance': NotifyXML}), ('xmls://user:pass@localhost', {'instance': NotifyXML, 'privacy_url': 'xmls://user:****@localhost'}), ('xml://user@localhost:8080/path/', {'instance': NotifyXML, 'privacy_url': 'xml://user@localhost:8080/path'}), ('xmls://localhost:8080/path/', {'instance': NotifyXML, 'privacy_url': 'xmls://localhost:8080/path/'}), ('xmls://user:pass@localhost:8080', {'instance': NotifyXML}), ('xml://localhost:8080/path?-ParamA=Value', {'instance': NotifyXML}), ('xml://localhost:8080/path?+HeaderKey=HeaderValue', {'instance': NotifyXML}), ('xml://user:pass@localhost:8081', {'instance': NotifyXML, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('xml://user:pass@localhost:8082', {'instance': NotifyXML, 'response': False, 'requests_response_code': 999}), ('xml://user:pass@localhost:8083', {'instance': NotifyXML, 'test_requests_exceptions': True}))

def test_plugin_custom_xml_urls():
    if False:
        print('Hello World!')
    '\n    NotifyXML() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_notify_xml_plugin_attachments(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifyXML() Attachments\n\n    '
    okay_response = requests.Request()
    okay_response.status_code = requests.codes.ok
    okay_response.content = ''
    mock_post.return_value = okay_response
    obj = Apprise.instantiate('xml://localhost.localdomain/')
    assert isinstance(obj, NotifyXML)
    path = os.path.join(TEST_VAR_DIR, 'apprise-test.gif')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    path = (os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'), os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    attach = AppriseAttachment(path)
    mock_post.side_effect = None
    mock_post.return_value = okay_response
    with mock.patch('builtins.open', side_effect=OSError()):
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is False
    obj = Apprise.instantiate('xml://no-reply@example.com/')
    assert isinstance(obj, NotifyXML)
    mock_post.reset_mock()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 1

@mock.patch('requests.post')
@mock.patch('requests.get')
def test_plugin_custom_xml_edge_cases(mock_get, mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyXML() Edge Cases\n\n    '
    response = requests.Request()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    mock_get.return_value = response
    results = NotifyXML.parse_url('xml://localhost:8080/command?:Message=Body&method=GET&:Key=value&:,=invalid&:MessageType=')
    assert isinstance(results, dict)
    assert results['user'] is None
    assert results['password'] is None
    assert results['port'] == 8080
    assert results['host'] == 'localhost'
    assert results['fullpath'] == '/command'
    assert results['path'] == '/'
    assert results['query'] == 'command'
    assert results['schema'] == 'xml'
    assert results['url'] == 'xml://localhost:8080/command'
    assert isinstance(results['qsd:'], dict) is True
    assert results['qsd:']['Message'] == 'Body'
    assert results['qsd:']['Key'] == 'value'
    assert results['qsd:'][','] == 'invalid'
    instance = NotifyXML(**results)
    assert isinstance(instance, NotifyXML)
    assert instance.xsd_url is None
    response = instance.send(title='title', body='body')
    assert response is True
    assert mock_post.call_count == 0
    assert mock_get.call_count == 1
    details = mock_get.call_args_list[0]
    assert details[0][0] == 'http://localhost:8080/command'
    assert instance.url(privacy=False).startswith('xml://localhost:8080/command?')
    new_results = NotifyXML.parse_url(instance.url(safe=False))
    for k in ('user', 'password', 'port', 'host', 'fullpath', 'path', 'query', 'schema', 'url', 'method'):
        assert new_results[k] == results[k]
    assert re.search('<Version>[1-9]+\\.[0-9]+</Version>', details[1]['data'])
    assert re.search('<Subject>title</Subject>', details[1]['data'])
    assert re.search('<Message>test</Message>', details[1]['data']) is None
    assert re.search('<Message>', details[1]['data']) is None
    assert re.search('<MessageType>', details[1]['data']) is None
    assert re.search('<Body>body</Body>', details[1]['data'])
    assert re.search('<Key>value</Key>', details[1]['data'])
    mock_post.reset_mock()
    mock_get.reset_mock()
    results = NotifyXML.parse_url('xml://localhost:8081/command?method=POST&:New=Value')
    assert isinstance(results, dict)
    assert results['user'] is None
    assert results['password'] is None
    assert results['port'] == 8081
    assert results['host'] == 'localhost'
    assert results['fullpath'] == '/command'
    assert results['path'] == '/'
    assert results['query'] == 'command'
    assert results['schema'] == 'xml'
    assert results['url'] == 'xml://localhost:8081/command'
    assert isinstance(results['qsd:'], dict) is True
    assert results['qsd:']['New'] == 'Value'
    instance = NotifyXML(**results)
    assert isinstance(instance, NotifyXML)
    assert instance.xsd_url is None
    response = instance.send(title='title', body='body')
    assert response is True
    assert mock_post.call_count == 1
    assert mock_get.call_count == 0
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'http://localhost:8081/command'
    assert instance.url(privacy=False).startswith('xml://localhost:8081/command?')
    new_results = NotifyXML.parse_url(instance.url(safe=False))
    for k in ('user', 'password', 'port', 'host', 'fullpath', 'path', 'query', 'schema', 'url', 'method'):
        assert new_results[k] == results[k]
    assert re.search('<Version>[1-9]+\\.[0-9]+</Version>', details[1]['data'])
    assert re.search('<MessageType>info</MessageType>', details[1]['data'])
    assert re.search('<Subject>title</Subject>', details[1]['data'])
    assert re.search('<Message>body</Message>', details[1]['data'])
    mock_post.reset_mock()
    mock_get.reset_mock()
    results = NotifyXML.parse_url('xmls://localhost?method=POST&:Message=Body&:Subject=Title&:Version')
    assert isinstance(results, dict)
    assert results['user'] is None
    assert results['password'] is None
    assert results['port'] is None
    assert results['host'] == 'localhost'
    assert results['fullpath'] is None
    assert results['path'] is None
    assert results['query'] is None
    assert results['schema'] == 'xmls'
    assert results['url'] == 'xmls://localhost'
    assert isinstance(results['qsd:'], dict) is True
    assert results['qsd:']['Version'] == ''
    assert results['qsd:']['Message'] == 'Body'
    assert results['qsd:']['Subject'] == 'Title'
    instance = NotifyXML(**results)
    assert isinstance(instance, NotifyXML)
    assert instance.xsd_url is None
    response = instance.send(title='title', body='body')
    assert response is True
    assert mock_post.call_count == 1
    assert mock_get.call_count == 0
    details = mock_post.call_args_list[0]
    assert details[0][0] == 'https://localhost'
    assert instance.url(privacy=False).startswith('xmls://localhost')
    new_results = NotifyXML.parse_url(instance.url(safe=False))
    assert re.search('<Version>[1-9]+\\.[0-9]+</Version>', details[1]['data']) is None
    assert re.search('<MessageType>info</MessageType>', details[1]['data'])
    assert re.search('<Subject>title</Subject>', details[1]['data']) is None
    assert re.search('<Title>title</Title>', details[1]['data'])
    assert re.search('<Message>body</Message>', details[1]['data']) is None
    assert re.search('<Body>body</Body>', details[1]['data'])