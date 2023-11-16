from unittest import mock
import json
import requests
import pytest
from apprise import Apprise
from apprise import AppriseConfig
from apprise import NotifyType
from apprise.plugins.NotifyMSTeams import NotifyMSTeams
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
UUID4 = '8b799edf-6f98-4d3a-9be7-2862fb4e5752'
apprise_url_tests = (('msteams://', {'instance': TypeError}), ('msteams://:@/', {'instance': TypeError}), ('msteams://{}'.format(UUID4), {'instance': TypeError}), ('msteams://{}@{}/'.format(UUID4, UUID4), {'instance': TypeError}), ('msteams://{}@{}/{}'.format(UUID4, UUID4, 'a' * 32), {'instance': TypeError}), ('msteams://{}@{}/{}/{}?t1'.format(UUID4, UUID4, 'b' * 32, UUID4), {'instance': NotifyMSTeams}), ('https://outlook.office.com/webhook/{}@{}/IncomingWebhook/{}/{}'.format(UUID4, UUID4, 'k' * 32, UUID4), {'instance': NotifyMSTeams, 'privacy_url': 'msteams://8...2/k...k/8...2/'}), ('https://myteam.webhook.office.com/webhookb2/{}@{}/IncomingWebhook/{}/{}'.format(UUID4, UUID4, 'm' * 32, UUID4), {'instance': NotifyMSTeams, 'privacy_url': 'msteams://myteam/8...2/m...m/8...2/'}), ('msteams://{}@{}/{}/{}?t2'.format(UUID4, UUID4, 'c' * 32, UUID4), {'instance': NotifyMSTeams, 'include_image': False}), ('msteams://{}@{}/{}/{}?image=No'.format(UUID4, UUID4, 'd' * 32, UUID4), {'instance': NotifyMSTeams, 'privacy_url': 'msteams://8...2/d...d/8...2/'}), ('msteams://apprise/{}@{}/{}/{}'.format(UUID4, UUID4, 'e' * 32, UUID4), {'instance': NotifyMSTeams, 'privacy_url': 'msteams://apprise/8...2/e...e/8...2/'}), ('msteams://{}@{}/{}/{}?team=teamname'.format(UUID4, UUID4, 'f' * 32, UUID4), {'instance': NotifyMSTeams, 'privacy_url': 'msteams://teamname/8...2/f...f/8...2/'}), ('msteams://apprise/{}@{}/{}/{}?version=1'.format(UUID4, UUID4, 'e' * 32, UUID4), {'instance': NotifyMSTeams, 'privacy_url': 'msteams://8...2/e...e/8...2/'}), ('msteams://apprise/{}@{}/{}/{}?version=999'.format(UUID4, UUID4, 'e' * 32, UUID4), {'instance': TypeError}), ('msteams://apprise/{}@{}/{}/{}?version=invalid'.format(UUID4, UUID4, 'e' * 32, UUID4), {'instance': TypeError}), ('msteams://{}@{}/{}/{}?tx'.format(UUID4, UUID4, 'x' * 32, UUID4), {'instance': NotifyMSTeams, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('msteams://{}@{}/{}/{}?ty'.format(UUID4, UUID4, 'y' * 32, UUID4), {'instance': NotifyMSTeams, 'response': False, 'requests_response_code': 999}), ('msteams://{}@{}/{}/{}?tz'.format(UUID4, UUID4, 'z' * 32, UUID4), {'instance': NotifyMSTeams, 'test_requests_exceptions': True}))

def test_plugin_msteams_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyMSTeams() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@pytest.fixture
def msteams_url():
    if False:
        for i in range(10):
            print('nop')
    return 'msteams://{}@{}/{}/{}'.format(UUID4, UUID4, 'a' * 32, UUID4)

@pytest.fixture
def request_mock(mocker):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prepare requests mock.\n    '
    mock_post = mocker.patch('requests.post')
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    return mock_post

@pytest.fixture
def simple_template(tmpdir):
    if False:
        print('Hello World!')
    template = tmpdir.join('simple.json')
    template.write('\n    {\n      "@type": "MessageCard",\n      "@context": "https://schema.org/extensions",\n      "summary": "{{name}}",\n      "themeColor": "{{app_color}}",\n      "sections": [\n        {\n          "activityImage": null,\n          "activityTitle": "{{title}}",\n          "text": "{{body}}"\n        }\n      ]\n    }\n    ')
    return template

def test_plugin_msteams_templating_basic_success(request_mock, msteams_url, tmpdir):
    if False:
        while True:
            i = 10
    '\n    NotifyMSTeams() Templating - success.\n    Test cases where URL and JSON is valid.\n    '
    template = tmpdir.join('simple.json')
    template.write('\n    {\n      "@type": "MessageCard",\n      "@context": "https://schema.org/extensions",\n      "summary": "{{app_id}}",\n      "themeColor": "{{app_color}}",\n      "sections": [\n        {\n          "activityImage": null,\n          "activityTitle": "{{app_title}}",\n          "text": "{{app_body}}"\n        }\n      ]\n    }\n    ')
    obj = Apprise.instantiate('{url}/?template={template}&{kwargs}'.format(url=msteams_url, template=str(template), kwargs=':key1=token&:key2=token'))
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert request_mock.called is True
    assert request_mock.call_args_list[0][0][0].startswith('https://outlook.office.com/webhook/')
    posted_json = json.loads(request_mock.call_args_list[0][1]['data'])
    assert 'summary' in posted_json
    assert posted_json['summary'] == 'Apprise'
    assert posted_json['themeColor'] == '#3AA3E3'
    assert posted_json['sections'][0]['activityTitle'] == 'title'
    assert posted_json['sections'][0]['text'] == 'body'

def test_plugin_msteams_templating_invalid_json(request_mock, msteams_url, tmpdir):
    if False:
        while True:
            i = 10
    '\n    NotifyMSTeams() Templating - invalid JSON.\n    '
    template = tmpdir.join('invalid.json')
    template.write('}')
    obj = Apprise.instantiate('{url}/?template={template}&{kwargs}'.format(url=msteams_url, template=str(template), kwargs=':key1=token&:key2=token'))
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False

def test_plugin_msteams_templating_json_missing_type(request_mock, msteams_url, tmpdir):
    if False:
        i = 10
        return i + 15
    "\n    NotifyMSTeams() Templating - invalid JSON.\n    Test case where we're missing the @type part of the URL.\n    "
    template = tmpdir.join('missing_type.json')
    template.write('\n    {\n      "@context": "https://schema.org/extensions",\n      "summary": "{{app_id}}",\n      "themeColor": "{{app_color}}",\n      "sections": [\n        {\n          "activityImage": null,\n          "activityTitle": "{{app_title}}",\n          "text": "{{app_body}}"\n        }\n      ]\n    }\n    ')
    obj = Apprise.instantiate('{url}/?template={template}&{kwargs}'.format(url=msteams_url, template=str(template), kwargs=':key1=token&:key2=token'))
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False

def test_plugin_msteams_templating_json_missing_context(request_mock, msteams_url, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    "\n    NotifyMSTeams() Templating - invalid JSON.\n    Test cases where we're missing the @context part of the URL.\n    "
    template = tmpdir.join('missing_context.json')
    template.write('\n    {\n      "@type": "MessageCard",\n      "summary": "{{app_id}}",\n      "themeColor": "{{app_color}}",\n      "sections": [\n        {\n          "activityImage": null,\n          "activityTitle": "{{app_title}}",\n          "text": "{{app_body}}"\n        }\n      ]\n    }\n    ')
    obj = Apprise.instantiate('{url}/?template={template}&{kwargs}'.format(url=msteams_url, template=str(template), kwargs=':key1=token&:key2=token'))
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False

def test_plugin_msteams_templating_load_json_failure(request_mock, msteams_url, tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    NotifyMSTeams() Templating - template loading failure.\n    Test a case where we can not access the file.\n    '
    template = tmpdir.join('empty.json')
    template.write('')
    obj = Apprise.instantiate('{url}/?template={template}'.format(url=msteams_url, template=str(template)))
    with mock.patch('json.loads', side_effect=OSError):
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False

def test_plugin_msteams_templating_target_success(request_mock, msteams_url, tmpdir):
    if False:
        while True:
            i = 10
    '\n    NotifyMSTeams() Templating - success with target.\n    A more complicated example; uses a target.\n    '
    template = tmpdir.join('more_complicated_example.json')
    template.write('\n    {\n      "@type": "MessageCard",\n      "@context": "https://schema.org/extensions",\n      "summary": "{{app_desc}}",\n      "themeColor": "{{app_color}}",\n      "sections": [\n        {\n          "activityImage": null,\n          "activityTitle": "{{app_title}}",\n          "text": "{{app_body}}"\n        }\n      ],\n     "potentialAction": [{\n        "@type": "ActionCard",\n        "name": "Add a comment",\n        "inputs": [{\n            "@type": "TextInput",\n            "id": "comment",\n            "isMultiline": false,\n            "title": "Add a comment here for this task."\n        }],\n        "actions": [{\n            "@type": "HttpPOST",\n            "name": "Add Comment",\n            "target": "{{ target }}"\n        }]\n     }]\n    }\n    ')
    obj = Apprise.instantiate('{url}/?template={template}&{kwargs}'.format(url=msteams_url, template=str(template), kwargs=':key1=token&:key2=token&:target=http://localhost'))
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert request_mock.called is True
    assert request_mock.call_args_list[0][0][0].startswith('https://outlook.office.com/webhook/')
    posted_json = json.loads(request_mock.call_args_list[0][1]['data'])
    assert 'summary' in posted_json
    assert posted_json['summary'] == 'Apprise Notifications'
    assert posted_json['themeColor'] == '#3AA3E3'
    assert posted_json['sections'][0]['activityTitle'] == 'title'
    assert posted_json['sections'][0]['text'] == 'body'
    assert posted_json['potentialAction'][0]['actions'][0]['target'] == 'http://localhost'

def test_msteams_yaml_config_invalid_template_filename(request_mock, msteams_url, simple_template, tmpdir):
    if False:
        while True:
            i = 10
    '\n    NotifyMSTeams() YAML Configuration Entries - invalid template filename.\n    '
    config = tmpdir.join('msteams01.yml')
    config.write("\n    urls:\n      - {url}:\n        - tag: 'msteams'\n          template:  {template}.missing\n          :name: 'Template.Missing'\n          :body: 'test body'\n          :title: 'test title'\n    ".format(url=msteams_url, template=str(simple_template)))
    cfg = AppriseConfig()
    cfg.add(str(config))
    assert len(cfg) == 1
    assert len(cfg[0]) == 1
    obj = cfg[0][0]
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert request_mock.called is False

def test_msteams_yaml_config_token_identifiers(request_mock, msteams_url, simple_template, tmpdir):
    if False:
        return 10
    '\n    NotifyMSTeams() YAML Configuration Entries - test token identifiers.\n    '
    config = tmpdir.join('msteams01.yml')
    config.write("\n    urls:\n      - {url}:\n        - tag: 'msteams'\n          template:  {template}\n          :name: 'Testing'\n          :body: 'test body'\n          :title: 'test title'\n    ".format(url=msteams_url, template=str(simple_template)))
    cfg = AppriseConfig()
    cfg.add(str(config))
    assert len(cfg) == 1
    assert len(cfg[0]) == 1
    obj = cfg[0][0]
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert request_mock.called is True
    assert request_mock.call_args_list[0][0][0].startswith('https://outlook.office.com/webhook/')
    posted_json = json.loads(request_mock.call_args_list[0][1]['data'])
    assert 'summary' in posted_json
    assert posted_json['summary'] == 'Testing'
    assert posted_json['themeColor'] == '#3AA3E3'
    assert posted_json['sections'][0]['activityTitle'] == 'test title'
    assert posted_json['sections'][0]['text'] == 'test body'

def test_msteams_yaml_config_no_bullet_under_url_1(request_mock, msteams_url, simple_template, tmpdir):
    if False:
        return 10
    '\n    NotifyMSTeams() YAML Configuration Entries - no bullet 1.\n    Now again but without a bullet under the url definition.\n    '
    config = tmpdir.join('msteams02.yml')
    config.write("\n    urls:\n      - {url}:\n          tag: 'msteams'\n          template:  {template}\n          :name: 'Testing2'\n          :body: 'test body2'\n          :title: 'test title2'\n    ".format(url=msteams_url, template=str(simple_template)))
    cfg = AppriseConfig()
    cfg.add(str(config))
    assert len(cfg) == 1
    assert len(cfg[0]) == 1
    obj = cfg[0][0]
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert request_mock.called is True
    assert request_mock.call_args_list[0][0][0].startswith('https://outlook.office.com/webhook/')
    posted_json = json.loads(request_mock.call_args_list[0][1]['data'])
    assert 'summary' in posted_json
    assert posted_json['summary'] == 'Testing2'
    assert posted_json['themeColor'] == '#3AA3E3'
    assert posted_json['sections'][0]['activityTitle'] == 'test title2'
    assert posted_json['sections'][0]['text'] == 'test body2'

def test_msteams_yaml_config_dictionary_file(request_mock, msteams_url, simple_template, tmpdir):
    if False:
        while True:
            i = 10
    '\n    NotifyMSTeams() YAML Configuration Entries.\n    Try again but store the content as a dictionary in the configuration file.\n    '
    config = tmpdir.join('msteams03.yml')
    config.write("\n    urls:\n      - {url}:\n        - tag: 'msteams'\n          template:  {template}\n          tokens:\n            name: 'Testing3'\n            body: 'test body3'\n            title: 'test title3'\n    ".format(url=msteams_url, template=str(simple_template)))
    cfg = AppriseConfig()
    cfg.add(str(config))
    assert len(cfg) == 1
    assert len(cfg[0]) == 1
    obj = cfg[0][0]
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert request_mock.called is True
    assert request_mock.call_args_list[0][0][0].startswith('https://outlook.office.com/webhook/')
    posted_json = json.loads(request_mock.call_args_list[0][1]['data'])
    assert 'summary' in posted_json
    assert posted_json['summary'] == 'Testing3'
    assert posted_json['themeColor'] == '#3AA3E3'
    assert posted_json['sections'][0]['activityTitle'] == 'test title3'
    assert posted_json['sections'][0]['text'] == 'test body3'

def test_msteams_yaml_config_no_bullet_under_url_2(request_mock, msteams_url, simple_template, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyMSTeams() YAML Configuration Entries - no bullet 2.\n    Now again but without a bullet under the url definition.\n    '
    config = tmpdir.join('msteams04.yml')
    config.write("\n    urls:\n      - {url}:\n          tag: 'msteams'\n          template:  {template}\n          tokens:\n            name: 'Testing4'\n            body: 'test body4'\n            title: 'test title4'\n    ".format(url=msteams_url, template=str(simple_template)))
    cfg = AppriseConfig()
    cfg.add(str(config))
    assert len(cfg) == 1
    assert len(cfg[0]) == 1
    obj = cfg[0][0]
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert request_mock.called is True
    assert request_mock.call_args_list[0][0][0].startswith('https://outlook.office.com/webhook/')
    posted_json = json.loads(request_mock.call_args_list[0][1]['data'])
    assert 'summary' in posted_json
    assert posted_json['summary'] == 'Testing4'
    assert posted_json['themeColor'] == '#3AA3E3'
    assert posted_json['sections'][0]['activityTitle'] == 'test title4'
    assert posted_json['sections'][0]['text'] == 'test body4'

def test_msteams_yaml_config_combined(request_mock, msteams_url, simple_template, tmpdir):
    if False:
        print('Hello World!')
    "\n    NotifyMSTeams() YAML Configuration Entries.\n    Now let's do a combination of the two.\n    "
    config = tmpdir.join('msteams05.yml')
    config.write("\n    urls:\n      - {url}:\n        - tag: 'msteams'\n          template:  {template}\n          tokens:\n              body: 'test body5'\n              title: 'test title5'\n          :name: 'Testing5'\n    ".format(url=msteams_url, template=str(simple_template)))
    cfg = AppriseConfig()
    cfg.add(str(config))
    assert len(cfg) == 1
    assert len(cfg[0]) == 1
    obj = cfg[0][0]
    assert isinstance(obj, NotifyMSTeams)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is True
    assert request_mock.called is True
    assert request_mock.call_args_list[0][0][0].startswith('https://outlook.office.com/webhook/')
    posted_json = json.loads(request_mock.call_args_list[0][1]['data'])
    assert 'summary' in posted_json
    assert posted_json['summary'] == 'Testing5'
    assert posted_json['themeColor'] == '#3AA3E3'
    assert posted_json['sections'][0]['activityTitle'] == 'test title5'
    assert posted_json['sections'][0]['text'] == 'test body5'

def test_msteams_yaml_config_token_mismatch(request_mock, msteams_url, simple_template, tmpdir):
    if False:
        return 10
    "\n    NotifyMSTeams() YAML Configuration Entries.\n    Now let's do a test where our tokens is not the\n    expected dictionary we want to see.\n    "
    config = tmpdir.join('msteams06.yml')
    config.write("\n    urls:\n      - {url}:\n        - tag: 'msteams'\n          template:  {template}\n          # Not a dictionary\n          tokens:\n            body\n    ".format(url=msteams_url, template=str(simple_template)))
    cfg = AppriseConfig()
    cfg.add(str(config))
    assert len(cfg) == 1
    assert len(cfg[0]) == 0

def test_plugin_msteams_edge_cases():
    if False:
        i = 10
        return i + 15
    '\n    NotifyMSTeams() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifyMSTeams(token_a=None, token_b='abcd', token_c='abcd')
    with pytest.raises(TypeError):
        NotifyMSTeams(token_a='  ', token_b='abcd', token_c='abcd')
    with pytest.raises(TypeError):
        NotifyMSTeams(token_a='abcd', token_b=None, token_c='abcd')
    with pytest.raises(TypeError):
        NotifyMSTeams(token_a='abcd', token_b='  ', token_c='abcd')
    with pytest.raises(TypeError):
        NotifyMSTeams(token_a='abcd', token_b='abcd', token_c=None)
    with pytest.raises(TypeError):
        NotifyMSTeams(token_a='abcd', token_b='abcd', token_c='  ')
    uuid4 = '8b799edf-6f98-4d3a-9be7-2862fb4e5752'
    token_a = '{}@{}'.format(uuid4, uuid4)
    token_b = 'A' * 32
    obj = NotifyMSTeams(token_a=token_a, token_b=token_b, token_c=uuid4)
    assert isinstance(obj, NotifyMSTeams)