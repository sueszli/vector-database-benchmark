import os
import pytest
import requests
from json import dumps
from json import loads
from unittest import mock
from apprise import Apprise
from apprise import AppriseAttachment
from apprise import AppriseAsset
from apprise import NotifyType
from apprise import NotifyFormat
from apprise.plugins.NotifyTelegram import NotifyTelegram
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('tgram://', {'instance': None}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/', {'instance': NotifyTelegram, 'include_image': False}), ('tgram://123456789:abcdefg_hijklmnop/id1/id2/', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/?to=id1,id2', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/%$/', {'instance': NotifyTelegram, 'response': False}), ('tgram://123456789:abcdefg_hijklmnop/id1/id2/23423/-30/', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/id1/id2/23423/-30/', {'instance': NotifyTelegram, 'include_image': False}), ('tgram://bottest@123456789:abcdefg_hijklmnop/lead2gold/', {'instance': NotifyTelegram}), ('tgram://bottest@123456789:abcdefg_hijklmnop/id1/?topic=12345', {'instance': NotifyTelegram}), ('tgram://bottest@123456789:abcdefg_hijklmnop/id1/?topic=invalid', {'instance': TypeError}), ('tgram://bottest@123456789:abcdefg_hijklmnop/id1/?content=invalid', {'instance': TypeError}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?image=Yes', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?format=invalid', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?format=', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?format=markdown', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?format=html', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?format=text', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?silent=yes', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?silent=no', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?preview=yes', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?preview=no', {'instance': NotifyTelegram}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/', {'instance': NotifyTelegram, 'include_image': False}), ('tgram://alpha:abcdefg_hijklmnop/lead2gold/', {'instance': None}), ('tgram://:@/', {'instance': None}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/', {'instance': NotifyTelegram, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?image=Yes', {'instance': NotifyTelegram, 'include_image': False, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('tgram://123456789:abcdefg_hijklmnop/id1/id2/', {'instance': NotifyTelegram, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('tgram://123456789:abcdefg_hijklmnop/id1/id2/', {'instance': NotifyTelegram, 'include_image': False, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/', {'instance': NotifyTelegram, 'response': False, 'requests_response_code': 999}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/', {'instance': NotifyTelegram, 'include_image': False, 'response': False, 'requests_response_code': 999}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?image=Yes', {'instance': NotifyTelegram, 'include_image': True, 'response': False, 'requests_response_code': 999}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/', {'instance': NotifyTelegram, 'test_requests_exceptions': True}), ('tgram://123456789:abcdefg_hijklmnop/lead2gold/?image=Yes', {'instance': NotifyTelegram, 'include_image': True, 'test_requests_exceptions': True}))

def test_plugin_telegram_urls():
    if False:
        print('Hello World!')
    '\n    NotifyTelegram() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_telegram_general(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyTelegram() General Tests\n\n    '
    bot_token = '123456789:abcdefg_hijklmnop'
    invalid_bot_token = 'abcd:123'
    chat_ids = 'l2g, lead2gold'
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.content = '{}'
    with pytest.raises(TypeError):
        NotifyTelegram(bot_token=None, targets=chat_ids)
    mock_post.return_value.content = '}'
    obj = NotifyTelegram(bot_token=bot_token, targets=None)
    obj.notify(title='hello', body='world')
    mock_post.return_value.status_code = requests.codes.internal_server_error
    obj = NotifyTelegram(bot_token=bot_token, targets=None)
    obj.notify(title='hello', body='world')
    mock_post.return_value.status_code = requests.codes.ok
    with pytest.raises(TypeError):
        NotifyTelegram(bot_token=invalid_bot_token, targets=chat_ids)
    obj = NotifyTelegram(bot_token=bot_token, targets=chat_ids, include_image=True)
    assert isinstance(obj, NotifyTelegram) is True
    assert len(obj.targets) == 2
    mock_post.side_effect = IOError()
    assert not obj.send_media(obj.targets[0], NotifyType.INFO)
    mock_post.side_effect = requests.HTTPError
    assert not obj.send_media(obj.targets[0], NotifyType.INFO)
    mock_post.side_effect = None
    mock_post.return_value.content = '{}'
    assert isinstance(obj.url(), str) is True
    assert isinstance(obj.url(privacy=True), str) is True
    assert obj.url(privacy=True).startswith('tgram://1...p/') is True
    obj = NotifyTelegram(**NotifyTelegram.parse_url(obj.url()))
    assert isinstance(obj, NotifyTelegram) is True
    response = mock.Mock()
    response.status_code = requests.codes.internal_server_error
    response.content = dumps({'description': 'test'})
    mock_post.return_value = response
    nimg_obj = NotifyTelegram(bot_token=bot_token, targets=chat_ids)
    nimg_obj.asset = AppriseAsset(image_path_mask=False, image_url_mask=False)
    assert obj.body_maxlen == NotifyTelegram.body_maxlen
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert nimg_obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    obj = NotifyTelegram(bot_token=bot_token, targets='l2g')
    nimg_obj = NotifyTelegram(bot_token=bot_token, targets='l2g')
    nimg_obj.asset = AppriseAsset(image_path_mask=False, image_url_mask=False)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    assert nimg_obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    mock_post.return_value.content = dumps({'ok': True, 'result': [{'update_id': 645421319}, {'update_id': 645421320, 'message': {'message_id': 2, 'chat': {'id': 532389719, 'first_name': 'Chris', 'type': 'private'}, 'date': 1519694394, 'text': '/start', 'entities': [{'offset': 0, 'length': 6, 'type': 'bot_command'}]}}, {'update_id': 645421321, 'message': {'message_id': 2, 'from': {'id': 532389719, 'is_bot': False, 'first_name': 'Chris', 'language_code': 'en-US'}, 'chat': {'id': 532389719, 'first_name': 'Chris', 'type': 'private'}, 'date': 1519694394, 'text': '/start', 'entities': [{'offset': 0, 'length': 6, 'type': 'bot_command'}]}}]})
    mock_post.return_value.status_code = requests.codes.ok
    obj = NotifyTelegram(bot_token=bot_token, targets='12345')
    assert len(obj.targets) == 1
    assert obj.targets[0] == '12345'
    mock_post.reset_mock()
    body = '<p>\'"This can\'t\t\r\nfail&nbsp;us"\'</p>'
    assert obj.notify(body=body, title='special characters', notify_type=NotifyType.INFO) is True
    assert mock_post.call_count == 1
    payload = loads(mock_post.call_args_list[0][1]['data'])
    assert payload['text'] == '<b>special characters</b>\r\n\'"This can\'t\t\r\nfail us"\'\r\n'
    for content in ('before', 'after'):
        obj = NotifyTelegram(bot_token=bot_token, targets='12345', content=content)
        mock_post.reset_mock()
        attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
        path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
        attach = AppriseAttachment(path)
        assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    obj = NotifyTelegram(bot_token=bot_token, targets=None)
    assert len(obj.targets) == 0
    assert obj.notify(title='hello', body='world') is True
    assert len(obj.targets) == 1
    assert obj.targets[0] == '532389719'
    mock_post.return_value.content = dumps({'ok': True, 'result': []})
    obj = NotifyTelegram(bot_token=bot_token, targets=None)
    assert len(obj.targets) == 0
    assert obj.notify(title='hello', body='world') is False
    assert len(obj.targets) == 0
    mock_post.return_value.content = dumps({'ok': False, 'result': [{'update_id': 645421321, 'message': {'message_id': 2, 'from': {'id': 532389719, 'is_bot': False, 'first_name': 'Chris', 'language_code': 'en-US'}, 'chat': {'id': 532389719, 'first_name': 'Chris', 'type': 'private'}, 'date': 1519694394, 'text': '/start', 'entities': [{'offset': 0, 'length': 6, 'type': 'bot_command'}]}}]})
    obj = NotifyTelegram(bot_token=bot_token, targets=None)
    assert len(obj.targets) == 0
    assert obj.notify(title='hello', body='world') is False
    assert len(obj.targets) == 0
    mock_post.return_value.content = dumps({'ok': True})
    obj = NotifyTelegram(bot_token=bot_token, targets=None)
    assert len(obj.targets) == 0
    assert obj.notify(title='hello', body='world') is False
    assert len(obj.targets) == 0
    mock_post.return_value.content = dumps({})
    obj.detect_bot_owner()
    mock_post.return_value.status_code = requests.codes.internal_server_error
    obj = NotifyTelegram(bot_token=bot_token, targets=None)
    assert len(obj.targets) == 0
    assert obj.notify(title='hello', body='world') is False
    assert len(obj.targets) == 0
    mock_post.return_value.status_code = 999
    NotifyTelegram(bot_token=bot_token, targets=None)
    assert len(obj.targets) == 0
    assert obj.notify(title='hello', body='world') is False
    assert len(obj.targets) == 0
    mock_post.return_value.content = dumps({'description': 'Failure Message'})
    NotifyTelegram(bot_token=bot_token, targets=None)
    assert len(obj.targets) == 0
    assert obj.notify(title='hello', body='world') is False
    assert len(obj.targets) == 0
    obj = NotifyTelegram(bot_token=bot_token, targets=['@abcd'])
    assert nimg_obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    mock_post.side_effect = requests.HTTPError
    obj = NotifyTelegram(bot_token=bot_token, targets=None)
    assert len(obj.targets) == 0
    assert obj.notify(title='hello', body='world') is False
    assert len(obj.targets) == 0
    obj = Apprise.instantiate('tgram://123456789:ABCdefghijkl123456789opqyz/-123456789525')
    assert isinstance(obj, NotifyTelegram)
    assert len(obj.targets) == 1
    assert '-123456789525' in obj.targets

@mock.patch('requests.post')
def test_plugin_telegram_formatting(mock_post):
    if False:
        return 10
    '\n    NotifyTelegram() formatting tests\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.content = '{}'
    mock_post.return_value.content = dumps({'ok': True, 'result': [{'update_id': 645421321, 'message': {'message_id': 2, 'from': {'id': 532389719, 'is_bot': False, 'first_name': 'Chris', 'language_code': 'en-US'}, 'chat': {'id': 532389719, 'first_name': 'Chris', 'type': 'private'}, 'date': 1519694394, 'text': '/start', 'entities': [{'offset': 0, 'length': 6, 'type': 'bot_command'}]}}]})
    mock_post.return_value.status_code = requests.codes.ok
    results = NotifyTelegram.parse_url('tgram://123456789:abcdefg_hijklmnop/')
    instance = NotifyTelegram(**results)
    assert isinstance(instance, NotifyTelegram)
    response = instance.send(title='title', body='body')
    assert response is True
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://api.telegram.org/bot123456789:abcdefg_hijklmnop/getUpdates'
    assert mock_post.call_args_list[1][0][0] == 'https://api.telegram.org/bot123456789:abcdefg_hijklmnop/sendMessage'
    mock_post.reset_mock()
    aobj = Apprise()
    aobj.add('tgram://123456789:abcdefg_hijklmnop/')
    assert len(aobj) == 1
    title = '🚨 Change detected for <i>Apprise Test Title</i>'
    body = '<a href="http://localhost"><i>Apprise Body Title</i></a> had <a href="http://127.0.0.1">a change</a>'
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.TEXT)
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://api.telegram.org/bot123456789:abcdefg_hijklmnop/getUpdates'
    assert mock_post.call_args_list[1][0][0] == 'https://api.telegram.org/bot123456789:abcdefg_hijklmnop/sendMessage'
    payload = loads(mock_post.call_args_list[1][1]['data'])
    assert payload['text'] == '<b>🚨 Change detected for &lt;i&gt;Apprise Test Title&lt;/i&gt;</b>\r\n&lt;a href="http://localhost"&gt;&lt;i&gt;Apprise Body Title&lt;/i&gt;&lt;/a&gt; had &lt;a href="http://127.0.0.1"&gt;a change&lt;/a&gt;'
    mock_post.reset_mock()
    aobj = Apprise()
    aobj.add('tgram://123456789:abcdefg_hijklmnop/?format=html')
    assert len(aobj) == 1
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.HTML)
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://api.telegram.org/bot123456789:abcdefg_hijklmnop/getUpdates'
    assert mock_post.call_args_list[1][0][0] == 'https://api.telegram.org/bot123456789:abcdefg_hijklmnop/sendMessage'
    payload = loads(mock_post.call_args_list[1][1]['data'])
    assert payload['text'] == '<b>🚨 Change detected for <i>Apprise Test Title</i></b>\r\n<a href="http://localhost"><i>Apprise Body Title</i></a> had <a href="http://127.0.0.1">a change</a>'
    mock_post.reset_mock()
    title = '# 🚨 Change detected for _Apprise Test Title_'
    body = '_[Apprise Body Title](http://localhost)_ had [a change](http://127.0.0.1)'
    aobj = Apprise()
    aobj.add('tgram://123456789:abcdefg_hijklmnop/?format=markdown')
    assert len(aobj) == 1
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.MARKDOWN)
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://api.telegram.org/bot123456789:abcdefg_hijklmnop/getUpdates'
    assert mock_post.call_args_list[1][0][0] == 'https://api.telegram.org/bot123456789:abcdefg_hijklmnop/sendMessage'
    payload = loads(mock_post.call_args_list[1][1]['data'])
    assert payload['text'] == '# 🚨 Change detected for _Apprise Test Title_\r\n_[Apprise Body Title](http://localhost)_ had [a change](http://127.0.0.1)'
    mock_post.reset_mock()
    aobj = Apprise()
    aobj.add('tgram://987654321:abcdefg_hijklmnop/?format=html')
    assert len(aobj) == 1
    title = '# 🚨 Another Change detected for _Apprise Test Title_'
    body = '_[Apprise Body Title](http://localhost)_ had [a change](http://127.0.0.2)'
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.MARKDOWN)
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://api.telegram.org/bot987654321:abcdefg_hijklmnop/getUpdates'
    assert mock_post.call_args_list[1][0][0] == 'https://api.telegram.org/bot987654321:abcdefg_hijklmnop/sendMessage'
    payload = loads(mock_post.call_args_list[1][1]['data'])
    assert payload['text'] == '<b>\r\n<b>🚨 Another Change detected for <i>Apprise Test Title</i></b>\r\n</b>\r\n<i><a href="http://localhost">Apprise Body Title</a></i> had <a href="http://127.0.0.2">a change</a>\r\n'
    mock_post.reset_mock()
    aobj = Apprise()
    aobj.add('tgram://987654321:abcdefg_hijklmnop/?format=markdown')
    assert len(aobj) == 1
    title = '# '
    body = '_[Apprise Body Title](http://localhost)_ had [a change](http://127.0.0.2)'
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.TEXT)
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://api.telegram.org/bot987654321:abcdefg_hijklmnop/getUpdates'
    assert mock_post.call_args_list[1][0][0] == 'https://api.telegram.org/bot987654321:abcdefg_hijklmnop/sendMessage'
    payload = loads(mock_post.call_args_list[1][1]['data'])
    assert payload['text'] == '_[Apprise Body Title](http://localhost)_ had [a change](http://127.0.0.2)'
    mock_post.reset_mock()
    aobj = Apprise()
    aobj.add('tgram://987654321:abcdefg_hijklmnop/?format=markdown')
    assert len(aobj) == 1
    title = '# A Great Title'
    body = '_[Apprise Body Title](http://localhost)_ had [a change](http://127.0.0.2)'
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.TEXT)
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://api.telegram.org/bot987654321:abcdefg_hijklmnop/getUpdates'
    assert mock_post.call_args_list[1][0][0] == 'https://api.telegram.org/bot987654321:abcdefg_hijklmnop/sendMessage'
    payload = loads(mock_post.call_args_list[1][1]['data'])
    assert payload['text'] == '# A Great Title\r\n_[Apprise Body Title](http://localhost)_ had [a change](http://127.0.0.2)'
    mock_post.reset_mock()
    title = 'Test Message Title'
    body = 'Test Message Body <br/> ok</br>'
    aobj = Apprise()
    aobj.add('tgram://1234:aaaaaaaaa/-1123456245134')
    assert len(aobj) == 1
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.MARKDOWN)
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://api.telegram.org/bot1234:aaaaaaaaa/sendMessage'
    payload = loads(mock_post.call_args_list[0][1]['data'])
    assert payload['text'] == '<b>Test Message Title\r\n</b>\r\nTest Message Body\r\nok\r\n'
    mock_post.reset_mock()
    aobj = Apprise()
    aobj.add('tgram://1234:aaaaaaaaa/-1123456245134')
    assert len(aobj) == 1
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.TEXT)
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://api.telegram.org/bot1234:aaaaaaaaa/sendMessage'
    payload = loads(mock_post.call_args_list[0][1]['data'])
    assert payload['text'] == '<b>Test Message Title</b>\r\nTest Message Body &lt;br/&gt; ok&lt;/br&gt;'
    mock_post.reset_mock()
    aobj = Apprise()
    aobj.add('tgram://1234:aaaaaaaaa/-1123456245134')
    assert len(aobj) == 1
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.HTML)
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://api.telegram.org/bot1234:aaaaaaaaa/sendMessage'
    payload = loads(mock_post.call_args_list[0][1]['data'])
    assert payload['text'] == '<b>Test Message Title</b>\r\nTest Message Body\r\nok\r\n'

@mock.patch('requests.post')
def test_plugin_telegram_html_formatting(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifyTelegram() HTML Formatting\n\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    mock_post.return_value.content = '{}'
    mock_post.return_value.content = dumps({'ok': True, 'result': [{'update_id': 645421321, 'message': {'message_id': 2, 'from': {'id': 532389719, 'is_bot': False, 'first_name': 'Chris', 'language_code': 'en-US'}, 'chat': {'id': 532389719, 'first_name': 'Chris', 'type': 'private'}, 'date': 1519694394, 'text': '/start', 'entities': [{'offset': 0, 'length': 6, 'type': 'bot_command'}]}}]})
    mock_post.return_value.status_code = requests.codes.ok
    aobj = Apprise()
    aobj.add('tgram://123456789:abcdefg_hijklmnop/')
    assert len(aobj) == 1
    assert isinstance(aobj[0], NotifyTelegram)
    title = '<title>&apos;information&apos</title>'
    body = '<em>&quot;This is in Italic&quot</em><br/><h5>&emsp;&emspHeadings&nbsp;are dropped and&nbspconverted to bold</h5>'
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.HTML)
    assert mock_post.call_count == 2
    payload = loads(mock_post.call_args_list[1][1]['data'])
    assert payload['text'] == '<b>\r\n<b>\'information\'</b>\r\n</b>\r\n<i>"This is in Italic"</i>\r\n<b>      Headings are dropped and converted to bold</b>\r\n'
    mock_post.reset_mock()
    assert aobj.notify(title=title, body=body, body_format=NotifyFormat.TEXT)
    assert mock_post.call_count == 1
    payload = loads(mock_post.call_args_list[0][1]['data'])
    assert payload['text'] == '<b>&lt;title&gt;&amp;apos;information&amp;apos&lt;/title&gt;</b>\r\n&lt;em&gt;&amp;quot;This is in Italic&amp;quot&lt;/em&gt;&lt;br/&gt;&lt;h5&gt;&amp;emsp;&amp;emspHeadings&amp;nbsp;are dropped and&amp;nbspconverted to bold&lt;/h5&gt;'
    mock_post.reset_mock()
    test_file_01 = os.path.join(TEST_VAR_DIR, '01_test_example.html')
    with open(test_file_01) as html_file:
        assert aobj.notify(body=html_file.read(), body_format=NotifyFormat.HTML)
    assert mock_post.call_count == 1
    payload = loads(mock_post.call_args_list[0][1]['data'])
    assert payload['text'] == '\r\n<b>Bootstrap 101 Template</b>\r\n<b>My Title</b>\r\n<b>Heading 1</b>\r\n-Bullet 1\r\n-Bullet 2\r\n-Bullet 3\r\n-Bullet 1\r\n-Bullet 2\r\n-Bullet 3\r\n<b>Heading 2</b>\r\nA div entry\r\nA div entry\r\n<code>A pre entry</code>\r\n<b>Heading 3</b>\r\n<b>Heading 4</b>\r\n<b>Heading 5</b>\r\n<b>Heading 6</b>\r\nA set of text\r\nAnother line after the set of text\r\nMore text\r\nlabel'