import re
from collections import OrderedDict
from typing import Any, Optional, Union
from unittest import mock
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import responses
from django.test import override_settings
from django.utils.html import escape
from pyoembed.providers import get_provider
from requests.exceptions import ConnectionError
from typing_extensions import override
from zerver.actions.message_delete import do_delete_messages
from zerver.lib.cache import cache_delete, cache_get, preview_url_cache_key
from zerver.lib.camo import get_camo_url
from zerver.lib.queue import queue_json_publish
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import mock_queue_publish
from zerver.lib.url_preview.oembed import get_oembed_data, strip_cdata
from zerver.lib.url_preview.parsers import GenericParser, OpenGraphParser
from zerver.lib.url_preview.preview import get_link_embed_data
from zerver.lib.url_preview.types import UrlEmbedData, UrlOEmbedData
from zerver.models import Message, Realm, UserProfile
from zerver.worker.queue_processors import FetchLinksEmbedData

def reconstruct_url(url: str, maxwidth: int=640, maxheight: int=480) -> str:
    if False:
        print('Hello World!')
    provider = get_provider(str(url))
    oembed_url = provider.oembed_url(url)
    (scheme, netloc, path, query_string, fragment) = urlsplit(oembed_url)
    query_params = OrderedDict(parse_qsl(query_string))
    query_params['maxwidth'] = str(maxwidth)
    query_params['maxheight'] = str(maxheight)
    final_url = urlunsplit((scheme, netloc, path, urlencode(query_params, True), fragment))
    return final_url

@override_settings(INLINE_URL_EMBED_PREVIEW=True)
class OembedTestCase(ZulipTestCase):

    @responses.activate
    def test_present_provider(self) -> None:
        if False:
            print('Hello World!')
        response_data = {'type': 'rich', 'thumbnail_url': 'https://scontent.cdninstagram.com/t51.2885-15/n.jpg', 'thumbnail_width': 640, 'thumbnail_height': 426, 'title': 'NASA', 'html': '<p>test</p>', 'version': '1.0', 'width': 658, 'height': 400}
        url = 'http://instagram.com/p/BLtI2WdAymy'
        reconstructed_url = reconstruct_url(url)
        responses.add(responses.GET, reconstructed_url, json=response_data, status=200)
        data = get_oembed_data(url)
        assert data is not None
        self.assertIsInstance(data, UrlEmbedData)
        self.assertEqual(data.title, response_data['title'])

    @responses.activate
    def test_photo_provider(self) -> None:
        if False:
            while True:
                i = 10
        response_data = {'type': 'photo', 'thumbnail_url': 'https://scontent.cdninstagram.com/t51.2885-15/n.jpg', 'url': 'https://scontent.cdninstagram.com/t51.2885-15/n.jpg', 'thumbnail_width': 640, 'thumbnail_height': 426, 'title': 'NASA', 'html': '<p>test</p>', 'version': '1.0', 'width': 658, 'height': 400}
        url = 'http://imgur.com/photo/158727223'
        reconstructed_url = reconstruct_url(url)
        responses.add(responses.GET, reconstructed_url, json=response_data, status=200)
        data = get_oembed_data(url)
        assert data is not None
        self.assertIsInstance(data, UrlOEmbedData)
        self.assertEqual(data.title, response_data['title'])

    @responses.activate
    def test_video_provider(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        response_data = {'type': 'video', 'thumbnail_url': 'https://scontent.cdninstagram.com/t51.2885-15/n.jpg', 'thumbnail_width': 640, 'thumbnail_height': 426, 'title': 'NASA', 'html': '<p>test</p>', 'version': '1.0', 'width': 658, 'height': 400}
        url = 'http://blip.tv/video/158727223'
        reconstructed_url = reconstruct_url(url)
        responses.add(responses.GET, reconstructed_url, json=response_data, status=200)
        data = get_oembed_data(url)
        assert data is not None
        self.assertIsInstance(data, UrlOEmbedData)
        self.assertEqual(data.title, response_data['title'])

    @responses.activate
    def test_connect_error_request(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        url = 'http://instagram.com/p/BLtI2WdAymy'
        reconstructed_url = reconstruct_url(url)
        responses.add(responses.GET, reconstructed_url, body=ConnectionError())
        data = get_oembed_data(url)
        self.assertIsNone(data)

    @responses.activate
    def test_400_error_request(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        url = 'http://instagram.com/p/BLtI2WdAymy'
        reconstructed_url = reconstruct_url(url)
        responses.add(responses.GET, reconstructed_url, status=400)
        data = get_oembed_data(url)
        self.assertIsNone(data)

    @responses.activate
    def test_500_error_request(self) -> None:
        if False:
            return 10
        url = 'http://instagram.com/p/BLtI2WdAymy'
        reconstructed_url = reconstruct_url(url)
        responses.add(responses.GET, reconstructed_url, status=500)
        data = get_oembed_data(url)
        self.assertIsNone(data)

    @responses.activate
    def test_invalid_json_in_response(self) -> None:
        if False:
            i = 10
            return i + 15
        url = 'http://instagram.com/p/BLtI2WdAymy'
        reconstructed_url = reconstruct_url(url)
        responses.add(responses.GET, reconstructed_url, json='{invalid json}', status=200)
        data = get_oembed_data(url)
        self.assertIsNone(data)

    def test_oembed_html(self) -> None:
        if False:
            i = 10
            return i + 15
        html = '<iframe src="//www.instagram.com/embed.js"></iframe>'
        stripped_html = strip_cdata(html)
        self.assertEqual(html, stripped_html)

    def test_autodiscovered_oembed_xml_format_html(self) -> None:
        if False:
            while True:
                i = 10
        iframe_content = '<iframe src="https://w.soundcloud.com/player"></iframe>'
        html = f'<![CDATA[{iframe_content}]]>'
        stripped_html = strip_cdata(html)
        self.assertEqual(iframe_content, stripped_html)

class OpenGraphParserTestCase(ZulipTestCase):

    def test_page_with_og(self) -> None:
        if False:
            while True:
                i = 10
        html = b'<html>\n          <head>\n          <meta property="og:title" content="The Rock" />\n          <meta property="og:type" content="video.movie" />\n          <meta property="og:url" content="http://www.imdb.com/title/tt0117500/" />\n          <meta property="og:image" content="http://ia.media-imdb.com/images/rock.jpg" />\n          <meta property="og:description" content="The Rock film" />\n          </head>\n        </html>'
        parser = OpenGraphParser(html, 'text/html; charset=UTF-8')
        result = parser.extract_data()
        self.assertEqual(result.title, 'The Rock')
        self.assertEqual(result.description, 'The Rock film')

    def test_charset_in_header(self) -> None:
        if False:
            return 10
        html = '<html>\n          <head>\n            <meta property="og:title" content="中文" />\n          </head>\n        </html>'.encode('big5')
        parser = OpenGraphParser(html, 'text/html; charset=Big5')
        result = parser.extract_data()
        self.assertEqual(result.title, '中文')

    def test_charset_in_meta(self) -> None:
        if False:
            print('Hello World!')
        html = '<html>\n          <head>\n            <meta content-type="text/html; charset=Big5" />\n            <meta property="og:title" content="中文" />\n          </head>\n        </html>'.encode('big5')
        parser = OpenGraphParser(html, 'text/html')
        result = parser.extract_data()
        self.assertEqual(result.title, '中文')

class GenericParserTestCase(ZulipTestCase):

    def test_parser(self) -> None:
        if False:
            return 10
        html = b'\n          <html>\n            <head><title>Test title</title></head>\n            <body>\n                <h1>Main header</h1>\n                <p>Description text</p>\n            </body>\n          </html>\n        '
        parser = GenericParser(html, 'text/html; charset=UTF-8')
        result = parser.extract_data()
        self.assertEqual(result.title, 'Test title')
        self.assertEqual(result.description, 'Description text')

    def test_extract_image(self) -> None:
        if False:
            while True:
                i = 10
        html = b'\n          <html>\n            <body>\n                <h1>Main header</h1>\n                <img data-src="Not an image">\n                <img src="http://test.com/test.jpg">\n                <div>\n                    <p>Description text</p>\n                </div>\n            </body>\n          </html>\n        '
        parser = GenericParser(html, 'text/html; charset=UTF-8')
        result = parser.extract_data()
        self.assertEqual(result.title, 'Main header')
        self.assertEqual(result.description, 'Description text')
        self.assertEqual(result.image, 'http://test.com/test.jpg')

    def test_extract_bad_image(self) -> None:
        if False:
            i = 10
            return i + 15
        html = b'\n          <html>\n            <body>\n                <h1>Main header</h1>\n                <img data-src="Not an image">\n                <img src="http://[bad url/test.jpg">\n                <div>\n                    <p>Description text</p>\n                </div>\n            </body>\n          </html>\n        '
        parser = GenericParser(html, 'text/html; charset=UTF-8')
        result = parser.extract_data()
        self.assertEqual(result.title, 'Main header')
        self.assertEqual(result.description, 'Description text')
        self.assertIsNone(result.image)

    def test_extract_description(self) -> None:
        if False:
            i = 10
            return i + 15
        html = b'\n          <html>\n            <body>\n                <div>\n                    <div>\n                        <p>Description text</p>\n                    </div>\n                </div>\n            </body>\n          </html>\n        '
        parser = GenericParser(html, 'text/html; charset=UTF-8')
        result = parser.extract_data()
        self.assertEqual(result.description, 'Description text')
        html = b'\n          <html>\n            <head><meta name="description" content="description 123"</head>\n            <body></body>\n          </html>\n        '
        parser = GenericParser(html, 'text/html; charset=UTF-8')
        result = parser.extract_data()
        self.assertEqual(result.description, 'description 123')
        html = b'<html><body></body></html>'
        parser = GenericParser(html, 'text/html; charset=UTF-8')
        result = parser.extract_data()
        self.assertIsNone(result.description)

class PreviewTestCase(ZulipTestCase):
    open_graph_html = '\n          <html>\n            <head>\n                <title>Test title</title>\n                <meta property="og:title" content="The Rock" />\n                <meta property="og:type" content="video.movie" />\n                <meta property="og:url" content="http://www.imdb.com/title/tt0117500/" />\n                <meta property="og:image" content="http://ia.media-imdb.com/images/rock.jpg" />\n                <meta http-equiv="refresh" content="30" />\n                <meta property="notog:extra-text" content="Extra!" />\n            </head>\n            <body>\n                <h1>Main header</h1>\n                <p>Description text</p>\n            </body>\n          </html>\n        '

    @override
    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        Realm.objects.all().update(inline_url_embed_preview=True)

    @classmethod
    def create_mock_response(cls, url: str, status: int=200, relative_url: bool=False, content_type: str='text/html', body: Optional[Union[str, ConnectionError]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if body is None:
            body = cls.open_graph_html
        if relative_url is True and isinstance(body, str):
            body = body.replace('http://ia.media-imdb.com', '')
        responses.add(responses.GET, url, body=body, status=status, content_type=content_type)

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_edit_message_history(self) -> None:
        if False:
            i = 10
            return i + 15
        user = self.example_user('hamlet')
        self.login_user(user)
        msg_id = self.send_stream_message(user, 'Denmark', topic_name='editing', content='original')
        url = 'http://test.org/'
        self.create_mock_response(url)
        with mock_queue_publish('zerver.actions.message_edit.queue_json_publish') as patched:
            result = self.client_patch('/json/messages/' + str(msg_id), {'content': url})
            self.assert_json_success(result)
            patched.assert_called_once()
            queue = patched.call_args[0][0]
            self.assertEqual(queue, 'embed_links')
            event = patched.call_args[0][1]
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/: ' in info_logs.output[0])
        embedded_link = f'<a href="{url}" title="The Rock">The Rock</a>'
        msg = Message.objects.select_related('sender').get(id=msg_id)
        assert msg.rendered_content is not None
        self.assertIn(embedded_link, msg.rendered_content)

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def _send_message_with_test_org_url(self, sender: UserProfile, queue_should_run: bool=True, relative_url: bool=False) -> Message:
        if False:
            i = 10
            return i + 15
        url = 'http://test.org/'
        cache_delete(preview_url_cache_key(url))
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish') as patched:
            msg_id = self.send_personal_message(sender, self.example_user('cordelia'), content=url)
            if queue_should_run:
                patched.assert_called_once()
                queue = patched.call_args[0][0]
                self.assertEqual(queue, 'embed_links')
                event = patched.call_args[0][1]
            else:
                patched.assert_not_called()
                return Message.objects.select_related('sender').get(id=msg_id)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        assert msg.rendered_content is not None
        self.assertNotIn(f'<a href="{url}" title="The Rock">The Rock</a>', msg.rendered_content)
        self.create_mock_response(url, relative_url=relative_url)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/: ' in info_logs.output[0])
        msg = Message.objects.select_related('sender').get(id=msg_id)
        return msg

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_message_update_race_condition(self) -> None:
        if False:
            return 10
        user = self.example_user('hamlet')
        self.login_user(user)
        original_url = 'http://test.org/'
        edited_url = 'http://edited.org/'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish') as patched:
            msg_id = self.send_stream_message(user, 'Denmark', topic_name='foo', content=original_url)
            patched.assert_called_once()
            queue = patched.call_args[0][0]
            self.assertEqual(queue, 'embed_links')
            event = patched.call_args[0][1]

        def wrapped_queue_json_publish(*args: Any, **kwargs: Any) -> None:
            if False:
                print('Hello World!')
            self.create_mock_response(original_url)
            self.create_mock_response(edited_url)
            with self.settings(TEST_SUITE=False):
                with self.assertLogs(level='INFO') as info_logs:
                    FetchLinksEmbedData().consume(event)
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/: ' in info_logs.output[0])
            msg = Message.objects.select_related('sender').get(id=msg_id)
            assert msg.rendered_content is not None
            self.assertNotIn(f'<a href="{original_url}" title="The Rock">The Rock</a>', msg.rendered_content)
            self.assertTrue(responses.assert_call_count(edited_url, 0))
            with self.settings(TEST_SUITE=False):
                with self.assertLogs(level='INFO') as info_logs:
                    queue_json_publish(*args, **kwargs)
                    msg = Message.objects.select_related('sender').get(id=msg_id)
                    assert msg.rendered_content is not None
                    self.assertIn(f'<a href="{edited_url}" title="The Rock">The Rock</a>', msg.rendered_content)
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://edited.org/: ' in info_logs.output[0])
        with mock_queue_publish('zerver.actions.message_edit.queue_json_publish', wraps=wrapped_queue_json_publish):
            result = self.client_patch('/json/messages/' + str(msg_id), {'content': edited_url})
            self.assert_json_success(result)

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_message_deleted(self) -> None:
        if False:
            i = 10
            return i + 15
        user = self.example_user('hamlet')
        self.login_user(user)
        url = 'http://test.org/'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish') as patched:
            msg_id = self.send_stream_message(user, 'Denmark', topic_name='foo', content=url)
            patched.assert_called_once()
            queue = patched.call_args[0][0]
            self.assertEqual(queue, 'embed_links')
            event = patched.call_args[0][1]
        msg = Message.objects.select_related('sender').get(id=msg_id)
        do_delete_messages(msg.realm, [msg])
        self.create_mock_response(url)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
        self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/: ' in info_logs.output[0])

    def test_get_link_embed_data(self) -> None:
        if False:
            while True:
                i = 10
        url = 'http://test.org/'
        embedded_link = f'<a href="{url}" title="The Rock">The Rock</a>'
        msg = self._send_message_with_test_org_url(sender=self.example_user('hamlet'))
        self.assertIn(embedded_link, msg.rendered_content)
        msg = self._send_message_with_test_org_url(sender=self.example_user('webhook_bot'), queue_should_run=False)
        self.assertNotIn(embedded_link, msg.rendered_content)
        msg = self._send_message_with_test_org_url(sender=self.example_user('prospero'))
        self.assertIn(embedded_link, msg.rendered_content)

    @override_settings(CAMO_URI='')
    def test_inline_url_embed_preview(self) -> None:
        if False:
            i = 10
            return i + 15
        with_preview = '<p><a href="http://test.org/">http://test.org/</a></p>\n<div class="message_embed"><a class="message_embed_image" href="http://test.org/" style="background-image: url(http\\:\\/\\/ia\\.media-imdb\\.com\\/images\\/rock\\.jpg)"></a><div class="data-container"><div class="message_embed_title"><a href="http://test.org/" title="The Rock">The Rock</a></div><div class="message_embed_description">Description text</div></div></div>'
        without_preview = '<p><a href="http://test.org/">http://test.org/</a></p>'
        msg = self._send_message_with_test_org_url(sender=self.example_user('hamlet'))
        self.assertEqual(msg.rendered_content, with_preview)
        realm = msg.get_realm()
        realm.inline_url_embed_preview = False
        realm.save()
        msg = self._send_message_with_test_org_url(sender=self.example_user('prospero'), queue_should_run=False)
        self.assertEqual(msg.rendered_content, without_preview)

    def test_inline_url_embed_preview_with_camo(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        camo_url = re.sub('([^\\w-])', '\\\\\\1', get_camo_url('http://ia.media-imdb.com/images/rock.jpg'))
        with_preview = '<p><a href="http://test.org/">http://test.org/</a></p>\n<div class="message_embed"><a class="message_embed_image" href="http://test.org/" style="background-image: url(' + camo_url + ')"></a><div class="data-container"><div class="message_embed_title"><a href="http://test.org/" title="The Rock">The Rock</a></div><div class="message_embed_description">Description text</div></div></div>'
        msg = self._send_message_with_test_org_url(sender=self.example_user('hamlet'))
        self.assertEqual(msg.rendered_content, with_preview)

    @responses.activate
    @override_settings(CAMO_URI='')
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_link_preview_css_escaping_image(self) -> None:
        if False:
            print('Hello World!')
        user = self.example_user('hamlet')
        self.login_user(user)
        url = 'http://test.org/'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish') as patched:
            msg_id = self.send_stream_message(user, 'Denmark', topic_name='foo', content=url)
            patched.assert_called_once()
            queue = patched.call_args[0][0]
            self.assertEqual(queue, 'embed_links')
            event = patched.call_args[0][1]
        html = re.sub('rock\\.jpg', 'rock).jpg', self.open_graph_html)
        self.create_mock_response(url, body=html)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/: ' in info_logs.output[0])
        msg = Message.objects.select_related('sender').get(id=msg_id)
        with_preview = '<p><a href="http://test.org/">http://test.org/</a></p>\n<div class="message_embed"><a class="message_embed_image" href="http://test.org/" style="background-image: url(http\\:\\/\\/ia\\.media-imdb\\.com\\/images\\/rock\\)\\.jpg)"></a><div class="data-container"><div class="message_embed_title"><a href="http://test.org/" title="The Rock">The Rock</a></div><div class="message_embed_description">Description text</div></div></div>'
        self.assertEqual(with_preview, msg.rendered_content)

    @override_settings(CAMO_URI='')
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_inline_relative_url_embed_preview(self) -> None:
        if False:
            i = 10
            return i + 15
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish') as patched:
            self.send_personal_message(self.example_user('prospero'), self.example_user('cordelia'), content='http://zulip.testserver/api/')
            patched.assert_not_called()

    @override_settings(CAMO_URI='')
    def test_inline_url_embed_preview_with_relative_image_url(self) -> None:
        if False:
            return 10
        with_preview_relative = '<p><a href="http://test.org/">http://test.org/</a></p>\n<div class="message_embed"><a class="message_embed_image" href="http://test.org/" style="background-image: url(http\\:\\/\\/test\\.org\\/images\\/rock\\.jpg)"></a><div class="data-container"><div class="message_embed_title"><a href="http://test.org/" title="The Rock">The Rock</a></div><div class="message_embed_description">Description text</div></div></div>'
        msg = self._send_message_with_test_org_url(sender=self.example_user('prospero'), relative_url=True)
        self.assertEqual(msg.rendered_content, with_preview_relative)

    @responses.activate
    def test_http_error_get_data(self) -> None:
        if False:
            return 10
        url = 'http://test.org/'
        msg_id = self.send_personal_message(self.example_user('hamlet'), self.example_user('cordelia'), content=url)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        event = {'message_id': msg_id, 'urls': [url], 'message_realm_id': msg.sender.realm_id, 'message_content': url}
        self.create_mock_response(url, body=ConnectionError())
        with self.settings(INLINE_URL_EMBED_PREVIEW=True, TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/: ' in info_logs.output[0])
        msg = Message.objects.get(id=msg_id)
        self.assertEqual('<p><a href="http://test.org/">http://test.org/</a></p>', msg.rendered_content)

    def test_invalid_link(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.settings(INLINE_URL_EMBED_PREVIEW=True, TEST_SUITE=False):
            self.assertIsNone(get_link_embed_data('com.notvalidlink'))
            self.assertIsNone(get_link_embed_data('μένει.com.notvalidlink'))

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_link_preview_non_html_data(self) -> None:
        if False:
            return 10
        user = self.example_user('hamlet')
        self.login_user(user)
        url = 'http://test.org/audio.mp3'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish') as patched:
            msg_id = self.send_stream_message(user, 'Denmark', topic_name='foo', content=url)
            patched.assert_called_once()
            queue = patched.call_args[0][0]
            self.assertEqual(queue, 'embed_links')
            event = patched.call_args[0][1]
        content_type = 'application/octet-stream'
        self.create_mock_response(url, content_type=content_type)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
                cached_data = cache_get(preview_url_cache_key(url))[0]
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/audio.mp3: ' in info_logs.output[0])
        self.assertIsNone(cached_data)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        self.assertEqual('<p><a href="http://test.org/audio.mp3">http://test.org/audio.mp3</a></p>', msg.rendered_content)

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_link_preview_no_open_graph_image(self) -> None:
        if False:
            i = 10
            return i + 15
        user = self.example_user('hamlet')
        self.login_user(user)
        url = 'http://test.org/foo.html'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish') as patched:
            msg_id = self.send_stream_message(user, 'Denmark', topic_name='foo', content=url)
            patched.assert_called_once()
            queue = patched.call_args[0][0]
            self.assertEqual(queue, 'embed_links')
            event = patched.call_args[0][1]
        html = '\n'.join((line for line in self.open_graph_html.splitlines() if 'og:image' not in line))
        self.create_mock_response(url, body=html)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
                cached_data = cache_get(preview_url_cache_key(url))[0]
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/foo.html: ' in info_logs.output[0])
        assert cached_data is not None
        self.assertIsNotNone(cached_data.title)
        self.assertIsNone(cached_data.image)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        self.assertEqual('<p><a href="http://test.org/foo.html">http://test.org/foo.html</a></p>', msg.rendered_content)

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_link_preview_open_graph_image_bad_url(self) -> None:
        if False:
            while True:
                i = 10
        user = self.example_user('hamlet')
        self.login_user(user)
        url = 'http://test.org/foo.html'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish') as patched:
            msg_id = self.send_stream_message(user, 'Denmark', topic_name='foo', content=url)
            patched.assert_called_once()
            queue = patched.call_args[0][0]
            self.assertEqual(queue, 'embed_links')
            event = patched.call_args[0][1]
        html = '\n'.join((line if 'og:image' not in line else '<meta property="og:image" content="http://[bad url/" />' for line in self.open_graph_html.splitlines()))
        self.create_mock_response(url, body=html)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
                cached_data = cache_get(preview_url_cache_key(url))[0]
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/foo.html: ' in info_logs.output[0])
        assert cached_data is not None
        self.assertIsNotNone(cached_data.title)
        self.assertIsNone(cached_data.image)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        self.assertEqual('<p><a href="http://test.org/foo.html">http://test.org/foo.html</a></p>', msg.rendered_content)

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_link_preview_open_graph_image_missing_content(self) -> None:
        if False:
            while True:
                i = 10
        user = self.example_user('hamlet')
        self.login_user(user)
        url = 'http://test.org/foo.html'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish') as patched:
            msg_id = self.send_stream_message(user, 'Denmark', topic_name='foo', content=url)
            patched.assert_called_once()
            queue = patched.call_args[0][0]
            self.assertEqual(queue, 'embed_links')
            event = patched.call_args[0][1]
        html = '\n'.join((line if 'og:image' not in line else '<meta property="og:image"/>' for line in self.open_graph_html.splitlines()))
        self.create_mock_response(url, body=html)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
                cached_data = cache_get(preview_url_cache_key(url))[0]
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/foo.html: ' in info_logs.output[0])
        assert cached_data is not None
        self.assertIsNotNone(cached_data.title)
        self.assertIsNone(cached_data.image)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        self.assertEqual('<p><a href="http://test.org/foo.html">http://test.org/foo.html</a></p>', msg.rendered_content)

    @responses.activate
    @override_settings(CAMO_URI='')
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_link_preview_no_content_type_header(self) -> None:
        if False:
            i = 10
            return i + 15
        user = self.example_user('hamlet')
        self.login_user(user)
        url = 'http://test.org/'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish') as patched:
            msg_id = self.send_stream_message(user, 'Denmark', topic_name='foo', content=url)
            patched.assert_called_once()
            queue = patched.call_args[0][0]
            self.assertEqual(queue, 'embed_links')
            event = patched.call_args[0][1]
        self.create_mock_response(url)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
                cached_data = cache_get(preview_url_cache_key(url))[0]
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/: ' in info_logs.output[0])
        assert cached_data is not None
        msg = Message.objects.select_related('sender').get(id=msg_id)
        assert msg.rendered_content is not None
        self.assertIn(cached_data.title, msg.rendered_content)
        assert cached_data.image is not None
        self.assertIn(re.sub('([^\\w-])', '\\\\\\1', cached_data.image), msg.rendered_content)

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_valid_content_type_error_get_data(self) -> None:
        if False:
            return 10
        url = 'http://test.org/'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish'):
            msg_id = self.send_personal_message(self.example_user('hamlet'), self.example_user('cordelia'), content=url)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        event = {'message_id': msg_id, 'urls': [url], 'message_realm_id': msg.sender.realm_id, 'message_content': url}
        self.create_mock_response(url, body=ConnectionError())
        with mock.patch('zerver.lib.url_preview.preview.get_oembed_data', side_effect=lambda *args, **kwargs: None):
            with mock.patch('zerver.lib.url_preview.preview.valid_content_type', side_effect=lambda k: True):
                with self.settings(TEST_SUITE=False):
                    with self.assertLogs(level='INFO') as info_logs:
                        FetchLinksEmbedData().consume(event)
                    self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/: ' in info_logs.output[0])
                    cached_data = cache_get(preview_url_cache_key(url))
                    self.assertIsNone(cached_data)
        msg.refresh_from_db()
        self.assertEqual('<p><a href="http://test.org/">http://test.org/</a></p>', msg.rendered_content)

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_invalid_url(self) -> None:
        if False:
            i = 10
            return i + 15
        url = 'http://test.org/'
        error_url = 'http://test.org/x'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish'):
            msg_id = self.send_personal_message(self.example_user('hamlet'), self.example_user('cordelia'), content=error_url)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        event = {'message_id': msg_id, 'urls': [error_url], 'message_realm_id': msg.sender.realm_id, 'message_content': error_url}
        self.create_mock_response(error_url, status=404)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                FetchLinksEmbedData().consume(event)
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/x: ' in info_logs.output[0])
            cached_data = cache_get(preview_url_cache_key(error_url))[0]
        self.assertIsNone(cached_data)
        msg.refresh_from_db()
        self.assertEqual('<p><a href="http://test.org/x">http://test.org/x</a></p>', msg.rendered_content)
        self.assertTrue(responses.assert_call_count(url, 0))

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_safe_oembed_html_url(self) -> None:
        if False:
            print('Hello World!')
        url = 'http://test.org/'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish'):
            msg_id = self.send_personal_message(self.example_user('hamlet'), self.example_user('cordelia'), content=url)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        event = {'message_id': msg_id, 'urls': [url], 'message_realm_id': msg.sender.realm_id, 'message_content': url}
        mocked_data = UrlOEmbedData(html=f'<iframe src="{url}"></iframe>', type='video', image=f'{url}/image.png')
        self.create_mock_response(url)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                with mock.patch('zerver.lib.url_preview.preview.get_oembed_data', lambda *args, **kwargs: mocked_data):
                    FetchLinksEmbedData().consume(event)
                    cached_data = cache_get(preview_url_cache_key(url))[0]
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for http://test.org/: ' in info_logs.output[0])
        self.assertEqual(cached_data, mocked_data)
        msg.refresh_from_db()
        assert msg.rendered_content is not None
        self.assertIn(f'a data-id="{escape(mocked_data.html)}"', msg.rendered_content)

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_youtube_url_title_replaces_url(self) -> None:
        if False:
            i = 10
            return i + 15
        url = 'https://www.youtube.com/watch?v=eSJTXC7Ixgg'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish'):
            msg_id = self.send_personal_message(self.example_user('hamlet'), self.example_user('cordelia'), content=url)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        event = {'message_id': msg_id, 'urls': [url], 'message_realm_id': msg.sender.realm_id, 'message_content': url}
        mocked_data = UrlEmbedData(title='Clearer Code at Scale - Static Types at Zulip and Dropbox')
        self.create_mock_response(url)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                with mock.patch('zerver.worker.queue_processors.url_preview.get_link_embed_data', lambda *args, **kwargs: mocked_data):
                    FetchLinksEmbedData().consume(event)
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for https://www.youtube.com/watch?v=eSJTXC7Ixgg:' in info_logs.output[0])
        msg.refresh_from_db()
        expected_content = f'''<p><a href="https://www.youtube.com/watch?v=eSJTXC7Ixgg">YouTube - Clearer Code at Scale - Static Types at Zulip and Dropbox</a></p>\n<div class="youtube-video message_inline_image"><a data-id="eSJTXC7Ixgg" href="https://www.youtube.com/watch?v=eSJTXC7Ixgg"><img src="{get_camo_url('https://i.ytimg.com/vi/eSJTXC7Ixgg/default.jpg')}"></a></div>'''
        self.assertEqual(expected_content, msg.rendered_content)

    @responses.activate
    @override_settings(INLINE_URL_EMBED_PREVIEW=True)
    def test_custom_title_replaces_youtube_url_title(self) -> None:
        if False:
            while True:
                i = 10
        url = '[YouTube link](https://www.youtube.com/watch?v=eSJTXC7Ixgg)'
        with mock_queue_publish('zerver.actions.message_send.queue_json_publish'):
            msg_id = self.send_personal_message(self.example_user('hamlet'), self.example_user('cordelia'), content=url)
        msg = Message.objects.select_related('sender').get(id=msg_id)
        event = {'message_id': msg_id, 'urls': [url], 'message_realm_id': msg.sender.realm_id, 'message_content': url}
        mocked_data = UrlEmbedData(title='Clearer Code at Scale - Static Types at Zulip and Dropbox')
        self.create_mock_response(url)
        with self.settings(TEST_SUITE=False):
            with self.assertLogs(level='INFO') as info_logs:
                with mock.patch('zerver.worker.queue_processors.url_preview.get_link_embed_data', lambda *args, **kwargs: mocked_data):
                    FetchLinksEmbedData().consume(event)
            self.assertTrue('INFO:root:Time spent on get_link_embed_data for [YouTube link](https://www.youtube.com/watch?v=eSJTXC7Ixgg):' in info_logs.output[0])
        msg.refresh_from_db()
        expected_content = f'''<p><a href="https://www.youtube.com/watch?v=eSJTXC7Ixgg">YouTube link</a></p>\n<div class="youtube-video message_inline_image"><a data-id="eSJTXC7Ixgg" href="https://www.youtube.com/watch?v=eSJTXC7Ixgg"><img src="{get_camo_url('https://i.ytimg.com/vi/eSJTXC7Ixgg/default.jpg')}"></a></div>'''
        self.assertEqual(expected_content, msg.rendered_content)