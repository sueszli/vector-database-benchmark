from types import SimpleNamespace
from typing import Dict
from unittest.mock import MagicMock, patch
from django.http import HttpRequest
from django.http.response import HttpResponse
from typing_extensions import override
from zerver.actions.streams import do_rename_stream
from zerver.decorator import webhook_view
from zerver.lib.exceptions import InvalidJSONError, JsonableError
from zerver.lib.send_email import FromAddress
from zerver.lib.test_classes import WebhookTestCase, ZulipTestCase
from zerver.lib.test_helpers import HostRequestMock
from zerver.lib.users import get_api_key
from zerver.lib.webhooks.common import INVALID_JSON_MESSAGE, MISSING_EVENT_HEADER_MESSAGE, MissingHTTPEventHeaderError, get_fixture_http_headers, standardize_headers, validate_extract_webhook_http_header
from zerver.models import UserProfile, get_realm, get_user

class WebhooksCommonTestCase(ZulipTestCase):

    def test_webhook_http_header_header_exists(self) -> None:
        if False:
            while True:
                i = 10
        webhook_bot = get_user('webhook-bot@zulip.com', get_realm('zulip'))
        request = HostRequestMock()
        request.META['HTTP_X_CUSTOM_HEADER'] = 'custom_value'
        request.user = webhook_bot
        header_value = validate_extract_webhook_http_header(request, 'X-Custom-Header', 'test_webhook')
        self.assertEqual(header_value, 'custom_value')

    def test_webhook_http_header_header_does_not_exist(self) -> None:
        if False:
            while True:
                i = 10
        realm = get_realm('zulip')
        webhook_bot = get_user('webhook-bot@zulip.com', realm)
        webhook_bot.last_reminder = None
        notification_bot = self.notification_bot(realm)
        request = HostRequestMock()
        request.user = webhook_bot
        request.path = 'some/random/path'
        exception_msg = "Missing the HTTP event header 'X-Custom-Header'"
        with self.assertRaisesRegex(MissingHTTPEventHeaderError, exception_msg):
            validate_extract_webhook_http_header(request, 'X-Custom-Header', 'test_webhook')
        msg = self.get_last_message()
        expected_message = MISSING_EVENT_HEADER_MESSAGE.format(bot_name=webhook_bot.full_name, request_path=request.path, header_name='X-Custom-Header', integration_name='test_webhook', support_email=FromAddress.SUPPORT).rstrip()
        self.assertEqual(msg.sender.id, notification_bot.id)
        self.assertEqual(msg.content, expected_message)

    def test_notify_bot_owner_on_invalid_json(self) -> None:
        if False:
            return 10

        @webhook_view('ClientName', notify_bot_owner_on_invalid_json=False)
        def my_webhook_no_notify(request: HttpRequest, user_profile: UserProfile) -> HttpResponse:
            if False:
                for i in range(10):
                    print('nop')
            raise InvalidJSONError('Malformed JSON')

        @webhook_view('ClientName', notify_bot_owner_on_invalid_json=True)
        def my_webhook_notify(request: HttpRequest, user_profile: UserProfile) -> HttpResponse:
            if False:
                return 10
            raise InvalidJSONError('Malformed JSON')
        webhook_bot_email = 'webhook-bot@zulip.com'
        webhook_bot_realm = get_realm('zulip')
        webhook_bot = get_user(webhook_bot_email, webhook_bot_realm)
        webhook_bot_api_key = get_api_key(webhook_bot)
        request = HostRequestMock()
        request.POST['api_key'] = webhook_bot_api_key
        request.host = 'zulip.testserver'
        expected_msg = INVALID_JSON_MESSAGE.format(webhook_name='ClientName')
        last_message_id = self.get_last_message().id
        with self.assertRaisesRegex(JsonableError, 'Malformed JSON'):
            my_webhook_no_notify(request)
        msg = self.get_last_message()
        self.assertEqual(msg.id, last_message_id)
        self.assertNotEqual(msg.content, expected_msg.strip())
        request = HostRequestMock()
        request.POST['api_key'] = webhook_bot_api_key
        request.host = 'zulip.testserver'
        with self.assertRaisesRegex(JsonableError, 'Malformed JSON'):
            my_webhook_notify(request)
        msg = self.get_last_message()
        self.assertNotEqual(msg.id, last_message_id)
        self.assertEqual(msg.sender.id, self.notification_bot(webhook_bot_realm).id)
        self.assertEqual(msg.content, expected_msg.strip())

    @patch('zerver.lib.webhooks.common.importlib.import_module')
    def test_get_fixture_http_headers_for_success(self, import_module_mock: MagicMock) -> None:
        if False:
            print('Hello World!')

        def fixture_to_headers(fixture_name: str) -> Dict[str, str]:
            if False:
                i = 10
                return i + 15
            return {'key': 'value'}
        fake_module = SimpleNamespace(fixture_to_headers=fixture_to_headers)
        import_module_mock.return_value = fake_module
        headers = get_fixture_http_headers('some_integration', 'complex_fixture')
        self.assertEqual(headers, {'key': 'value'})

    def test_get_fixture_http_headers_for_non_existent_integration(self) -> None:
        if False:
            while True:
                i = 10
        headers = get_fixture_http_headers('some_random_nonexistent_integration', 'fixture_name')
        self.assertEqual(headers, {})

    @patch('zerver.lib.webhooks.common.importlib.import_module')
    def test_get_fixture_http_headers_with_no_fixtures_to_headers_function(self, import_module_mock: MagicMock) -> None:
        if False:
            return 10
        fake_module = SimpleNamespace()
        import_module_mock.return_value = fake_module
        self.assertEqual(get_fixture_http_headers('some_integration', 'simple_fixture'), {})

    def test_standardize_headers(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(standardize_headers({}), {})
        raw_headers = {'Content-Type': 'text/plain', 'X-Event-Type': 'ping'}
        djangoified_headers = standardize_headers(raw_headers)
        expected_djangoified_headers = {'CONTENT_TYPE': 'text/plain', 'HTTP_X_EVENT_TYPE': 'ping'}
        self.assertEqual(djangoified_headers, expected_djangoified_headers)

class WebhookURLConfigurationTestCase(WebhookTestCase):
    STREAM_NAME = 'helloworld'
    WEBHOOK_DIR_NAME = 'helloworld'
    URL_TEMPLATE = '/api/v1/external/helloworld?stream={stream}&api_key={api_key}'

    @override
    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        stream = self.subscribe(self.test_user, self.STREAM_NAME)
        self.STREAM_NAME = str(stream.id)
        do_rename_stream(stream, 'helloworld_renamed', self.test_user)
        self.url = self.build_webhook_url()

    def test_trigger_stream_message_by_id(self) -> None:
        if False:
            i = 10
            return i + 15
        payload = self.get_body('hello')
        self.send_webhook_payload(self.test_user, self.url, payload, content_type='application/json')
        expected_topic = 'Hello World'
        expected_message = 'Hello! I am happy to be here! :smile:\nThe Wikipedia featured article for today is **[Marilyn Monroe](https://en.wikipedia.org/wiki/Marilyn_Monroe)**'
        msg = self.get_last_message()
        self.assert_stream_message(message=msg, stream_name='helloworld_renamed', topic_name=expected_topic, content=expected_message)

class MissingEventHeaderTestCase(WebhookTestCase):
    STREAM_NAME = 'groove'
    URL_TEMPLATE = '/api/v1/external/groove?stream={stream}&api_key={api_key}'

    def test_missing_event_header(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.subscribe(self.test_user, self.STREAM_NAME)
        with self.assertLogs('zulip.zerver.webhooks.anomalous', level='INFO') as webhook_logs:
            result = self.client_post(self.url, self.get_body('ticket_state_changed'), content_type='application/x-www-form-urlencoded')
        self.assertTrue("Missing the HTTP event header 'X-Groove-Event'" in webhook_logs.output[0])
        self.assert_json_error(result, "Missing the HTTP event header 'X-Groove-Event'")
        realm = get_realm('zulip')
        webhook_bot = get_user('webhook-bot@zulip.com', realm)
        webhook_bot.last_reminder = None
        notification_bot = self.notification_bot(realm)
        msg = self.get_last_message()
        expected_message = MISSING_EVENT_HEADER_MESSAGE.format(bot_name=webhook_bot.full_name, request_path='/api/v1/external/groove', header_name='X-Groove-Event', integration_name='Groove', support_email=FromAddress.SUPPORT).rstrip()
        if msg.sender.id != notification_bot.id:
            print(msg)
            print(msg.content)
        self.assertEqual(msg.sender.id, notification_bot.id)
        self.assertEqual(msg.content, expected_message)

    @override
    def get_body(self, fixture_name: str) -> str:
        if False:
            i = 10
            return i + 15
        return self.webhook_fixture_data('groove', fixture_name, file_type='json')