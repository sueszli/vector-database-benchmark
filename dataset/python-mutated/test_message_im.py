import responses
from sentry.models.identity import Identity, IdentityProvider, IdentityStatus
from sentry.silo import SiloMode
from sentry.silo.util import PROXY_BASE_URL_HEADER, PROXY_OI_HEADER, PROXY_SIGNATURE_HEADER
from sentry.testutils.cases import IntegratedApiTestCase
from sentry.testutils.helpers import get_response_text
from sentry.testutils.silo import assume_test_silo_mode, region_silo_test
from sentry.utils import json
from . import BaseEventTest
MESSAGE_IM_EVENT = '{\n    "type": "message",\n    "channel": "DOxxxxxx",\n    "user": "Uxxxxxxx",\n    "text": "helloo",\n    "message_ts": "123456789.9875"\n}'
MESSAGE_IM_EVENT_NO_TEXT = '{\n    "type": "message",\n    "channel": "DOxxxxxx",\n    "user": "Uxxxxxxx",\n    "message_ts": "123456789.9875"\n}'
MESSAGE_IM_EVENT_UNLINK = '{\n    "type": "message",\n    "text": "unlink",\n    "user": "UXXXXXXX1",\n    "team": "TXXXXXXX1",\n    "channel": "DTPJWTJ2D"\n}'
MESSAGE_IM_EVENT_LINK = '{\n    "type": "message",\n    "text": "link",\n    "user": "UXXXXXXX1",\n    "team": "TXXXXXXX1",\n    "channel": "DTPJWTJ2D"\n}'
MESSAGE_IM_BOT_EVENT = '{\n    "type": "message",\n    "channel": "DOxxxxxx",\n    "user": "Uxxxxxxx",\n    "text": "helloo",\n    "bot_id": "bot_id",\n    "message_ts": "123456789.9875"\n}'

@region_silo_test(stable=True)
class MessageIMEventTest(BaseEventTest, IntegratedApiTestCase):

    def get_block_section_text(self, data):
        if False:
            print('Hello World!')
        blocks = data['blocks']
        return (blocks[0]['text']['text'], blocks[1]['text']['text'])

    @responses.activate
    def test_identifying_channel_correctly(self):
        if False:
            print('Hello World!')
        responses.add(responses.POST, 'https://slack.com/api/chat.postMessage', json={'ok': True})
        event_data = json.loads(MESSAGE_IM_EVENT)
        self.post_webhook(event_data=event_data)
        request = responses.calls[0].request
        data = json.loads(request.body)
        assert data.get('channel') == event_data['channel']

    def _check_proxying(self) -> None:
        if False:
            while True:
                i = 10
        assert len(responses.calls) == 1
        request = responses.calls[0].request
        assert request.headers[PROXY_OI_HEADER] == str(self.organization_integration.id)
        assert request.headers[PROXY_BASE_URL_HEADER] == 'https://slack.com/api'
        assert PROXY_SIGNATURE_HEADER in request.headers

    @responses.activate
    def test_user_message_im_notification_platform(self):
        if False:
            return 10
        responses.add(responses.POST, 'https://slack.com/api/chat.postMessage', json={'ok': True})
        resp = self.post_webhook(event_data=json.loads(MESSAGE_IM_EVENT))
        assert resp.status_code == 200, resp.content
        if self.should_call_api_without_proxying():
            assert len(responses.calls) == 1
            request = responses.calls[0].request
            assert request.headers['Authorization'] == 'Bearer xoxb-xxxxxxxxx-xxxxxxxxxx-xxxxxxxxxxxx'
            data = json.loads(request.body)
            (heading, contents) = self.get_block_section_text(data)
            assert heading == 'Unknown command: `helloo`'
            assert contents == 'Here are the commands you can use. Commands not working? Re-install the app!'
        else:
            self._check_proxying()

    @responses.activate
    def test_user_message_link(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that when a user types in "link" to the DM we reply with the correct response.\n        '
        with assume_test_silo_mode(SiloMode.CONTROL):
            IdentityProvider.objects.create(type='slack', external_id='TXXXXXXX1', config={})
        responses.add(responses.POST, 'https://slack.com/api/chat.postMessage', json={'ok': True})
        resp = self.post_webhook(event_data=json.loads(MESSAGE_IM_EVENT_LINK))
        assert resp.status_code == 200, resp.content
        if self.should_call_api_without_proxying():
            assert len(responses.calls) == 1
            request = responses.calls[0].request
            assert request.headers['Authorization'] == 'Bearer xoxb-xxxxxxxxx-xxxxxxxxxx-xxxxxxxxxxxx'
            data = json.loads(request.body)
            assert 'Link your Slack identity' in get_response_text(data)
        else:
            self._check_proxying()

    @responses.activate
    def test_user_message_already_linked(self):
        if False:
            while True:
                i = 10
        '\n        Test that when a user who has already linked their identity types in\n        "link" to the DM we reply with the correct response.\n        '
        with assume_test_silo_mode(SiloMode.CONTROL):
            idp = IdentityProvider.objects.create(type='slack', external_id='TXXXXXXX1', config={})
            Identity.objects.create(external_id='UXXXXXXX1', idp=idp, user=self.user, status=IdentityStatus.VALID, scopes=[])
        responses.add(responses.POST, 'https://slack.com/api/chat.postMessage', json={'ok': True})
        resp = self.post_webhook(event_data=json.loads(MESSAGE_IM_EVENT_LINK))
        assert resp.status_code == 200, resp.content
        if self.should_call_api_without_proxying():
            assert len(responses.calls) == 1
            request = responses.calls[0].request
            assert request.headers['Authorization'] == 'Bearer xoxb-xxxxxxxxx-xxxxxxxxxx-xxxxxxxxxxxx'
            data = json.loads(request.body)
            assert 'You are already linked' in get_response_text(data)
        else:
            self._check_proxying()

    @responses.activate
    def test_user_message_unlink(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that when a user types in "unlink" to the DM we reply with the correct response.\n        '
        with assume_test_silo_mode(SiloMode.CONTROL):
            idp = IdentityProvider.objects.create(type='slack', external_id='TXXXXXXX1', config={})
            Identity.objects.create(external_id='UXXXXXXX1', idp=idp, user=self.user, status=IdentityStatus.VALID, scopes=[])
        responses.add(responses.POST, 'https://slack.com/api/chat.postMessage', json={'ok': True})
        resp = self.post_webhook(event_data=json.loads(MESSAGE_IM_EVENT_UNLINK))
        assert resp.status_code == 200, resp.content
        if self.should_call_api_without_proxying():
            assert len(responses.calls) == 1
            request = responses.calls[0].request
            assert request.headers['Authorization'] == 'Bearer xoxb-xxxxxxxxx-xxxxxxxxxx-xxxxxxxxxxxx'
            data = json.loads(request.body)
            assert 'Click here to unlink your identity' in get_response_text(data)
        else:
            self._check_proxying()

    @responses.activate
    def test_user_message_already_unlinked(self):
        if False:
            while True:
                i = 10
        '\n        Test that when a user without an Identity types in "unlink" to the DM we\n        reply with the correct response.\n        '
        with assume_test_silo_mode(SiloMode.CONTROL):
            IdentityProvider.objects.create(type='slack', external_id='TXXXXXXX1', config={})
        responses.add(responses.POST, 'https://slack.com/api/chat.postMessage', json={'ok': True})
        resp = self.post_webhook(event_data=json.loads(MESSAGE_IM_EVENT_UNLINK))
        assert resp.status_code == 200, resp.content
        if self.should_call_api_without_proxying():
            assert len(responses.calls) == 1
            request = responses.calls[0].request
            assert request.headers['Authorization'] == 'Bearer xoxb-xxxxxxxxx-xxxxxxxxxx-xxxxxxxxxxxx'
            data = json.loads(request.body)
            assert 'You do not have a linked identity to unlink' in get_response_text(data)
        else:
            self._check_proxying()

    def test_bot_message_im(self):
        if False:
            while True:
                i = 10
        resp = self.post_webhook(event_data=json.loads(MESSAGE_IM_BOT_EVENT))
        assert resp.status_code == 200, resp.content

    @responses.activate
    def test_user_message_im_no_text(self):
        if False:
            for i in range(10):
                print('nop')
        responses.add(responses.POST, 'https://slack.com/api/chat.postMessage', json={'ok': True})
        resp = self.post_webhook(event_data=json.loads(MESSAGE_IM_EVENT_NO_TEXT))
        assert resp.status_code == 200, resp.content
        assert len(responses.calls) == 0