import time
from datetime import datetime
import pytest
from telegram import LoginUrl, WebhookInfo
from telegram._utils.datetime import UTC, from_timestamp
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def webhook_info():
    if False:
        i = 10
        return i + 15
    return WebhookInfo(url=TestWebhookInfoBase.url, has_custom_certificate=TestWebhookInfoBase.has_custom_certificate, pending_update_count=TestWebhookInfoBase.pending_update_count, ip_address=TestWebhookInfoBase.ip_address, last_error_date=TestWebhookInfoBase.last_error_date, max_connections=TestWebhookInfoBase.max_connections, allowed_updates=TestWebhookInfoBase.allowed_updates, last_synchronization_error_date=TestWebhookInfoBase.last_synchronization_error_date)

class TestWebhookInfoBase:
    url = 'http://www.google.com'
    has_custom_certificate = False
    pending_update_count = 5
    ip_address = '127.0.0.1'
    last_error_date = time.time()
    max_connections = 42
    allowed_updates = ['type1', 'type2']
    last_synchronization_error_date = time.time()

class TestWebhookInfoWithoutRequest(TestWebhookInfoBase):

    def test_slot_behaviour(self, webhook_info):
        if False:
            print('Hello World!')
        for attr in webhook_info.__slots__:
            assert getattr(webhook_info, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(webhook_info)) == len(set(mro_slots(webhook_info))), 'duplicate slot'

    def test_to_dict(self, webhook_info):
        if False:
            for i in range(10):
                print('nop')
        webhook_info_dict = webhook_info.to_dict()
        assert isinstance(webhook_info_dict, dict)
        assert webhook_info_dict['url'] == self.url
        assert webhook_info_dict['pending_update_count'] == self.pending_update_count
        assert webhook_info_dict['last_error_date'] == self.last_error_date
        assert webhook_info_dict['max_connections'] == self.max_connections
        assert webhook_info_dict['allowed_updates'] == self.allowed_updates
        assert webhook_info_dict['ip_address'] == self.ip_address
        assert webhook_info_dict['last_synchronization_error_date'] == self.last_synchronization_error_date

    def test_de_json(self, bot):
        if False:
            print('Hello World!')
        json_dict = {'url': self.url, 'has_custom_certificate': self.has_custom_certificate, 'pending_update_count': self.pending_update_count, 'last_error_date': self.last_error_date, 'max_connections': self.max_connections, 'allowed_updates': self.allowed_updates, 'ip_address': self.ip_address, 'last_synchronization_error_date': self.last_synchronization_error_date}
        webhook_info = WebhookInfo.de_json(json_dict, bot)
        assert webhook_info.api_kwargs == {}
        assert webhook_info.url == self.url
        assert webhook_info.has_custom_certificate == self.has_custom_certificate
        assert webhook_info.pending_update_count == self.pending_update_count
        assert isinstance(webhook_info.last_error_date, datetime)
        assert webhook_info.last_error_date == from_timestamp(self.last_error_date)
        assert webhook_info.max_connections == self.max_connections
        assert webhook_info.allowed_updates == tuple(self.allowed_updates)
        assert webhook_info.ip_address == self.ip_address
        assert isinstance(webhook_info.last_synchronization_error_date, datetime)
        assert webhook_info.last_synchronization_error_date == from_timestamp(self.last_synchronization_error_date)
        none = WebhookInfo.de_json(None, bot)
        assert none is None

    def test_de_json_localization(self, bot, raw_bot, tz_bot):
        if False:
            while True:
                i = 10
        json_dict = {'url': self.url, 'has_custom_certificate': self.has_custom_certificate, 'pending_update_count': self.pending_update_count, 'last_error_date': self.last_error_date, 'max_connections': self.max_connections, 'allowed_updates': self.allowed_updates, 'ip_address': self.ip_address, 'last_synchronization_error_date': self.last_synchronization_error_date}
        webhook_info_bot = WebhookInfo.de_json(json_dict, bot)
        webhook_info_raw = WebhookInfo.de_json(json_dict, raw_bot)
        webhook_info_tz = WebhookInfo.de_json(json_dict, tz_bot)
        last_error_date_offset = webhook_info_tz.last_error_date.utcoffset()
        last_error_tz_bot_offset = tz_bot.defaults.tzinfo.utcoffset(webhook_info_tz.last_error_date.replace(tzinfo=None))
        sync_error_date_offset = webhook_info_tz.last_synchronization_error_date.utcoffset()
        sync_error_date_tz_bot_offset = tz_bot.defaults.tzinfo.utcoffset(webhook_info_tz.last_synchronization_error_date.replace(tzinfo=None))
        assert webhook_info_raw.last_error_date.tzinfo == UTC
        assert webhook_info_bot.last_error_date.tzinfo == UTC
        assert last_error_date_offset == last_error_tz_bot_offset
        assert webhook_info_raw.last_synchronization_error_date.tzinfo == UTC
        assert webhook_info_bot.last_synchronization_error_date.tzinfo == UTC
        assert sync_error_date_offset == sync_error_date_tz_bot_offset

    def test_always_tuple_allowed_updates(self):
        if False:
            return 10
        webhook_info = WebhookInfo(self.url, self.has_custom_certificate, self.pending_update_count)
        assert webhook_info.allowed_updates == ()

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        a = WebhookInfo(url=self.url, has_custom_certificate=self.has_custom_certificate, pending_update_count=self.pending_update_count, last_error_date=self.last_error_date, max_connections=self.max_connections)
        b = WebhookInfo(url=self.url, has_custom_certificate=self.has_custom_certificate, pending_update_count=self.pending_update_count, last_error_date=self.last_error_date, max_connections=self.max_connections)
        c = WebhookInfo(url='http://github.com', has_custom_certificate=True, pending_update_count=78, last_error_date=0, max_connections=1)
        d = WebhookInfo(url='http://github.com', has_custom_certificate=True, pending_update_count=78, last_error_date=0, max_connections=1, last_synchronization_error_date=123)
        e = LoginUrl('text.com')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)