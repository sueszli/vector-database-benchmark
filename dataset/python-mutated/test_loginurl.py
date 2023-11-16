import pytest
from telegram import LoginUrl
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def login_url():
    if False:
        for i in range(10):
            print('nop')
    return LoginUrl(url=TestLoginUrlBase.url, forward_text=TestLoginUrlBase.forward_text, bot_username=TestLoginUrlBase.bot_username, request_write_access=TestLoginUrlBase.request_write_access)

class TestLoginUrlBase:
    url = 'http://www.google.com'
    forward_text = 'Send me forward!'
    bot_username = 'botname'
    request_write_access = True

class TestLoginUrlWithoutRequest(TestLoginUrlBase):

    def test_slot_behaviour(self, login_url):
        if False:
            i = 10
            return i + 15
        for attr in login_url.__slots__:
            assert getattr(login_url, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(login_url)) == len(set(mro_slots(login_url))), 'duplicate slot'

    def test_to_dict(self, login_url):
        if False:
            while True:
                i = 10
        login_url_dict = login_url.to_dict()
        assert isinstance(login_url_dict, dict)
        assert login_url_dict['url'] == self.url
        assert login_url_dict['forward_text'] == self.forward_text
        assert login_url_dict['bot_username'] == self.bot_username
        assert login_url_dict['request_write_access'] == self.request_write_access

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        a = LoginUrl(self.url, self.forward_text, self.bot_username, self.request_write_access)
        b = LoginUrl(self.url, self.forward_text, self.bot_username, self.request_write_access)
        c = LoginUrl(self.url)
        d = LoginUrl('text.com', self.forward_text, self.bot_username, self.request_write_access)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a == c
        assert hash(a) == hash(c)
        assert a != d
        assert hash(a) != hash(d)