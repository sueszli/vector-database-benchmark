import pytest
from telegram import WebAppInfo
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def web_app_info():
    if False:
        while True:
            i = 10
    return WebAppInfo(url=TestWebAppInfoBase.url)

class TestWebAppInfoBase:
    url = 'https://www.example.com'

class TestWebAppInfoWithoutRequest(TestWebAppInfoBase):

    def test_slot_behaviour(self, web_app_info):
        if False:
            return 10
        for attr in web_app_info.__slots__:
            assert getattr(web_app_info, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(web_app_info)) == len(set(mro_slots(web_app_info))), 'duplicate slot'

    def test_to_dict(self, web_app_info):
        if False:
            i = 10
            return i + 15
        web_app_info_dict = web_app_info.to_dict()
        assert isinstance(web_app_info_dict, dict)
        assert web_app_info_dict['url'] == self.url

    def test_de_json(self, bot):
        if False:
            for i in range(10):
                print('nop')
        json_dict = {'url': self.url}
        web_app_info = WebAppInfo.de_json(json_dict, bot)
        assert web_app_info.api_kwargs == {}
        assert web_app_info.url == self.url

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        a = WebAppInfo(self.url)
        b = WebAppInfo(self.url)
        c = WebAppInfo('')
        d = WebAppInfo('not_url')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)