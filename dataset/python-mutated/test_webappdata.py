import pytest
from telegram import WebAppData
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def web_app_data():
    if False:
        i = 10
        return i + 15
    return WebAppData(data=TestWebAppDataBase.data, button_text=TestWebAppDataBase.button_text)

class TestWebAppDataBase:
    data = 'data'
    button_text = 'button_text'

class TestWebAppDataWithoutRequest(TestWebAppDataBase):

    def test_slot_behaviour(self, web_app_data):
        if False:
            return 10
        for attr in web_app_data.__slots__:
            assert getattr(web_app_data, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(web_app_data)) == len(set(mro_slots(web_app_data))), 'duplicate slot'

    def test_to_dict(self, web_app_data):
        if False:
            for i in range(10):
                print('nop')
        web_app_data_dict = web_app_data.to_dict()
        assert isinstance(web_app_data_dict, dict)
        assert web_app_data_dict['data'] == self.data
        assert web_app_data_dict['button_text'] == self.button_text

    def test_de_json(self, bot):
        if False:
            for i in range(10):
                print('nop')
        json_dict = {'data': self.data, 'button_text': self.button_text}
        web_app_data = WebAppData.de_json(json_dict, bot)
        assert web_app_data.api_kwargs == {}
        assert web_app_data.data == self.data
        assert web_app_data.button_text == self.button_text

    def test_equality(self):
        if False:
            print('Hello World!')
        a = WebAppData(self.data, self.button_text)
        b = WebAppData(self.data, self.button_text)
        c = WebAppData('', '')
        d = WebAppData('not_data', 'not_button_text')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)