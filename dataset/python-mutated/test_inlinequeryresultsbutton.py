import pytest
from telegram import InlineQueryResultsButton, WebAppInfo
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def inline_query_results_button():
    if False:
        for i in range(10):
            print('nop')
    return InlineQueryResultsButton(text=TestInlineQueryResultsButtonBase.text, start_parameter=TestInlineQueryResultsButtonBase.start_parameter, web_app=TestInlineQueryResultsButtonBase.web_app)

class TestInlineQueryResultsButtonBase:
    text = 'text'
    start_parameter = 'start_parameter'
    web_app = WebAppInfo(url='https://python-telegram-bot.org')

class TestInlineQueryResultsButtonWithoutRequest(TestInlineQueryResultsButtonBase):

    def test_slot_behaviour(self, inline_query_results_button):
        if False:
            return 10
        inst = inline_query_results_button
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_to_dict(self, inline_query_results_button):
        if False:
            return 10
        inline_query_results_button_dict = inline_query_results_button.to_dict()
        assert isinstance(inline_query_results_button_dict, dict)
        assert inline_query_results_button_dict['text'] == self.text
        assert inline_query_results_button_dict['start_parameter'] == self.start_parameter
        assert inline_query_results_button_dict['web_app'] == self.web_app.to_dict()

    def test_de_json(self, bot):
        if False:
            print('Hello World!')
        assert InlineQueryResultsButton.de_json(None, bot) is None
        assert InlineQueryResultsButton.de_json({}, bot) is None
        json_dict = {'text': self.text, 'start_parameter': self.start_parameter, 'web_app': self.web_app.to_dict()}
        inline_query_results_button = InlineQueryResultsButton.de_json(json_dict, bot)
        assert inline_query_results_button.text == self.text
        assert inline_query_results_button.start_parameter == self.start_parameter
        assert inline_query_results_button.web_app == self.web_app

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        a = InlineQueryResultsButton(self.text, self.start_parameter, self.web_app)
        b = InlineQueryResultsButton(self.text, self.start_parameter, self.web_app)
        c = InlineQueryResultsButton(self.text, '', self.web_app)
        d = InlineQueryResultsButton(self.text, self.start_parameter, None)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)