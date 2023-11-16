import pytest
from telegram import SentWebAppMessage
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def sent_web_app_message():
    if False:
        for i in range(10):
            print('nop')
    return SentWebAppMessage(inline_message_id=TestSentWebAppMessageBase.inline_message_id)

class TestSentWebAppMessageBase:
    inline_message_id = '123'

class TestSentWebAppMessageWithoutRequest(TestSentWebAppMessageBase):

    def test_slot_behaviour(self, sent_web_app_message):
        if False:
            print('Hello World!')
        inst = sent_web_app_message
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_to_dict(self, sent_web_app_message):
        if False:
            while True:
                i = 10
        sent_web_app_message_dict = sent_web_app_message.to_dict()
        assert isinstance(sent_web_app_message_dict, dict)
        assert sent_web_app_message_dict['inline_message_id'] == self.inline_message_id

    def test_de_json(self, bot):
        if False:
            for i in range(10):
                print('nop')
        data = {'inline_message_id': self.inline_message_id}
        m = SentWebAppMessage.de_json(data, None)
        assert m.api_kwargs == {}
        assert m.inline_message_id == self.inline_message_id

    def test_equality(self):
        if False:
            while True:
                i = 10
        a = SentWebAppMessage(self.inline_message_id)
        b = SentWebAppMessage(self.inline_message_id)
        c = SentWebAppMessage('')
        d = SentWebAppMessage('not_inline_message_id')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)