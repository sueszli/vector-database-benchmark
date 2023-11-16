import pytest
from telegram import ForceReply, ReplyKeyboardRemove
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def force_reply():
    if False:
        return 10
    return ForceReply(TestForceReplyBase.selective, TestForceReplyBase.input_field_placeholder)

class TestForceReplyBase:
    force_reply = True
    selective = True
    input_field_placeholder = 'force replies can be annoying if not used properly'

class TestForceReplyWithoutRequest(TestForceReplyBase):

    def test_slot_behaviour(self, force_reply):
        if False:
            for i in range(10):
                print('nop')
        for attr in force_reply.__slots__:
            assert getattr(force_reply, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(force_reply)) == len(set(mro_slots(force_reply))), 'duplicate slot'

    def test_expected(self, force_reply):
        if False:
            for i in range(10):
                print('nop')
        assert force_reply.force_reply == self.force_reply
        assert force_reply.selective == self.selective
        assert force_reply.input_field_placeholder == self.input_field_placeholder

    def test_to_dict(self, force_reply):
        if False:
            i = 10
            return i + 15
        force_reply_dict = force_reply.to_dict()
        assert isinstance(force_reply_dict, dict)
        assert force_reply_dict['force_reply'] == force_reply.force_reply
        assert force_reply_dict['selective'] == force_reply.selective
        assert force_reply_dict['input_field_placeholder'] == force_reply.input_field_placeholder

    def test_equality(self):
        if False:
            while True:
                i = 10
        a = ForceReply(True, 'test')
        b = ForceReply(False, 'pass')
        c = ForceReply(True)
        d = ReplyKeyboardRemove()
        assert a != b
        assert hash(a) != hash(b)
        assert a == c
        assert hash(a) == hash(c)
        assert a != d
        assert hash(a) != hash(d)

class TestForceReplyWithRequest(TestForceReplyBase):

    async def test_send_message_with_force_reply(self, bot, chat_id, force_reply):
        message = await bot.send_message(chat_id, 'text', reply_markup=force_reply)
        assert message.text == 'text'