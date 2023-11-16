import pytest
from telegram import ReplyKeyboardRemove
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def reply_keyboard_remove():
    if False:
        i = 10
        return i + 15
    return ReplyKeyboardRemove(selective=TestReplyKeyboardRemoveBase.selective)

class TestReplyKeyboardRemoveBase:
    remove_keyboard = True
    selective = True

class TestReplyKeyboardRemoveWithoutRequest(TestReplyKeyboardRemoveBase):

    def test_slot_behaviour(self, reply_keyboard_remove):
        if False:
            return 10
        inst = reply_keyboard_remove
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, reply_keyboard_remove):
        if False:
            i = 10
            return i + 15
        assert reply_keyboard_remove.remove_keyboard == self.remove_keyboard
        assert reply_keyboard_remove.selective == self.selective

    def test_to_dict(self, reply_keyboard_remove):
        if False:
            for i in range(10):
                print('nop')
        reply_keyboard_remove_dict = reply_keyboard_remove.to_dict()
        assert reply_keyboard_remove_dict['remove_keyboard'] == reply_keyboard_remove.remove_keyboard
        assert reply_keyboard_remove_dict['selective'] == reply_keyboard_remove.selective

class TestReplyKeyboardRemoveWithRequest(TestReplyKeyboardRemoveBase):

    async def test_send_message_with_reply_keyboard_remove(self, bot, chat_id, reply_keyboard_remove):
        message = await bot.send_message(chat_id, 'Text', reply_markup=reply_keyboard_remove)
        assert message.text == 'Text'