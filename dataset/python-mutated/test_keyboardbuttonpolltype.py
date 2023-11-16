import pytest
from telegram import KeyboardButtonPollType, Poll
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def keyboard_button_poll_type():
    if False:
        i = 10
        return i + 15
    return KeyboardButtonPollType(TestKeyboardButtonPollTypeBase.type)

class TestKeyboardButtonPollTypeBase:
    type = Poll.QUIZ

class TestKeyboardButtonPollTypeWithoutRequest(TestKeyboardButtonPollTypeBase):

    def test_slot_behaviour(self, keyboard_button_poll_type):
        if False:
            print('Hello World!')
        inst = keyboard_button_poll_type
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_to_dict(self, keyboard_button_poll_type):
        if False:
            while True:
                i = 10
        keyboard_button_poll_type_dict = keyboard_button_poll_type.to_dict()
        assert isinstance(keyboard_button_poll_type_dict, dict)
        assert keyboard_button_poll_type_dict['type'] == self.type

    def test_equality(self):
        if False:
            print('Hello World!')
        a = KeyboardButtonPollType(Poll.QUIZ)
        b = KeyboardButtonPollType(Poll.QUIZ)
        c = KeyboardButtonPollType(Poll.REGULAR)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)