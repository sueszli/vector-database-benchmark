import pytest
from telegram import BotName
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def bot_name(bot):
    if False:
        print('Hello World!')
    return BotName(TestBotNameBase.name)

class TestBotNameBase:
    name = 'This is a test name'

class TestBotNameWithoutRequest(TestBotNameBase):

    def test_slot_behaviour(self, bot_name):
        if False:
            i = 10
            return i + 15
        for attr in bot_name.__slots__:
            assert getattr(bot_name, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(bot_name)) == len(set(mro_slots(bot_name))), 'duplicate slot'

    def test_to_dict(self, bot_name):
        if False:
            print('Hello World!')
        bot_name_dict = bot_name.to_dict()
        assert isinstance(bot_name_dict, dict)
        assert bot_name_dict['name'] == self.name

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        a = BotName(self.name)
        b = BotName(self.name)
        c = BotName('text.com')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)