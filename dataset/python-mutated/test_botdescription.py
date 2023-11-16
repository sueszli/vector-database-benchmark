import pytest
from telegram import BotDescription, BotShortDescription
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def bot_description(bot):
    if False:
        return 10
    return BotDescription(TestBotDescriptionBase.description)

@pytest.fixture(scope='module')
def bot_short_description(bot):
    if False:
        return 10
    return BotShortDescription(TestBotDescriptionBase.short_description)

class TestBotDescriptionBase:
    description = 'This is a test description'
    short_description = 'This is a test short description'

class TestBotDescriptionWithoutRequest(TestBotDescriptionBase):

    def test_slot_behaviour(self, bot_description):
        if False:
            return 10
        for attr in bot_description.__slots__:
            assert getattr(bot_description, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(bot_description)) == len(set(mro_slots(bot_description))), 'duplicate slot'

    def test_to_dict(self, bot_description):
        if False:
            while True:
                i = 10
        bot_description_dict = bot_description.to_dict()
        assert isinstance(bot_description_dict, dict)
        assert bot_description_dict['description'] == self.description

    def test_equality(self):
        if False:
            print('Hello World!')
        a = BotDescription(self.description)
        b = BotDescription(self.description)
        c = BotDescription('text.com')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)

class TestBotShortDescriptionWithoutRequest(TestBotDescriptionBase):

    def test_slot_behaviour(self, bot_short_description):
        if False:
            print('Hello World!')
        for attr in bot_short_description.__slots__:
            assert getattr(bot_short_description, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(bot_short_description)) == len(set(mro_slots(bot_short_description))), 'duplicate slot'

    def test_to_dict(self, bot_short_description):
        if False:
            while True:
                i = 10
        bot_short_description_dict = bot_short_description.to_dict()
        assert isinstance(bot_short_description_dict, dict)
        assert bot_short_description_dict['short_description'] == self.short_description

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        a = BotShortDescription(self.short_description)
        b = BotShortDescription(self.short_description)
        c = BotShortDescription('text.com')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)