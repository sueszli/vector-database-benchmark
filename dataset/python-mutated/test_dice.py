import pytest
from telegram import BotCommand, Dice
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module', params=Dice.ALL_EMOJI)
def dice(request):
    if False:
        print('Hello World!')
    return Dice(value=5, emoji=request.param)

class TestDiceBase:
    value = 4

class TestDiceWithoutRequest(TestDiceBase):

    def test_slot_behaviour(self, dice):
        if False:
            print('Hello World!')
        for attr in dice.__slots__:
            assert getattr(dice, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(dice)) == len(set(mro_slots(dice))), 'duplicate slot'

    @pytest.mark.parametrize('emoji', Dice.ALL_EMOJI)
    def test_de_json(self, bot, emoji):
        if False:
            for i in range(10):
                print('nop')
        json_dict = {'value': self.value, 'emoji': emoji}
        dice = Dice.de_json(json_dict, bot)
        assert dice.api_kwargs == {}
        assert dice.value == self.value
        assert dice.emoji == emoji
        assert Dice.de_json(None, bot) is None

    def test_to_dict(self, dice):
        if False:
            print('Hello World!')
        dice_dict = dice.to_dict()
        assert isinstance(dice_dict, dict)
        assert dice_dict['value'] == dice.value
        assert dice_dict['emoji'] == dice.emoji

    def test_equality(self):
        if False:
            while True:
                i = 10
        a = Dice(3, '🎯')
        b = Dice(3, '🎯')
        c = Dice(3, '🎲')
        d = Dice(4, '🎯')
        e = BotCommand('start', 'description')
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)