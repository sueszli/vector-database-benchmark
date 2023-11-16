import pytest
from telegram import BotCommand, Dice
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def bot_command():
    if False:
        for i in range(10):
            print('nop')
    return BotCommand(command='start', description='A command')

class TestBotCommandWithoutRequest:
    command = 'start'
    description = 'A command'

    def test_slot_behaviour(self, bot_command):
        if False:
            for i in range(10):
                print('nop')
        for attr in bot_command.__slots__:
            assert getattr(bot_command, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(bot_command)) == len(set(mro_slots(bot_command))), 'duplicate slot'

    def test_de_json(self, bot):
        if False:
            i = 10
            return i + 15
        json_dict = {'command': self.command, 'description': self.description}
        bot_command = BotCommand.de_json(json_dict, bot)
        assert bot_command.api_kwargs == {}
        assert bot_command.command == self.command
        assert bot_command.description == self.description
        assert BotCommand.de_json(None, bot) is None

    def test_to_dict(self, bot_command):
        if False:
            print('Hello World!')
        bot_command_dict = bot_command.to_dict()
        assert isinstance(bot_command_dict, dict)
        assert bot_command_dict['command'] == bot_command.command
        assert bot_command_dict['description'] == bot_command.description

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        a = BotCommand('start', 'some description')
        b = BotCommand('start', 'some description')
        c = BotCommand('start', 'some other description')
        d = BotCommand('hepl', 'some description')
        e = Dice(4, 'emoji')
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)