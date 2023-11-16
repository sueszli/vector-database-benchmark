from copy import deepcopy
import pytest
from telegram import BotCommandScope, BotCommandScopeAllChatAdministrators, BotCommandScopeAllGroupChats, BotCommandScopeAllPrivateChats, BotCommandScopeChat, BotCommandScopeChatAdministrators, BotCommandScopeChatMember, BotCommandScopeDefault, Dice
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module', params=['str', 'int'])
def chat_id(request):
    if False:
        print('Hello World!')
    if request.param == 'str':
        return '@supergroupusername'
    return 43

@pytest.fixture(scope='class', params=[BotCommandScope.DEFAULT, BotCommandScope.ALL_PRIVATE_CHATS, BotCommandScope.ALL_GROUP_CHATS, BotCommandScope.ALL_CHAT_ADMINISTRATORS, BotCommandScope.CHAT, BotCommandScope.CHAT_ADMINISTRATORS, BotCommandScope.CHAT_MEMBER])
def scope_type(request):
    if False:
        return 10
    return request.param

@pytest.fixture(scope='module', params=[BotCommandScopeDefault, BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats, BotCommandScopeAllChatAdministrators, BotCommandScopeChat, BotCommandScopeChatAdministrators, BotCommandScopeChatMember], ids=[BotCommandScope.DEFAULT, BotCommandScope.ALL_PRIVATE_CHATS, BotCommandScope.ALL_GROUP_CHATS, BotCommandScope.ALL_CHAT_ADMINISTRATORS, BotCommandScope.CHAT, BotCommandScope.CHAT_ADMINISTRATORS, BotCommandScope.CHAT_MEMBER])
def scope_class(request):
    if False:
        while True:
            i = 10
    return request.param

@pytest.fixture(scope='module', params=[(BotCommandScopeDefault, BotCommandScope.DEFAULT), (BotCommandScopeAllPrivateChats, BotCommandScope.ALL_PRIVATE_CHATS), (BotCommandScopeAllGroupChats, BotCommandScope.ALL_GROUP_CHATS), (BotCommandScopeAllChatAdministrators, BotCommandScope.ALL_CHAT_ADMINISTRATORS), (BotCommandScopeChat, BotCommandScope.CHAT), (BotCommandScopeChatAdministrators, BotCommandScope.CHAT_ADMINISTRATORS), (BotCommandScopeChatMember, BotCommandScope.CHAT_MEMBER)], ids=[BotCommandScope.DEFAULT, BotCommandScope.ALL_PRIVATE_CHATS, BotCommandScope.ALL_GROUP_CHATS, BotCommandScope.ALL_CHAT_ADMINISTRATORS, BotCommandScope.CHAT, BotCommandScope.CHAT_ADMINISTRATORS, BotCommandScope.CHAT_MEMBER])
def scope_class_and_type(request):
    if False:
        print('Hello World!')
    return request.param

@pytest.fixture(scope='module')
def bot_command_scope(scope_class_and_type, chat_id):
    if False:
        while True:
            i = 10
    return scope_class_and_type[0].de_json({'type': scope_class_and_type[1], 'chat_id': chat_id, 'user_id': 42}, bot=None)

class TestBotCommandScopeWithoutRequest:

    def test_slot_behaviour(self, bot_command_scope):
        if False:
            for i in range(10):
                print('nop')
        for attr in bot_command_scope.__slots__:
            assert getattr(bot_command_scope, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(bot_command_scope)) == len(set(mro_slots(bot_command_scope))), 'duplicate slot'

    def test_de_json(self, bot, scope_class_and_type, chat_id):
        if False:
            for i in range(10):
                print('nop')
        cls = scope_class_and_type[0]
        type_ = scope_class_and_type[1]
        assert cls.de_json({}, bot) is None
        json_dict = {'type': type_, 'chat_id': chat_id, 'user_id': 42}
        bot_command_scope = BotCommandScope.de_json(json_dict, bot)
        assert set(bot_command_scope.api_kwargs.keys()) == {'chat_id', 'user_id'} - set(cls.__slots__)
        assert isinstance(bot_command_scope, BotCommandScope)
        assert isinstance(bot_command_scope, cls)
        assert bot_command_scope.type == type_
        if 'chat_id' in cls.__slots__:
            assert bot_command_scope.chat_id == chat_id
        if 'user_id' in cls.__slots__:
            assert bot_command_scope.user_id == 42

    def test_de_json_invalid_type(self, bot):
        if False:
            while True:
                i = 10
        json_dict = {'type': 'invalid', 'chat_id': chat_id, 'user_id': 42}
        bot_command_scope = BotCommandScope.de_json(json_dict, bot)
        assert type(bot_command_scope) is BotCommandScope
        assert bot_command_scope.type == 'invalid'

    def test_de_json_subclass(self, scope_class, bot, chat_id):
        if False:
            i = 10
            return i + 15
        'This makes sure that e.g. BotCommandScopeDefault(data) never returns a\n        BotCommandScopeChat instance.'
        json_dict = {'type': 'invalid', 'chat_id': chat_id, 'user_id': 42}
        assert type(scope_class.de_json(json_dict, bot)) is scope_class

    def test_to_dict(self, bot_command_scope):
        if False:
            for i in range(10):
                print('nop')
        bot_command_scope_dict = bot_command_scope.to_dict()
        assert isinstance(bot_command_scope_dict, dict)
        assert bot_command_scope['type'] == bot_command_scope.type
        if hasattr(bot_command_scope, 'chat_id'):
            assert bot_command_scope['chat_id'] == bot_command_scope.chat_id
        if hasattr(bot_command_scope, 'user_id'):
            assert bot_command_scope['user_id'] == bot_command_scope.user_id

    def test_equality(self, bot_command_scope, bot):
        if False:
            for i in range(10):
                print('nop')
        a = BotCommandScope('base_type')
        b = BotCommandScope('base_type')
        c = bot_command_scope
        d = deepcopy(bot_command_scope)
        e = Dice(4, 'emoji')
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)
        assert c == d
        assert hash(c) == hash(d)
        assert c != e
        assert hash(c) != hash(e)
        if hasattr(c, 'chat_id'):
            json_dict = c.to_dict()
            json_dict['chat_id'] = 0
            f = c.__class__.de_json(json_dict, bot)
            assert c != f
            assert hash(c) != hash(f)
        if hasattr(c, 'user_id'):
            json_dict = c.to_dict()
            json_dict['user_id'] = 0
            g = c.__class__.de_json(json_dict, bot)
            assert c != g
            assert hash(c) != hash(g)