import pytest
from telegram import SwitchInlineQueryChosenChat
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def switch_inline_query_chosen_chat():
    if False:
        while True:
            i = 10
    return SwitchInlineQueryChosenChat(query=TestSwitchInlineQueryChosenChatBase.query, allow_user_chats=TestSwitchInlineQueryChosenChatBase.allow_user_chats, allow_bot_chats=TestSwitchInlineQueryChosenChatBase.allow_bot_chats, allow_channel_chats=TestSwitchInlineQueryChosenChatBase.allow_channel_chats, allow_group_chats=TestSwitchInlineQueryChosenChatBase.allow_group_chats)

class TestSwitchInlineQueryChosenChatBase:
    query = 'query'
    allow_user_chats = True
    allow_bot_chats = True
    allow_channel_chats = False
    allow_group_chats = True

class TestSwitchInlineQueryChosenChat(TestSwitchInlineQueryChosenChatBase):

    def test_slot_behaviour(self, switch_inline_query_chosen_chat):
        if False:
            print('Hello World!')
        inst = switch_inline_query_chosen_chat
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, switch_inline_query_chosen_chat):
        if False:
            for i in range(10):
                print('nop')
        assert switch_inline_query_chosen_chat.query == self.query
        assert switch_inline_query_chosen_chat.allow_user_chats == self.allow_user_chats
        assert switch_inline_query_chosen_chat.allow_bot_chats == self.allow_bot_chats
        assert switch_inline_query_chosen_chat.allow_channel_chats == self.allow_channel_chats
        assert switch_inline_query_chosen_chat.allow_group_chats == self.allow_group_chats

    def test_to_dict(self, switch_inline_query_chosen_chat):
        if False:
            i = 10
            return i + 15
        siqcc = switch_inline_query_chosen_chat.to_dict()
        assert isinstance(siqcc, dict)
        assert siqcc['query'] == switch_inline_query_chosen_chat.query
        assert siqcc['allow_user_chats'] == switch_inline_query_chosen_chat.allow_user_chats
        assert siqcc['allow_bot_chats'] == switch_inline_query_chosen_chat.allow_bot_chats
        assert siqcc['allow_channel_chats'] == switch_inline_query_chosen_chat.allow_channel_chats
        assert siqcc['allow_group_chats'] == switch_inline_query_chosen_chat.allow_group_chats

    def test_equality(self):
        if False:
            print('Hello World!')
        siqcc = SwitchInlineQueryChosenChat
        a = siqcc(self.query, self.allow_user_chats, self.allow_bot_chats)
        b = siqcc(self.query, self.allow_user_chats, self.allow_bot_chats)
        c = siqcc(self.query, self.allow_user_chats)
        d = siqcc('', self.allow_user_chats, self.allow_bot_chats)
        e = siqcc(self.query, self.allow_user_chats, self.allow_bot_chats, self.allow_group_chats)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)