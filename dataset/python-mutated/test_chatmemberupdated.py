import datetime
import inspect
import pytest
from telegram import Chat, ChatInviteLink, ChatMember, ChatMemberAdministrator, ChatMemberBanned, ChatMemberOwner, ChatMemberUpdated, User
from telegram._utils.datetime import UTC, to_timestamp
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def user():
    if False:
        print('Hello World!')
    return User(1, 'First name', False)

@pytest.fixture(scope='module')
def chat():
    if False:
        i = 10
        return i + 15
    return Chat(1, Chat.SUPERGROUP, 'Chat')

@pytest.fixture(scope='module')
def old_chat_member(user):
    if False:
        i = 10
        return i + 15
    return ChatMember(user, TestChatMemberUpdatedBase.old_status)

@pytest.fixture(scope='module')
def new_chat_member(user):
    if False:
        print('Hello World!')
    return ChatMemberAdministrator(user, TestChatMemberUpdatedBase.new_status, True, True, True, True, True, True, True, True, True)

@pytest.fixture(scope='module')
def time():
    if False:
        i = 10
        return i + 15
    return datetime.datetime.now(tz=UTC)

@pytest.fixture(scope='module')
def invite_link(user):
    if False:
        print('Hello World!')
    return ChatInviteLink('link', user, False, True, True)

@pytest.fixture(scope='module')
def chat_member_updated(user, chat, old_chat_member, new_chat_member, invite_link, time):
    if False:
        i = 10
        return i + 15
    return ChatMemberUpdated(chat, user, time, old_chat_member, new_chat_member, invite_link, True)

class TestChatMemberUpdatedBase:
    old_status = ChatMember.MEMBER
    new_status = ChatMember.ADMINISTRATOR

class TestChatMemberUpdatedWithoutRequest(TestChatMemberUpdatedBase):

    def test_slot_behaviour(self, chat_member_updated):
        if False:
            i = 10
            return i + 15
        action = chat_member_updated
        for attr in action.__slots__:
            assert getattr(action, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(action)) == len(set(mro_slots(action))), 'duplicate slot'

    def test_de_json_required_args(self, bot, user, chat, old_chat_member, new_chat_member, time):
        if False:
            i = 10
            return i + 15
        json_dict = {'chat': chat.to_dict(), 'from': user.to_dict(), 'date': to_timestamp(time), 'old_chat_member': old_chat_member.to_dict(), 'new_chat_member': new_chat_member.to_dict()}
        chat_member_updated = ChatMemberUpdated.de_json(json_dict, bot)
        assert chat_member_updated.api_kwargs == {}
        assert chat_member_updated.chat == chat
        assert chat_member_updated.from_user == user
        assert abs(chat_member_updated.date - time) < datetime.timedelta(seconds=1)
        assert to_timestamp(chat_member_updated.date) == to_timestamp(time)
        assert chat_member_updated.old_chat_member == old_chat_member
        assert chat_member_updated.new_chat_member == new_chat_member
        assert chat_member_updated.invite_link is None
        assert chat_member_updated.via_chat_folder_invite_link is None

    def test_de_json_all_args(self, bot, user, time, invite_link, chat, old_chat_member, new_chat_member):
        if False:
            return 10
        json_dict = {'chat': chat.to_dict(), 'from': user.to_dict(), 'date': to_timestamp(time), 'old_chat_member': old_chat_member.to_dict(), 'new_chat_member': new_chat_member.to_dict(), 'invite_link': invite_link.to_dict(), 'via_chat_folder_invite_link': True}
        chat_member_updated = ChatMemberUpdated.de_json(json_dict, bot)
        assert chat_member_updated.api_kwargs == {}
        assert chat_member_updated.chat == chat
        assert chat_member_updated.from_user == user
        assert abs(chat_member_updated.date - time) < datetime.timedelta(seconds=1)
        assert to_timestamp(chat_member_updated.date) == to_timestamp(time)
        assert chat_member_updated.old_chat_member == old_chat_member
        assert chat_member_updated.new_chat_member == new_chat_member
        assert chat_member_updated.invite_link == invite_link
        assert chat_member_updated.via_chat_folder_invite_link is True

    def test_de_json_localization(self, bot, raw_bot, tz_bot, user, chat, old_chat_member, new_chat_member, time, invite_link):
        if False:
            return 10
        json_dict = {'chat': chat.to_dict(), 'from': user.to_dict(), 'date': to_timestamp(time), 'old_chat_member': old_chat_member.to_dict(), 'new_chat_member': new_chat_member.to_dict(), 'invite_link': invite_link.to_dict()}
        chat_member_updated_bot = ChatMemberUpdated.de_json(json_dict, bot)
        chat_member_updated_raw = ChatMemberUpdated.de_json(json_dict, raw_bot)
        chat_member_updated_tz = ChatMemberUpdated.de_json(json_dict, tz_bot)
        message_offset = chat_member_updated_tz.date.utcoffset()
        tz_bot_offset = tz_bot.defaults.tzinfo.utcoffset(chat_member_updated_tz.date.replace(tzinfo=None))
        assert chat_member_updated_raw.date.tzinfo == UTC
        assert chat_member_updated_bot.date.tzinfo == UTC
        assert message_offset == tz_bot_offset

    def test_to_dict(self, chat_member_updated):
        if False:
            i = 10
            return i + 15
        chat_member_updated_dict = chat_member_updated.to_dict()
        assert isinstance(chat_member_updated_dict, dict)
        assert chat_member_updated_dict['chat'] == chat_member_updated.chat.to_dict()
        assert chat_member_updated_dict['from'] == chat_member_updated.from_user.to_dict()
        assert chat_member_updated_dict['date'] == to_timestamp(chat_member_updated.date)
        assert chat_member_updated_dict['old_chat_member'] == chat_member_updated.old_chat_member.to_dict()
        assert chat_member_updated_dict['new_chat_member'] == chat_member_updated.new_chat_member.to_dict()
        assert chat_member_updated_dict['invite_link'] == chat_member_updated.invite_link.to_dict()
        assert chat_member_updated_dict['via_chat_folder_invite_link'] == chat_member_updated.via_chat_folder_invite_link

    def test_equality(self, time, old_chat_member, new_chat_member, invite_link):
        if False:
            print('Hello World!')
        a = ChatMemberUpdated(Chat(1, 'chat'), User(1, '', False), time, old_chat_member, new_chat_member, invite_link)
        b = ChatMemberUpdated(Chat(1, 'chat'), User(1, '', False), time, old_chat_member, new_chat_member)
        c = ChatMemberUpdated(Chat(1, 'chat'), User(1, '', False), time + datetime.timedelta(hours=1), old_chat_member, new_chat_member)
        d = ChatMemberUpdated(Chat(42, 'wrong_chat'), User(42, 'wrong_user', False), time, old_chat_member, new_chat_member)
        e = ChatMemberUpdated(Chat(1, 'chat'), User(1, '', False), time, ChatMember(User(1, '', False), ChatMember.OWNER), new_chat_member)
        f = ChatMemberUpdated(Chat(1, 'chat'), User(1, '', False), time, old_chat_member, ChatMember(User(1, '', False), ChatMember.OWNER))
        g = ChatMember(User(1, '', False), ChatMember.OWNER)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        for other in [c, d, e, f, g]:
            assert a != other
            assert hash(a) != hash(other)

    def test_difference_required(self, user, chat):
        if False:
            for i in range(10):
                print('nop')
        old_chat_member = ChatMember(user, 'old_status')
        new_chat_member = ChatMember(user, 'new_status')
        chat_member_updated = ChatMemberUpdated(chat, user, datetime.datetime.utcnow(), old_chat_member, new_chat_member)
        assert chat_member_updated.difference() == {'status': ('old_status', 'new_status')}
        new_user = User(1, 'First name', False, last_name='last name')
        new_chat_member = ChatMember(new_user, 'new_status')
        chat_member_updated = ChatMemberUpdated(chat, user, datetime.datetime.utcnow(), old_chat_member, new_chat_member)
        assert chat_member_updated.difference() == {'status': ('old_status', 'new_status'), 'user': (user, new_user)}

    @pytest.mark.parametrize('optional_attribute', [name for (name, param) in inspect.signature(ChatMemberAdministrator).parameters.items() if name not in ['self', 'api_kwargs'] and param.default != inspect.Parameter.empty])
    def test_difference_optionals(self, optional_attribute, user, chat):
        if False:
            while True:
                i = 10
        old_value = 'old_value'
        new_value = 'new_value'
        trues = tuple((True for _ in range(9)))
        old_chat_member = ChatMemberAdministrator(user, *trues, **{optional_attribute: old_value})
        new_chat_member = ChatMemberAdministrator(user, *trues, **{optional_attribute: new_value})
        chat_member_updated = ChatMemberUpdated(chat, user, datetime.datetime.utcnow(), old_chat_member, new_chat_member)
        assert chat_member_updated.difference() == {optional_attribute: (old_value, new_value)}

    def test_difference_different_classes(self, user, chat):
        if False:
            return 10
        old_chat_member = ChatMemberOwner(user=user, is_anonymous=False)
        new_chat_member = ChatMemberBanned(user=user, until_date=datetime.datetime(2021, 1, 1))
        chat_member_updated = ChatMemberUpdated(chat=chat, from_user=user, date=datetime.datetime.utcnow(), old_chat_member=old_chat_member, new_chat_member=new_chat_member)
        diff = chat_member_updated.difference()
        assert diff.pop('is_anonymous') == (False, None)
        assert diff.pop('until_date') == (None, datetime.datetime(2021, 1, 1))
        assert diff.pop('status') == (ChatMember.OWNER, ChatMember.BANNED)
        assert diff == {}