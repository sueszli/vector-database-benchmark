import pytest
from telegram import ChatAdministratorRights
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def chat_admin_rights():
    if False:
        i = 10
        return i + 15
    return ChatAdministratorRights(can_change_info=True, can_delete_messages=True, can_invite_users=True, can_pin_messages=True, can_promote_members=True, can_restrict_members=True, can_post_messages=True, can_edit_messages=True, can_manage_chat=True, can_manage_video_chats=True, can_manage_topics=True, is_anonymous=True, can_post_stories=True, can_edit_stories=True, can_delete_stories=True)

class TestChatAdministratorRightsWithoutRequest:

    def test_slot_behaviour(self, chat_admin_rights):
        if False:
            for i in range(10):
                print('nop')
        inst = chat_admin_rights
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_de_json(self, bot, chat_admin_rights):
        if False:
            i = 10
            return i + 15
        json_dict = {'can_change_info': True, 'can_delete_messages': True, 'can_invite_users': True, 'can_pin_messages': True, 'can_promote_members': True, 'can_restrict_members': True, 'can_post_messages': True, 'can_edit_messages': True, 'can_manage_chat': True, 'can_manage_video_chats': True, 'can_manage_topics': True, 'is_anonymous': True, 'can_post_stories': True, 'can_edit_stories': True, 'can_delete_stories': True}
        chat_administrator_rights_de = ChatAdministratorRights.de_json(json_dict, bot)
        assert chat_administrator_rights_de.api_kwargs == {}
        assert chat_admin_rights == chat_administrator_rights_de

    def test_to_dict(self, chat_admin_rights):
        if False:
            i = 10
            return i + 15
        car = chat_admin_rights
        admin_rights_dict = car.to_dict()
        assert isinstance(admin_rights_dict, dict)
        assert admin_rights_dict['can_change_info'] == car.can_change_info
        assert admin_rights_dict['can_delete_messages'] == car.can_delete_messages
        assert admin_rights_dict['can_invite_users'] == car.can_invite_users
        assert admin_rights_dict['can_pin_messages'] == car.can_pin_messages
        assert admin_rights_dict['can_promote_members'] == car.can_promote_members
        assert admin_rights_dict['can_restrict_members'] == car.can_restrict_members
        assert admin_rights_dict['can_post_messages'] == car.can_post_messages
        assert admin_rights_dict['can_edit_messages'] == car.can_edit_messages
        assert admin_rights_dict['can_manage_chat'] == car.can_manage_chat
        assert admin_rights_dict['is_anonymous'] == car.is_anonymous
        assert admin_rights_dict['can_manage_video_chats'] == car.can_manage_video_chats
        assert admin_rights_dict['can_manage_topics'] == car.can_manage_topics
        assert admin_rights_dict['can_post_stories'] == car.can_post_stories
        assert admin_rights_dict['can_edit_stories'] == car.can_edit_stories
        assert admin_rights_dict['can_delete_stories'] == car.can_delete_stories

    def test_equality(self):
        if False:
            return 10
        a = ChatAdministratorRights(True, *(False,) * 11)
        b = ChatAdministratorRights(True, *(False,) * 11)
        c = ChatAdministratorRights(*(False,) * 12)
        d = ChatAdministratorRights(True, True, *(False,) * 10)
        e = ChatAdministratorRights(True, True, *(False,) * 10)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert d == e
        assert hash(d) == hash(e)

    def test_all_rights(self):
        if False:
            while True:
                i = 10
        f = ChatAdministratorRights(True, True, True, True, True, True, True, True, True)
        t = ChatAdministratorRights.all_rights()
        assert dir(f) == dir(t)
        for key in t.__slots__:
            assert t[key] is True
        assert f != t

    def test_no_rights(self):
        if False:
            return 10
        f = ChatAdministratorRights(False, False, False, False, False, False, False, False, False)
        t = ChatAdministratorRights.no_rights()
        assert dir(f) == dir(t)
        for key in t.__slots__:
            assert t[key] is False
        assert f != t