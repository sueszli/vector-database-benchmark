import pytest
from telegram import ChatPermissions, User
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def chat_permissions():
    if False:
        return 10
    return ChatPermissions(can_send_messages=True, can_send_polls=True, can_send_other_messages=True, can_add_web_page_previews=True, can_change_info=True, can_invite_users=True, can_pin_messages=True, can_manage_topics=True, can_send_audios=True, can_send_documents=True, can_send_photos=True, can_send_videos=True, can_send_video_notes=True, can_send_voice_notes=True)

class TestChatPermissionsBase:
    can_send_messages = True
    can_send_polls = True
    can_send_other_messages = False
    can_add_web_page_previews = False
    can_change_info = False
    can_invite_users = None
    can_pin_messages = None
    can_manage_topics = None
    can_send_audios = True
    can_send_documents = False
    can_send_photos = None
    can_send_videos = True
    can_send_video_notes = False
    can_send_voice_notes = None

class TestChatPermissionsWithoutRequest(TestChatPermissionsBase):

    def test_slot_behaviour(self, chat_permissions):
        if False:
            print('Hello World!')
        inst = chat_permissions
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_de_json(self, bot):
        if False:
            while True:
                i = 10
        json_dict = {'can_send_messages': self.can_send_messages, 'can_send_media_messages': 'can_send_media_messages', 'can_send_polls': self.can_send_polls, 'can_send_other_messages': self.can_send_other_messages, 'can_add_web_page_previews': self.can_add_web_page_previews, 'can_change_info': self.can_change_info, 'can_invite_users': self.can_invite_users, 'can_pin_messages': self.can_pin_messages, 'can_send_audios': self.can_send_audios, 'can_send_documents': self.can_send_documents, 'can_send_photos': self.can_send_photos, 'can_send_videos': self.can_send_videos, 'can_send_video_notes': self.can_send_video_notes, 'can_send_voice_notes': self.can_send_voice_notes}
        permissions = ChatPermissions.de_json(json_dict, bot)
        assert permissions.api_kwargs == {'can_send_media_messages': 'can_send_media_messages'}
        assert permissions.can_send_messages == self.can_send_messages
        assert permissions.can_send_polls == self.can_send_polls
        assert permissions.can_send_other_messages == self.can_send_other_messages
        assert permissions.can_add_web_page_previews == self.can_add_web_page_previews
        assert permissions.can_change_info == self.can_change_info
        assert permissions.can_invite_users == self.can_invite_users
        assert permissions.can_pin_messages == self.can_pin_messages
        assert permissions.can_manage_topics == self.can_manage_topics
        assert permissions.can_send_audios == self.can_send_audios
        assert permissions.can_send_documents == self.can_send_documents
        assert permissions.can_send_photos == self.can_send_photos
        assert permissions.can_send_videos == self.can_send_videos
        assert permissions.can_send_video_notes == self.can_send_video_notes
        assert permissions.can_send_voice_notes == self.can_send_voice_notes

    def test_to_dict(self, chat_permissions):
        if False:
            i = 10
            return i + 15
        permissions_dict = chat_permissions.to_dict()
        assert isinstance(permissions_dict, dict)
        assert permissions_dict['can_send_messages'] == chat_permissions.can_send_messages
        assert permissions_dict['can_send_polls'] == chat_permissions.can_send_polls
        assert permissions_dict['can_send_other_messages'] == chat_permissions.can_send_other_messages
        assert permissions_dict['can_add_web_page_previews'] == chat_permissions.can_add_web_page_previews
        assert permissions_dict['can_change_info'] == chat_permissions.can_change_info
        assert permissions_dict['can_invite_users'] == chat_permissions.can_invite_users
        assert permissions_dict['can_pin_messages'] == chat_permissions.can_pin_messages
        assert permissions_dict['can_manage_topics'] == chat_permissions.can_manage_topics
        assert permissions_dict['can_send_audios'] == chat_permissions.can_send_audios
        assert permissions_dict['can_send_documents'] == chat_permissions.can_send_documents
        assert permissions_dict['can_send_photos'] == chat_permissions.can_send_photos
        assert permissions_dict['can_send_videos'] == chat_permissions.can_send_videos
        assert permissions_dict['can_send_video_notes'] == chat_permissions.can_send_video_notes
        assert permissions_dict['can_send_voice_notes'] == chat_permissions.can_send_voice_notes

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        a = ChatPermissions(can_send_messages=True, can_send_polls=True, can_send_other_messages=False)
        b = ChatPermissions(can_send_polls=True, can_send_other_messages=False, can_send_messages=True)
        c = ChatPermissions(can_send_messages=False, can_send_polls=True, can_send_other_messages=False)
        d = User(123, '', False)
        e = ChatPermissions(can_send_messages=True, can_send_polls=True, can_send_other_messages=False, can_send_audios=True, can_send_documents=True, can_send_photos=True, can_send_videos=True, can_send_video_notes=True, can_send_voice_notes=True)
        f = ChatPermissions(can_send_messages=True, can_send_polls=True, can_send_other_messages=False, can_send_audios=True, can_send_documents=True, can_send_photos=True, can_send_videos=True, can_send_video_notes=True, can_send_voice_notes=True)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)
        assert e == f
        assert hash(e) == hash(f)

    def test_all_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        f = ChatPermissions()
        t = ChatPermissions.all_permissions()
        assert dir(f) == dir(t)
        for key in t.__slots__:
            assert t[key] is True
        assert f != t

    def test_no_permissions(self):
        if False:
            return 10
        f = ChatPermissions()
        t = ChatPermissions.no_permissions()
        assert dir(f) == dir(t)
        for key in t.__slots__:
            assert t[key] is False
        assert f != t