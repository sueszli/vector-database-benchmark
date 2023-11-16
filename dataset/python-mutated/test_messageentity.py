import pytest
from telegram import MessageEntity, User
from telegram.constants import MessageEntityType
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module', params=MessageEntity.ALL_TYPES)
def message_entity(request):
    if False:
        i = 10
        return i + 15
    type_ = request.param
    url = None
    if type_ == MessageEntity.TEXT_LINK:
        url = 't.me'
    user = None
    if type_ == MessageEntity.TEXT_MENTION:
        user = User(1, 'test_user', False)
    language = None
    if type_ == MessageEntity.PRE:
        language = 'python'
    return MessageEntity(type_, 1, 3, url=url, user=user, language=language)

class TestMessageEntityBase:
    type_ = 'url'
    offset = 1
    length = 2
    url = 'url'

class TestMessageEntityWithoutRequest(TestMessageEntityBase):

    def test_slot_behaviour(self, message_entity):
        if False:
            for i in range(10):
                print('nop')
        inst = message_entity
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_de_json(self, bot):
        if False:
            return 10
        json_dict = {'type': self.type_, 'offset': self.offset, 'length': self.length}
        entity = MessageEntity.de_json(json_dict, bot)
        assert entity.api_kwargs == {}
        assert entity.type == self.type_
        assert entity.offset == self.offset
        assert entity.length == self.length

    def test_to_dict(self, message_entity):
        if False:
            print('Hello World!')
        entity_dict = message_entity.to_dict()
        assert isinstance(entity_dict, dict)
        assert entity_dict['type'] == message_entity.type
        assert entity_dict['offset'] == message_entity.offset
        assert entity_dict['length'] == message_entity.length
        if message_entity.url:
            assert entity_dict['url'] == message_entity.url
        if message_entity.user:
            assert entity_dict['user'] == message_entity.user.to_dict()
        if message_entity.language:
            assert entity_dict['language'] == message_entity.language

    def test_enum_init(self):
        if False:
            i = 10
            return i + 15
        entity = MessageEntity(type='foo', offset=0, length=1)
        assert entity.type == 'foo'
        entity = MessageEntity(type='url', offset=0, length=1)
        assert entity.type is MessageEntityType.URL

    def test_equality(self):
        if False:
            return 10
        a = MessageEntity(MessageEntity.BOLD, 2, 3)
        b = MessageEntity(MessageEntity.BOLD, 2, 3)
        c = MessageEntity(MessageEntity.CODE, 2, 3)
        d = MessageEntity(MessageEntity.CODE, 5, 6)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)