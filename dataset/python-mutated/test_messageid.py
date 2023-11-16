import pytest
from telegram import MessageId, User
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def message_id():
    if False:
        return 10
    return MessageId(message_id=TestMessageIdWithoutRequest.m_id)

class TestMessageIdWithoutRequest:
    m_id = 1234

    def test_slot_behaviour(self, message_id):
        if False:
            print('Hello World!')
        for attr in message_id.__slots__:
            assert getattr(message_id, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(message_id)) == len(set(mro_slots(message_id))), 'duplicate slot'

    def test_de_json(self):
        if False:
            i = 10
            return i + 15
        json_dict = {'message_id': self.m_id}
        message_id = MessageId.de_json(json_dict, None)
        assert message_id.api_kwargs == {}
        assert message_id.message_id == self.m_id

    def test_to_dict(self, message_id):
        if False:
            while True:
                i = 10
        message_id_dict = message_id.to_dict()
        assert isinstance(message_id_dict, dict)
        assert message_id_dict['message_id'] == message_id.message_id

    def test_equality(self):
        if False:
            print('Hello World!')
        a = MessageId(message_id=1)
        b = MessageId(message_id=1)
        c = MessageId(message_id=2)
        d = User(id=1, first_name='name', is_bot=False)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)