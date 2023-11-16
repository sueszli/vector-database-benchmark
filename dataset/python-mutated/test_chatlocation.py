import pytest
from telegram import ChatLocation, Location, User
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def chat_location():
    if False:
        while True:
            i = 10
    return ChatLocation(TestChatLocationBase.location, TestChatLocationBase.address)

class TestChatLocationBase:
    location = Location(123, 456)
    address = 'The Shire'

class TestChatLocationWithoutRequest(TestChatLocationBase):

    def test_slot_behaviour(self, chat_location):
        if False:
            while True:
                i = 10
        inst = chat_location
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_de_json(self, bot):
        if False:
            print('Hello World!')
        json_dict = {'location': self.location.to_dict(), 'address': self.address}
        chat_location = ChatLocation.de_json(json_dict, bot)
        assert chat_location.api_kwargs == {}
        assert chat_location.location == self.location
        assert chat_location.address == self.address

    def test_to_dict(self, chat_location):
        if False:
            print('Hello World!')
        chat_location_dict = chat_location.to_dict()
        assert isinstance(chat_location_dict, dict)
        assert chat_location_dict['location'] == chat_location.location.to_dict()
        assert chat_location_dict['address'] == chat_location.address

    def test_equality(self, chat_location):
        if False:
            while True:
                i = 10
        a = chat_location
        b = ChatLocation(self.location, self.address)
        c = ChatLocation(self.location, 'Mordor')
        d = ChatLocation(Location(456, 132), self.address)
        e = User(456, '', False)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a == c
        assert hash(a) == hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)