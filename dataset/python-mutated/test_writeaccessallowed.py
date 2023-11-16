from telegram import WriteAccessAllowed
from tests.auxil.slots import mro_slots

class TestWriteAccessAllowed:

    def test_slot_behaviour(self):
        if False:
            i = 10
            return i + 15
        action = WriteAccessAllowed()
        for attr in action.__slots__:
            assert getattr(action, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(action)) == len(set(mro_slots(action))), 'duplicate slot'

    def test_de_json(self):
        if False:
            for i in range(10):
                print('nop')
        action = WriteAccessAllowed.de_json({}, None)
        assert action.api_kwargs == {}
        assert isinstance(action, WriteAccessAllowed)

    def test_to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        action = WriteAccessAllowed()
        action_dict = action.to_dict()
        assert action_dict == {}

    def test_equality(self):
        if False:
            return 10
        a = WriteAccessAllowed()
        b = WriteAccessAllowed()
        c = WriteAccessAllowed(web_app_name='foo')
        d = WriteAccessAllowed(web_app_name='foo')
        e = WriteAccessAllowed(web_app_name='bar')
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert c == d
        assert hash(c) == hash(d)
        assert c != e
        assert hash(c) != hash(e)