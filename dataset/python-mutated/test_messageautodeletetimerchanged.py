from telegram import MessageAutoDeleteTimerChanged, VideoChatEnded
from tests.auxil.slots import mro_slots

class TestMessageAutoDeleteTimerChangedWithoutRequest:
    message_auto_delete_time = 100

    def test_slot_behaviour(self):
        if False:
            i = 10
            return i + 15
        action = MessageAutoDeleteTimerChanged(self.message_auto_delete_time)
        for attr in action.__slots__:
            assert getattr(action, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(action)) == len(set(mro_slots(action))), 'duplicate slot'

    def test_de_json(self):
        if False:
            i = 10
            return i + 15
        json_dict = {'message_auto_delete_time': self.message_auto_delete_time}
        madtc = MessageAutoDeleteTimerChanged.de_json(json_dict, None)
        assert madtc.api_kwargs == {}
        assert madtc.message_auto_delete_time == self.message_auto_delete_time

    def test_to_dict(self):
        if False:
            return 10
        madtc = MessageAutoDeleteTimerChanged(self.message_auto_delete_time)
        madtc_dict = madtc.to_dict()
        assert isinstance(madtc_dict, dict)
        assert madtc_dict['message_auto_delete_time'] == self.message_auto_delete_time

    def test_equality(self):
        if False:
            print('Hello World!')
        a = MessageAutoDeleteTimerChanged(100)
        b = MessageAutoDeleteTimerChanged(100)
        c = MessageAutoDeleteTimerChanged(50)
        d = VideoChatEnded(25)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)