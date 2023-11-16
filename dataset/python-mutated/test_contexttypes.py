import pytest
from telegram.ext import CallbackContext, ContextTypes
from tests.auxil.slots import mro_slots

class SubClass(CallbackContext):
    pass

class TestContextTypes:

    def test_slot_behaviour(self):
        if False:
            while True:
                i = 10
        instance = ContextTypes()
        for attr in instance.__slots__:
            assert getattr(instance, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(instance)) == len(set(mro_slots(instance))), 'duplicate slot'

    def test_data_init(self):
        if False:
            for i in range(10):
                print('nop')
        ct = ContextTypes(SubClass, int, float, bool)
        assert ct.context is SubClass
        assert ct.bot_data is int
        assert ct.chat_data is float
        assert ct.user_data is bool
        with pytest.raises(ValueError, match='subclass of CallbackContext'):
            ContextTypes(context=bool)

    def test_data_assignment(self):
        if False:
            i = 10
            return i + 15
        ct = ContextTypes()
        with pytest.raises(AttributeError):
            ct.bot_data = bool
        with pytest.raises(AttributeError):
            ct.user_data = bool
        with pytest.raises(AttributeError):
            ct.chat_data = bool