import pytest
from telegram import User
from telegram._utils.defaultvalue import DefaultValue
from tests.auxil.slots import mro_slots

class TestDefaultValue:

    def test_slot_behaviour(self):
        if False:
            i = 10
            return i + 15
        inst = DefaultValue(1)
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_identity(self):
        if False:
            return 10
        df_1 = DefaultValue(1)
        df_2 = DefaultValue(2)
        assert df_1 is not df_2
        assert df_1 != df_2

    @pytest.mark.parametrize(('value', 'expected'), [({}, False), ({1: 2}, True), (None, False), (True, True), (1, True), (0, False), (False, False), ([], False), ([1], True)])
    def test_truthiness(self, value, expected):
        if False:
            return 10
        assert bool(DefaultValue(value)) == expected

    @pytest.mark.parametrize('value', ['string', 1, True, [1, 2, 3], {1: 3}, DefaultValue(1), User(1, 'first', False)])
    def test_string_representations(self, value):
        if False:
            i = 10
            return i + 15
        df = DefaultValue(value)
        assert str(df) == f'DefaultValue({value})'
        assert repr(df) == repr(value)

    def test_as_function_argument(self):
        if False:
            for i in range(10):
                print('nop')
        default_one = DefaultValue(1)

        def foo(arg=default_one):
            if False:
                return 10
            if arg is default_one:
                return 1
            return 2
        assert foo() == 1
        assert foo(None) == 2
        assert foo(1) == 2