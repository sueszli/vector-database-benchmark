import pytest
from telegram import PassportElementErrorReverseSide, PassportElementErrorSelfie
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def passport_element_error_reverse_side():
    if False:
        for i in range(10):
            print('nop')
    return PassportElementErrorReverseSide(TestPassportElementErrorReverseSideBase.type_, TestPassportElementErrorReverseSideBase.file_hash, TestPassportElementErrorReverseSideBase.message)

class TestPassportElementErrorReverseSideBase:
    source = 'reverse_side'
    type_ = 'test_type'
    file_hash = 'file_hash'
    message = 'Error message'

class TestPassportElementErrorReverseSideWithoutRequest(TestPassportElementErrorReverseSideBase):

    def test_slot_behaviour(self, passport_element_error_reverse_side):
        if False:
            i = 10
            return i + 15
        inst = passport_element_error_reverse_side
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, passport_element_error_reverse_side):
        if False:
            print('Hello World!')
        assert passport_element_error_reverse_side.source == self.source
        assert passport_element_error_reverse_side.type == self.type_
        assert passport_element_error_reverse_side.file_hash == self.file_hash
        assert passport_element_error_reverse_side.message == self.message

    def test_to_dict(self, passport_element_error_reverse_side):
        if False:
            i = 10
            return i + 15
        passport_element_error_reverse_side_dict = passport_element_error_reverse_side.to_dict()
        assert isinstance(passport_element_error_reverse_side_dict, dict)
        assert passport_element_error_reverse_side_dict['source'] == passport_element_error_reverse_side.source
        assert passport_element_error_reverse_side_dict['type'] == passport_element_error_reverse_side.type
        assert passport_element_error_reverse_side_dict['file_hash'] == passport_element_error_reverse_side.file_hash
        assert passport_element_error_reverse_side_dict['message'] == passport_element_error_reverse_side.message

    def test_equality(self):
        if False:
            while True:
                i = 10
        a = PassportElementErrorReverseSide(self.type_, self.file_hash, self.message)
        b = PassportElementErrorReverseSide(self.type_, self.file_hash, self.message)
        c = PassportElementErrorReverseSide(self.type_, '', '')
        d = PassportElementErrorReverseSide('', self.file_hash, '')
        e = PassportElementErrorReverseSide('', '', self.message)
        f = PassportElementErrorSelfie(self.type_, self.file_hash, self.message)
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a != c
        assert hash(a) != hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)
        assert a != f
        assert hash(a) != hash(f)