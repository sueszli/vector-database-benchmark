import pytest
from telegram import PassportElementErrorFrontSide, PassportElementErrorSelfie
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def passport_element_error_front_side():
    if False:
        print('Hello World!')
    return PassportElementErrorFrontSide(TestPassportElementErrorFrontSideBase.type_, TestPassportElementErrorFrontSideBase.file_hash, TestPassportElementErrorFrontSideBase.message)

class TestPassportElementErrorFrontSideBase:
    source = 'front_side'
    type_ = 'test_type'
    file_hash = 'file_hash'
    message = 'Error message'

class TestPassportElementErrorFrontSideWithoutRequest(TestPassportElementErrorFrontSideBase):

    def test_slot_behaviour(self, passport_element_error_front_side):
        if False:
            return 10
        inst = passport_element_error_front_side
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, passport_element_error_front_side):
        if False:
            while True:
                i = 10
        assert passport_element_error_front_side.source == self.source
        assert passport_element_error_front_side.type == self.type_
        assert passport_element_error_front_side.file_hash == self.file_hash
        assert passport_element_error_front_side.message == self.message

    def test_to_dict(self, passport_element_error_front_side):
        if False:
            for i in range(10):
                print('nop')
        passport_element_error_front_side_dict = passport_element_error_front_side.to_dict()
        assert isinstance(passport_element_error_front_side_dict, dict)
        assert passport_element_error_front_side_dict['source'] == passport_element_error_front_side.source
        assert passport_element_error_front_side_dict['type'] == passport_element_error_front_side.type
        assert passport_element_error_front_side_dict['file_hash'] == passport_element_error_front_side.file_hash
        assert passport_element_error_front_side_dict['message'] == passport_element_error_front_side.message

    def test_equality(self):
        if False:
            while True:
                i = 10
        a = PassportElementErrorFrontSide(self.type_, self.file_hash, self.message)
        b = PassportElementErrorFrontSide(self.type_, self.file_hash, self.message)
        c = PassportElementErrorFrontSide(self.type_, '', '')
        d = PassportElementErrorFrontSide('', self.file_hash, '')
        e = PassportElementErrorFrontSide('', '', self.message)
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