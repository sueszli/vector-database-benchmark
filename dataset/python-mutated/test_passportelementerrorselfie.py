import pytest
from telegram import PassportElementErrorDataField, PassportElementErrorSelfie
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def passport_element_error_selfie():
    if False:
        i = 10
        return i + 15
    return PassportElementErrorSelfie(TestPassportElementErrorSelfieBase.type_, TestPassportElementErrorSelfieBase.file_hash, TestPassportElementErrorSelfieBase.message)

class TestPassportElementErrorSelfieBase:
    source = 'selfie'
    type_ = 'test_type'
    file_hash = 'file_hash'
    message = 'Error message'

class TestPassportElementErrorSelfieWithoutRequest(TestPassportElementErrorSelfieBase):

    def test_slot_behaviour(self, passport_element_error_selfie):
        if False:
            print('Hello World!')
        inst = passport_element_error_selfie
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, passport_element_error_selfie):
        if False:
            while True:
                i = 10
        assert passport_element_error_selfie.source == self.source
        assert passport_element_error_selfie.type == self.type_
        assert passport_element_error_selfie.file_hash == self.file_hash
        assert passport_element_error_selfie.message == self.message

    def test_to_dict(self, passport_element_error_selfie):
        if False:
            while True:
                i = 10
        passport_element_error_selfie_dict = passport_element_error_selfie.to_dict()
        assert isinstance(passport_element_error_selfie_dict, dict)
        assert passport_element_error_selfie_dict['source'] == passport_element_error_selfie.source
        assert passport_element_error_selfie_dict['type'] == passport_element_error_selfie.type
        assert passport_element_error_selfie_dict['file_hash'] == passport_element_error_selfie.file_hash
        assert passport_element_error_selfie_dict['message'] == passport_element_error_selfie.message

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        a = PassportElementErrorSelfie(self.type_, self.file_hash, self.message)
        b = PassportElementErrorSelfie(self.type_, self.file_hash, self.message)
        c = PassportElementErrorSelfie(self.type_, '', '')
        d = PassportElementErrorSelfie('', self.file_hash, '')
        e = PassportElementErrorSelfie('', '', self.message)
        f = PassportElementErrorDataField(self.type_, '', '', self.message)
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