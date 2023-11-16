import pytest
from telegram import PassportElementErrorDataField, PassportElementErrorUnspecified
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def passport_element_error_unspecified():
    if False:
        return 10
    return PassportElementErrorUnspecified(TestPassportElementErrorUnspecifiedBase.type_, TestPassportElementErrorUnspecifiedBase.element_hash, TestPassportElementErrorUnspecifiedBase.message)

class TestPassportElementErrorUnspecifiedBase:
    source = 'unspecified'
    type_ = 'test_type'
    element_hash = 'element_hash'
    message = 'Error message'

class TestPassportElementErrorUnspecifiedWithoutRequest(TestPassportElementErrorUnspecifiedBase):

    def test_slot_behaviour(self, passport_element_error_unspecified):
        if False:
            for i in range(10):
                print('nop')
        inst = passport_element_error_unspecified
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, passport_element_error_unspecified):
        if False:
            return 10
        assert passport_element_error_unspecified.source == self.source
        assert passport_element_error_unspecified.type == self.type_
        assert passport_element_error_unspecified.element_hash == self.element_hash
        assert passport_element_error_unspecified.message == self.message

    def test_to_dict(self, passport_element_error_unspecified):
        if False:
            while True:
                i = 10
        passport_element_error_unspecified_dict = passport_element_error_unspecified.to_dict()
        assert isinstance(passport_element_error_unspecified_dict, dict)
        assert passport_element_error_unspecified_dict['source'] == passport_element_error_unspecified.source
        assert passport_element_error_unspecified_dict['type'] == passport_element_error_unspecified.type
        assert passport_element_error_unspecified_dict['element_hash'] == passport_element_error_unspecified.element_hash
        assert passport_element_error_unspecified_dict['message'] == passport_element_error_unspecified.message

    def test_equality(self):
        if False:
            print('Hello World!')
        a = PassportElementErrorUnspecified(self.type_, self.element_hash, self.message)
        b = PassportElementErrorUnspecified(self.type_, self.element_hash, self.message)
        c = PassportElementErrorUnspecified(self.type_, '', '')
        d = PassportElementErrorUnspecified('', self.element_hash, '')
        e = PassportElementErrorUnspecified('', '', self.message)
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