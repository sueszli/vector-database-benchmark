import pytest
from telegram import PassportElementErrorDataField, PassportElementErrorSelfie
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def passport_element_error_data_field():
    if False:
        while True:
            i = 10
    return PassportElementErrorDataField(TestPassportElementErrorDataFieldBase.type_, TestPassportElementErrorDataFieldBase.field_name, TestPassportElementErrorDataFieldBase.data_hash, TestPassportElementErrorDataFieldBase.message)

class TestPassportElementErrorDataFieldBase:
    source = 'data'
    type_ = 'test_type'
    field_name = 'test_field'
    data_hash = 'data_hash'
    message = 'Error message'

class TestPassportElementErrorDataFieldWithoutRequest(TestPassportElementErrorDataFieldBase):

    def test_slot_behaviour(self, passport_element_error_data_field):
        if False:
            while True:
                i = 10
        inst = passport_element_error_data_field
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, passport_element_error_data_field):
        if False:
            while True:
                i = 10
        assert passport_element_error_data_field.source == self.source
        assert passport_element_error_data_field.type == self.type_
        assert passport_element_error_data_field.field_name == self.field_name
        assert passport_element_error_data_field.data_hash == self.data_hash
        assert passport_element_error_data_field.message == self.message

    def test_to_dict(self, passport_element_error_data_field):
        if False:
            print('Hello World!')
        passport_element_error_data_field_dict = passport_element_error_data_field.to_dict()
        assert isinstance(passport_element_error_data_field_dict, dict)
        assert passport_element_error_data_field_dict['source'] == passport_element_error_data_field.source
        assert passport_element_error_data_field_dict['type'] == passport_element_error_data_field.type
        assert passport_element_error_data_field_dict['field_name'] == passport_element_error_data_field.field_name
        assert passport_element_error_data_field_dict['data_hash'] == passport_element_error_data_field.data_hash
        assert passport_element_error_data_field_dict['message'] == passport_element_error_data_field.message

    def test_equality(self):
        if False:
            while True:
                i = 10
        a = PassportElementErrorDataField(self.type_, self.field_name, self.data_hash, self.message)
        b = PassportElementErrorDataField(self.type_, self.field_name, self.data_hash, self.message)
        c = PassportElementErrorDataField(self.type_, '', '', '')
        d = PassportElementErrorDataField('', self.field_name, '', '')
        e = PassportElementErrorDataField('', '', self.data_hash, '')
        f = PassportElementErrorDataField('', '', '', self.message)
        g = PassportElementErrorSelfie(self.type_, '', self.message)
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
        assert a != g
        assert hash(a) != hash(g)