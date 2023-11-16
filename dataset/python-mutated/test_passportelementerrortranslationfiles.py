import pytest
from telegram import PassportElementErrorSelfie, PassportElementErrorTranslationFiles
from telegram.warnings import PTBDeprecationWarning
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def passport_element_error_translation_files():
    if False:
        return 10
    return PassportElementErrorTranslationFiles(TestPassportElementErrorTranslationFilesBase.type_, TestPassportElementErrorTranslationFilesBase.file_hashes, TestPassportElementErrorTranslationFilesBase.message)

class TestPassportElementErrorTranslationFilesBase:
    source = 'translation_files'
    type_ = 'test_type'
    file_hashes = ['hash1', 'hash2']
    message = 'Error message'

class TestPassportElementErrorTranslationFilesWithoutRequest(TestPassportElementErrorTranslationFilesBase):

    def test_slot_behaviour(self, passport_element_error_translation_files):
        if False:
            return 10
        inst = passport_element_error_translation_files
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, passport_element_error_translation_files):
        if False:
            print('Hello World!')
        assert passport_element_error_translation_files.source == self.source
        assert passport_element_error_translation_files.type == self.type_
        assert isinstance(passport_element_error_translation_files.file_hashes, list)
        assert passport_element_error_translation_files.file_hashes == self.file_hashes
        assert passport_element_error_translation_files.message == self.message

    def test_to_dict(self, passport_element_error_translation_files):
        if False:
            print('Hello World!')
        passport_element_error_translation_files_dict = passport_element_error_translation_files.to_dict()
        assert isinstance(passport_element_error_translation_files_dict, dict)
        assert passport_element_error_translation_files_dict['source'] == passport_element_error_translation_files.source
        assert passport_element_error_translation_files_dict['type'] == passport_element_error_translation_files.type
        assert passport_element_error_translation_files_dict['message'] == passport_element_error_translation_files.message
        assert passport_element_error_translation_files_dict['file_hashes'] == passport_element_error_translation_files.file_hashes

    def test_equality(self):
        if False:
            return 10
        a = PassportElementErrorTranslationFiles(self.type_, self.file_hashes, self.message)
        b = PassportElementErrorTranslationFiles(self.type_, self.file_hashes, self.message)
        c = PassportElementErrorTranslationFiles(self.type_, '', '')
        d = PassportElementErrorTranslationFiles('', self.file_hashes, '')
        e = PassportElementErrorTranslationFiles('', '', self.message)
        f = PassportElementErrorSelfie(self.type_, '', self.message)
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

    def test_file_hashes_deprecated(self, passport_element_error_translation_files, recwarn):
        if False:
            while True:
                i = 10
        passport_element_error_translation_files.file_hashes
        assert len(recwarn) == 1
        assert 'The attribute `file_hashes` will return a tuple instead of a list in future major versions.' in str(recwarn[0].message)
        assert recwarn[0].category is PTBDeprecationWarning
        assert recwarn[0].filename == __file__