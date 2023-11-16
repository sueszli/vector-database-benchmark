import pytest
from telegram import Bot, File, PassportElementError, PassportFile
from telegram.warnings import PTBDeprecationWarning
from tests.auxil.bot_method_checks import check_defaults_handling, check_shortcut_call, check_shortcut_signature
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='class')
def passport_file(bot):
    if False:
        while True:
            i = 10
    pf = PassportFile(file_id=TestPassportFileBase.file_id, file_unique_id=TestPassportFileBase.file_unique_id, file_size=TestPassportFileBase.file_size, file_date=TestPassportFileBase.file_date)
    pf.set_bot(bot)
    return pf

class TestPassportFileBase:
    file_id = 'data'
    file_unique_id = 'adc3145fd2e84d95b64d68eaa22aa33e'
    file_size = 50
    file_date = 1532879128

class TestPassportFileWithoutRequest(TestPassportFileBase):

    def test_slot_behaviour(self, passport_file):
        if False:
            while True:
                i = 10
        inst = passport_file
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, passport_file):
        if False:
            i = 10
            return i + 15
        assert passport_file.file_id == self.file_id
        assert passport_file.file_unique_id == self.file_unique_id
        assert passport_file.file_size == self.file_size
        assert passport_file.file_date == self.file_date

    def test_to_dict(self, passport_file):
        if False:
            print('Hello World!')
        passport_file_dict = passport_file.to_dict()
        assert isinstance(passport_file_dict, dict)
        assert passport_file_dict['file_id'] == passport_file.file_id
        assert passport_file_dict['file_unique_id'] == passport_file.file_unique_id
        assert passport_file_dict['file_size'] == passport_file.file_size
        assert passport_file_dict['file_date'] == passport_file.file_date

    def test_equality(self):
        if False:
            return 10
        a = PassportFile(self.file_id, self.file_unique_id, self.file_size, self.file_date)
        b = PassportFile('', self.file_unique_id, self.file_size, self.file_date)
        c = PassportFile(self.file_id, self.file_unique_id, '', '')
        d = PassportFile('', '', self.file_size, self.file_date)
        e = PassportElementError('source', 'type', 'message')
        assert a == b
        assert hash(a) == hash(b)
        assert a is not b
        assert a == c
        assert hash(a) == hash(c)
        assert a != d
        assert hash(a) != hash(d)
        assert a != e
        assert hash(a) != hash(e)

    def test_file_date_deprecated(self, passport_file, recwarn):
        if False:
            i = 10
            return i + 15
        passport_file.file_date
        assert len(recwarn) == 1
        assert 'The attribute `file_date` will return a datetime instead of an integer in future major versions.' in str(recwarn[0].message)
        assert recwarn[0].category is PTBDeprecationWarning
        assert recwarn[0].filename == __file__

    async def test_get_file_instance_method(self, monkeypatch, passport_file):

        async def make_assertion(*_, **kwargs):
            result = kwargs['file_id'] == passport_file.file_id
            return File(file_id=result, file_unique_id=result)
        assert check_shortcut_signature(PassportFile.get_file, Bot.get_file, ['file_id'], [])
        assert await check_shortcut_call(passport_file.get_file, passport_file.get_bot(), 'get_file')
        assert await check_defaults_handling(passport_file.get_file, passport_file.get_bot())
        monkeypatch.setattr(passport_file.get_bot(), 'get_file', make_assertion)
        assert (await passport_file.get_file()).file_id == 'True'