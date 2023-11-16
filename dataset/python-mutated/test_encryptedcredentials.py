import pytest
from telegram import EncryptedCredentials, PassportElementError
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def encrypted_credentials():
    if False:
        print('Hello World!')
    return EncryptedCredentials(TestEncryptedCredentialsBase.data, TestEncryptedCredentialsBase.hash, TestEncryptedCredentialsBase.secret)

class TestEncryptedCredentialsBase:
    data = 'data'
    hash = 'hash'
    secret = 'secret'

class TestEncryptedCredentialsWithoutRequest(TestEncryptedCredentialsBase):

    def test_slot_behaviour(self, encrypted_credentials):
        if False:
            return 10
        inst = encrypted_credentials
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_expected_values(self, encrypted_credentials):
        if False:
            return 10
        assert encrypted_credentials.data == self.data
        assert encrypted_credentials.hash == self.hash
        assert encrypted_credentials.secret == self.secret

    def test_to_dict(self, encrypted_credentials):
        if False:
            i = 10
            return i + 15
        encrypted_credentials_dict = encrypted_credentials.to_dict()
        assert isinstance(encrypted_credentials_dict, dict)
        assert encrypted_credentials_dict['data'] == encrypted_credentials.data
        assert encrypted_credentials_dict['hash'] == encrypted_credentials.hash
        assert encrypted_credentials_dict['secret'] == encrypted_credentials.secret

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        a = EncryptedCredentials(self.data, self.hash, self.secret)
        b = EncryptedCredentials(self.data, self.hash, self.secret)
        c = EncryptedCredentials(self.data, '', '')
        d = EncryptedCredentials('', self.hash, '')
        e = EncryptedCredentials('', '', self.secret)
        f = PassportElementError('source', 'type', 'message')
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