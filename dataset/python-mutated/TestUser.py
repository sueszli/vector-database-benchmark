import pytest
from Crypt import CryptBitcoin

@pytest.mark.usefixtures('resetSettings')
class TestUser:

    def testAddress(self, user):
        if False:
            return 10
        assert user.master_address == '15E5rhcAUD69WbiYsYARh4YHJ4sLm2JEyc'
        address_index = 1458664252141532163166741013621928587528255888800826689784628722366466547364755811
        assert user.getAddressAuthIndex('15E5rhcAUD69WbiYsYARh4YHJ4sLm2JEyc') == address_index

    def testNewSite(self, user):
        if False:
            return 10
        (address, address_index, site_data) = user.getNewSiteData()
        assert CryptBitcoin.hdPrivatekey(user.master_seed, address_index) == site_data['privatekey']
        user.sites = {}
        assert user.getSiteData(address)['auth_address'] != address
        assert user.getSiteData(address)['auth_privatekey'] == site_data['auth_privatekey']

    def testAuthAddress(self, user):
        if False:
            while True:
                i = 10
        auth_address = user.getAuthAddress('1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr')
        assert auth_address == '1MyJgYQjeEkR9QD66nkfJc9zqi9uUy5Lr2'
        auth_privatekey = user.getAuthPrivatekey('1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr')
        assert CryptBitcoin.privatekeyToAddress(auth_privatekey) == auth_address

    def testCert(self, user):
        if False:
            while True:
                i = 10
        cert_auth_address = user.getAuthAddress('1iD5ZQJMNXu43w1qLB8sfdHVKppVMduGz')
        user.addCert(cert_auth_address, 'zeroid.bit', 'faketype', 'fakeuser', 'fakesign')
        user.setCert('1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr', 'zeroid.bit')
        assert user.getAuthAddress('1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr') == cert_auth_address
        auth_privatekey = user.getAuthPrivatekey('1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr')
        assert CryptBitcoin.privatekeyToAddress(auth_privatekey) == cert_auth_address
        assert '1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr' in user.sites
        user.deleteSiteData('1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr')
        assert '1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr' not in user.sites
        assert not user.getAuthAddress('1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr') == cert_auth_address
        assert user.getAuthAddress('1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr') == '1MyJgYQjeEkR9QD66nkfJc9zqi9uUy5Lr2'