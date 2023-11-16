import os
from Config import config
from Crypt import CryptConnection

class TestCryptConnection:

    def testSslCert(self):
        if False:
            for i in range(10):
                print('nop')
        if os.path.isfile('%s/cert-rsa.pem' % config.data_dir):
            os.unlink('%s/cert-rsa.pem' % config.data_dir)
        if os.path.isfile('%s/key-rsa.pem' % config.data_dir):
            os.unlink('%s/key-rsa.pem' % config.data_dir)
        CryptConnection.manager.loadCerts()
        assert 'tls-rsa' in CryptConnection.manager.crypt_supported
        assert CryptConnection.manager.selectCrypt(['tls-rsa', 'unknown']) == 'tls-rsa'
        assert os.path.isfile('%s/cert-rsa.pem' % config.data_dir)
        assert os.path.isfile('%s/key-rsa.pem' % config.data_dir)