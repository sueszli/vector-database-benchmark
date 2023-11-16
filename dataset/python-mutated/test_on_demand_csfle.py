"""Test client side encryption with on demand credentials."""
from __future__ import annotations
import os
import sys
import unittest
sys.path[0:0] = ['']
from test import IntegrationTest, client_context
from bson.codec_options import CodecOptions
from pymongo.encryption import _HAVE_PYMONGOCRYPT, ClientEncryption, EncryptionError

class TestonDemandGCPCredentials(IntegrationTest):

    @classmethod
    @unittest.skipUnless(_HAVE_PYMONGOCRYPT, 'pymongocrypt is not installed')
    @client_context.require_version_min(4, 2, -1)
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.master_key = {'projectId': 'devprod-drivers', 'location': 'global', 'keyRing': 'key-ring-csfle', 'keyName': 'key-name-csfle'}

    @unittest.skipIf(not os.getenv('TEST_FLE_GCP_AUTO'), 'Not testing FLE GCP auto')
    def test_01_failure(self):
        if False:
            while True:
                i = 10
        if os.environ['SUCCESS'].lower() == 'true':
            self.skipTest('Expecting success')
        self.client_encryption = ClientEncryption(kms_providers={'gcp': {}}, key_vault_namespace='keyvault.datakeys', key_vault_client=client_context.client, codec_options=CodecOptions())
        with self.assertRaises(EncryptionError):
            self.client_encryption.create_data_key('gcp', self.master_key)

    @unittest.skipIf(not os.getenv('TEST_FLE_GCP_AUTO'), 'Not testing FLE GCP auto')
    def test_02_success(self):
        if False:
            print('Hello World!')
        if os.environ['SUCCESS'].lower() == 'false':
            self.skipTest('Expecting failure')
        self.client_encryption = ClientEncryption(kms_providers={'gcp': {}}, key_vault_namespace='keyvault.datakeys', key_vault_client=client_context.client, codec_options=CodecOptions())
        self.client_encryption.create_data_key('gcp', self.master_key)

class TestonDemandAzureCredentials(IntegrationTest):

    @classmethod
    @unittest.skipUnless(_HAVE_PYMONGOCRYPT, 'pymongocrypt is not installed')
    @client_context.require_version_min(4, 2, -1)
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.master_key = {'keyVaultEndpoint': os.environ['KEY_VAULT_ENDPOINT'], 'keyName': os.environ['KEY_NAME']}

    @unittest.skipIf(not os.getenv('TEST_FLE_AZURE_AUTO'), 'Not testing FLE Azure auto')
    def test_01_failure(self):
        if False:
            print('Hello World!')
        if os.environ['SUCCESS'].lower() == 'true':
            self.skipTest('Expecting success')
        self.client_encryption = ClientEncryption(kms_providers={'azure': {}}, key_vault_namespace='keyvault.datakeys', key_vault_client=client_context.client, codec_options=CodecOptions())
        with self.assertRaises(EncryptionError):
            self.client_encryption.create_data_key('azure', self.master_key)

    @unittest.skipIf(not os.getenv('TEST_FLE_AZURE_AUTO'), 'Not testing FLE Azure auto')
    def test_02_success(self):
        if False:
            while True:
                i = 10
        if os.environ['SUCCESS'].lower() == 'false':
            self.skipTest('Expecting failure')
        self.client_encryption = ClientEncryption(kms_providers={'azure': {}}, key_vault_namespace='keyvault.datakeys', key_vault_client=client_context.client, codec_options=CodecOptions())
        self.client_encryption.create_data_key('azure', self.master_key)
if __name__ == '__main__':
    unittest.main(verbosity=2)