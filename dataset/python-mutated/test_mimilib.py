import pytest
import unittest
from tests.dcerpc import DCERPCTests
from Cryptodome.Cipher import ARC4
from impacket.dcerpc.v5 import mimilib
from impacket.dcerpc.v5.rpcrt import RPC_C_AUTHN_LEVEL_PKT_INTEGRITY, RPC_C_AUTHN_LEVEL_PKT_PRIVACY

@pytest.mark.remote
class MimiKatzTests(DCERPCTests, unittest.TestCase):
    timeout = 30000
    iface_uuid = mimilib.MSRPC_UUID_MIMIKATZ
    protocol = 'ncacn_ip_tcp'
    string_binding_formatting = DCERPCTests.STRING_BINDING_MAPPER
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR
    mimikatz_command = 'token::whoami'

    def get_dh_public_key(self):
        if False:
            for i in range(10):
                print('nop')
        dh = mimilib.MimiDiffeH()
        blob = mimilib.PUBLICKEYBLOB()
        blob['y'] = dh.genPublicKey()[::-1]
        public_key = mimilib.MIMI_PUBLICKEY()
        public_key['sessionType'] = mimilib.CALG_RC4
        public_key['cbPublicKey'] = 144
        public_key['pbPublicKey'] = blob.getData()
        return (dh, public_key)

    def get_handle_key(self, dce):
        if False:
            i = 10
            return i + 15
        (dh, public_key) = self.get_dh_public_key()
        resp = mimilib.hMimiBind(dce, public_key)
        blob = mimilib.PUBLICKEYBLOB(b''.join(resp['serverPublicKey']['pbPublicKey']))
        key = dh.getSharedSecret(blob['y'][::-1])
        pHandle = resp['phMimi']
        return (pHandle, key[-16:])

    def test_MimiBind(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        (dh, public_key) = self.get_dh_public_key()
        request = mimilib.MimiBind()
        request['clientPublicKey'] = public_key
        resp = dce.request(request)
        self.assertEqual(resp['ErrorCode'], 0)
        self.assertEqual(resp['serverPublicKey']['sessionType'], mimilib.CALG_RC4)
        blob = mimilib.PUBLICKEYBLOB(b''.join(resp['serverPublicKey']['pbPublicKey']))
        key = dh.getSharedSecret(blob['y'][::-1])
        pHandle = resp['phMimi']
        self.assertIsInstance(pHandle, bytes)
        self.assertIsInstance(key, bytes)
        dce.disconnect()
        rpc_transport.disconnect()

    def test_hMimiBind(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        (dh, public_key) = self.get_dh_public_key()
        resp = mimilib.hMimiBind(dce, public_key)
        self.assertEqual(resp['ErrorCode'], 0)
        self.assertEqual(resp['serverPublicKey']['sessionType'], mimilib.CALG_RC4)
        dce.disconnect()
        rpc_transport.disconnect()

    def test_MimiCommand(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        (pHandle, key) = self.get_handle_key(dce)
        cipher = ARC4.new(key[::-1])
        command = cipher.encrypt('{}\x00'.format(self.mimikatz_command).encode('utf-16le'))
        request = mimilib.MimiCommand()
        request['phMimi'] = pHandle
        request['szEncCommand'] = len(command)
        request['encCommand'] = list(command)
        resp = dce.request(request)
        self.assertEqual(resp['ErrorCode'], 0)
        self.assertEqual(len(resp['encResult']), resp['szEncResult'])
        cipherText = b''.join(resp['encResult'])
        cipher = ARC4.new(key[::-1])
        plain = cipher.decrypt(cipherText)
        dce.disconnect()
        rpc_transport.disconnect()

    def test_hMimiCommand(self):
        if False:
            return 10
        (dce, rpc_transport) = self.connect()
        (pHandle, key) = self.get_handle_key(dce)
        cipher = ARC4.new(key[::-1])
        command = cipher.encrypt('{}\x00'.format(self.mimikatz_command).encode('utf-16le'))
        resp = mimilib.hMimiCommand(dce, pHandle, command)
        self.assertEqual(resp['ErrorCode'], 0)
        self.assertEqual(len(resp['encResult']), resp['szEncResult'])
        dce.disconnect()
        rpc_transport.disconnect()

    def test_MimiUnBind(self):
        if False:
            return 10
        (dce, rpc_transport) = self.connect()
        (pHandle, key) = self.get_handle_key(dce)
        request = mimilib.MimiUnbind()
        request['phMimi'] = pHandle
        resp = dce.request(request)
        self.assertEqual(resp['ErrorCode'], 0)
        dce.disconnect()
        rpc_transport.disconnect()

class MimiKatzTestsAuthn(MimiKatzTests):
    authn = True

class MimiKatzTestsIntegrity(MimiKatzTestsAuthn):
    authn_level = RPC_C_AUTHN_LEVEL_PKT_INTEGRITY

class MimiKatzTestsPrivacy(MimiKatzTestsAuthn):
    authn_level = RPC_C_AUTHN_LEVEL_PKT_PRIVACY
if __name__ == '__main__':
    unittest.main(verbosity=1)