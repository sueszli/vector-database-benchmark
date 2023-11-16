from __future__ import division
from __future__ import print_function
import pytest
import unittest
from tests.dcerpc import DCERPCTests
from impacket.dcerpc.v5 import bkrp
from impacket.dcerpc.v5.rpcrt import RPC_C_AUTHN_LEVEL_PKT_PRIVACY
from impacket.dcerpc.v5.dtypes import NULL
try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print('In order to run these test cases you need the cryptography package')

class BKRPTests(DCERPCTests):
    iface_uuid = bkrp.MSRPC_UUID_BKRP
    string_binding = 'ncacn_np:{0.machine}[\\PIPE\\protected_storage]'
    authn = True
    authn_level = RPC_C_AUTHN_LEVEL_PKT_PRIVACY
    data_in = b"Huh? wait wait, let me, let me explain something to you. Uh, I am not Mr. Lebowski; you're Mr. Lebowski. I'm the Dude. So that's what you call me. You know, uh, That, or uh, his Dudeness, or uh Duder, or uh El Duderino, if, you know, you're not into the whole brevity thing--uh."

    def test_BackuprKey_BACKUPKEY_BACKUP_GUID_BACKUPKEY_RESTORE_GUID(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        request = bkrp.BackuprKey()
        request['pguidActionAgent'] = bkrp.BACKUPKEY_BACKUP_GUID
        request['pDataIn'] = self.data_in
        request['cbDataIn'] = len(self.data_in)
        request['dwParam'] = 0
        resp = dce.request(request)
        resp.dump()
        wrapped = bkrp.WRAPPED_SECRET()
        wrapped.fromString(b''.join(resp['ppDataOut']))
        wrapped.dump()
        request = bkrp.BackuprKey()
        request['pguidActionAgent'] = bkrp.BACKUPKEY_RESTORE_GUID
        request['pDataIn'] = b''.join(resp['ppDataOut'])
        request['cbDataIn'] = resp['pcbDataOut']
        request['dwParam'] = 0
        resp = dce.request(request)
        resp.dump()
        self.assertEqual(self.data_in, b''.join(resp['ppDataOut']))

    def test_hBackuprKey_BACKUPKEY_BACKUP_GUID_BACKUPKEY_RESTORE_GUID(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        resp = bkrp.hBackuprKey(dce, bkrp.BACKUPKEY_BACKUP_GUID, self.data_in)
        resp.dump()
        wrapped = bkrp.WRAPPED_SECRET()
        wrapped.fromString(b''.join(resp['ppDataOut']))
        wrapped.dump()
        resp = bkrp.hBackuprKey(dce, bkrp.BACKUPKEY_RESTORE_GUID, b''.join(resp['ppDataOut']))
        resp.dump()
        self.assertEqual(self.data_in, b''.join(resp['ppDataOut']))

    def test_BackuprKey_BACKUPKEY_BACKUP_GUID_BACKUPKEY_RESTORE_GUID_WIN2K(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = bkrp.BackuprKey()
        request['pguidActionAgent'] = bkrp.BACKUPKEY_BACKUP_GUID
        request['pDataIn'] = self.data_in
        request['cbDataIn'] = len(self.data_in)
        request['dwParam'] = 0
        resp = dce.request(request)
        resp.dump()
        wrapped = bkrp.WRAPPED_SECRET()
        wrapped.fromString(b''.join(resp['ppDataOut']))
        wrapped.dump()
        request = bkrp.BackuprKey()
        request['pguidActionAgent'] = bkrp.BACKUPKEY_RESTORE_GUID_WIN2K
        request['pDataIn'] = b''.join(resp['ppDataOut'])
        request['cbDataIn'] = resp['pcbDataOut']
        request['dwParam'] = 0
        resp = dce.request(request)
        resp.dump()
        self.assertEqual(self.data_in, b''.join(resp['ppDataOut']))

    def test_hBackuprKey_BACKUPKEY_BACKUP_GUID_BACKUPKEY_RESTORE_GUID_WIN2K(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        resp = bkrp.hBackuprKey(dce, bkrp.BACKUPKEY_BACKUP_GUID, self.data_in)
        resp.dump()
        wrapped = bkrp.WRAPPED_SECRET()
        wrapped.fromString(b''.join(resp['ppDataOut']))
        wrapped.dump()
        resp = bkrp.hBackuprKey(dce, bkrp.BACKUPKEY_RESTORE_GUID_WIN2K, b''.join(resp['ppDataOut']))
        resp.dump()
        self.assertEqual(self.data_in, b''.join(resp['ppDataOut']))

    def test_BackuprKey_BACKUPKEY_RETRIEVE_BACKUP_KEY_GUID(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = bkrp.BackuprKey()
        request['pguidActionAgent'] = bkrp.BACKUPKEY_RETRIEVE_BACKUP_KEY_GUID
        request['pDataIn'] = NULL
        request['cbDataIn'] = 0
        request['dwParam'] = 0
        resp = dce.request(request)
        resp.dump()
        cert = x509.load_der_x509_certificate(b''.join(resp['ppDataOut']), default_backend())
        print(cert.subject)
        print(cert.issuer)
        print(cert.signature)

    def test_hBackuprKey_BACKUPKEY_RETRIEVE_BACKUP_KEY_GUID(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = bkrp.BackuprKey()
        request['pguidActionAgent'] = bkrp.BACKUPKEY_RETRIEVE_BACKUP_KEY_GUID
        request['pDataIn'] = NULL
        request['cbDataIn'] = 0
        request['dwParam'] = 0
        resp = bkrp.hBackuprKey(dce, bkrp.BACKUPKEY_RETRIEVE_BACKUP_KEY_GUID, NULL)
        resp.dump()
        cert = x509.load_der_x509_certificate(b''.join(resp['ppDataOut']), default_backend())
        print(cert.subject)
        print(cert.issuer)
        print(cert.signature)

@pytest.mark.remote
class BKRPTestsSMBTransport(BKRPTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class BKRPTestsSMBTransport64(BKRPTestsSMBTransport):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64
if __name__ == '__main__':
    unittest.main(verbosity=1)