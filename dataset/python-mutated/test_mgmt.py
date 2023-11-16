from __future__ import division
from __future__ import print_function
import pytest
import unittest
from six import assertRaisesRegex
from tests.dcerpc import DCERPCTests
from impacket.dcerpc.v5 import mgmt
from impacket.dcerpc.v5.rpcrt import DCERPCException

class MGMTTests(DCERPCTests):
    iface_uuid = mgmt.MSRPC_UUID_MGMT
    string_binding = 'ncacn_np:{0.machine}[\\pipe\\epmapper]'
    authn = True

    def test_inq_if_ids(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, transport) = self.connect()
        request = mgmt.inq_if_ids()
        resp = dce.request(request)
        resp.dump()

    def test_hinq_if_ids(self):
        if False:
            print('Hello World!')
        (dce, transport) = self.connect()
        resp = mgmt.hinq_if_ids(dce)
        resp.dump()

    def test_inq_stats(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, transport) = self.connect()
        request = mgmt.inq_stats()
        request['count'] = 40
        resp = dce.request(request)
        resp.dump()

    def test_hinq_stats(self):
        if False:
            while True:
                i = 10
        (dce, transport) = self.connect()
        resp = mgmt.hinq_stats(dce)
        resp.dump()

    def test_is_server_listening(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, transport) = self.connect()
        request = mgmt.is_server_listening()
        resp = dce.request(request, checkError=False)
        resp.dump()

    def test_his_server_listening(self):
        if False:
            print('Hello World!')
        (dce, transport) = self.connect()
        resp = mgmt.his_server_listening(dce)
        resp.dump()

    def test_stop_server_listening(self):
        if False:
            return 10
        (dce, transport) = self.connect()
        request = mgmt.stop_server_listening()
        with assertRaisesRegex(self, DCERPCException, 'rpc_s_access_denied'):
            dce.request(request)

    def test_hstop_server_listening(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, transport) = self.connect()
        with assertRaisesRegex(self, DCERPCException, 'rpc_s_access_denied'):
            mgmt.hstop_server_listening(dce)

    def test_inq_princ_name(self):
        if False:
            while True:
                i = 10
        (dce, transport) = self.connect()
        request = mgmt.inq_princ_name()
        request['authn_proto'] = 0
        request['princ_name_size'] = 32
        resp = dce.request(request, checkError=False)
        resp.dump()

    def test_hinq_princ_name(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, transport) = self.connect()
        resp = mgmt.hinq_princ_name(dce)
        resp.dump()

@pytest.mark.remote
class MGMTTestsSMBTransport(MGMTTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class MGMTTestsSMBTransport64(MGMTTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64

@pytest.mark.remote
class MGMTTestsTCPTransport(MGMTTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR
    string_binding = 'ncacn_ip_tcp:{0.machine}[135]'

@pytest.mark.remote
class MGMTTestsTCPTransport64(MGMTTests, unittest.TestCase):
    string_binding = 'ncacn_ip_tcp:{0.machine}[135]'
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64
if __name__ == '__main__':
    unittest.main(verbosity=1)