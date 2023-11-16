import unittest
import pytest
from tests.dcerpc import DCERPCTests
from impacket.dcerpc.v5.rpcrt import RPC_C_AUTHN_LEVEL_PKT_PRIVACY
fasp = None

@pytest.mark.skip(reason='fasp module unavailable')
class FASPTests(DCERPCTests):
    authn = True
    authn_level = RPC_C_AUTHN_LEVEL_PKT_PRIVACY

    def test_FWOpenPolicyStore(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpc_transport) = self.connect()
        request = fasp.FWOpenPolicyStore()
        request['BinaryVersion'] = 512
        request['StoreType'] = fasp.FW_STORE_TYPE.FW_STORE_TYPE_LOCAL
        request['AccessRight'] = fasp.FW_POLICY_ACCESS_RIGHT.FW_POLICY_ACCESS_RIGHT_READ
        request['dwFlags'] = 0
        resp = dce.request(request)
        resp.dump()

    def test_hFWOpenPolicyStore(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        resp = fasp.hFWOpenPolicyStore(dce)
        resp.dump()

    def test_FWClosePolicyStore(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        resp = fasp.hFWOpenPolicyStore(dce)
        request = fasp.FWClosePolicyStore()
        request['phPolicyStore'] = resp['phPolicyStore']
        resp = dce.request(request)
        resp.dump()

    def test_hFWClosePolicyStore(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        resp = fasp.hFWOpenPolicyStore(dce)
        resp = fasp.hFWClosePolicyStore(dce, resp['phPolicyStore'])
        resp.dump()

@pytest.mark.remote
class FASPTestsTCPTransport(FASPTests, unittest.TestCase):
    protocol = 'ncacn_ip_tcp'
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class FASPTestsTCPTransport64(FASPTests, unittest.TestCase):
    protocol = 'ncacn_ip_tcp'
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64
if __name__ == '__main__':
    unittest.main(verbosity=1)