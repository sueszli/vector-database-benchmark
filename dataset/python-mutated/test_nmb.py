import pytest
import unittest
from tests import RemoteTestCase
from impacket import nmb
from impacket.structure import hexdump

@pytest.mark.remote
class NMBTests(RemoteTestCase, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(NMBTests, self).setUp()
        self.set_transport_config()

    def create_connection(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_encodedecodename(self):
        if False:
            i = 10
            return i + 15
        name = 'THISISAVERYLONGLONGNAME'
        encoded = nmb.encode_name(name, nmb.TYPE_SERVER, None)
        hexdump(encoded)
        decoded = nmb.decode_name(encoded)
        hexdump(bytearray(decoded[1], 'utf-8'))
        self.assertEqual(name[:15], decoded[1].strip())

    def test_getnetbiosname(self):
        if False:
            while True:
                i = 10
        n = nmb.NetBIOS()
        res = n.getnetbiosname(self.machine)
        print(repr(res))
        self.assertEqual(self.serverName, res)

    def test_getnodestatus(self):
        if False:
            print('Hello World!')
        n = nmb.NetBIOS()
        resp = n.getnodestatus(self.serverName.upper(), self.machine)
        for r in resp:
            r.dump()
        print(resp)

    def test_gethostbyname(self):
        if False:
            while True:
                i = 10
        n = nmb.NetBIOS()
        n.set_nameserver(self.serverName)
        resp = n.gethostbyname(self.serverName, nmb.TYPE_SERVER)
        print(resp.entries)

    def test_name_registration_request(self):
        if False:
            while True:
                i = 10
        n = nmb.NetBIOS()
        try:
            resp = n.name_registration_request('*JSMBSERVER', self.serverName, nmb.TYPE_WORKSTATION, None, nmb.NB_FLAGS_ONT_P, '1.1.1.2')
            resp.dump()
        except Exception as e:
            print(str(e))
            if str(e).find('NETBIOS') <= 0:
                raise e

    def test_name_query_request(self):
        if False:
            print('Hello World!')
        n = nmb.NetBIOS()
        resp = n.name_query_request(self.serverName, self.machine)
        print(resp.entries)
if __name__ == '__main__':
    unittest.main(verbosity=1)