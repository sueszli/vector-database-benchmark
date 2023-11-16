import unittest
from binascii import unhexlify
import impacket.dot11
import impacket.ImpactPacket
from impacket.Dot11KeyManager import KeyManager

class TestDot11WEPData(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.dot11 = impacket.dot11.Dot11(FCS_at_end=False)
        self.dot11.set_version(0)
        self.dot11.set_type_n_subtype(impacket.dot11.Dot11Types.DOT11_TYPE_DATA_SUBTYPE_DATA)
        self.dot11.set_fromDS(0)
        self.dot11.set_toDS(1)
        self.dot11.set_moreFrag(0)
        self.dot11.set_retry(0)
        self.dot11.set_powerManagement(0)
        self.dot11.set_moreData(0)
        self.dot11.set_protectedFrame(1)
        self.dot11.set_order(0)
        self.dot11data = impacket.dot11.Dot11DataFrame()
        self.dot11data.set_duration(44)
        self.dot11data.set_address1([0, 33, 41, 104, 51, 93])
        self.dot11data.set_address2([0, 24, 222, 124, 55, 159])
        self.dot11data.set_address3([0, 33, 41, 104, 51, 93])
        self.dot11data.set_fragment_number(0)
        self.dot11data.set_sequence_number(3439)
        self.wep = impacket.dot11.Dot11WEP()
        self.wep.set_iv(799077)
        self.wep.set_keyid(0)
        self.wepdata = impacket.dot11.Dot11WEPData()
        self.llc = impacket.dot11.LLC()
        self.llc.set_DSAP(170)
        self.llc.set_SSAP(170)
        self.llc.set_control(3)
        self.snap = impacket.dot11.SNAP()
        self.snap.set_OUI(0)
        self.snap.set_protoID(2048)
        self.ip = impacket.ImpactPacket.IP()
        self.ip.set_ip_v(4)
        self.ip.set_ip_tos(0)
        self.ip.set_ip_id(42503)
        self.ip.set_ip_rf(0)
        self.ip.set_ip_df(0)
        self.ip.set_ip_mf(0)
        self.ip.set_ip_off(0)
        self.ip.set_ip_ttl(128)
        self.ip.set_ip_p(1)
        self.ip.set_ip_src('192.168.1.102')
        self.ip.set_ip_dst('64.233.163.103')
        self.icmp = impacket.ImpactPacket.ICMP()
        self.icmp.set_icmp_type(self.icmp.ICMP_ECHO)
        self.icmp.set_icmp_code(0)
        self.icmp.set_icmp_id(1024)
        self.icmp.set_icmp_seq(33797)
        datastring = b'abcdefghijklmnopqrstuvwabcdefghi'
        self.data = impacket.ImpactPacket.Data(datastring)
        self.dot11.contains(self.dot11data)
        self.dot11data.contains(self.wep)
        self.wep.contains(self.wepdata)
        self.wepdata.contains(self.llc)
        self.llc.contains(self.snap)
        self.snap.contains(self.ip)
        self.ip.contains(self.icmp)
        self.icmp.contains(self.data)
        self.km = KeyManager()
        self.km.add_key([0, 33, 41, 104, 51, 91], unhexlify('999cbb701ca2ef030e302dcc35'))

    def test_02(self):
        if False:
            for i in range(10):
                print('nop')
        'Test ICV methods'
        self.assertEqual(self.wepdata.get_icv(), 0)
        self.assertEqual(self.wepdata.get_computed_icv(), 2717464965)
        self.wepdata.set_icv(2717464965)
        self.assertEqual(self.wepdata.get_icv(), self.wepdata.get_computed_icv())
        self.wepdata.set_icv(16909060)
        self.assertEqual(self.wepdata.get_icv(), 16909060)

    def test_03(self):
        if False:
            i = 10
            return i + 15
        'Test WEPData creation from scratch with encryption'
        self.wepdata.set_icv(2717464965)
        wep_enc = self.wep.get_encrypted_data(unhexlify('999cbb701ca2ef030e302dcc35'))
        self.assertEqual(wep_enc, unhexlify('8d2381e9251cb5aa83d2c716ba6ee18e7d3a2c71c00f6ab82fbc54c4b014ab03115edeccab2b18ebeb250f75eb6bf57fd65cb9e1b26e50ba4bb48b9f3471da9ecf12cb8f361b0253'))
        self.wep.encrypt_frame(unhexlify('999cbb701ca2ef030e302dcc35'))
if __name__ == '__main__':
    unittest.main(verbosity=1)