import unittest
import array
from impacket import wps

class TestTLVContainer(unittest.TestCase):

    def testNormalUsageContainer(self):
        if False:
            for i in range(10):
                print('nop')
        BUILDERS = {1: wps.StringBuilder(), 2: wps.ByteBuilder(), 3: wps.NumBuilder(2)}
        tlvc = wps.TLVContainer(builders=BUILDERS)
        KINDS_N_VALUES = ((1, b'Sarlanga'), (2, 1), (3, 1024), (4, array.array('B', [1, 2, 3])))
        for (k, v) in KINDS_N_VALUES:
            tlvc.append(k, v)
        tlvc2 = wps.TLVContainer(builders=BUILDERS)
        tlvc2.from_ary(tlvc.to_ary())
        for (k, v) in KINDS_N_VALUES:
            self.assertEqual(v, tlvc2.first(k))
        self.assertEqual(tlvc.to_ary(), tlvc2.to_ary())
        self.assertEqual(b'Sarlanga', tlvc.first(1))
if __name__ == '__main__':
    unittest.main(verbosity=1)