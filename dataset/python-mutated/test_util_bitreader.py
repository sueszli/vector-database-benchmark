from io import BytesIO
from test.picardtestcase import PicardTestCase
from picard.util.bitreader import LSBBitReader, MSBBitReader

class LsbBitReaderTest(PicardTestCase):

    def test_msb_bit_reader(self):
        if False:
            while True:
                i = 10
        data = BytesIO(b'\x8b\xc0\x17\x10')
        reader = MSBBitReader(data)
        self.assertEqual(8944, reader.bits(14))
        self.assertEqual(369, reader.bits(14))
        self.assertEqual(0, reader.bits(4))

    def test_lsb_bit_reader(self):
        if False:
            for i in range(10):
                print('nop')
        data = BytesIO(b'\x8b\xc0\x17\x10')
        reader = LSBBitReader(data)
        self.assertEqual(139, reader.bits(14))
        self.assertEqual(95, reader.bits(14))
        self.assertEqual(1, reader.bits(4))