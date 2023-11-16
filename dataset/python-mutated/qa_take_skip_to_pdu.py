from builtins import range
from gnuradio import gr, gr_unittest, blocks, pdu
import pmt
import time
import numpy

class qa_take_skip_to_pdu_X(gr_unittest.TestCase):

    def assertEqualPDU(self, pdu1, pdu2):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(pmt.equal(pdu1, pdu2))
        self.assertTrue(pmt.equal(pmt.car(pdu1), pmt.car(pdu2)))
        vec1 = pmt.cdr(pdu1)
        vec2 = pmt.cdr(pdu2)
        self.assertTrue(pmt.equal(vec1, vec2))
        self.assertTrue((pmt.to_python(vec1) == pmt.to_python(vec2)).all(), msg=f'vectors not equal? {vec1!r} {vec2!r}')

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_001_f_32(self):
        if False:
            i = 10
            return i + 15
        self.source = blocks.vector_source_f(range(0, 32 * 3), False, 1, [])
        self.ts_pdu = pdu.take_skip_to_pdu_f(32, 32)
        self.debug = blocks.message_debug()
        self.tb.connect((self.source, 0), (self.ts_pdu, 0))
        self.tb.msg_connect((self.ts_pdu, 'pdus'), (self.debug, 'store'))
        dic = pmt.dict_add(pmt.make_dict(), pmt.intern('pdu_num'), pmt.from_uint64(0))
        vec = pmt.init_f32vector(32, range(0, 32))
        expected = pmt.cons(dic, vec)
        self.tb.run()
        actual = self.debug.get_message(0)
        self.assertEqualPDU(actual, expected)

    def test_002_c_80(self):
        if False:
            for i in range(10):
                print('nop')
        self.source = blocks.vector_source_c(range(0, 32 * 3), False, 1, [])
        self.ts_pdu = pdu.take_skip_to_pdu_c(80, 32)
        self.debug = blocks.message_debug()
        self.tb.connect((self.source, 0), (self.ts_pdu, 0))
        self.tb.msg_connect((self.ts_pdu, 'pdus'), (self.debug, 'store'))
        dic = pmt.dict_add(pmt.make_dict(), pmt.intern('pdu_num'), pmt.from_uint64(0))
        vec = pmt.init_c32vector(80, range(0, 80))
        expected = pmt.cons(dic, vec)
        self.tb.run()
        actual = self.debug.get_message(0)
        self.assertEqualPDU(actual, expected)

    def test_003_s_2_11_7(self):
        if False:
            for i in range(10):
                print('nop')
        self.source = blocks.vector_source_s(range(0, 32 * 3), False, 1, [])
        self.ts_pdu = pdu.take_skip_to_pdu_s(2, 11)
        self.debug = blocks.message_debug()
        self.tb.connect((self.source, 0), (self.ts_pdu, 0))
        self.tb.msg_connect((self.ts_pdu, 'pdus'), (self.debug, 'store'))
        dic = pmt.dict_add(pmt.make_dict(), pmt.intern('pdu_num'), pmt.from_uint64(7))
        vec = pmt.init_s16vector(2, list(range(91, 93)))
        expected = pmt.cons(dic, vec)
        self.tb.run()
        actual = self.debug.get_message(7)
        self.assertEqualPDU(actual, expected)

    def test_004_b_512(self):
        if False:
            while True:
                i = 10
        self.source = blocks.vector_source_b(list(range(0, 256)) * 4, False, 1, [])
        self.ts_pdu = pdu.take_skip_to_pdu_b(512, 1)
        self.debug = blocks.message_debug()
        self.tb.connect((self.source, 0), (self.ts_pdu, 0))
        self.tb.msg_connect((self.ts_pdu, 'pdus'), (self.debug, 'store'))
        dic = pmt.dict_add(pmt.make_dict(), pmt.intern('pdu_num'), pmt.from_uint64(0))
        vec = pmt.init_u8vector(512, list(range(0, 256)) * 2)
        expected = pmt.cons(dic, vec)
        self.tb.run()
        actual = self.debug.get_message(0)
        self.assertEqualPDU(actual, expected)
if __name__ == '__main__':
    gr_unittest.run(qa_take_skip_to_pdu_X)