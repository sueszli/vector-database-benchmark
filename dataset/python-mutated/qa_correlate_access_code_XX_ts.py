from gnuradio import gr, gr_unittest, digital, blocks
import pmt
default_access_code = '¬Ý¤âò\x8c ü'

def string_to_1_0_list(s):
    if False:
        for i in range(10):
            print('nop')
    r = []
    for ch in s:
        x = ord(ch)
        for i in range(8):
            t = x >> i & 1
            r.append(t)
    return r

def to_1_0_string(L):
    if False:
        while True:
            i = 10
    return ''.join([chr(x + ord('0')) for x in L])

class test_correlate_access_code_XX_ts(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test_001(self):
        if False:
            while True:
                i = 10
        payload = 'test packet'
        header = '\x00Ð\x00Ð'
        packet = header + payload
        pad = (0,) * 64
        src_data = (0, 0, 1, 1, 1, 1, 0, 1, 1) + tuple(string_to_1_0_list(packet)) + pad
        expected = list(map(int, src_data[9 + 32:-len(pad)]))
        src = blocks.vector_source_b(src_data)
        op = digital.correlate_access_code_bb_ts('1011', 0, 'sync')
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        result_tags = dst.tags()
        self.assertEqual(len(result_data), len(payload) * 8)
        self.assertEqual(result_tags[0].offset, 0)
        self.assertEqual(pmt.to_long(result_tags[0].value), len(payload) * 8)
        self.assertEqual(result_data, expected)

    def test_bb_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        payload = 'test packet'
        header = '\x00Ð\x00Ð'
        packet = header + payload
        pad = (0,) * 64
        src_data = (0, 1, 1, 1, 0, 0, 0, 1, 1) + tuple(string_to_1_0_list(packet)) + pad
        expected = list(map(int, src_data[9 + 32:-len(pad)]))
        src = blocks.vector_source_b(src_data)
        op = digital.correlate_access_code_bb_ts('0011', 0, 'sync')
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        result_tags = dst.tags()
        self.assertEqual(len(result_data), len(payload) * 8)
        self.assertEqual(result_tags[0].offset, 0)
        self.assertEqual(pmt.to_long(result_tags[0].value), len(payload) * 8)
        self.assertEqual(result_data, expected)

    def test_bb_immediate(self):
        if False:
            while True:
                i = 10
        'Test that packets at start of stream match'
        payload = 'test packet'
        header = '\x00Ð\x00Ð'
        packet = header + payload
        pad = (0,) * 64
        src_data = (0, 0, 1, 1) + tuple(string_to_1_0_list(packet)) + pad
        expected = list(map(int, src_data[4 + 32:-len(pad)]))
        src = blocks.vector_source_b(src_data)
        op = digital.correlate_access_code_bb_ts('0011', 0, 'sync')
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        result_tags = dst.tags()
        self.assertEqual(result_tags[0].offset, 0)
        self.assertEqual(result_data, expected)

    def test_002(self):
        if False:
            print('Hello World!')
        payload = 'test packet'
        header = '\x00Ð\x00Ð'
        packet = header + payload
        pad = (0,) * 64
        src_data = (0, 0, 1, 1, 1, 1, 0, 1, 1) + tuple(string_to_1_0_list(packet)) + pad
        src_floats = tuple((2 * b - 1 for b in src_data))
        expected = src_floats[9 + 32:-len(pad)]
        src = blocks.vector_source_f(src_floats)
        op = digital.correlate_access_code_ff_ts('1011', 0, 'sync')
        dst = blocks.vector_sink_f()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        result_tags = dst.tags()
        self.assertEqual(len(result_data), len(payload) * 8)
        self.assertEqual(result_tags[0].offset, 0)
        self.assertEqual(pmt.to_long(result_tags[0].value), len(payload) * 8)
        self.assertFloatTuplesAlmostEqual(result_data, expected, 5)

    def test_ff_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        payload = 'test packet'
        header = '\x00Ð\x00Ð'
        packet = header + payload
        pad = (0,) * 64
        src_data = (0, 1, 1, 1, 1, 0, 0, 1, 1) + tuple(string_to_1_0_list(packet)) + pad
        src_floats = tuple((2 * b - 1 for b in src_data))
        expected = src_floats[9 + 32:-len(pad)]
        src = blocks.vector_source_f(src_floats)
        op = digital.correlate_access_code_ff_ts('0011', 0, 'sync')
        dst = blocks.vector_sink_f()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        result_tags = dst.tags()
        self.assertEqual(len(result_data), len(payload) * 8)
        self.assertEqual(result_tags[0].offset, 0)
        self.assertEqual(pmt.to_long(result_tags[0].value), len(payload) * 8)
        self.assertFloatTuplesAlmostEqual(result_data, expected, 5)

    def test_ff_immediate(self):
        if False:
            print('Hello World!')
        'Test that packets at start of stream match'
        payload = 'test packet'
        header = '\x00Ð\x00Ð'
        packet = header + payload
        pad = (0,) * 64
        src_data = (0, 0, 1, 1) + tuple(string_to_1_0_list(packet)) + pad
        src_floats = tuple((2 * b - 1 for b in src_data))
        expected = src_floats[4 + 32:-len(pad)]
        src = blocks.vector_source_f(src_floats)
        op = digital.correlate_access_code_ff_ts('0011', 0, 'sync')
        dst = blocks.vector_sink_f()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        result_tags = dst.tags()
        self.assertEqual(len(result_data), len(payload) * 8)
        self.assertEqual(result_tags[0].offset, 0)
        self.assertEqual(pmt.to_long(result_tags[0].value), len(payload) * 8)
        self.assertFloatTuplesAlmostEqual(result_data, expected, 5)
if __name__ == '__main__':
    gr_unittest.run(test_correlate_access_code_XX_ts)