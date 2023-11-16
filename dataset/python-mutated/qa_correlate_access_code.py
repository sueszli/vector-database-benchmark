from gnuradio import gr, gr_unittest, digital, blocks
default_access_code = '¬Ý¤âò\x8c ü'

def string_to_1_0_list(s):
    if False:
        i = 10
        return i + 15
    r = []
    for ch in s:
        x = ord(ch)
        for i in range(8):
            t = x >> i & 1
            r.append(t)
    return r

def to_1_0_string(L):
    if False:
        return 10
    return ''.join([chr(x + ord('0')) for x in L])

class test_correlate_access_code(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_001(self):
        if False:
            i = 10
            return i + 15
        pad = [0] * 64
        src_data = [1, 0, 1, 1, 1, 1, 0, 1, 1] + pad + [0] * 7
        expected_result = pad + [1, 0, 1, 1, 3, 1, 0, 1, 1, 2] + [0] * 6
        src = blocks.vector_source_b(src_data)
        op = digital.correlate_access_code_bb('1011', 0)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(expected_result, result_data)

    def test_002(self):
        if False:
            return 10
        code = list(string_to_1_0_list(default_access_code))
        access_code = to_1_0_string(code)
        pad = [0] * 64
        src_data = code + [1, 0, 1, 1] + pad
        expected_result = pad + code + [3, 0, 1, 1]
        src = blocks.vector_source_b(src_data)
        op = digital.correlate_access_code_bb(access_code, 0)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(expected_result, result_data)

    def test_003(self):
        if False:
            return 10
        code = list(string_to_1_0_list(default_access_code))
        access_code = to_1_0_string(code)
        pad = [0] * 64
        src_data = code + [1, 0, 1, 1] + pad
        expected_result = code + [1, 0, 1, 1] + pad
        src = blocks.vector_source_b(src_data)
        op = digital.correlate_access_code_tag_bb(access_code, 0, 'test')
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(expected_result, result_data)

    def test_004(self):
        if False:
            i = 10
            return i + 15
        code = list(string_to_1_0_list(default_access_code))
        access_code = to_1_0_string(code)
        pad = [0] * 64
        src_bits = code + [1, 0, 1, 1] + pad
        src_data = [2.0 * x - 1.0 for x in src_bits]
        expected_result_bits = code + [1, 0, 1, 1] + pad
        expected_result = [2.0 * x - 1.0 for x in expected_result_bits]
        src = blocks.vector_source_f(src_data)
        op = digital.correlate_access_code_tag_ff(access_code, 0, 'test')
        dst = blocks.vector_sink_f()
        self.tb.connect(src, op, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertFloatTuplesAlmostEqual(expected_result, result_data, 5)
if __name__ == '__main__':
    gr_unittest.run(test_correlate_access_code)