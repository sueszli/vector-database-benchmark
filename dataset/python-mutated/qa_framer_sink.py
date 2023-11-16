from gnuradio import gr, gr_unittest, digital, blocks
default_access_code = '¬Ý¤âò\x8c ü'

def string_to_1_0_list(s):
    if False:
        print('Hello World!')
    r = []
    for ch in s:
        x = ord(ch)
        for i in range(8):
            t = x >> i & 1
            r.append(t)
    return r

def to_1_0_string(L):
    if False:
        i = 10
        return i + 15
    return ''.join([chr(x + ord('0')) for x in L])

class test_framer_sink(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test_001(self):
        if False:
            return 10
        code = (1, 1, 0, 1)
        access_code = to_1_0_string(code)
        header = tuple(2 * [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        pad = (0,) * 100
        src_data = code + header + (0, 1, 0, 0, 0, 0, 0, 1) + pad
        expected_data = b'A'
        rcvd_pktq = gr.msg_queue()
        src = blocks.vector_source_b(src_data)
        correlator = digital.correlate_access_code_bb(access_code, 0)
        framer_sink = digital.framer_sink_1(rcvd_pktq)
        vsnk = blocks.vector_sink_b()
        self.tb.connect(src, correlator, framer_sink)
        self.tb.connect(correlator, vsnk)
        self.tb.run()
        result_data = rcvd_pktq.delete_head()
        result_data = result_data.to_string()
        self.assertEqual(expected_data, result_data)

    def test_002(self):
        if False:
            i = 10
            return i + 15
        code = tuple(string_to_1_0_list(default_access_code))
        access_code = to_1_0_string(code)
        header = tuple(2 * [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        pad = (0,) * 100
        src_data = code + header + (0, 1, 0, 0, 1, 0, 0, 0) + (0, 1, 0, 0, 1, 0, 0, 1) + pad
        expected_data = b'HI'
        rcvd_pktq = gr.msg_queue()
        src = blocks.vector_source_b(src_data)
        correlator = digital.correlate_access_code_bb(access_code, 0)
        framer_sink = digital.framer_sink_1(rcvd_pktq)
        vsnk = blocks.vector_sink_b()
        self.tb.connect(src, correlator, framer_sink)
        self.tb.connect(correlator, vsnk)
        self.tb.run()
        result_data = rcvd_pktq.delete_head()
        result_data = result_data.to_string()
        self.assertEqual(expected_data, result_data)
if __name__ == '__main__':
    gr_unittest.run(test_framer_sink)