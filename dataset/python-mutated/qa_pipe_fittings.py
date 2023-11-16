from gnuradio import gr, gr_unittest, blocks

def calc_expected_result(src_data, n):
    if False:
        while True:
            i = 10
    assert len(src_data) % n == 0
    result = [list() for x in range(n)]
    for i in range(len(src_data)):
        result[i % n].append(src_data[i])
    return result

class test_pipe_fittings(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_001(self):
        if False:
            i = 10
            return i + 15
        '\n        Test stream_to_streams.\n        '
        n = 8
        src_len = n * 8
        src_data = list(range(src_len))
        expected_results = calc_expected_result(src_data, n)
        src = blocks.vector_source_i(src_data)
        op = blocks.stream_to_streams(gr.sizeof_int, n)
        self.tb.connect(src, op)
        dsts = []
        for i in range(n):
            dst = blocks.vector_sink_i()
            self.tb.connect((op, i), (dst, 0))
            dsts.append(dst)
        self.tb.run()
        for d in range(n):
            self.assertEqual(expected_results[d], dsts[d].data())

    def test_002(self):
        if False:
            print('Hello World!')
        n = 8
        src_len = n * 8
        src_data = list(range(src_len))
        expected_results = src_data
        src = blocks.vector_source_i(src_data)
        op1 = blocks.stream_to_streams(gr.sizeof_int, n)
        op2 = blocks.streams_to_stream(gr.sizeof_int, n)
        dst = blocks.vector_sink_i()
        self.tb.connect(src, op1)
        for i in range(n):
            self.tb.connect((op1, i), (op2, i))
        self.tb.connect(op2, dst)
        self.tb.run()
        self.assertEqual(expected_results, dst.data())

    def test_003(self):
        if False:
            while True:
                i = 10
        n = 8
        src_len = n * 8
        src_data = list(range(src_len))
        expected_results = src_data
        src = blocks.vector_source_i(src_data)
        op1 = blocks.stream_to_streams(gr.sizeof_int, n)
        op2 = blocks.streams_to_vector(gr.sizeof_int, n)
        op3 = blocks.vector_to_stream(gr.sizeof_int, n)
        dst = blocks.vector_sink_i()
        self.tb.connect(src, op1)
        for i in range(n):
            self.tb.connect((op1, i), (op2, i))
        self.tb.connect(op2, op3, dst)
        self.tb.run()
        self.assertEqual(expected_results, dst.data())

    def test_004(self):
        if False:
            while True:
                i = 10
        n = 8
        src_len = n * 8
        src_data = list(range(src_len))
        expected_results = src_data
        src = blocks.vector_source_i(src_data)
        op1 = blocks.stream_to_vector(gr.sizeof_int, n)
        op2 = blocks.vector_to_streams(gr.sizeof_int, n)
        op3 = blocks.streams_to_stream(gr.sizeof_int, n)
        dst = blocks.vector_sink_i()
        self.tb.connect(src, op1, op2)
        for i in range(n):
            self.tb.connect((op2, i), (op3, i))
        self.tb.connect(op3, dst)
        self.tb.run()
        self.assertEqual(expected_results, dst.data())
if __name__ == '__main__':
    gr_unittest.run(test_pipe_fittings)