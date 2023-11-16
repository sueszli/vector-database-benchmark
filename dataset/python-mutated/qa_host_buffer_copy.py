from gnuradio import gr, gr_unittest, blocks

def makeExpected(history, src_data):
    if False:
        for i in range(10):
            print('nop')
    ' Helper function to make expected result from history and source\n    data parameters.\n    '
    if history - 1 > 0:
        expected_data = [0.0] * (history - 1)
        expected_data.extend(src_data[:-(history - 1)])
    else:
        expected_data = src_data
    return [float(x) for x in expected_data]

class test_copy(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_simple_hostbuf_copy(self):
        if False:
            i = 10
            return i + 15
        history = 1
        src_data = 10 * [x for x in range(16384)]
        expected_data = makeExpected(history, src_data)
        src = blocks.vector_source_f(src_data)
        copy = blocks.host_buffer_copy(history)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, copy, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(expected_data, result_data)

    def test_simple_hostbuf_copy_hist100(self):
        if False:
            print('Hello World!')
        history = 100
        src_data = 10 * [x for x in range(16384)]
        expected_data = makeExpected(history, src_data)
        src = blocks.vector_source_f(src_data)
        copy = blocks.host_buffer_copy(history)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, copy, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(expected_data, result_data)

    def test_simple_hostbuf_copy_hist113(self):
        if False:
            for i in range(10):
                print('nop')
        history = 113
        src_data = 10 * [x for x in range(16384)]
        expected_data = makeExpected(history, src_data)
        src = blocks.vector_source_f(src_data)
        copy = blocks.host_buffer_copy(history)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, copy, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(expected_data, result_data)

    def test_simple_hostbuf_copy_hist1057_big(self):
        if False:
            print('Hello World!')
        history = 1057
        src_data = 17 * [x for x in range(91537)]
        expected_data = makeExpected(history, src_data)
        src = blocks.vector_source_f(src_data)
        copy = blocks.host_buffer_copy(history)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, copy, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertEqual(expected_data, result_data)
if __name__ == '__main__':
    gr_unittest.run(test_copy)