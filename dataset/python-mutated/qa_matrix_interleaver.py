from gnuradio import gr, gr_unittest
from gnuradio import blocks

class qa_matrix_interleaver(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_interleave(self):
        if False:
            return 10
        tb = self.tb
        (cols, rows) = (4, 10)
        vec = sum((cols * [x] for x in range(rows)), [])
        expected = cols * list(range(rows))
        src = blocks.vector_source_f(vec, False)
        itlv = blocks.matrix_interleaver(gr.sizeof_float, rows=rows, cols=cols)
        snk = blocks.vector_sink_f()
        tb.connect(src, itlv, snk)
        tb.run()
        result = snk.data()
        self.assertFloatTuplesAlmostEqual(expected, result)

    def test_deinterleave(self):
        if False:
            i = 10
            return i + 15
        tb = self.tb
        (cols, rows) = (4, 10)
        vec = sum((rows * [x] for x in range(cols)), [])
        expected = rows * list(range(cols))
        src = blocks.vector_source_f(vec, False)
        itlv = blocks.matrix_interleaver(gr.sizeof_float, rows=rows, cols=cols, deint=True)
        snk = blocks.vector_sink_f()
        tb.connect(src, itlv, snk)
        tb.run()
        result = snk.data()
        self.assertFloatTuplesAlmostEqual(expected, result)
if __name__ == '__main__':
    gr_unittest.run(qa_matrix_interleaver)