from gnuradio import gr, gr_unittest, blocks
import pmt
import math

class test_tags_strobe(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_001(self):
        if False:
            return 10
        N = 10000
        nsamps = 1000
        ntags = N / nsamps - 1
        src = blocks.tags_strobe(gr.sizeof_float, pmt.intern('TEST'), nsamps)
        hed = blocks.head(gr.sizeof_float, N)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, hed, dst)
        self.tb.run()
        self.assertEqual(ntags, len(dst.tags()))
        n_expected = nsamps
        for tag in dst.tags():
            self.assertEqual(tag.offset, n_expected)
            n_expected += nsamps

    def test_002(self):
        if False:
            print('Hello World!')
        N = 10000
        nsamps = 123
        ntags = N // nsamps
        src = blocks.tags_strobe(gr.sizeof_float, pmt.intern('TEST'), nsamps)
        hed = blocks.head(gr.sizeof_float, N)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, hed, dst)
        self.tb.run()
        self.assertEqual(ntags, len(dst.tags()))
        n_expected = nsamps
        for tag in dst.tags():
            self.assertEqual(tag.offset, n_expected)
            n_expected += nsamps

    def test_003(self):
        if False:
            i = 10
            return i + 15
        N = 100000
        nsamps = 10000
        ntags = N / nsamps - 1
        src = blocks.tags_strobe(gr.sizeof_float, pmt.intern('TEST'), nsamps)
        hed = blocks.head(gr.sizeof_float, N)
        dst = blocks.vector_sink_f()
        self.tb.connect(src, hed, dst)
        self.tb.run()
        self.assertEqual(ntags, len(dst.tags()))
        n_expected = nsamps
        for tag in dst.tags():
            self.assertEqual(tag.offset, n_expected)
            n_expected += nsamps
if __name__ == '__main__':
    gr_unittest.run(test_tags_strobe)