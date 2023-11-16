from gnuradio import gr, gr_unittest, blocks
import math

class test_null_sink_source(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
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
        src = blocks.null_source(gr.sizeof_float)
        hed = blocks.head(gr.sizeof_float, 100)
        dst = blocks.null_sink(gr.sizeof_float)
        self.tb.connect(src, hed, dst)
        self.tb.run()
if __name__ == '__main__':
    gr_unittest.run(test_null_sink_source)