from gnuradio import gr, gr_unittest, blocks

class test_block_behavior(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_000(self):
        if False:
            while True:
                i = 10
        '\n        Tests the max noutput sizes set by the scheduler. When creating\n        the block, there is no block_detail and so the max buffer size\n        is 0. When the top_block is run, it builds the detail and\n        buffers and sets the max value. test_0001 tests when the\n        max_noutput_items is set by hand.\n\n        '
        src = blocks.null_source(gr.sizeof_float)
        op = blocks.head(gr.sizeof_float, 100)
        snk = blocks.null_sink(gr.sizeof_float)
        maxn_pre = op.max_noutput_items()
        self.tb.connect(src, op, snk)
        self.tb.run()
        maxn_post = op.max_noutput_items()
        self.assertEqual(maxn_pre, 0)
        self.assertEqual(maxn_post, 16384)

    def test_001(self):
        if False:
            print('Hello World!')
        '\n        Tests the max noutput size when being explicitly set.\n        '
        src = blocks.null_source(gr.sizeof_float)
        op = blocks.head(gr.sizeof_float, 100)
        snk = blocks.null_sink(gr.sizeof_float)
        op.set_max_noutput_items(1024)
        maxn_pre = op.max_noutput_items()
        self.tb.connect(src, op, snk)
        self.tb.run()
        maxn_post = op.max_noutput_items()
        self.assertEqual(maxn_pre, 1024)
        self.assertEqual(maxn_post, 1024)
if __name__ == '__main__':
    gr_unittest.run(test_block_behavior)