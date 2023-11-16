from gnuradio import gr, gr_unittest, blocks

class test_logger(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def set_and_assert_log_level(self, block, level, ref=None):
        if False:
            for i in range(10):
                print('nop')
        if ref is None:
            ref = level
        block.set_log_level(level)
        self.assertEqual(block.log_level(), ref)

    def test_log_level_for_block(self):
        if False:
            for i in range(10):
                print('nop')
        ns = blocks.null_source(1)
        self.set_and_assert_log_level(ns, 'debug')
        self.set_and_assert_log_level(ns, 'info')
        self.set_and_assert_log_level(ns, 'notice', 'info')
        self.set_and_assert_log_level(ns, 'warn', 'warning')
        self.set_and_assert_log_level(ns, 'warning')
        self.set_and_assert_log_level(ns, 'error')
        self.set_and_assert_log_level(ns, 'crit', 'critical')
        self.set_and_assert_log_level(ns, 'critical')
        self.set_and_assert_log_level(ns, 'alert', 'critical')
        self.set_and_assert_log_level(ns, 'emerg', 'critical')
        ns.set_log_level('off')
        self.assertEqual(ns.log_level(), 'off')

    def test_log_level_for_tb(self):
        if False:
            print('Hello World!')
        nsrc = blocks.null_source(4)
        nsnk = blocks.null_sink(4)
        nsrc.set_log_level('debug')
        nsnk.set_log_level('debug')
        tb = gr.top_block()
        tb.connect(nsrc, nsnk)
        self.assertEqual(tb.log_level(), 'debug')
        tb.set_log_level('critical')
        self.assertEqual(tb.log_level(), 'critical')
        self.assertEqual(nsrc.log_level(), 'critical')
        self.assertEqual(nsnk.log_level(), 'critical')

    def test_log_level_for_hier_block(self):
        if False:
            return 10
        nsrc = blocks.null_source(4)
        nsnk = blocks.null_sink(4)
        b = blocks.stream_to_vector_decimator(4, 1, 1, 1)
        tb = gr.top_block()
        tb.connect(nsrc, b, nsnk)
        tb.set_log_level('debug')
        self.assertEqual(tb.log_level(), 'debug')
        self.assertEqual(nsrc.log_level(), 'debug')
        self.assertEqual(nsnk.log_level(), 'debug')
        self.assertEqual(b.one_in_n.log_level(), 'debug')
        tb.set_log_level('critical')
        self.assertEqual(tb.log_level(), 'critical')
        self.assertEqual(nsrc.log_level(), 'critical')
        self.assertEqual(nsnk.log_level(), 'critical')
        self.assertEqual(b.one_in_n.log_level(), 'critical')
if __name__ == '__main__':
    gr_unittest.run(test_logger)