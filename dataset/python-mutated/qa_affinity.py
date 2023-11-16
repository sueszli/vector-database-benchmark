from gnuradio import gr, gr_unittest, blocks

class test_affinity(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_000(self):
        if False:
            print('Hello World!')
        src_data = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        src = blocks.vector_source_f(src_data)
        snk = blocks.vector_sink_f()
        src.set_processor_affinity([0])
        self.tb.connect(src, snk)
        self.tb.run()
        a = src.processor_affinity()
        self.assertEqual([0], a)
if __name__ == '__main__':
    gr_unittest.run(test_affinity)