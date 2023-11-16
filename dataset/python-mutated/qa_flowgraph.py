from gnuradio import gr, gr_unittest

class test_flowgraph(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_000_empty_fg(self):
        if False:
            i = 10
            return i + 15
        self.tb.start()
        self.tb.stop()
if __name__ == '__main__':
    gr_unittest.run(test_flowgraph)