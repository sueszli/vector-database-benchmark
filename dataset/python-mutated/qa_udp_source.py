from gnuradio import gr, gr_unittest, blocks, network
import time

class qa_udp_source(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_restart(self):
        if False:
            while True:
                i = 10
        udp_source = network.udp_source(gr.sizeof_gr_complex, 1, 1234, 0, 1472, False, False, False)
        null_sink = blocks.null_sink(gr.sizeof_gr_complex)
        self.tb.connect(udp_source, null_sink)
        self.tb.start()
        time.sleep(0.1)
        self.tb.stop()
        time.sleep(0.1)
        self.tb.start()
        time.sleep(0.1)
        self.tb.stop()
if __name__ == '__main__':
    gr_unittest.run(qa_udp_source)