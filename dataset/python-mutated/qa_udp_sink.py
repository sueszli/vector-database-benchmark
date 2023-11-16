from gnuradio import gr, gr_unittest, blocks, network
import time

class qa_udp_sink(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_restart(self):
        if False:
            print('Hello World!')
        null_source = blocks.null_source(gr.sizeof_gr_complex)
        throttle = blocks.throttle(gr.sizeof_gr_complex, 320000, True)
        udp_sink = network.udp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 2000, 0, 1472, False)
        self.tb.connect(null_source, throttle, udp_sink)
        self.tb.start()
        time.sleep(0.1)
        self.tb.stop()
        time.sleep(0.1)
        self.tb.start()
        time.sleep(0.1)
        self.tb.stop()
if __name__ == '__main__':
    gr_unittest.run(qa_udp_sink)