"""
gr-uhd sanity checking
"""
from gnuradio import gr, gr_unittest, uhd

class test_uhd(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_000_nop(self):
        if False:
            i = 10
            return i + 15
        "Just see if we can import the module...\n        They may not have a UHD device connected, etc.  Don't try to run anything"
        pass

    def test_time_spec_t(self):
        if False:
            while True:
                i = 10
        seconds = 42.0
        time = uhd.time_spec_t(seconds)
        twice_time = time + time
        zero_time = time - time
        self.assertEqual(time.get_real_secs() * 2, seconds * 2)
        self.assertEqual(time.get_real_secs() - time.get_real_secs(), 0.0)

    def test_stream_args_channel_foo(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Try to manipulate the stream args channels for proper swig'ing checks.\n        "
        sa = uhd.stream_args_t()
        sa.channels = [1, 0]
        print(sa.channels)
        self.assertEqual(len(sa.channels), 2)
        self.assertEqual(sa.channels[0], 1)
        self.assertEqual(sa.channels[1], 0)
if __name__ == '__main__':
    gr_unittest.run(test_uhd)