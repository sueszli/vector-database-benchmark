from gnuradio import gr, gr_unittest
from gnuradio import blocks
import pmt
import time

class qa_message_strobe(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test_001_t(self):
        if False:
            i = 10
            return i + 15
        test_str = 'test_msg'
        new_msg = 'new_msg'
        message_period_ms = 100
        msg_strobe = blocks.message_strobe(pmt.intern(test_str), message_period_ms)
        msg_debug = blocks.message_debug()
        self.tb.msg_connect(msg_strobe, 'strobe', msg_debug, 'store')
        self.tb.start()
        self.assertAlmostEqual(msg_debug.num_messages(), 0, delta=2)
        time.sleep(1.05)
        self.assertAlmostEqual(msg_debug.num_messages(), 10, delta=8)
        time.sleep(1)
        self.assertAlmostEqual(msg_debug.num_messages(), 20, delta=10)
        msg_strobe.to_basic_block()._post(pmt.intern('set_msg'), pmt.intern(new_msg))
        time.sleep(1)
        self.tb.stop()
        self.tb.wait()
        self.assertAlmostEqual(pmt.to_python(msg_debug.get_message(0)), test_str, 'mismatch initial test string')
        no_msgs = msg_debug.num_messages()
        self.assertAlmostEqual(pmt.to_python(msg_debug.get_message(no_msgs - 1)), new_msg, 'failed to update string')
if __name__ == '__main__':
    gr_unittest.run(qa_message_strobe)