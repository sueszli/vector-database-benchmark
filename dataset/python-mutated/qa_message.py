import time
from gnuradio import gr, gr_unittest, blocks
import pmt

class test_message(gr_unittest.TestCase):

    def test_100(self):
        if False:
            while True:
                i = 10
        msg = gr.message(0, 1.5, 2.3)
        self.assertEqual(0, msg.type())
        self.assertAlmostEqual(1.5, msg.arg1())
        self.assertAlmostEqual(2.3, msg.arg2())
        self.assertEqual(0, msg.length())

    def test_101(self):
        if False:
            print('Hello World!')
        s = b'This is a test'
        msg = gr.message_from_string(s.decode('utf8'))
        self.assertEqual(s, msg.to_string())

    def test_102_unicodechars(self):
        if False:
            for i in range(10):
                print('nop')
        s = u'(╯°□°)╯︵ ┻━┻'
        msg = gr.message_from_string(s)
        self.assertEqual(s.encode('utf8'), msg.to_string())

    def body_202(self):
        if False:
            i = 10
            return i + 15
        msg = gr.message(666)

    def test_300(self):
        if False:
            i = 10
            return i + 15
        input_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        src = blocks.vector_source_b(input_data)
        dst = blocks.vector_sink_b()
        tb = gr.top_block()
        tb.connect(src, dst)
        tb.run()
        self.assertEqual(input_data, dst.data())

    def test_debug_401(self):
        if False:
            for i in range(10):
                print('nop')
        msg = pmt.intern('TESTING')
        src = blocks.message_strobe(msg, 500)
        snk = blocks.message_debug()
        tb = gr.top_block()
        tb.msg_connect(src, 'strobe', snk, 'store')
        tb.start()
        time.sleep(1)
        tb.stop()
        tb.wait()
        rec_msg = snk.get_message(0)
        self.assertTrue(pmt.eqv(rec_msg, msg))
if __name__ == '__main__':
    gr_unittest.run(test_message)