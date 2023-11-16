from gnuradio import gr, gr_unittest, digital, blocks
import pmt
import numpy
from time import sleep

class test_hdlc_framer(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_001(self):
        if False:
            while True:
                i = 10
        npkts = 20
        src_data = [254, 218, 172, 41, 127, 162, 144, 15, 248]
        frame = digital.hdlc_framer_pb('wat')
        deframe = digital.hdlc_deframer_bp(8, 500)
        debug = blocks.message_debug()
        self.tb.connect(frame, deframe)
        self.tb.msg_connect(deframe, 'out', debug, 'store')
        self.tb.start()
        msg = pmt.cons(pmt.PMT_NIL, pmt.init_u8vector(len(src_data), src_data))
        for i in range(npkts):
            frame.to_basic_block()._post(pmt.intern('in'), msg)
        sleep(0.2)
        self.tb.stop()
        self.tb.wait()
        rxmsg = debug.get_message(0)
        result_len = pmt.blob_length(pmt.cdr(rxmsg))
        msg_data = []
        for j in range(result_len):
            msg_data.append(pmt.u8vector_ref(pmt.cdr(rxmsg), j))
        self.assertEqual(src_data, msg_data)
if __name__ == '__main__':
    gr_unittest.run(test_hdlc_framer)