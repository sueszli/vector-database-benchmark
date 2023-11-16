from gnuradio import gr, gr_unittest, blocks, pdu
from gnuradio import network
import random
import pmt
import time

class qa_socket_pdu(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        random.seed(0)
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test_001(self):
        if False:
            i = 10
            return i + 15
        port = str(random.Random().randint(0, 30000) + 10000)
        self.pdu_send = network.socket_pdu('UDP_CLIENT', 'localhost', port)
        self.pdu_recv = network.socket_pdu('UDP_SERVER', 'localhost', port)
        self.pdu_send = None
        self.pdu_recv = None

    def test_002(self):
        if False:
            while True:
                i = 10
        port = str(random.Random().randint(0, 30000) + 10000)
        srcdata = (100, 111, 103, 101)
        data = pmt.init_u8vector(srcdata.__len__(), srcdata)
        pdu_msg = pmt.cons(pmt.PMT_NIL, data)
        self.pdu_source = blocks.message_strobe(pdu_msg, 500)
        self.pdu_recv = network.socket_pdu('UDP_SERVER', 'localhost', port)
        self.pdu_send = network.socket_pdu('UDP_CLIENT', 'localhost', port)
        self.dbg = blocks.message_debug()
        self.tb.msg_connect(self.pdu_source, 'strobe', self.pdu_send, 'pdus')
        self.tb.msg_connect(self.pdu_recv, 'pdus', self.dbg, 'store')
        self.tb.start()
        time.sleep(1)
        self.tb.stop()
        self.tb.wait()
        self.pdu_send = None
        self.pdu_recv = None
        received = self.dbg.get_message(0)
        received_data = pmt.cdr(received)
        msg_data = []
        for i in range(4):
            msg_data.append(pmt.u8vector_ref(received_data, i))
        self.assertEqual(srcdata, tuple(msg_data))

    def test_003(self):
        if False:
            while True:
                i = 10
        port = str(random.Random().randint(0, 30000) + 10000)
        srcdata = (115, 117, 99, 104, 116, 101, 115, 116, 118, 101, 114, 121, 112, 97, 115, 115)
        tag_dict = {'offset': 0}
        tag_dict['key'] = pmt.intern('len')
        tag_dict['value'] = pmt.from_long(8)
        tag1 = gr.python_to_tag(tag_dict)
        tag_dict['offset'] = 8
        tag2 = gr.python_to_tag(tag_dict)
        tags = [tag1, tag2]
        src = blocks.vector_source_b(srcdata, False, 1, tags)
        ts_to_pdu = pdu.tagged_stream_to_pdu(gr.types.byte_t, 'len')
        pdu_send = network.socket_pdu('UDP_CLIENT', 'localhost', '4141')
        pdu_to_ts = pdu.pdu_to_tagged_stream(gr.types.byte_t, 'len')
        head = blocks.head(gr.sizeof_char, 10)
        sink = blocks.vector_sink_b(1)
        self.tb.connect(src, ts_to_pdu)
        self.tb.msg_connect(ts_to_pdu, 'pdus', pdu_send, 'pdus')
        self.tb.run()

    def test_004(self):
        if False:
            i = 10
            return i + 15
        port = str(random.Random().randint(0, 30000) + 10000)
        mtu = 10000
        srcdata = tuple((x % 256 for x in range(mtu)))
        data = pmt.init_u8vector(srcdata.__len__(), srcdata)
        pdu_msg = pmt.cons(pmt.PMT_NIL, data)
        self.pdu_source = blocks.message_strobe(pdu_msg, 500)
        self.pdu_send = network.socket_pdu('TCP_SERVER', 'localhost', port, mtu)
        self.pdu_recv = network.socket_pdu('TCP_CLIENT', 'localhost', port, mtu)
        self.pdu_sink = blocks.message_debug()
        self.tb.msg_connect(self.pdu_source, 'strobe', self.pdu_send, 'pdus')
        self.tb.msg_connect(self.pdu_recv, 'pdus', self.pdu_sink, 'store')
        self.tb.start()
        time.sleep(1)
        self.tb.stop()
        self.tb.wait()
        received = self.pdu_sink.get_message(0)
        received_data = pmt.cdr(received)
        msg_data = []
        for i in range(mtu):
            msg_data.append(pmt.u8vector_ref(received_data, i))
        self.assertEqual(srcdata, tuple(msg_data))
if __name__ == '__main__':
    gr_unittest.run(qa_socket_pdu)