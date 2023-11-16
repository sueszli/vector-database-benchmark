import random
from gnuradio import gr, gr_unittest, blocks, digital
from gnuradio.digital.utils import tagged_streams
import pmt

class qa_packet_headerparser_b(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        random.seed(0)
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tb = None

    def test_001_t(self):
        if False:
            while True:
                i = 10
        '\n        First header: Packet length 4, packet num 0\n        Second header: Packet 2, packet num 1\n        Third header: Invalid (CRC does not check) (would be len 4, num 2)\n        '
        encoded_headers = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)
        packet_len_tagname = 'packet_len'
        random_tag = gr.tag_t()
        random_tag.offset = 5
        random_tag.key = pmt.string_to_symbol('foo')
        random_tag.value = pmt.from_long(42)
        src = blocks.vector_source_b(encoded_headers, tags=(random_tag,))
        parser = digital.packet_headerparser_b(32, packet_len_tagname)
        sink = blocks.message_debug()
        self.tb.connect(src, parser)
        self.tb.msg_connect(parser, 'header_data', sink, 'store')
        self.tb.start()
        self.waitFor(lambda : sink.num_messages() == 3)
        self.tb.stop()
        self.tb.wait()
        self.assertEqual(sink.num_messages(), 3)
        msg1 = pmt.to_python(sink.get_message(0))
        msg2 = pmt.to_python(sink.get_message(1))
        msg3 = pmt.to_python(sink.get_message(2))
        self.assertEqual(msg1, {'packet_len': 4, 'packet_num': 0, 'foo': 42})
        self.assertEqual(msg2, {'packet_len': 2, 'packet_num': 1})
        self.assertEqual(msg3, False)

    def test_002_pipe(self):
        if False:
            while True:
                i = 10
        '\n        Create N packets of random length, pipe them through header generator,\n        back to header parser, make sure output is the same.\n        '
        N = 20
        header_len = 32
        packet_len_tagname = 'packet_len'
        packet_lengths = [random.randint(1, 100) for x in range(N)]
        (data, tags) = tagged_streams.packets_to_vectors([list(range(packet_lengths[i])) for i in range(N)], packet_len_tagname)
        src = blocks.vector_source_b(data, False, 1, tags)
        header_gen = digital.packet_headergenerator_bb(header_len, packet_len_tagname)
        header_parser = digital.packet_headerparser_b(header_len, packet_len_tagname)
        sink = blocks.message_debug()
        self.tb.connect(src, header_gen, header_parser)
        self.tb.msg_connect(header_parser, 'header_data', sink, 'store')
        self.tb.start()
        self.waitFor(lambda : sink.num_messages() == N)
        self.tb.stop()
        self.tb.wait()
        self.assertEqual(sink.num_messages(), N)
        for i in range(N):
            msg = pmt.to_python(sink.get_message(i))
            self.assertEqual(msg, {'packet_len': packet_lengths[i], 'packet_num': i})

    def test_003_ofdm(self):
        if False:
            i = 10
            return i + 15
        ' Header 1: 193 bytes\n        Header 2: 8 bytes\n        2 bits per complex symbol, 32 carriers => 64 bits = 8 bytes per OFDM symbol\n                                    4 carriers =>  8 bits = 1 byte  per OFDM symbol\n                                    8 carriers => 16 bits = 2 bytes per OFDM symbol\n        Means we need 52 carriers to store the 193 bytes.\n        '
        encoded_headers = (1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0)
        packet_len_tagname = 'packet_len'
        frame_len_tagname = 'frame_len'
        src = blocks.vector_source_b(encoded_headers)
        header_formatter = digital.packet_header_ofdm((list(range(32)), list(range(4)), list(range(8))), 1, packet_len_tagname, frame_len_tagname, 'packet_num', 1, 2)
        parser = digital.packet_headerparser_b(header_formatter.base())
        sink = blocks.message_debug()
        self.tb.connect(src, parser)
        self.tb.msg_connect(parser, 'header_data', sink, 'store')
        self.tb.start()
        self.waitFor(lambda : sink.num_messages() == 2)
        self.tb.stop()
        self.tb.wait()
        self.assertEqual(sink.num_messages(), 2)
        msg1 = pmt.to_python(sink.get_message(0))
        msg2 = pmt.to_python(sink.get_message(1))
        self.assertEqual(msg1, {'packet_len': 193 * 4, 'frame_len': 52, 'packet_num': 0})
        self.assertEqual(msg2, {'packet_len': 8 * 4, 'frame_len': 1, 'packet_num': 1})

    def test_004_ofdm_scramble(self):
        if False:
            while True:
                i = 10
        '\n        Test scrambling for OFDM header gen\n        '
        header_len = 32
        packet_length = 23
        packet_len_tagname = 'packet_len'
        frame_len_tagname = 'frame_len'
        (data, tags) = tagged_streams.packets_to_vectors([list(range(packet_length)), list(range(packet_length))], packet_len_tagname)
        src = blocks.vector_source_b(data, False, 1, tags)
        header_formatter = digital.packet_header_ofdm((list(range(32)),), 1, packet_len_tagname, frame_len_tagname, 'packet_num', 1, 2, scramble_header=True)
        header_gen = digital.packet_headergenerator_bb(header_formatter.base())
        header_parser = digital.packet_headerparser_b(header_formatter.base())
        sink = blocks.message_debug()
        self.tb.connect(src, header_gen, header_parser)
        self.tb.msg_connect(header_parser, 'header_data', sink, 'store')
        self.tb.start()
        self.waitFor(lambda : sink.num_messages() == 2)
        self.tb.stop()
        self.tb.wait()
        msg = pmt.to_python(sink.get_message(0))
        self.assertEqual(msg, {'packet_len': packet_length, 'packet_num': 0, 'frame_len': 4})
        msg = pmt.to_python(sink.get_message(1))
        self.assertEqual(msg, {'packet_len': packet_length, 'packet_num': 1, 'frame_len': 4})
if __name__ == '__main__':
    gr_unittest.run(qa_packet_headerparser_b)