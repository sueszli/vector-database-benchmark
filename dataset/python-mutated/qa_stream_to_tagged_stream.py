from gnuradio import gr, gr_unittest
from gnuradio import blocks

class qa_stream_to_tagged_stream(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tb = None

    def test_001_t(self):
        if False:
            i = 10
            return i + 15
        src_data = [1] * 50
        packet_len = 10
        len_tag_key = 'packet_len'
        src = blocks.vector_source_f(src_data, False, 1)
        tagger = blocks.stream_to_tagged_stream(gr.sizeof_float, 1, packet_len, len_tag_key)
        sink = blocks.vector_sink_f()
        self.tb.connect(src, tagger, sink)
        self.tb.run()
        self.assertEqual(sink.data(), src_data)
        tags = [gr.tag_to_python(x) for x in sink.tags()]
        tags = sorted([(x.offset, x.key, x.value) for x in tags])
        expected_tags = [(int(pos), 'packet_len', packet_len) for pos in range(0, 50, 10)]
        self.assertEqual(tags, expected_tags)
if __name__ == '__main__':
    gr_unittest.run(qa_stream_to_tagged_stream)