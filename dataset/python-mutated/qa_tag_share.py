from gnuradio import gr, gr_unittest
from gnuradio import blocks
import pmt

class qa_tag_share(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_001_t(self):
        if False:
            for i in range(10):
                print('nop')
        tag_key = 'in1_tag'
        tag_value = 0
        tag_offset = 0
        in0_value = 1.0 + 1j
        in1_value = 2.717
        in0_data = [in0_value] * 10
        in1_data = [in1_value] * 10
        sink_data = in0_data
        tag = gr.tag_t()
        tag.key = pmt.to_pmt(tag_key)
        tag.value = pmt.to_pmt(tag_value)
        tag.offset = tag_offset
        in0 = blocks.vector_source_c(in0_data, False, 1)
        in1 = blocks.vector_source_f(in1_data, False, 1, (tag,))
        tag_share = blocks.tag_share(gr.sizeof_gr_complex, gr.sizeof_float)
        sink = blocks.vector_sink_c(1)
        self.tb.connect(in0, (tag_share, 0))
        self.tb.connect(in1, (tag_share, 1))
        self.tb.connect(tag_share, sink)
        self.tb.run()
        self.assertEqual(len(sink.tags()), 1)
        received_tag = sink.tags()[0]
        self.assertEqual(pmt.to_python(received_tag.key), tag_key)
        self.assertEqual(pmt.to_python(received_tag.value), tag_value)
        self.assertEqual(received_tag.offset, tag_offset)
        self.assertEqual(sink.data(), sink_data)
if __name__ == '__main__':
    gr_unittest.run(qa_tag_share)