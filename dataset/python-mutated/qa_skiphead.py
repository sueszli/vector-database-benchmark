from gnuradio import gr, gr_unittest, blocks
import pmt
import numpy

def make_tag(key, value, offset, srcid=None):
    if False:
        while True:
            i = 10
    tag = gr.tag_t()
    tag.key = pmt.string_to_symbol(key)
    tag.value = pmt.to_pmt(value)
    tag.offset = offset
    if srcid is not None:
        tag.srcid = pmt.to_pmt(srcid)
    return tag

class test_skiphead(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()
        self.src_data = [int(x) for x in range(65536)]

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_skip_0(self):
        if False:
            print('Hello World!')
        skip_cnt = 0
        expected_result = self.src_data[skip_cnt:]
        src1 = blocks.vector_source_i(self.src_data)
        op = blocks.skiphead(gr.sizeof_int, skip_cnt)
        dst1 = blocks.vector_sink_i()
        self.tb.connect(src1, op, dst1)
        self.tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_skip_1(self):
        if False:
            while True:
                i = 10
        skip_cnt = 1
        expected_result = self.src_data[skip_cnt:]
        src1 = blocks.vector_source_i(self.src_data)
        op = blocks.skiphead(gr.sizeof_int, skip_cnt)
        dst1 = blocks.vector_sink_i()
        self.tb.connect(src1, op, dst1)
        self.tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_skip_1023(self):
        if False:
            return 10
        skip_cnt = 1023
        expected_result = self.src_data[skip_cnt:]
        src1 = blocks.vector_source_i(self.src_data)
        op = blocks.skiphead(gr.sizeof_int, skip_cnt)
        dst1 = blocks.vector_sink_i()
        self.tb.connect(src1, op, dst1)
        self.tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_skip_6339(self):
        if False:
            print('Hello World!')
        skip_cnt = 6339
        expected_result = self.src_data[skip_cnt:]
        src1 = blocks.vector_source_i(self.src_data)
        op = blocks.skiphead(gr.sizeof_int, skip_cnt)
        dst1 = blocks.vector_sink_i()
        self.tb.connect(src1, op, dst1)
        self.tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_skip_12678(self):
        if False:
            for i in range(10):
                print('nop')
        skip_cnt = 12678
        expected_result = self.src_data[skip_cnt:]
        src1 = blocks.vector_source_i(self.src_data)
        op = blocks.skiphead(gr.sizeof_int, skip_cnt)
        dst1 = blocks.vector_sink_i()
        self.tb.connect(src1, op, dst1)
        self.tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_skip_all(self):
        if False:
            return 10
        skip_cnt = len(self.src_data)
        expected_result = self.src_data[skip_cnt:]
        src1 = blocks.vector_source_i(self.src_data)
        op = blocks.skiphead(gr.sizeof_int, skip_cnt)
        dst1 = blocks.vector_sink_i()
        self.tb.connect(src1, op, dst1)
        self.tb.run()
        dst_data = dst1.data()
        self.assertEqual(expected_result, dst_data)

    def test_skip_tags(self):
        if False:
            print('Hello World!')
        skip_cnt = 25
        expected_result = self.src_data[skip_cnt:]
        src_tags = tuple([make_tag('foo', 'bar', 1, 'src'), make_tag('baz', 'qux', 50, 'src')])
        src1 = blocks.vector_source_i(self.src_data, tags=src_tags)
        op = blocks.skiphead(gr.sizeof_int, skip_cnt)
        dst1 = blocks.vector_sink_i()
        self.tb.connect(src1, op, dst1)
        self.tb.run()
        dst_data = dst1.data()
        dst_tags = dst1.tags()
        self.assertEqual(expected_result, dst_data)
        self.assertEqual(dst_tags[0].offset, 25, 'Tag offset is incorrect')
        self.assertEqual(len(dst_tags), 1, 'Wrong number of tags received')
        self.assertEqual(pmt.to_python(dst_tags[0].key), 'baz', 'Tag key is incorrect')
        self.assertEqual(pmt.to_python(dst_tags[0].value), 'qux', 'Tag value is incorrect')
if __name__ == '__main__':
    gr_unittest.run(test_skiphead)