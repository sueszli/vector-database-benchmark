from gnuradio import gr, gr_unittest
import pmt

class test_tag_utils(gr_unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test_001(self):
        if False:
            i = 10
            return i + 15
        t = gr.tag_t()
        t.offset = 10
        t.key = pmt.string_to_symbol('key')
        t.value = pmt.from_long(23)
        t.srcid = pmt.from_bool(False)
        pt = gr.tag_to_python(t)
        self.assertEqual(pt.key, 'key')
        self.assertEqual(pt.value, 23)
        self.assertEqual(pt.offset, 10)

    def test_comparison(self):
        if False:
            for i in range(10):
                print('nop')
        t = gr.tag_t()
        t.offset = 10
        t.key = pmt.string_to_symbol('key')
        t.value = pmt.from_long(23)
        t.srcid = pmt.from_bool(False)
        t2 = gr.tag_t()
        t2.offset = 100
        t2.key = pmt.string_to_symbol('aaa')
        t2.value = pmt.from_long(230)
        t2.srcid = pmt.from_bool(True)
        self.assertTrue(t < t2)
        self.assertTrue(t == t)
        self.assertTrue(t != t2)
        self.assertFalse(t > t2)
        self.assertFalse(t < t)

    def test_002(self):
        if False:
            print('Hello World!')
        offset = 10
        key = pmt.string_to_symbol('key')
        value = pmt.from_long(23)
        srcid = pmt.from_bool(False)
        format_dict = {'offset': offset, 'key': key, 'value': value, 'srcid': srcid}
        format_list = [offset, key, value, srcid]
        format_tuple = (offset, key, value, srcid)
        t_dict = gr.python_to_tag(format_dict)
        t_list = gr.python_to_tag(format_list)
        t_tuple = gr.python_to_tag(format_tuple)
        self.assertTrue(pmt.equal(t_dict.key, key))
        self.assertTrue(pmt.equal(t_dict.value, value))
        self.assertEqual(t_dict.offset, offset)
        self.assertTrue(pmt.equal(t_list.key, key))
        self.assertTrue(pmt.equal(t_list.value, value))
        self.assertEqual(t_list.offset, offset)
        self.assertTrue(pmt.equal(t_tuple.key, key))
        self.assertTrue(pmt.equal(t_tuple.value, value))
        self.assertEqual(t_tuple.offset, offset)

    def test_003(self):
        if False:
            print('Hello World!')
        offsets = (6, 3, 8)
        key = pmt.string_to_symbol('key')
        srcid = pmt.string_to_symbol('qa_tag_utils')
        tags = []
        for k in offsets:
            t = gr.tag_t()
            t.offset = k
            t.key = key
            t.value = pmt.from_long(k)
            t.srcid = srcid
            tags.append(t)
        for (k, t) in zip(sorted(offsets), sorted(tags)):
            self.assertEqual(t.offset, k)
            self.assertTrue(pmt.equal(t.key, key))
            self.assertTrue(pmt.equal(t.value, pmt.from_long(k)))
            self.assertTrue(pmt.equal(t.srcid, srcid))
        tmin = min(tags)
        self.assertEqual(tmin.offset, min(offsets))
        self.assertTrue(pmt.equal(tmin.key, key))
        self.assertTrue(pmt.equal(tmin.value, pmt.from_long(min(offsets))))
        self.assertTrue(pmt.equal(tmin.srcid, srcid))
        tmax = max(tags)
        self.assertEqual(tmax.offset, max(offsets))
        self.assertTrue(pmt.equal(tmax.key, key))
        self.assertTrue(pmt.equal(tmax.value, pmt.from_long(max(offsets))))
        self.assertTrue(pmt.equal(tmax.srcid, srcid))
if __name__ == '__main__':
    gr_unittest.run(test_tag_utils)