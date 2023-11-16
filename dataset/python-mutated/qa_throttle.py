import time
import pmt
from gnuradio import gr, gr_unittest, blocks

class test_throttle(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            return 10
        self.tb = None

    def test_throttling(self):
        if False:
            return 10
        src_data = list(range(200))
        src = blocks.vector_source_c(src_data)
        thr = blocks.throttle(gr.sizeof_gr_complex, 100)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, thr, dst)
        start_time = time.perf_counter()
        self.tb.run()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        self.assertGreater(total_time, 2.0)
        self.assertLess(total_time, 3.0)
        dst_data = dst.data()
        self.assertEqual(src_data, dst_data)

    def test_rx_rate_tag(self):
        if False:
            while True:
                i = 10
        src_data = list(range(400))
        tag = gr.tag_t()
        tag.key = pmt.string_to_symbol('rx_rate')
        tag.value = pmt.to_pmt(200)
        tag.offset = 0
        src = blocks.vector_source_c(src_data, tags=(tag,))
        thr = blocks.throttle(gr.sizeof_gr_complex, 100, ignore_tags=False)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, thr, dst)
        start_time = time.perf_counter()
        self.tb.run()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        self.assertGreater(total_time, 2.0)
        self.assertLess(total_time, 3.0)
        dst_data = dst.data()
        self.assertEqual(src_data, dst_data)

    def test_limited_chunk_throttling(self):
        if False:
            print('Hello World!')
        src_data = [1, 2, 3]
        rate = 50
        chunksize = 10
        src = blocks.vector_source_c(src_data, repeat=True)
        thr = blocks.throttle(gr.sizeof_gr_complex, rate, ignore_tags=True, maximum_items_per_chunk=chunksize)
        dst = blocks.vector_sink_c()
        self.tb.connect(src, thr, dst)
        total_time = 0.5
        self.tb.start()
        time.sleep(total_time)
        self.tb.stop()
        dst_data = dst.data()
        num = len(dst_data)
        self.assertLess(abs(total_time * rate - num) + 1, chunksize)
if __name__ == '__main__':
    gr_unittest.run(test_throttle)