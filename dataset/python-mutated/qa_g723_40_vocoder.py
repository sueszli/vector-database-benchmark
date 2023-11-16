from gnuradio import gr, gr_unittest, vocoder, blocks

class test_g723_40_vocoder(gr_unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tb = None

    def test001_module_load(self):
        if False:
            return 10
        data = (0, 8, 36, 72, 100, 152, 228, 316, 404, 528)
        src = blocks.vector_source_s(data)
        enc = vocoder.g723_40_encode_sb()
        dec = vocoder.g723_40_decode_bs()
        snk = blocks.vector_sink_s()
        self.tb.connect(src, enc, dec, snk)
        self.tb.run()
        actual_result = snk.data()
        self.assertEqual(list(data), actual_result)
if __name__ == '__main__':
    gr_unittest.run(test_g723_40_vocoder)