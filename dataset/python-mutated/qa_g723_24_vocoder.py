from gnuradio import gr, gr_unittest, vocoder, blocks

class test_g723_24_vocoder(gr_unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tb = gr.top_block()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def test001_module_load(self):
        if False:
            while True:
                i = 10
        data = (0, 16, 36, 40, 68, 104, 148, 220, 320, 512)
        src = blocks.vector_source_s(data)
        enc = vocoder.g723_24_encode_sb()
        dec = vocoder.g723_24_decode_bs()
        snk = blocks.vector_sink_s()
        self.tb.connect(src, enc, dec, snk)
        self.tb.run()
        actual_result = snk.data()
        self.assertEqual(list(data), actual_result)
if __name__ == '__main__':
    gr_unittest.run(test_g723_24_vocoder)